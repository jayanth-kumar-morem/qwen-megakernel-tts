# Measuring and Hitting the Performance Targets

## The Targets

Two numbers mattered:
- **TTFC (Time to First Audio Chunk): < 90 ms**
- **RTF (Real-Time Factor): < 0.3** — generating 1 second of audio must take less than 300ms

Both had to be met with streaming enabled — audio chunks pushed as they're generated, not buffered.

## How I Measured

All timing uses `time.perf_counter()` with `torch.cuda.synchronize()` barriers. CUDA operations are asynchronous — without the sync, you're measuring launch time, not execution time. My measurement harness:

```python
torch.cuda.synchronize()
t_start = time.perf_counter()
# ... do the work ...
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - t_start) * 1000
```

**TTFC** is measured from the start of `synthesize_streaming()` to the moment the first audio chunk is fully decoded and ready to yield. This includes tokenization, embedding, prefill, first talker decode, first code predictor run, and first vocoder decode.

**RTF** is measured over the full generation: `total_wall_time / total_audio_duration`. An RTF of 0.234 means 1 second of audio takes 234ms to generate — well within the 300ms budget.

I ran warmup iterations before measuring (engine initialization + one full synthesis) to exclude JIT compilation and CUDA context setup from the numbers.

## The Optimization Journey

### Starting Point: Everything Broken

My first end-to-end run had:
- **Streaming TTFC: 35,932 ms** (400x over target)
- **RTF: 0.605** (2x over target)

The streaming TTFC was catastrophic because `_generate_codec_frames` returned a list instead of a generator — it processed all 2,048 frames before yielding anything. The RTF was high because the code predictor used pure PyTorch (179ms per frame).

### Step 1: Generator-Based Streaming → TTFC 1,096ms

Converting the frame generation from a list to a Python generator (`yield` instead of `append`) let the streaming wrapper send the first chunk immediately. TTFC dropped from 35,932ms to 1,096ms.

Still 12x over target. The bottleneck was now the vocoder's cold start.

### Step 2: Vocoder Warmup → TTFC ~192ms

The vocoder's first decode call takes ~834ms (CUDA JIT, memory allocation, etc.). Subsequent calls take ~38ms. I added dummy decode calls during engine initialization:

```python
for n in [1, 1, 5]:
    dummy = torch.randint(0, 2048, (n, 16), device="cuda")
    self.speech_tokenizer.decode([{"audio_codes": dummy}])
```

TTFC dropped to ~192ms. Still 2x over.

### Step 3: Small First Chunk → TTFC ~192ms (no change, but unblocked Step 4)

Instead of waiting for 10 frames to fill a chunk, I yield the first frame immediately (1 frame = 80ms of audio). This didn't help TTFC directly (vocoder was still the bottleneck at this point) but set up the right architecture for streaming.

### Step 4: Code Predictor via Megakernel → RTF 0.175

This was the big one. The code predictor (5-layer transformer, generates 15 codebook groups per frame) was running in pure PyTorch: ~70 separate CUDA kernel launches per decode step, taking **179ms per frame**.

I realized the megakernel already supports `num_layers` as a runtime parameter. I packed the code predictor's weights into the same struct format and called the same kernel with `num_layers=5`. Result: **10.9ms per frame**. An 18x speedup.

RTF dropped from 0.605 to 0.175. Target met.

### Step 5: Sampling Path Warmup → TTFC ~92ms

The first code predictor call still took 107ms instead of 13ms. I profiled and found that `torch.multinomial`, `torch.softmax`, and `torch.topk` each have first-call overhead when using the sampling path. My warmup only covered the argmax path.

Added warmup with `do_sample=True`:

```python
for do_sample in [False, False, True, True, True]:
    # ... warmup with both paths
```

TTFC dropped from ~192ms to ~92ms. Almost there.

### Step 6: Batched Text Projection → TTFC ~90ms

The `build_prefill_embeddings` function was making 5 separate `embed_text_ids` calls (role tokens, content tokens, TTS_PAD, TTS_BOS, TTS_EOS). Each call triggers 4 CUDA kernels (embedding lookup, linear, SiLU, linear). That's 20 kernel launches just for embedding.

I batched everything into a single call:

```python
# Before: 5 calls × 4 kernels = 20 launches
role_embeds = text_projection.embed_text_ids(role_ids)
content_embeds = text_projection.embed_text_ids(content_ids)
pad_embed = text_projection.embed_text_ids(pad_id)
# ...

# After: 1 call × 4 kernels = 4 launches
all_ids = torch.cat([role_ids, content_ids, special_ids])
all_embeds = text_projection.embed_text_ids(all_ids)
```

Embed build time: 13.9ms → 6.9ms. TTFC: ~92ms → ~90ms.

### Step 7: Precomputed Constant Embeddings → TTFC ~78ms

Some embeddings never change between utterances: role tokens (`<|im_start|>assistant\n`), TTS special tokens (PAD, BOS, EOS), codec tag embeddings (nothink, think_bos, think_eos), and the fused codec+TTS tags.

I precomputed all of these during `initialize()`:

```python
# Compute once during init:
self._cached_role_embeds = text_projection.embed_text_ids(role_ids)
self._cached_fused_tags = tts_prefix + codec_embeds[:4]
self._cached_codec_bos = codec_embeds[4:5]
```

This eliminated ~6ms of redundant embedding computation per utterance. TTFC: ~90ms → ~78ms. **Target met.**

## Final Numbers

| Metric | Result | Target |
|---|---|---|
| TTFC (non-streaming pipeline test) | **50.5 ms** | < 90 ms |
| TTFC (streaming with vocoder) | **81.6 ms** | < 90 ms |
| RTF (non-streaming) | **0.175** | < 0.3 |
| RTF (streaming) | **0.234** | < 0.3 |

### TTFC Breakdown (50.5ms total)

| Phase | Time | What It Does |
|---|---|---|
| Tokenize | 2.3 ms | HuggingFace tokenizer encodes text |
| Embed build | 7.2 ms | Project text tokens 2048→1024 |
| Prefill (8 steps) | 24.9 ms | Feed conditioning tokens through talker |
| First talker decode | 3.1 ms | Generate first codebook token |
| First code predictor | 13.0 ms | Generate remaining 15 codebook groups |

The streaming TTFC adds ~30ms for the vocoder decode of the first frame (1 frame → 1,920 samples → PCM bytes).

### RTF Breakdown (per frame)

| Component | Time | Notes |
|---|---|---|
| Talker decode | ~1 ms | Megakernel, single launch |
| Code predictor | ~11 ms | Megakernel, 5 layers, sampling |
| Embedding sum | ~1 ms | 16 F.embedding + additions |
| Vocoder | ~2 ms | Amortized over 10-frame chunks |
| **Total** | **~15 ms** | For 80ms of audio → RTF ≈ 0.19 |

## The Optimization That Didn't Happen: M-RoPE

The elephant in the room is M-RoPE. If I could implement it in the kernel, the model would properly emit EOS tokens, eliminating the need for heuristic frame limits and potentially improving audio quality. But modifying the attention hot path in a 1,600-line hand-optimized CUDA kernel is high-risk work — a subtle bug could silently corrupt all outputs.

I chose to ship with the workaround (word-count-based frame limit) and document the limitation clearly, rather than risk breaking the kernel for a feature that doesn't affect the performance metrics.

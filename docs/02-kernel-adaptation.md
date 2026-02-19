# Adapting the Megakernel for Qwen3-TTS

## Understanding What Needed to Change

The first thing I did was study both codebases side by side: AlpinDale's megakernel (targeting Qwen3-0.6B text generation) and the Qwen3-TTS model architecture.

The good news hit me almost immediately — the talker decoder inside Qwen3-TTS **is** a Qwen3-0.6B model. Same hidden size (1024), same number of layers (28), same number of attention heads (16 query, 8 KV), same head dimension (128), same intermediate size (3072). The transformer backbone is identical.

The differences turned out to be small but critical:

| Parameter | Qwen3-0.6B (text) | Qwen3-TTS (talker) |
|---|---|---|
| Vocab size | 151,936 | 3,072 |
| RoPE theta | 10,000 | 1,000,000 |
| LM head | Tied to embeddings | Separate `codec_head` |
| RoPE type | Standard | M-RoPE (mrope_section: [24, 20, 20]) |

## Change 1: Vocab Size (The Easy Win)

The text model has a 151K vocabulary. The TTS talker uses a 3,072-token codec vocabulary. This means the LM head (the matrix multiply that converts hidden states to logits over the vocabulary) is **48x smaller**.

In the kernel, the LM head runs as a separate kernel launch after the main transformer. It uses `LDG_LM_NUM_BLOCKS` thread blocks to scan the vocabulary. For 151K tokens, AlpinDale used 1,280 blocks. For 3,072 tokens, I only needed 16.

I created `build_tts.py` (a modified version of `build.py`) that passes these compile-time constants:

```python
"-DLDG_VOCAB_SIZE=3072",      # was 151936
"-DLDG_LM_NUM_BLOCKS=16",     # was 1280
```

That's it. No kernel code changes needed for the vocab size — the kernel reads `LDG_VOCAB_SIZE` from the compile-time define.

## Change 2: The Embedding Sentinel (The Clever Hack)

This is where things got interesting. In standard text generation, the kernel's input is a **token ID** — it looks up the embedding from a table. But in TTS, the input to each decode step is a **sum of embeddings**: all 16 codebook embeddings from the previous frame, plus a trailing text embedding. There's no single token ID to look up.

I had two options:
1. Add a separate embedding kernel launch before each decode step (extra launch overhead)
2. Patch the kernel to optionally read from a precomputed buffer

I went with option 2. The patch was exactly 3 lines in `kernel.cu`:

```c
// Sentinel: if token_id < 0, use hidden_buffer (precomputed embedding)
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;
```

When Python passes `token_id = -1`, the kernel skips the embedding table lookup and reads directly from the `hidden_buffer` (where Python has already written the summed embedding). When `token_id >= 0`, it works exactly as before. Zero overhead, fully backward-compatible.

This enabled a clean Python API:

```python
# Standard decode (codec token lookup):
next_token, hidden = talker.step(token_id=2149)  # CODEC_BOS

# Precomputed embedding (sum of codec embeds + text):
next_token, hidden = talker.step_with_embed(summed_embedding)
```

## Change 3: Runtime num_layers (Already There)

This was the discovery that made the biggest performance difference, and I didn't have to change a single line in the kernel.

The megakernel's decode function already accepts `num_layers` as a **runtime parameter**. AlpinDale designed it this way for flexibility. The kernel loop simply runs `for (int layer = 0; layer < num_layers; layer++)`.

I realized this meant I could reuse the **exact same compiled kernel** for both:
- The **talker decoder** (28 layers) — the main speech generation model
- The **code predictor** (5 layers) — a smaller transformer that generates codebook groups

My first implementation of the code predictor used pure PyTorch (70+ separate CUDA kernel launches per step): **179 ms per frame**. When I switched to the megakernel with `num_layers=5`: **10.9 ms per frame**. An **18x speedup** from a zero-line kernel change.

The trick was packing the code predictor's weights into the same `LDGLayerWeights` struct format the kernel expects, allocating a separate (smaller) KV cache, and calling the same `decode` op with `num_layers=5`.

## Change 4: RoPE Theta

The talker uses `rope_theta = 1,000,000` instead of the text model's `10,000`. This only affects the precomputed cos/sin tables — no kernel changes needed. I compute the tables in Python during weight loading:

```python
inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, 128, 2) / 128))
```

## The M-RoPE Problem (What I Couldn't Fix)

Here's where I hit a wall. The talker decoder's config specifies M-RoPE (Multimodal RoPE) with `mrope_section: [24, 20, 20]` and `interleaved: True`. This means the 64 head dimension pairs are split into 3 groups (24, 20, 20), each using a potentially different position ID.

The megakernel implements standard 1D RoPE — all pairs use the same position. Implementing M-RoPE in the kernel would require:
1. Splitting the RoPE rotation into 3 sections within the attention computation
2. Passing 3 position IDs per step instead of 1
3. Modifying the cos/sin lookup logic

This is a non-trivial kernel change (touching the attention hot path), and I decided the risk of breaking the carefully-tuned kernel outweighed the benefit. Instead, I noted that for text-only TTS (no vision input), the 3 M-RoPE position IDs are identical — so M-RoPE reduces to standard RoPE with the same cos/sin values.

**The catch**: the model was *trained* with M-RoPE, and the attention patterns still diverge over long sequences. The practical effect is that the model never reliably emits the EOS token. I worked around this with a word-count-based frame limit (see the performance optimization doc).

## Weight Loading

The weight loading was straightforward but tedious — mapping 478 tensors from the HuggingFace safetensors format to the megakernel's expected layout:

```python
# Per-layer weights (11 tensors per layer, same order as LDGLayerWeights struct)
for i in range(28):
    p = f"talker.model.layers.{i}."
    layer_weights.extend([
        state[p + "input_layernorm.weight"],
        state[p + "self_attn.q_proj.weight"],
        state[p + "self_attn.k_proj.weight"],
        # ... 8 more per layer
    ])
```

The order matters — the kernel expects the exact struct layout defined in `kernel.cu`. One wrong tensor and you get garbage output (or a crash).

I also loaded the text projection weights (a 2-layer MLP that maps text embeddings from dimension 2048 to 1024), the code predictor weights (5 transformer layers + 15 per-group LM heads and embeddings), and the speaker encoder weights (loaded but not used in this implementation).

## Summary of Kernel Changes

| Change | Location | Lines Changed | Impact |
|---|---|---|---|
| Vocab size | `build_tts.py` (compile flags) | 2 | LM head 48x smaller |
| Embedding sentinel | `kernel.cu` | 3 | Precomputed embedding input |
| RoPE theta | `model_tts.py` (Python) | 1 | Correct rotation tables |
| num_layers reuse | (none — already supported) | 0 | 18x code predictor speedup |

Total kernel code changes: **3 lines**. Everything else was Python-side integration.

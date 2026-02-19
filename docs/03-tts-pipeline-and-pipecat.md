# Building the TTS Pipeline & Pipecat Integration

## Understanding the Qwen3-TTS Inference Flow

Before writing any pipeline code, I needed to understand exactly how Qwen3-TTS turns text into audio. I read through the official `modeling_qwen3_tts.py` (shipped with the model under `trust_remote_code=True`) and traced the generation flow:

1. **Tokenize** the input text with the chat template: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
2. **Project** text token embeddings from dimension 2048 to 1024 via a learned SiLU-activated MLP
3. **Prefill** the talker decoder with a carefully constructed sequence of role tokens, codec thinking tokens, and the first text token fused with codec BOS
4. **Autoregressive decode**: each step the talker produces a codebook token + hidden state. The code predictor takes the hidden state and generates 15 more codebook groups. All 16 groups form one audio "frame" at 12.5 Hz
5. **Vocoder**: accumulated codec frames are decoded to 24 kHz PCM audio

## The Prefill Format Discovery

Getting the prefill format right was one of the trickiest parts. I initially used a simplified format — just role tokens, some padding, and the text. The model generated audio, but it sounded wrong and EOS never triggered.

I went back to the official source code and found the exact construction (lines 2136-2160 of `modeling_qwen3_tts.py`):

```
Position 0: role_embed[0] (from "<|im_start|>")
Position 1: role_embed[1] (from "assistant")
Position 2: role_embed[2] (from "\n")
Position 3: tts_pad_embed + codec_embed(nothink)     ← "thinking" tokens!
Position 4: tts_pad_embed + codec_embed(think_bos)
Position 5: tts_pad_embed + codec_embed(think_eos)
Position 6: tts_bos_embed + codec_embed(pad)
Position 7: first_text_embed + codec_embed(bos)       ← text starts here
```

The key insight was that positions 3-6 aren't just padding — they're **thinking tokens** (nothink=2155, think_bos=2156, think_eos=2157) that the model was trained to expect. My initial version used `[PAD, PAD, PAD]` for the codec tags, which gave the model a completely wrong conditioning signal.

After fixing this to match the official format, the prefill went from 7 steps to the correct 8 steps. Audio quality improved immediately.

## The Trailing Text Mechanism

During autoregressive decode, each frame's input embedding isn't just the sum of the previous frame's codec embeddings — it also includes a **trailing text embedding**. The model consumes one text token per frame as additional conditioning:

```python
if trailing_idx < trailing_text.shape[0]:
    embed_sum = embed_sum + trailing_text[trailing_idx]
    trailing_idx += 1
else:
    embed_sum = embed_sum + tts_pad_embed  # text exhausted
```

I discovered from the official code that the trailing text should be `input_ids[4:-5]` — strip the first token (already in prefill) and the last 5 format-end tokens (`<|im_end|>\n<|im_start|>assistant\n`). Getting this wrong meant feeding chat template tokens as "speech content," which caused the model to generate garbage after the actual text.

## Making It Stream: The Generator Revelation

My first implementation of `_generate_codec_frames` returned a list:

```python
def _generate_codec_frames(self, text):
    frames = []
    for step in range(max_frames):
        # ... generate frame ...
        frames.append(all_codes)
    return frames
```

The streaming wrapper called `synthesize_streaming()` which iterated over the returned list. The problem? **It blocked until ALL frames were generated before yielding the first chunk.** The streaming TTFC was 35,932ms — the model generated 2,048 frames before any audio reached the consumer.

The fix was embarrassingly simple — convert it to a Python generator:

```python
def _generate_codec_frames(self, text):
    for step in range(max_frames):
        # ... generate frame ...
        yield all_codes  # yields immediately!
```

TTFC dropped from 35,932ms to 1,096ms instantly. The generator yields each frame as it's produced, so the streaming wrapper can decode and send the first chunk without waiting for the full utterance.

## The First-Chunk Optimization

Even at 1,096ms, the TTFC was too high. The bottleneck was now the vocoder's first decode call (~900ms cold start). I added three optimizations:

**1. Vocoder warmup during initialization:**
```python
# Warm up the vocoder with dummy decodes
for n in [1, 1, 5]:
    dummy_codes = torch.randint(0, 2048, (n, 16), device="cuda")
    self.speech_tokenizer.decode([{"audio_codes": dummy_codes}])
```

**2. Yield the first frame immediately** (1 frame instead of waiting for a full 10-frame chunk):
```python
target = 1 if first_chunk else chunk_size
if len(buffer) >= target:
    audio, sr = self._decode_to_audio(buffer)
    first_chunk = False
    yield audio, sr
```

**3. Warm up BOTH argmax and sampling code paths:**
```python
for do_sample in [False, False, True, True, True]:
    self.talker.reset()
    _, h = self.talker.step(CODEC_BOS)
    self.code_predictor.predict(h, 0, embed_weight, do_sample=do_sample)
```

I discovered the sampling path (torch.multinomial, softmax, topk) has its own first-call overhead separate from the argmax path. Without warming both, the code predictor took 107ms on the first call instead of 13ms.

These three changes brought streaming TTFC from 1,096ms down to ~80ms.

## Pipecat Integration

The Pipecat integration was the cleanest part of the project. Pipecat's `TTSService` interface is well-designed — you implement `run_tts(text, context_id)` as an async generator that yields frames:

```python
class MegakernelTTSService(TTSService):
    async def run_tts(self, text, context_id):
        yield TTSStartedFrame(context_id=context_id)

        async for audio_chunk, sr in engine.synthesize_streaming(text):
            pcm16 = _float32_to_pcm16(audio_chunk)
            yield TTSAudioRawFrame(
                audio=pcm16, sample_rate=sr,
                num_channels=1, context_id=context_id
            )

        yield TTSStoppedFrame(context_id=context_id)
```

The engine initialization happens lazily on first use (or eagerly via Pipecat's `start()` hook). The streaming synthesis runs in the async event loop — each chunk is yielded as soon as the vocoder decodes it.

The service handles:
- **TTFB metrics**: `start_ttfb_metrics()` / `stop_ttfb_metrics()` called around the first chunk
- **Usage metrics**: tracks text length for billing/monitoring
- **Error handling**: catches exceptions and yields `ErrorFrame` instead of crashing the pipeline
- **Float32 to PCM16 conversion**: Pipecat expects 16-bit PCM bytes, the vocoder outputs float32

## The Audio Format

Qwen3-TTS outputs 24 kHz mono audio. The vocoder (Qwen3-TTS-Tokenizer-12Hz) takes codec frames at 12.5 Hz and produces 1,920 PCM samples per frame (= 80ms of audio at 24 kHz). The Pipecat service converts this to 16-bit PCM bytes for transport.

Each streaming chunk contains either:
- 1 frame (1,920 samples = 80ms) for the first chunk (minimum TTFC)
- 10 frames (19,200 samples = 800ms) for subsequent chunks (efficient batching)

This gives a good balance between low latency (first chunk arrives fast) and efficiency (subsequent chunks are batched to reduce vocoder call overhead).

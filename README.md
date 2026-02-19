# RTX 5090 Megakernel for Qwen3-TTS on Pipecat

A CUDA megakernel that runs Qwen3-TTS's talker decoder at **~1,000 tok/s** on a single RTX 5090, integrated into a streaming Pipecat voice pipeline.

## Performance Results

| Metric | Result | Target | Status |
|---|---|---|---|
| **TTFC** (time to first audio chunk) | **50.5 ms** | < 90 ms | PASS |
| **TTFC** (streaming, incl. vocoder) | **81.6 ms** | < 90 ms | PASS |
| **RTF** (real-time factor) | **0.177** | < 0.3 | PASS |
| **RTF** (streaming) | **0.234** | < 0.3 | PASS |
| Code predictor (argmax) | 9.8 ms/frame | — | — |
| Code predictor (sampling) | 10.9 ms/frame | — | — |
| Talker decode | ~1 ms/step | — | — |

**TTFC breakdown** (non-streaming, 50.5 ms total):

| Phase | Time |
|---|---|
| Tokenize | 2.3 ms |
| Embed build | 7.2 ms |
| Prefill (8 steps) | 24.9 ms |
| First talker decode | 3.1 ms |
| First code predictor | 13.0 ms |

**Measurement methodology**: All timing uses `time.perf_counter()` with `torch.cuda.synchronize()` barriers. TTFC is measured from the start of generation to the first complete codec frame (16 codebook groups). RTF is measured over 50 frames of generation. Warmup runs are excluded from measurements.

## Architecture

```
Text input
    │
    ▼
┌─────────────────┐
│  Text Tokenizer  │  (HuggingFace AutoTokenizer)
│  + Projection    │  (151936→2048→1024, SiLU activation)
└─────────┬───────┘
          ▼
┌─────────────────┐
│  Talker Decoder  │  ◄── MEGAKERNEL (28-layer Qwen3, single CUDA launch)
│  (28 layers)     │      128 blocks × 512 threads, persistent
└─────────┬───────┘
          │ hidden state + first codebook token
          ▼
┌─────────────────┐
│ Code Predictor   │  ◄── MEGAKERNEL (5-layer Qwen3, reused kernel)
│ (5 layers)       │      Generates 15 more codebook groups per frame
└─────────┬───────┘
          │ 16 codebook tokens per frame
          ▼
┌─────────────────┐
│    Vocoder       │  (Qwen3-TTS-Tokenizer-12Hz)
│  codec → audio   │  → 24 kHz PCM streaming chunks
└─────────┬───────┘
          ▼
┌─────────────────┐
│    Pipecat       │  TTSStartedFrame → TTSAudioRawFrame chunks → TTSStoppedFrame
│   Pipeline       │
└─────────────────┘
```

### How it works

1. **Text tokenization**: Input text is tokenized and projected from the text embedding space (dim 2048) to the decoder hidden space (dim 1024) via a learned 2-layer projection with SiLU activation.

2. **Prefill** (8 megakernel steps): The input sequence `[role_tokens(3), fused_codec_tags(4), first_text+codec_bos(1)]` is fed step-by-step through the talker decoder via `step_with_embed()`.

3. **Autoregressive decode loop**: Each frame produces 16 codebook tokens:
   - The **talker decoder** (megakernel, ~1 ms/step) generates the first codebook token + a hidden state
   - The **code predictor** (megakernel, ~11 ms) takes the hidden state and generates 15 more codebook groups
   - The next input embedding is the sum of all 16 codec embeddings + trailing text

4. **Vocoder**: Accumulated codec frames are decoded to 24 kHz audio. The first chunk (1 frame) is sent immediately for low TTFC; subsequent chunks batch 10 frames (~0.8s each).

5. **Streaming**: Audio chunks are yielded via an async generator as they're produced — the full utterance is never buffered.

## Kernel Modifications

The original [qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) targets Qwen3-0.6B text generation. Two changes were needed for TTS:

### 1. Compile-time vocab size (`build_tts.py`)

```
LDG_VOCAB_SIZE: 151936 → 3072  (codec vocabulary)
LDG_LM_NUM_BLOCKS: 1280 → 16   (48x fewer rows to scan)
```

The LM head is 48x smaller, so we need far fewer thread blocks for the argmax reduction.

### 2. Embedding sentinel for precomputed inputs (`kernel.cu`)

The TTS decode loop feeds a *sum of embeddings* (codec + text) as input, not a single token ID. A 3-line kernel patch adds sentinel support:

```c
// Sentinel: if token_id < 0, use hidden_buffer (precomputed embedding)
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;
```

When `token_id == -1`, the kernel reads from `hidden_buffer` (where Python has pre-written the summed embedding) instead of doing an embedding table lookup. This eliminates the need for a separate embedding kernel launch.

### 3. Runtime `num_layers` parameter

The kernel already accepts `num_layers` as a runtime parameter. This is the key insight that enables reusing the **same compiled kernel** for both:
- The **talker decoder** (28 layers)
- The **code predictor** (5 layers)

The code predictor was initially a pure PyTorch implementation (~179 ms/frame). By reusing the megakernel with `num_layers=5`, it dropped to **~11 ms/frame** — an **18x speedup** that was critical for meeting the RTF target.

## Setup

### Requirements

- **GPU**: NVIDIA RTX 5090 (sm_120a / Blackwell)
- **CUDA**: 12.8+
- **PyTorch**: 2.7+ (with CUDA 12.8 support)
- **Python**: 3.10+

### Installation

```bash
# 1. Rent an RTX 5090 (e.g. on vast.ai, ~$0.30/hr)
# 2. SSH in and clone
git clone <this-repo> && cd qwen_megakernel

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo (first run triggers JIT kernel compilation, ~60s)
python3 demo_pipeline.py --text "Hello, this is a test."
```

The model weights (~1.2 GB) are downloaded automatically from HuggingFace on first run.

### Running the demo

```bash
# Streaming synthesis with timing
python3 demo_pipeline.py --text "Your text here" --output output.wav

# Non-streaming synthesis
python3 demo_tts.py "Your text here" --output output.wav

# Full benchmark (multiple runs, streaming + non-streaming)
python3 benchmark.py --text "Your text here" --runs 5

# End-to-end performance test
python3 test_e2e.py
```

### Pipecat integration

```python
from qwen_megakernel.pipecat_tts import MegakernelTTSService

tts = MegakernelTTSService(
    model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    chunk_frames=10,  # ~0.8s per audio chunk
)

# In a Pipecat pipeline:
pipeline = Pipeline([stt, llm, tts, transport.output()])
```

The service implements the standard Pipecat `TTSService` interface:
- `TTSStartedFrame` → `TTSAudioRawFrame` chunks → `TTSStoppedFrame`
- Audio is streamed as 16-bit PCM at 24 kHz
- First chunk is a single frame (~80ms audio) for minimum TTFC

## Key Design Decisions

**Megakernel reuse for the code predictor**: Rather than writing a separate kernel or using PyTorch for the 5-layer code predictor transformer, we reuse the talker megakernel with `num_layers=5`. The kernel's architecture (persistent thread blocks, fused layers, L1-bypass loads) provides the same speedup regardless of layer count. This turned a 179ms bottleneck into an 11ms operation.

**Precomputed constant embeddings**: Role tokens, TTS special tokens, codec tags, and their fused combinations are computed once during `initialize()` and cached. This eliminates ~6ms of redundant GPU kernel launches per utterance.

**Generator-based streaming**: `_generate_codec_frames()` is a Python generator that `yield`s each frame immediately. The streaming wrapper sends the first frame (1 frame = ~80ms audio) as soon as it's ready, then batches subsequent frames into 10-frame chunks (~0.8s each). This achieves an 81ms streaming TTFC.

**Pipeline warmup**: Both argmax and sampling code paths (torch.multinomial, softmax, topk) are warmed up during initialization. The vocoder is also pre-warmed with dummy decodes. Without this, TTFC was ~1,100ms due to CUDA JIT overhead on first call.

**Word-count-based frame limit**: Since the model does not reliably emit EOS (see Known Limitations), generation is capped at a heuristic based on word count: `estimated_speech_sec = word_count / 2.5` (150 WPM), then `max_frames = estimated_speech_sec * 12.5 * 2.0` (2x margin). This prevents runaway generation while allowing enough frames for the text content.

## Known Limitations

**EOS detection**: The talker decoder uses M-RoPE (Multimodal RoPE with `mrope_section: [24, 20, 20]`), which applies different rotary position encodings to different head dimensions. The megakernel implements standard RoPE, which means the model's attention patterns diverge from the reference implementation over long sequences. As a result, the model does not reliably emit the EOS token. We work around this with a word-count-based frame limit. Implementing M-RoPE in the kernel (splitting head dimensions into 3 sections with independent position counters) would fix this.

**No voice cloning**: The current implementation uses the base model without reference audio conditioning. The speaker encoder weights are loaded but not used.

**Audio quality**: Because of the RoPE mismatch noted above, generated speech is intelligible but may have quality differences from the reference HuggingFace implementation, especially for longer utterances.

## File Structure

```
qwen_megakernel/
├── csrc/
│   ├── kernel.cu              # CUDA megakernel (~1,600 lines)
│   └── torch_bindings.cpp     # PyTorch C++ extension bindings
├── qwen_megakernel/
│   ├── model_tts.py           # Weight loading, TTSDecoder, CodePredictorKernel
│   ├── tts_engine.py          # Full TTS pipeline orchestration
│   ├── pipecat_tts.py         # Pipecat TTSService implementation
│   ├── build_tts.py           # JIT compilation (VOCAB=3072, LM_BLOCKS=16)
│   ├── model.py               # Original Qwen3-0.6B decoder (unchanged)
│   ├── build.py               # Original build config (unchanged)
│   └── bench.py               # Original benchmark (unchanged)
├── benchmarks/
│   ├── measure_ttfc.py        # TTFC measurement
│   ├── measure_rtf.py         # RTF measurement
│   ├── measure_tok_s.py       # Decode throughput
│   └── measure_e2e.py         # End-to-end pipeline
├── demo_pipeline.py           # Streaming TTS demo
├── demo_tts.py                # Non-streaming TTS demo
├── benchmark.py               # All-in-one benchmark
├── test_e2e.py                # End-to-end test with all metrics
├── validate_kernel.py         # Kernel correctness validation
└── requirements.txt
```

## Detailed Documentation

For the full story of how this was built — including what didn't work, the debugging journey, and the reasoning behind each decision:

- [GPU Setup](docs/01-gpu-setup.md) — Finding and configuring the RTX 5090 on vast.ai
- [Kernel Adaptation](docs/02-kernel-adaptation.md) — What changed in the megakernel and why
- [TTS Pipeline & Pipecat](docs/03-tts-pipeline-and-pipecat.md) — Building the inference pipeline and Pipecat integration
- [Performance Optimization](docs/04-performance-optimization.md) — The step-by-step journey from 35,932ms to 50ms TTFC
- [Key Insights](docs/05-key-insights.md) — Unique approaches and hard-won lessons

## Credits

- Megakernel by [AlpinDale](https://blog.alpindale.net/posts/5090_decode_optimization/), based on [MegaQwen](https://github.com/Infatoshi/MegaQwen)
- Qwen3-TTS by [Qwen Team](https://huggingface.co/Qwen/Qwen3-TTS)
- [Pipecat](https://docs.pipecat.ai) by Daily

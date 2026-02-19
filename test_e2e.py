#!/usr/bin/env python3
"""End-to-end TTS pipeline test with full performance metrics.

Tests:
1. Vocoder loading
2. Code predictor (megakernel) performance
3. Full pipeline: text → codec frames → audio
4. TTFC and RTF measurements
"""

import time
import numpy as np
import torch

from qwen_megakernel.model_tts import (
    CODEC_BOS, CODEC_EOS, NUM_CODE_GROUPS,
    CodePredictorKernel, TTSDecoder, TextProjection,
    build_prefill_embeddings, load_tts_weights,
    TTS_PAD,
)


def test_vocoder():
    """Test vocoder loading and decode."""
    print("\n" + "=" * 60)
    print("VOCODER TEST")
    print("=" * 60)

    try:
        import transformers.utils.generic
        if not hasattr(transformers.utils.generic, 'check_model_inputs'):
            def _check_model_inputs(*a, **kw):
                def decorator(func): return func
                return decorator
            transformers.utils.generic.check_model_inputs = _check_model_inputs

        from transformers import AutoConfig, AutoModel
        from qwen_tts.core import (
            Qwen3TTSTokenizerV2Config,
            Qwen3TTSTokenizerV2Model,
        )

        try:
            AutoConfig.register('qwen3_tts_tokenizer_12hz', Qwen3TTSTokenizerV2Config)
            AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
        except ValueError:
            pass

        print("Loading speech tokenizer...")
        model = AutoModel.from_pretrained(
            'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
            subfolder='speech_tokenizer',
            device_map='cuda',
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        from qwen_tts import Qwen3TTSTokenizer
        tok = Qwen3TTSTokenizer()
        tok.model = model
        tok.feature_extractor = None
        tok.config = model.config
        tok.device = model.device
        sr = tok.get_output_sample_rate()
        print(f"  Loaded! Sample rate: {sr} Hz")

        # Test with dummy codec frames (16 codebooks)
        codes = torch.randint(0, 2048, (10, NUM_CODE_GROUPS), device='cuda')
        print(f"  Test codes: {codes.shape} ({NUM_CODE_GROUPS} codebooks)")

        t0 = time.perf_counter()
        wavs, sr_out = tok.decode([{"audio_codes": codes}])
        t1 = time.perf_counter()
        print(f"  Decoded: {len(wavs[0])} samples, sr={sr_out}, took {(t1-t0)*1000:.1f}ms")
        print(f"  Audio duration: {len(wavs[0])/sr_out:.3f}s")
        return tok
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_code_predictor_perf(weights):
    """Benchmark the megakernel code predictor."""
    print("\n" + "=" * 60)
    print("CODE PREDICTOR BENCHMARK")
    print("=" * 60)

    talker = TTSDecoder(weights=weights)
    talker.reset()
    token, hidden = talker.step(CODEC_BOS)

    cp = CodePredictorKernel(weights, device="cuda")

    # Warmup
    for _ in range(5):
        cp.predict(hidden, token, talker.embed_weight, do_sample=False)
    for _ in range(5):
        cp.predict(hidden, token, talker.embed_weight, do_sample=True)

    N = 30

    # Argmax
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        cp.predict(hidden, token, talker.embed_weight, do_sample=False)
    torch.cuda.synchronize()
    argmax_ms = (time.perf_counter() - t0) / N * 1000

    # Sampling
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        cp.predict(hidden, token, talker.embed_weight, do_sample=True, temperature=0.9, top_k=50)
    torch.cuda.synchronize()
    sample_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  Argmax:   {argmax_ms:.1f} ms/frame ({NUM_CODE_GROUPS} codebooks)")
    print(f"  Sampling: {sample_ms:.1f} ms/frame")
    return argmax_ms, sample_ms


def test_full_pipeline(weights, vocoder):
    """Full text → audio pipeline test."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    talker = TTSDecoder(weights=weights)
    text_proj = TextProjection(weights, device="cuda")
    cp = CodePredictorKernel(weights, device="cuda")
    talker_embed = weights["embed_weight"]

    # CP embeddings for the decode loop
    cp_embeds = []
    for g in range(NUM_CODE_GROUPS - 1):
        cp_embeds.append(weights["code_predictor"][f"codec_embedding.{g}.weight"])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    text = "Hello, this is a test of the megakernel TTS engine."
    print(f"  Text: '{text}'")

    # Warmup
    for _ in range(2):
        talker.reset()
        talker.step(CODEC_BOS)
        cp.predict(torch.randn(1024, device='cuda'), 0, talker_embed, do_sample=False)

    # === TTFC measurement ===
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # Tokenize
    formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer.encode(formatted, return_tensors="pt")[0].to("cuda")
    t_tokenize = time.perf_counter()

    # Build prefill embeddings
    prefill_embeds, trailing_text = build_prefill_embeddings(
        text_ids, text_proj, talker_embed, device="cuda",
    )
    t_embed = time.perf_counter()

    # Prefill
    talker.reset()
    for i in range(prefill_embeds.shape[0]):
        talker.step_with_embed(prefill_embeds[i])
    t_prefill = time.perf_counter()

    # First decode step
    first_token, hidden = talker.step(CODEC_BOS)
    t_first_decode = time.perf_counter()

    # First code predictor call
    all_codes = cp.predict(
        talker_hidden=hidden,
        first_codebook_token=first_token,
        talker_embed_weight=talker_embed,
        do_sample=True,
        temperature=0.9,
        top_k=50,
    )
    torch.cuda.synchronize()
    t_first_cp = time.perf_counter()

    ttfc_ms = (t_first_cp - t_start) * 1000
    print(f"\n  TTFC breakdown:")
    print(f"    Tokenize:       {(t_tokenize - t_start)*1000:.1f} ms")
    print(f"    Embed build:    {(t_embed - t_tokenize)*1000:.1f} ms")
    print(f"    Prefill ({prefill_embeds.shape[0]} steps): {(t_prefill - t_embed)*1000:.1f} ms")
    print(f"    First decode:   {(t_first_decode - t_prefill)*1000:.1f} ms")
    print(f"    First CP:       {(t_first_cp - t_first_decode)*1000:.1f} ms")
    print(f"    Total TTFC:     {ttfc_ms:.1f} ms {'✓ PASS' if ttfc_ms < 90 else '✗ FAIL'} (target < 90 ms)")

    # === Generate more frames for RTF ===
    tts_pad_embed = text_proj.embed_text_ids(
        torch.tensor([TTS_PAD], device="cuda"),
    ).squeeze(0).to(torch.bfloat16)

    frames = [all_codes]
    trailing_idx = 0
    prev_token = first_token

    max_frames = 50  # Generate 50 frames for RTF measurement
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()

    for step in range(max_frames - 1):
        # Build next input embedding (use slice indexing to avoid GPU→CPU sync)
        embed_sum = torch.nn.functional.embedding(all_codes[0:1], talker_embed).squeeze(0)
        for g in range(NUM_CODE_GROUPS - 1):
            embed_sum = embed_sum + torch.nn.functional.embedding(
                all_codes[g + 1:g + 2], cp_embeds[g],
            ).squeeze(0)

        if trailing_idx < trailing_text.shape[0]:
            embed_sum = embed_sum + trailing_text[trailing_idx].to(torch.bfloat16)
            trailing_idx += 1
        else:
            embed_sum = embed_sum + tts_pad_embed

        prev_token, hidden = talker.step_with_embed(embed_sum)
        if prev_token == CODEC_EOS:
            print(f"  EOS at frame {step + 1}")
            break

        all_codes = cp.predict(
            talker_hidden=hidden,
            first_codebook_token=prev_token,
            talker_embed_weight=talker_embed,
            do_sample=True,
            temperature=0.9,
            top_k=50,
        )
        frames.append(all_codes)

    torch.cuda.synchronize()
    t_gen_end = time.perf_counter()

    n_frames = len(frames)
    gen_time_ms = (t_gen_end - t_gen_start) * 1000
    audio_duration_s = n_frames / 12.5
    per_frame_ms = gen_time_ms / max(n_frames - 1, 1)  # exclude first frame (already measured)
    rtf = (per_frame_ms / 1000) / (1.0 / 12.5)

    print(f"\n  Generation stats ({n_frames} frames):")
    print(f"    Total gen time: {gen_time_ms:.1f} ms")
    print(f"    Audio duration: {audio_duration_s:.1f} s")
    print(f"    Per frame:      {per_frame_ms:.1f} ms")
    print(f"    RTF:            {rtf:.3f} {'✓ PASS' if rtf < 0.3 else '✗ FAIL'} (target < 0.3)")

    # === Vocoder decode ===
    if vocoder is not None:
        print(f"\n  Decoding {n_frames} frames with vocoder...")
        audio_codes = torch.stack(frames, dim=0)
        print(f"    Codes shape: {audio_codes.shape}")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wavs, sr = vocoder.decode([{"audio_codes": audio_codes}])
        torch.cuda.synchronize()
        vocoder_ms = (time.perf_counter() - t0) * 1000

        wav = wavs[0]
        print(f"    Vocoder decode: {vocoder_ms:.1f} ms")
        print(f"    Audio: {len(wav)} samples at {sr} Hz = {len(wav)/sr:.2f}s")

        # Save to file
        import soundfile as sf
        sf.write("/tmp/tts_output.wav", wav, sr)
        print(f"    Saved to /tmp/tts_output.wav")
    else:
        print("\n  [Vocoder not available — skipping audio decode]")

    return ttfc_ms, rtf, n_frames


def main():
    print("=" * 60)
    print("MEGAKERNEL TTS ENGINE — FULL PERFORMANCE TEST")
    print("=" * 60)

    # Load weights
    print("\nLoading TTS weights...")
    weights = load_tts_weights()

    # Test vocoder
    vocoder = test_vocoder()

    # Test code predictor performance
    argmax_ms, sample_ms = test_code_predictor_perf(weights)

    # Test full pipeline
    ttfc_ms, rtf, n_frames = test_full_pipeline(weights, vocoder)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Code predictor (argmax):  {argmax_ms:.1f} ms/frame")
    print(f"  Code predictor (sample):  {sample_ms:.1f} ms/frame")
    print(f"  TTFC:                     {ttfc_ms:.1f} ms {'✓' if ttfc_ms < 90 else '✗'} (target < 90)")
    print(f"  RTF:                      {rtf:.3f} {'✓' if rtf < 0.3 else '✗'} (target < 0.3)")
    print(f"  Frames generated:         {n_frames}")
    print(f"  Codebook groups:          {NUM_CODE_GROUPS}")
    print(f"  Vocoder:                  {'✓ loaded' if vocoder else '✗ unavailable'}")


if __name__ == "__main__":
    main()

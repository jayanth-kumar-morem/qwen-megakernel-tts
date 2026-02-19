#!/usr/bin/env python3
"""Measure Real-Time Factor (RTF) for TTS generation.

RTF = generation_time / audio_duration
RTF < 1.0 means faster than real-time.
Target: RTF < 0.3 on RTX 5090.

Measures both per-frame RTF (talker + code predictor + vocoder) and
overall pipeline RTF including all overhead.

Usage:
    python -m benchmarks.measure_rtf
    python -m benchmarks.measure_rtf --runs 5 --text "A longer sentence for testing."
"""

import argparse
import time

import numpy as np
import torch


def measure_per_frame_rtf(engine, text: str) -> dict:
    """Measure RTF with per-frame timing breakdown.

    Returns per-frame times for talker decode, code predictor, and embedding computation.
    """
    from qwen_megakernel.model_tts import (
        CODEC_BOS, CODEC_EOS, CODEC_PAD, NUM_CODE_GROUPS,
        HIDDEN_SIZE, build_prefill_embeddings,
    )

    cfg = engine.config
    engine.talker.reset()

    # Tokenize and build prefill
    formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = engine.tokenizer.encode(formatted_text, return_tensors="pt")[0].to(engine.device)
    prefill_embeds, trailing_text = build_prefill_embeddings(
        text_ids, engine.text_projection, engine._talker_embed, device=engine.device,
    )

    # Prefill
    for i in range(prefill_embeds.shape[0]):
        engine.talker.step_with_embed(prefill_embeds[i])

    first_token, hidden = engine.talker.step(CODEC_BOS)

    # Per-frame timing
    talker_times = []
    cp_times = []
    embed_times = []
    frames = []

    trailing_idx = 0
    tts_pad_embed = engine.text_projection.embed_text_ids(
        torch.tensor([151671], device=engine.device),
    ).squeeze(0)

    prev_token = first_token
    prev_hidden = hidden

    for step in range(cfg.max_new_tokens):
        if prev_token == CODEC_EOS:
            break

        # Code predictor timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_codes = engine.code_predictor.predict(
            talker_hidden=prev_hidden,
            first_codebook_token=prev_token,
            talker_embed_weight=engine._talker_embed,
            do_sample=cfg.subtalker_do_sample,
            temperature=cfg.subtalker_temperature,
            top_k=cfg.subtalker_top_k,
        )
        torch.cuda.synchronize()
        cp_times.append(time.perf_counter() - t0)
        frames.append(all_codes)

        # Embedding computation timing
        t0 = time.perf_counter()
        embed_sum = torch.nn.functional.embedding(
            all_codes[0:1], engine._talker_embed,
        ).squeeze(0)
        for g in range(NUM_CODE_GROUPS - 1):
            token_g = all_codes[g + 1]
            if token_g < engine._cp_embeds[g].shape[0]:
                embed_g = torch.nn.functional.embedding(
                    token_g.unsqueeze(0), engine._cp_embeds[g],
                ).squeeze(0)
                embed_sum = embed_sum + embed_g

        if trailing_idx < trailing_text.shape[0]:
            embed_sum = embed_sum + trailing_text[trailing_idx].to(torch.bfloat16)
            trailing_idx += 1
        else:
            embed_sum = embed_sum + tts_pad_embed.to(torch.bfloat16)
        torch.cuda.synchronize()
        embed_times.append(time.perf_counter() - t0)

        # Talker decode timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prev_token, prev_hidden = engine.talker.step_with_embed(embed_sum)
        torch.cuda.synchronize()
        talker_times.append(time.perf_counter() - t0)

    num_frames = len(frames)
    audio_duration = num_frames / 12.5  # 12.5 Hz frame rate

    return {
        "num_frames": num_frames,
        "audio_duration_s": audio_duration,
        "talker_ms_per_frame": np.mean(talker_times) * 1000 if talker_times else 0,
        "cp_ms_per_frame": np.mean(cp_times) * 1000 if cp_times else 0,
        "embed_ms_per_frame": np.mean(embed_times) * 1000 if embed_times else 0,
        "total_ms_per_frame": (np.mean(talker_times) + np.mean(cp_times) + np.mean(embed_times)) * 1000 if talker_times else 0,
        "frame_rtf": ((np.mean(talker_times) + np.mean(cp_times) + np.mean(embed_times)) / 0.08) if talker_times else 0,
        "talker_times_ms": [t * 1000 for t in talker_times],
        "cp_times_ms": [t * 1000 for t in cp_times],
    }


def measure_overall_rtf(engine, text: str, num_runs: int = 3) -> dict:
    """Measure overall pipeline RTF (wall clock)."""
    results = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        waveform, sr = engine.synthesize(text)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        gen_time = t_end - t_start
        audio_dur = len(waveform) / sr if sr > 0 and len(waveform) > 0 else 0
        rtf = gen_time / audio_dur if audio_dur > 0 else float('inf')
        results.append({
            "gen_time_s": gen_time,
            "audio_dur_s": audio_dur,
            "rtf": rtf,
            "tokens": engine.talker.position,
        })

    rtfs = [r["rtf"] for r in results]
    return {
        "rtf_mean": np.mean(rtfs),
        "rtf_std": np.std(rtfs),
        "rtf_min": np.min(rtfs),
        "rtf_max": np.max(rtfs),
        "per_run": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure TTS Real-Time Factor")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--text", type=str,
                        default="The quick brown fox jumps over the lazy dog. "
                                "This is a benchmark of real-time speech synthesis.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    from qwen_megakernel.tts_engine import MegakernelTTSEngine, TTSConfig

    config = TTSConfig(model_path=args.model)
    engine = MegakernelTTSEngine(config=config)

    print("Initializing engine...")
    engine.initialize()

    print(f"\n{'='*60}")
    print("Real-Time Factor (RTF) Measurement")
    print(f"{'='*60}")
    print(f"Text: \"{args.text[:60]}...\"")

    # Warmup
    for _ in range(args.warmup):
        engine.synthesize(args.text)

    # Per-frame breakdown
    print(f"\n--- Per-Frame Breakdown ---")
    frame_r = measure_per_frame_rtf(engine, args.text)
    print(f"  Frames generated: {frame_r['num_frames']}")
    print(f"  Audio duration:   {frame_r['audio_duration_s']:.2f}s")
    print(f"  Per-frame budget: 80.0 ms (12.5 Hz)")
    print(f"  Talker decode:    {frame_r['talker_ms_per_frame']:.2f} ms/frame")
    print(f"  Code predictor:   {frame_r['cp_ms_per_frame']:.2f} ms/frame")
    print(f"  Embed compute:    {frame_r['embed_ms_per_frame']:.2f} ms/frame")
    print(f"  Total per frame:  {frame_r['total_ms_per_frame']:.2f} ms/frame")
    print(f"  Frame RTF:        {frame_r['frame_rtf']:.3f}")

    # Overall pipeline RTF
    print(f"\n--- Overall Pipeline RTF ---")
    overall_r = measure_overall_rtf(engine, args.text, args.runs)
    for i, r in enumerate(overall_r["per_run"]):
        print(f"  Run {i+1}: RTF={r['rtf']:.3f} "
              f"(gen={r['gen_time_s']:.3f}s, audio={r['audio_dur_s']:.2f}s, "
              f"tokens={r['tokens']})")
    print(f"  Mean RTF: {overall_r['rtf_mean']:.3f} ± {overall_r['rtf_std']:.3f}")

    # Verdict
    print(f"\n{'='*60}")
    target = 0.3
    passed = overall_r["rtf_mean"] < target
    status = "PASS" if passed else "FAIL"
    print(f"Overall RTF: {overall_r['rtf_mean']:.3f} — {status} (target < {target})")
    print(f"Frame RTF:   {frame_r['frame_rtf']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

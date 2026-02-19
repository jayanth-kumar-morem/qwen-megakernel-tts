#!/usr/bin/env python3
"""Measure megakernel talker decode throughput (tokens/sec).

Uses CUDA events for precise GPU timing. Measures ONLY the kernel decode
step, excluding code predictor, vocoder, and text tokenization.

Usage:
    python -m benchmarks.measure_tok_s
    python -m benchmarks.measure_tok_s --steps 200 --runs 5
"""

import argparse
import time

import torch


def measure_kernel_tok_s(
    decoder,
    num_steps: int = 100,
    warmup_steps: int = 10,
    num_runs: int = 3,
) -> dict:
    """Measure raw megakernel decode throughput.

    Args:
        decoder: TTSDecoder instance (already initialized).
        num_steps: Decode steps per run.
        warmup_steps: Warmup steps before measurement.
        num_runs: Number of timed runs.

    Returns:
        Dict with tok_s_mean, tok_s_std, ms_per_tok_mean, ms_per_tok_std.
    """
    from qwen_megakernel.model_tts import CODEC_BOS, CODEC_PAD

    # Warmup: populate KV cache a bit
    decoder.reset()
    for _ in range(warmup_steps):
        decoder.step(CODEC_PAD)

    # Timed runs
    results = []
    for run in range(num_runs):
        decoder.reset()

        # Feed BOS to get started
        decoder.step(CODEC_BOS)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_steps):
            decoder.step(CODEC_PAD)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tok_s = num_steps / (elapsed_ms / 1000.0)
        ms_per_tok = elapsed_ms / num_steps
        results.append({"tok_s": tok_s, "ms_per_tok": ms_per_tok, "elapsed_ms": elapsed_ms})

    tok_s_vals = [r["tok_s"] for r in results]
    ms_vals = [r["ms_per_tok"] for r in results]
    return {
        "tok_s_mean": sum(tok_s_vals) / len(tok_s_vals),
        "tok_s_std": (sum((x - sum(tok_s_vals) / len(tok_s_vals)) ** 2 for x in tok_s_vals) / len(tok_s_vals)) ** 0.5,
        "ms_per_tok_mean": sum(ms_vals) / len(ms_vals),
        "ms_per_tok_std": (sum((x - sum(ms_vals) / len(ms_vals)) ** 2 for x in ms_vals) / len(ms_vals)) ** 0.5,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "per_run": results,
    }


def measure_step_with_embed_tok_s(
    decoder,
    num_steps: int = 100,
    num_runs: int = 3,
) -> dict:
    """Measure decode throughput using step_with_embed (precomputed embedding input).

    This is the actual mode used during TTS generation.
    """
    from qwen_megakernel.model_tts import CODEC_BOS, HIDDEN_SIZE

    results = []
    for _ in range(num_runs):
        decoder.reset()
        decoder.step(CODEC_BOS)

        # Create a dummy embedding (simulates codec_embed_sum + text)
        dummy_embed = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_steps):
            decoder.step_with_embed(dummy_embed)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        results.append({
            "tok_s": num_steps / (elapsed_ms / 1000.0),
            "ms_per_tok": elapsed_ms / num_steps,
        })

    tok_s_vals = [r["tok_s"] for r in results]
    return {
        "tok_s_mean": sum(tok_s_vals) / len(tok_s_vals),
        "ms_per_tok_mean": sum(r["ms_per_tok"] for r in results) / len(results),
        "num_steps": num_steps,
        "per_run": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure megakernel decode tok/s")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--steps", type=int, default=100, help="Decode steps per run")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup steps")
    args = parser.parse_args()

    from qwen_megakernel.model_tts import TTSDecoder, load_tts_weights

    print("Loading TTS weights...")
    weights = load_tts_weights(args.model)
    decoder = TTSDecoder(weights=weights)

    print(f"\n{'='*60}")
    print("Megakernel Talker Decode Throughput")
    print(f"{'='*60}")
    print(f"Steps per run: {args.steps}")
    print(f"Runs: {args.runs}")

    # Test 1: step() — token ID input
    print(f"\n--- step(token_id) mode ---")
    r1 = measure_kernel_tok_s(decoder, args.steps, args.warmup, args.runs)
    print(f"  {r1['tok_s_mean']:.0f} ± {r1['tok_s_std']:.0f} tok/s")
    print(f"  {r1['ms_per_tok_mean']:.3f} ± {r1['ms_per_tok_std']:.3f} ms/tok")

    # Test 2: step_with_embed() — precomputed embedding input (TTS mode)
    print(f"\n--- step_with_embed() mode (TTS production path) ---")
    r2 = measure_step_with_embed_tok_s(decoder, args.steps, args.runs)
    print(f"  {r2['tok_s_mean']:.0f} tok/s")
    print(f"  {r2['ms_per_tok_mean']:.3f} ms/tok")

    print(f"\n{'='*60}")
    print("Note: TTS frame rate is 12.5 Hz (80ms per frame).")
    print(f"At {r2['tok_s_mean']:.0f} tok/s, talker decode uses "
          f"{r2['ms_per_tok_mean']:.2f}ms of the 80ms budget.")
    remaining = 80.0 - r2["ms_per_tok_mean"]
    print(f"Remaining budget for code predictor + vocoder: {remaining:.1f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

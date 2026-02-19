#!/usr/bin/env python3
"""Measure end-to-end pipeline latency for TTS.

Tests both non-streaming (total latency) and streaming (per-chunk latency)
modes with various text lengths.

Usage:
    python -m benchmarks.measure_e2e
    python -m benchmarks.measure_e2e --runs 5
"""

import argparse
import asyncio
import time

import numpy as np
import torch


TEST_TEXTS = {
    "short": "Hello, how are you?",
    "medium": "The quick brown fox jumps over the lazy dog. This sentence is used to test every letter of the alphabet.",
    "long": (
        "In the beginning, the universe was created. This has made a lot of people "
        "very angry and been widely regarded as a bad move. Many people now think this "
        "was a bad idea because the universe turned out to be far more complex than "
        "anyone had anticipated."
    ),
}


def measure_non_streaming(engine, text: str, num_runs: int = 3) -> dict:
    """Measure non-streaming end-to-end latency."""
    results = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        waveform, sr = engine.synthesize(text)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        audio_dur = len(waveform) / sr if sr > 0 and len(waveform) > 0 else 0
        results.append({
            "latency_s": elapsed,
            "audio_dur_s": audio_dur,
            "rtf": elapsed / audio_dur if audio_dur > 0 else float('inf'),
        })

    latencies = [r["latency_s"] for r in results]
    return {
        "latency_mean_s": np.mean(latencies),
        "latency_std_s": np.std(latencies),
        "rtf_mean": np.mean([r["rtf"] for r in results]),
        "audio_dur_s": results[0]["audio_dur_s"] if results else 0,
    }


async def measure_streaming(engine, text: str, chunk_frames: int = 10, num_runs: int = 3) -> dict:
    """Measure streaming end-to-end latency and inter-chunk times."""
    results = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        chunk_arrivals = []
        total_samples = 0

        async for audio_chunk, sr in engine.synthesize_streaming(text, chunk_frames=chunk_frames):
            torch.cuda.synchronize()
            chunk_arrivals.append(time.perf_counter() - t_start)
            total_samples += len(audio_chunk)

        audio_dur = total_samples / sr if sr > 0 and total_samples > 0 else 0

        # Inter-chunk deltas
        deltas = [chunk_arrivals[0]] + [
            chunk_arrivals[i] - chunk_arrivals[i - 1]
            for i in range(1, len(chunk_arrivals))
        ]

        results.append({
            "ttfc_s": chunk_arrivals[0] if chunk_arrivals else float('inf'),
            "total_latency_s": chunk_arrivals[-1] if chunk_arrivals else float('inf'),
            "audio_dur_s": audio_dur,
            "num_chunks": len(chunk_arrivals),
            "inter_chunk_mean_ms": np.mean(deltas[1:]) * 1000 if len(deltas) > 1 else 0,
            "inter_chunk_max_ms": np.max(deltas[1:]) * 1000 if len(deltas) > 1 else 0,
            "inter_chunk_std_ms": np.std(deltas[1:]) * 1000 if len(deltas) > 1 else 0,
        })

    return {
        "ttfc_mean_ms": np.mean([r["ttfc_s"] for r in results]) * 1000,
        "total_latency_mean_s": np.mean([r["total_latency_s"] for r in results]),
        "inter_chunk_mean_ms": np.mean([r["inter_chunk_mean_ms"] for r in results]),
        "inter_chunk_max_ms": np.max([r["inter_chunk_max_ms"] for r in results]),
        "num_chunks": results[0]["num_chunks"] if results else 0,
        "audio_dur_s": results[0]["audio_dur_s"] if results else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure end-to-end TTS latency")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=10)
    args = parser.parse_args()

    from qwen_megakernel.tts_engine import MegakernelTTSEngine, TTSConfig

    config = TTSConfig(model_path=args.model, chunk_frames=args.chunk_frames)
    engine = MegakernelTTSEngine(config=config)

    print("Initializing engine...")
    engine.initialize()

    print(f"\n{'='*70}")
    print("End-to-End Pipeline Latency")
    print(f"{'='*70}")
    print(f"Chunk size: {args.chunk_frames} frames ({args.chunk_frames / 12.5:.1f}s per chunk)")

    # Warmup
    for _ in range(args.warmup):
        engine.synthesize(TEST_TEXTS["short"])

    # Test each text length
    for label, text in TEST_TEXTS.items():
        print(f"\n--- {label.upper()} ({len(text)} chars) ---")
        print(f"  \"{text[:60]}{'...' if len(text) > 60 else ''}\"")

        # Non-streaming
        ns = measure_non_streaming(engine, text, args.runs)
        print(f"  Non-streaming: {ns['latency_mean_s']:.3f}s ± {ns['latency_std_s']:.3f}s "
              f"(RTF={ns['rtf_mean']:.3f}, audio={ns['audio_dur_s']:.2f}s)")

        # Streaming
        st = asyncio.run(measure_streaming(engine, text, args.chunk_frames, args.runs))
        print(f"  Streaming:")
        print(f"    TTFC:          {st['ttfc_mean_ms']:.1f}ms")
        print(f"    Total:         {st['total_latency_mean_s']:.3f}s")
        print(f"    Chunks:        {st['num_chunks']}")
        print(f"    Inter-chunk:   {st['inter_chunk_mean_ms']:.1f}ms avg, "
              f"{st['inter_chunk_max_ms']:.1f}ms max")
        print(f"    Audio:         {st['audio_dur_s']:.2f}s")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Text':>10} | {'Non-stream':>12} | {'TTFC':>8} | {'RTF':>6} | {'Audio':>6}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
    for label, text in TEST_TEXTS.items():
        ns = measure_non_streaming(engine, text, 1)
        st = asyncio.run(measure_streaming(engine, text, args.chunk_frames, 1))
        print(f"{label:>10} | {ns['latency_mean_s']*1000:>9.0f} ms | {st['ttfc_mean_ms']:>5.0f} ms | "
              f"{ns['rtf_mean']:>5.3f} | {ns['audio_dur_s']:>4.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

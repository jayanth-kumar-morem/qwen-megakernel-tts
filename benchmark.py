#!/usr/bin/env python3
"""Benchmark the megakernel TTS pipeline.

Measures:
  - TTFC (time to first audio chunk)
  - RTF (real-time factor)
  - Talker decode tok/s
  - End-to-end latency
  - Streaming chunk latencies

Usage:
    python benchmark.py
    python benchmark.py --text "Custom text" --runs 5
"""

import argparse
import asyncio
import time
import sys

import numpy as np
import torch


async def benchmark_streaming(engine, text: str, chunk_frames: int = 10):
    """Benchmark streaming synthesis."""
    t_start = time.perf_counter()
    ttfc = None
    chunk_times = []
    total_audio_sec = 0
    total_samples = 0

    async for audio_chunk, sr in engine.synthesize_streaming(text, chunk_frames=chunk_frames):
        t_now = time.perf_counter()
        if ttfc is None:
            ttfc = (t_now - t_start) * 1000  # ms
        chunk_times.append(t_now)
        total_samples += len(audio_chunk)
        total_audio_sec = total_samples / sr if sr > 0 else 0

    t_end = time.perf_counter()
    total_time = t_end - t_start
    rtf = total_time / total_audio_sec if total_audio_sec > 0 else float('inf')

    return {
        "ttfc_ms": ttfc or 0,
        "rtf": rtf,
        "total_time_s": total_time,
        "audio_duration_s": total_audio_sec,
        "num_chunks": len(chunk_times),
        "tokens_decoded": engine.talker.position,
    }


def benchmark_non_streaming(engine, text: str):
    """Benchmark non-streaming synthesis."""
    engine.talker.reset()

    t_start = time.perf_counter()
    waveform, sr = engine.synthesize(text)
    t_end = time.perf_counter()

    total_time = t_end - t_start
    audio_duration = len(waveform) / sr if sr > 0 and len(waveform) > 0 else 0
    rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
    tok_s = engine.talker.position / total_time if total_time > 0 else 0

    return {
        "total_time_s": total_time,
        "audio_duration_s": audio_duration,
        "rtf": rtf,
        "tokens_per_sec": tok_s,
        "tokens_decoded": engine.talker.position,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Megakernel TTS")
    parser.add_argument("--text", type=str,
                        default="The quick brown fox jumps over the lazy dog. "
                                "This is a benchmark of the megakernel text to speech system.",
                        help="Text to benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--chunk-frames", type=int, default=10, help="Frames per streaming chunk")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (not counted)")
    args = parser.parse_args()

    from qwen_megakernel.tts_engine import MegakernelTTSEngine, TTSConfig

    config = TTSConfig(
        model_path=args.model,
        chunk_frames=args.chunk_frames,
    )
    engine = MegakernelTTSEngine(config=config)

    print("=" * 60)
    print("Qwen3-TTS Megakernel Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Text: {args.text[:80]}...")
    print(f"Runs: {args.runs} (+ {args.warmup} warmup)")
    print()

    print("Initializing engine...")
    t0 = time.perf_counter()
    engine.initialize()
    print(f"Engine initialized in {time.perf_counter() - t0:.2f}s")
    print()

    # Warmup
    print(f"Running {args.warmup} warmup run(s)...")
    for _ in range(args.warmup):
        engine.synthesize(args.text)
    print()

    # Non-streaming benchmark
    print("--- Non-Streaming Benchmark ---")
    ns_results = []
    for i in range(args.runs):
        result = benchmark_non_streaming(engine, args.text)
        ns_results.append(result)
        print(f"  Run {i+1}: {result['total_time_s']:.3f}s, "
              f"RTF={result['rtf']:.3f}, "
              f"{result['tokens_per_sec']:.0f} tok/s, "
              f"{result['audio_duration_s']:.2f}s audio")

    avg_rtf = np.mean([r['rtf'] for r in ns_results])
    avg_toks = np.mean([r['tokens_per_sec'] for r in ns_results])
    print(f"  Average: RTF={avg_rtf:.3f}, {avg_toks:.0f} tok/s")
    print()

    # Streaming benchmark
    print("--- Streaming Benchmark ---")
    st_results = []
    for i in range(args.runs):
        result = asyncio.run(benchmark_streaming(engine, args.text, args.chunk_frames))
        st_results.append(result)
        print(f"  Run {i+1}: TTFC={result['ttfc_ms']:.1f}ms, "
              f"RTF={result['rtf']:.3f}, "
              f"{result['num_chunks']} chunks, "
              f"{result['audio_duration_s']:.2f}s audio")

    avg_ttfc = np.mean([r['ttfc_ms'] for r in st_results])
    avg_rtf_s = np.mean([r['rtf'] for r in st_results])
    print(f"  Average: TTFC={avg_ttfc:.1f}ms, RTF={avg_rtf_s:.3f}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  TTFC (streaming):     {avg_ttfc:.1f} ms  (target < 90 ms)")
    print(f"  RTF (streaming):      {avg_rtf_s:.3f}     (target < 0.3)")
    print(f"  RTF (non-streaming):  {avg_rtf:.3f}")
    print(f"  Decode tok/s:         {avg_toks:.0f}")
    print(f"  Audio sample rate:    {engine.sample_rate} Hz")
    print("=" * 60)


if __name__ == "__main__":
    main()

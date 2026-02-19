#!/usr/bin/env python3
"""Demo: Megakernel TTS streaming synthesis.

Demonstrates the full pipeline:
  Text input -> Megakernel TTS -> Streaming audio output

Usage:
    python demo_pipeline.py
    python demo_pipeline.py --text "Hello world"
    python demo_pipeline.py --output /tmp/demo.wav
"""

import argparse
import asyncio
import time

import numpy as np
import soundfile as sf
import torch

from qwen_megakernel.tts_engine import MegakernelTTSEngine, TTSConfig


async def run_streaming_demo(text, output_path, chunk_frames=10):
    """Run streaming TTS and save output."""
    print(f"\n{'='*60}")
    print(f"MEGAKERNEL TTS STREAMING DEMO")
    print(f"{'='*60}")
    print(f"Text: '{text}'")
    print(f"Chunk size: {chunk_frames} frames (~{chunk_frames/12.5:.1f}s per chunk)")
    print()

    config = TTSConfig(chunk_frames=chunk_frames)
    engine = MegakernelTTSEngine(config=config)

    print("Initializing engine...")
    t0 = time.perf_counter()
    engine.initialize()
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"Engine initialized in {init_ms:.0f}ms")

    print(f"\nStreaming synthesis...")
    audio_chunks = []

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    first_chunk_time = None

    async for audio_chunk, sr in engine.synthesize_streaming(text, chunk_frames=chunk_frames):
        torch.cuda.synchronize()
        t_now = time.perf_counter()
        if first_chunk_time is None:
            first_chunk_time = t_now
            ttfc_ms = (first_chunk_time - t_start) * 1000
            print(f"  TTFC: {ttfc_ms:.1f}ms")

        audio_chunks.append(audio_chunk)
        chunk_ms = (t_now - t_start) * 1000
        chunk_dur = len(audio_chunk) / sr
        print(f"  Chunk {len(audio_chunks):2d}: {len(audio_chunk):6d} samples ({chunk_dur:.2f}s) @ {chunk_ms:.0f}ms")

    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t_start) * 1000

    full_audio = np.concatenate(audio_chunks)
    sf.write(output_path, full_audio, sr)

    audio_duration = len(full_audio) / sr
    rtf = (total_ms / 1000) / audio_duration

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    ttfc = (first_chunk_time - t_start) * 1000 if first_chunk_time else 0
    print(f"  Audio:        {len(full_audio)} samples, {audio_duration:.2f}s @ {sr}Hz")
    print(f"  Chunks:       {len(audio_chunks)}")
    print(f"  Total time:   {total_ms:.1f}ms")
    print(f"  TTFC:         {ttfc:.1f}ms {'PASS' if ttfc < 90 else 'FAIL'} (target < 90ms)")
    print(f"  RTF:          {rtf:.3f} {'PASS' if rtf < 0.3 else 'FAIL'} (target < 0.3)")
    print(f"  Saved:        {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Megakernel TTS Demo")
    parser.add_argument("--text", default="Hello, this is a test of the megakernel text to speech engine.")
    parser.add_argument("--output", default="/tmp/tts_demo.wav")
    parser.add_argument("--chunk-frames", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(run_streaming_demo(args.text, args.output, args.chunk_frames))

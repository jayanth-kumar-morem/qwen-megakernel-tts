#!/usr/bin/env python3
"""Standalone TTS demo: text → .wav file.

Usage:
    python demo_tts.py "Hello, this is a test of the megakernel TTS system."
    python demo_tts.py --output output.wav "Your text here."
"""

import argparse
import time
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Megakernel Demo")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Model path")
    parser.add_argument("--no-sample", action="store_true", help="Use greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    from qwen_megakernel.tts_engine import MegakernelTTSEngine, TTSConfig

    config = TTSConfig(
        model_path=args.model,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    engine = MegakernelTTSEngine(config=config)

    print(f"Text: {args.text}")
    print(f"Initializing engine...")
    t0 = time.perf_counter()
    engine.initialize()
    t_init = time.perf_counter() - t0
    print(f"Engine initialized in {t_init:.2f}s")

    print(f"Generating speech...")
    t0 = time.perf_counter()
    waveform, sr = engine.synthesize(args.text)
    t_gen = time.perf_counter() - t0

    duration = len(waveform) / sr if sr > 0 else 0
    print(f"Generated {duration:.2f}s of audio in {t_gen:.3f}s")
    print(f"RTF: {t_gen / duration:.3f}" if duration > 0 else "RTF: N/A")
    print(f"Tokens decoded: {engine.talker.position}")

    if len(waveform) > 0:
        import soundfile as sf
        sf.write(args.output, waveform, sr)
        print(f"Saved to {args.output}")
    else:
        print("Warning: No audio generated")


if __name__ == "__main__":
    main()

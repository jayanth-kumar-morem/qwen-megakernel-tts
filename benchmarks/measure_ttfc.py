#!/usr/bin/env python3
"""Measure Time-To-First-Chunk (TTFC) for streaming TTS.

TTFC = time from text input to first audio chunk emitted.
Includes: tokenization + prefill + first talker decode + first code predictor
+ first vocoder decode.

Target: < 90ms on RTX 5090.

Usage:
    python -m benchmarks.measure_ttfc
    python -m benchmarks.measure_ttfc --runs 10 --text "Hello world"
"""

import argparse
import asyncio
import time

import torch


def measure_ttfc_breakdown(engine, text: str) -> dict:
    """Measure TTFC with per-component breakdown.

    Returns timing for each phase: tokenize, prefill, first_decode, first_code_pred, first_vocoder.
    """
    from qwen_megakernel.model_tts import (
        CODEC_BOS, CODEC_EOS, HIDDEN_SIZE, NUM_CODE_GROUPS,
        build_prefill_embeddings,
    )

    cfg = engine.config
    engine.talker.reset()

    torch.cuda.synchronize()

    # Phase 1: Tokenize
    t0 = time.perf_counter()
    formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = engine.tokenizer.encode(formatted_text, return_tensors="pt")[0]
    text_ids = text_ids.to(engine.device)
    t_tokenize = time.perf_counter() - t0

    # Phase 2: Build prefill embeddings
    t0 = time.perf_counter()
    prefill_embeds, trailing_text = build_prefill_embeddings(
        text_ids, engine.text_projection, engine._talker_embed,
        device=engine.device,
    )
    torch.cuda.synchronize()
    t_embed_build = time.perf_counter() - t0

    # Phase 3: Prefill (feed all prefill embeddings)
    t0 = time.perf_counter()
    for i in range(prefill_embeds.shape[0]):
        engine.talker.step_with_embed(prefill_embeds[i])
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    # Phase 4: First talker decode step
    t0 = time.perf_counter()
    first_token, hidden = engine.talker.step(CODEC_BOS)
    torch.cuda.synchronize()
    t_first_decode = time.perf_counter() - t0

    # Phase 5: First code predictor
    t0 = time.perf_counter()
    if first_token != CODEC_EOS:
        all_codes = engine.code_predictor.predict(
            talker_hidden=hidden,
            first_codebook_token=first_token,
            talker_embed_weight=engine._talker_embed,
            do_sample=cfg.subtalker_do_sample,
            temperature=cfg.subtalker_temperature,
            top_k=cfg.subtalker_top_k,
        )
    torch.cuda.synchronize()
    t_code_pred = time.perf_counter() - t0

    # Phase 6: First vocoder decode (1 frame)
    t0 = time.perf_counter()
    if first_token != CODEC_EOS and engine.speech_tokenizer is not None:
        engine._decode_to_audio([all_codes])
    torch.cuda.synchronize()
    t_vocoder = time.perf_counter() - t0

    total = t_tokenize + t_embed_build + t_prefill + t_first_decode + t_code_pred + t_vocoder

    return {
        "total_ms": total * 1000,
        "tokenize_ms": t_tokenize * 1000,
        "embed_build_ms": t_embed_build * 1000,
        "prefill_ms": t_prefill * 1000,
        "first_decode_ms": t_first_decode * 1000,
        "code_predictor_ms": t_code_pred * 1000,
        "vocoder_ms": t_vocoder * 1000,
        "prefill_tokens": prefill_embeds.shape[0],
        "text_tokens": len(text_ids),
    }


async def measure_ttfc_streaming(engine, text: str) -> float:
    """Measure TTFC using the actual streaming API. Returns milliseconds."""
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    async for audio_chunk, sr in engine.synthesize_streaming(text, chunk_frames=1):
        torch.cuda.synchronize()
        ttfc = (time.perf_counter() - t_start) * 1000
        # We only need the first chunk
        return ttfc

    return float('inf')


def main():
    parser = argparse.ArgumentParser(description="Measure TTFC")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--text", type=str, default="Hello, how are you today?")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    from qwen_megakernel.tts_engine import MegakernelTTSEngine, TTSConfig

    config = TTSConfig(model_path=args.model)
    engine = MegakernelTTSEngine(config=config)

    print("Initializing engine...")
    engine.initialize()

    print(f"\n{'='*60}")
    print("Time-To-First-Chunk (TTFC) Measurement")
    print(f"{'='*60}")
    print(f"Text: \"{args.text}\"")
    print(f"Runs: {args.runs} (+ {args.warmup} warmup)")

    # Warmup
    for _ in range(args.warmup):
        engine.synthesize(args.text)

    # Breakdown measurement
    print(f"\n--- Component Breakdown ---")
    breakdown_results = []
    for i in range(args.runs):
        r = measure_ttfc_breakdown(engine, args.text)
        breakdown_results.append(r)
        print(f"  Run {i+1}: {r['total_ms']:.1f}ms total "
              f"(tok={r['tokenize_ms']:.1f}, emb={r['embed_build_ms']:.1f}, "
              f"prefill={r['prefill_ms']:.1f}, decode={r['first_decode_ms']:.1f}, "
              f"cp={r['code_predictor_ms']:.1f}, voc={r['vocoder_ms']:.1f})")

    avg = {k: sum(r[k] for r in breakdown_results) / len(breakdown_results)
           for k in breakdown_results[0] if k.endswith("_ms") or k.endswith("_tokens")}

    print(f"\n--- Average Breakdown ---")
    print(f"  Tokenize:        {avg['tokenize_ms']:6.2f} ms")
    print(f"  Embed build:     {avg['embed_build_ms']:6.2f} ms")
    print(f"  Prefill:         {avg['prefill_ms']:6.2f} ms  ({avg.get('prefill_tokens', 0):.0f} tokens)")
    print(f"  First decode:    {avg['first_decode_ms']:6.2f} ms")
    print(f"  Code predictor:  {avg['code_predictor_ms']:6.2f} ms")
    print(f"  Vocoder:         {avg['vocoder_ms']:6.2f} ms")
    print(f"  ────────────────────────")
    print(f"  TOTAL TTFC:      {avg['total_ms']:6.2f} ms  (target < 90 ms)")

    # Streaming API measurement
    print(f"\n--- Streaming API TTFC ---")
    streaming_ttfcs = []
    for i in range(args.runs):
        ttfc = asyncio.run(measure_ttfc_streaming(engine, args.text))
        streaming_ttfcs.append(ttfc)
        print(f"  Run {i+1}: {ttfc:.1f}ms")

    avg_streaming = sum(streaming_ttfcs) / len(streaming_ttfcs)
    print(f"  Average: {avg_streaming:.1f}ms")

    # Verdict
    print(f"\n{'='*60}")
    target = 90.0
    passed = avg["total_ms"] < target
    status = "PASS" if passed else "FAIL"
    print(f"TTFC: {avg['total_ms']:.1f}ms — {status} (target < {target}ms)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

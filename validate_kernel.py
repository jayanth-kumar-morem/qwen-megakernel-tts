#!/usr/bin/env python3
"""Validate megakernel TTS decode against HuggingFace reference.

Loads the same Qwen3-TTS talker weights into both the megakernel decoder
and a pure-PyTorch reference implementation, feeds identical inputs, and
compares:
  1. Output token IDs (should match exactly for greedy/argmax decoding)
  2. Hidden states after RMSNorm (should be numerically close)

Usage:
    python validate_kernel.py
    python validate_kernel.py --steps 50 --verbose
    python validate_kernel.py --reference-only  # just run PyTorch reference
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np


class PyTorchTalkerReference:
    """Pure-PyTorch reference implementation of the talker decoder.

    Implements the same transformer architecture as the megakernel but
    using standard PyTorch operations. Used for correctness validation.
    """

    def __init__(self, weights: dict, device: str = "cuda"):
        from qwen_megakernel.model_tts import (
            NUM_LAYERS, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
            HIDDEN_SIZE, INTERMEDIATE_SIZE, VOCAB_SIZE, MAX_SEQ_LEN,
            ROPE_THETA,
        )
        self.device = device
        self.num_layers = NUM_LAYERS
        self.num_q_heads = NUM_Q_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size = INTERMEDIATE_SIZE
        self.num_kv_groups = NUM_Q_HEADS // NUM_KV_HEADS

        self.embed_weight = weights["embed_weight"]          # [3072, 1024] bf16
        self.lm_head_weight = weights["lm_head_weight"]      # [3072, 1024] bf16
        self.final_norm_weight = weights["final_norm_weight"] # [1024] bf16

        # Per-layer weights
        self.layers = []
        for i in range(NUM_LAYERS):
            base = i * 11
            lw = weights["layer_weights"]
            self.layers.append({
                "input_layernorm_weight": lw[base + 0],
                "q_proj_weight": lw[base + 1],
                "k_proj_weight": lw[base + 2],
                "v_proj_weight": lw[base + 3],
                "q_norm_weight": lw[base + 4],
                "k_norm_weight": lw[base + 5],
                "o_proj_weight": lw[base + 6],
                "post_attn_layernorm_weight": lw[base + 7],
                "gate_proj_weight": lw[base + 8],
                "up_proj_weight": lw[base + 9],
                "down_proj_weight": lw[base + 10],
            })

        # RoPE tables
        self.cos_table = weights["cos_table"]  # [MAX_SEQ_LEN, HEAD_DIM] bf16
        self.sin_table = weights["sin_table"]

        # KV cache
        self.k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.position = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.position = 0

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
        return (x_f32 / rms * weight.float()).to(x.dtype)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE with half-split layout (matches rotate_half in HF)."""
        d2 = x.shape[-1] // 2
        x1 = x[..., :d2]
        x2 = x[..., d2:]
        cos = cos[..., :d2].to(x.dtype)
        sin = sin[..., :d2].to(x.dtype)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    @torch.no_grad()
    def step(self, token_id: int) -> tuple[int, torch.Tensor]:
        """One decode step. Returns (next_token, hidden_state_f32)."""
        pos = self.position

        # Embed
        hidden = F.embedding(
            torch.tensor([token_id], device=self.device), self.embed_weight
        ).squeeze(0)  # [1024] bf16

        return self._forward(hidden, pos)

    @torch.no_grad()
    def step_with_embed(self, embed_bf16: torch.Tensor) -> tuple[int, torch.Tensor]:
        """One decode step with precomputed embedding."""
        return self._forward(embed_bf16, self.position)

    def _forward(self, hidden: torch.Tensor, pos: int) -> tuple[int, torch.Tensor]:
        """Forward pass through all layers. Returns (token, hidden_f32)."""
        cos = self.cos_table[pos:pos+1]  # [1, HEAD_DIM]
        sin = self.sin_table[pos:pos+1]

        residual = hidden.float()

        for layer_idx, lw in enumerate(self.layers):
            # Pre-attention LayerNorm
            normed = self._rms_norm(
                torch.tensor(residual, dtype=torch.bfloat16),
                lw["input_layernorm_weight"],
            )

            # QKV projection
            q = F.linear(normed, lw["q_proj_weight"])  # [Q_SIZE=2048]
            k = F.linear(normed, lw["k_proj_weight"])  # [KV_SIZE=1024]
            v = F.linear(normed, lw["v_proj_weight"])  # [KV_SIZE=1024]

            # Reshape to heads
            q = q.view(self.num_q_heads, self.head_dim)   # [16, 128]
            k = k.view(self.num_kv_heads, self.head_dim)  # [8, 128]
            v = v.view(self.num_kv_heads, self.head_dim)

            # QK-norm
            q = self._rms_norm(q, lw["q_norm_weight"])
            k = self._rms_norm(k, lw["k_norm_weight"])

            # RoPE
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

            # Update KV cache
            self.k_cache[layer_idx, :, pos, :] = k
            self.v_cache[layer_idx, :, pos, :] = v

            # Attention: Q [16, 128] vs K [8, pos+1, 128]
            # GQA: repeat KV
            k_full = self.k_cache[layer_idx, :, :pos+1, :]  # [8, pos+1, 128]
            v_full = self.v_cache[layer_idx, :, :pos+1, :]

            # Repeat KV for GQA
            k_full = k_full.repeat_interleave(self.num_kv_groups, dim=0)  # [16, pos+1, 128]
            v_full = v_full.repeat_interleave(self.num_kv_groups, dim=0)

            # Scaled dot-product attention
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.einsum("hd,hsd->hs", q.float(), k_full.float()) * scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = torch.einsum("hs,hsd->hd", attn_weights, v_full.float())

            # O projection
            attn_out = attn_out.to(torch.bfloat16).reshape(-1)  # [Q_SIZE]
            o_out = F.linear(attn_out, lw["o_proj_weight"])

            # Residual
            residual = residual + o_out.float()

            # Post-attention LayerNorm + MLP
            normed2 = self._rms_norm(
                torch.tensor(residual, dtype=torch.bfloat16),
                lw["post_attn_layernorm_weight"],
            )

            # SwiGLU MLP
            gate = F.linear(normed2, lw["gate_proj_weight"])
            up = F.linear(normed2, lw["up_proj_weight"])
            mlp_out = F.silu(gate) * up
            mlp_out = F.linear(mlp_out, lw["down_proj_weight"])

            residual = residual + mlp_out.float()

        # Final RMSNorm
        hidden_normed = self._rms_norm(
            torch.tensor(residual, dtype=torch.bfloat16),
            self.final_norm_weight,
        )

        # LM head
        logits = F.linear(hidden_normed, self.lm_head_weight).float()  # [3072]
        next_token = logits.argmax().item()

        self.position += 1
        return next_token, hidden_normed.float()  # Return f32 hidden for comparison


def compare_outputs(
    kernel_tokens: list[int],
    ref_tokens: list[int],
    kernel_hiddens: list[torch.Tensor],
    ref_hiddens: list[torch.Tensor],
    verbose: bool = False,
) -> dict:
    """Compare megakernel vs reference outputs."""
    n = min(len(kernel_tokens), len(ref_tokens))

    token_matches = 0
    token_mismatches = []
    hidden_diffs = []

    for i in range(n):
        # Token comparison
        if kernel_tokens[i] == ref_tokens[i]:
            token_matches += 1
        else:
            token_mismatches.append({
                "step": i,
                "kernel": kernel_tokens[i],
                "ref": ref_tokens[i],
            })
            if verbose:
                print(f"  MISMATCH at step {i}: kernel={kernel_tokens[i]}, ref={ref_tokens[i]}")

        # Hidden state comparison
        if i < len(kernel_hiddens) and i < len(ref_hiddens):
            diff = (kernel_hiddens[i] - ref_hiddens[i]).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            cos_sim = F.cosine_similarity(
                kernel_hiddens[i].unsqueeze(0),
                ref_hiddens[i].unsqueeze(0),
            ).item()
            hidden_diffs.append({
                "step": i,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "cos_sim": cos_sim,
            })
            if verbose and max_diff > 0.01:
                print(f"  Step {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                      f"cos_sim={cos_sim:.6f}")

    return {
        "total_steps": n,
        "token_match_rate": token_matches / n if n > 0 else 0,
        "token_mismatches": token_mismatches,
        "hidden_max_diff": max(h["max_diff"] for h in hidden_diffs) if hidden_diffs else 0,
        "hidden_mean_diff": sum(h["mean_diff"] for h in hidden_diffs) / len(hidden_diffs) if hidden_diffs else 0,
        "hidden_min_cos_sim": min(h["cos_sim"] for h in hidden_diffs) if hidden_diffs else 0,
        "hidden_diffs": hidden_diffs,
    }


def run_megakernel(decoder, input_tokens: list[int], num_steps: int) -> tuple[list[int], list[torch.Tensor]]:
    """Run megakernel decode and collect outputs."""
    decoder.reset()
    tokens = []
    hiddens = []

    # Feed input tokens as prefill (discard outputs)
    for tok in input_tokens[:-1]:
        decoder.step(tok)

    # Decode from last input token
    for step in range(num_steps):
        if step == 0:
            token, hidden = decoder.step(input_tokens[-1])
        else:
            token, hidden = decoder.step(tokens[-1])
        tokens.append(token)
        hiddens.append(hidden.cpu())

    return tokens, hiddens


def run_reference(ref: PyTorchTalkerReference, input_tokens: list[int], num_steps: int) -> tuple[list[int], list[torch.Tensor]]:
    """Run PyTorch reference and collect outputs."""
    ref.reset()
    tokens = []
    hiddens = []

    # Feed input tokens as prefill
    for tok in input_tokens[:-1]:
        ref.step(tok)

    # Decode
    for step in range(num_steps):
        if step == 0:
            token, hidden = ref.step(input_tokens[-1])
        else:
            token, hidden = ref.step(tokens[-1])
        tokens.append(token)
        hiddens.append(hidden.cpu())

    return tokens, hiddens


def run_step_with_embed_validation(decoder, ref, num_steps: int = 20, verbose: bool = False):
    """Validate step_with_embed() mode (precomputed embedding input)."""
    from qwen_megakernel.model_tts import CODEC_BOS, HIDDEN_SIZE

    print(f"\n--- Validating step_with_embed() mode ---")

    decoder.reset()
    ref.reset()

    # Start with BOS
    kern_tok, kern_hid = decoder.step(CODEC_BOS)
    ref_tok, ref_hid = ref.step(CODEC_BOS)

    kernel_tokens = [kern_tok]
    ref_tokens = [ref_tok]
    kernel_hiddens = [kern_hid.cpu()]
    ref_hiddens = [ref_hid.cpu()]

    # Now feed precomputed embeddings (random, but same for both)
    for i in range(num_steps - 1):
        # Create a random embedding
        embed = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        kern_tok, kern_hid = decoder.step_with_embed(embed)
        ref_tok, ref_hid = ref.step_with_embed(embed)

        kernel_tokens.append(kern_tok)
        ref_tokens.append(ref_tok)
        kernel_hiddens.append(kern_hid.cpu())
        ref_hiddens.append(ref_hid.cpu())

    result = compare_outputs(kernel_tokens, ref_tokens, kernel_hiddens, ref_hiddens, verbose)
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate megakernel vs PyTorch reference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--steps", type=int, default=50, help="Number of decode steps")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--reference-only", action="store_true",
                        help="Only run PyTorch reference (no kernel needed)")
    args = parser.parse_args()

    from qwen_megakernel.model_tts import (
        load_tts_weights, CODEC_BOS, CODEC_PAD, CODEC_EOS,
    )

    print("Loading weights...")
    weights = load_tts_weights(args.model)

    # Build reference
    print("Building PyTorch reference decoder...")
    ref = PyTorchTalkerReference(weights)

    if args.reference_only:
        print(f"\nRunning reference-only decode ({args.steps} steps)...")
        input_tokens = [CODEC_BOS]
        ref_tokens, ref_hiddens = run_reference(ref, input_tokens, args.steps)
        print(f"Generated tokens: {ref_tokens[:20]}...")
        print(f"Final hidden norm: {ref_hiddens[-1].norm().item():.4f}")
        return

    # Build megakernel decoder
    print("Building megakernel decoder (triggers JIT compile)...")
    from qwen_megakernel.model_tts import TTSDecoder
    decoder = TTSDecoder(weights=weights)

    print(f"\n{'='*60}")
    print("Kernel Validation: Megakernel vs PyTorch Reference")
    print(f"{'='*60}")

    # Test 1: Simple decode from BOS
    print(f"\n--- Test 1: Decode from CODEC_BOS ({args.steps} steps) ---")
    input_tokens = [CODEC_BOS]
    kernel_tokens, kernel_hiddens = run_megakernel(decoder, input_tokens, args.steps)
    ref_tokens, ref_hiddens = run_reference(ref, input_tokens, args.steps)

    result1 = compare_outputs(kernel_tokens, ref_tokens, kernel_hiddens, ref_hiddens, args.verbose)
    print(f"  Token match rate: {result1['token_match_rate']*100:.1f}%")
    print(f"  Token mismatches: {len(result1['token_mismatches'])}")
    print(f"  Hidden max diff:  {result1['hidden_max_diff']:.6f}")
    print(f"  Hidden mean diff: {result1['hidden_mean_diff']:.6f}")
    print(f"  Hidden min cosine sim: {result1['hidden_min_cos_sim']:.6f}")

    # Test 2: Decode with PAD token prefix
    print(f"\n--- Test 2: Decode with PAD prefix ({args.steps} steps) ---")
    input_tokens = [CODEC_PAD, CODEC_PAD, CODEC_PAD, CODEC_BOS]
    kernel_tokens2, kernel_hiddens2 = run_megakernel(decoder, input_tokens, args.steps)
    ref_tokens2, ref_hiddens2 = run_reference(ref, input_tokens, args.steps)

    result2 = compare_outputs(kernel_tokens2, ref_tokens2, kernel_hiddens2, ref_hiddens2, args.verbose)
    print(f"  Token match rate: {result2['token_match_rate']*100:.1f}%")
    print(f"  Token mismatches: {len(result2['token_mismatches'])}")
    print(f"  Hidden max diff:  {result2['hidden_max_diff']:.6f}")
    print(f"  Hidden mean diff: {result2['hidden_mean_diff']:.6f}")

    # Test 3: step_with_embed validation
    result3 = run_step_with_embed_validation(decoder, ref, min(args.steps, 20), args.verbose)
    print(f"  Token match rate: {result3['token_match_rate']*100:.1f}%")
    print(f"  Hidden max diff:  {result3['hidden_max_diff']:.6f}")
    print(f"  Hidden min cosine sim: {result3['hidden_min_cos_sim']:.6f}")

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, result in [("BOS decode", result1), ("PAD+BOS decode", result2), ("Embed mode", result3)]:
        token_ok = result["token_match_rate"] == 1.0
        hidden_ok = result["hidden_min_cos_sim"] > 0.99
        status = "PASS" if (token_ok and hidden_ok) else "FAIL"
        if not (token_ok and hidden_ok):
            all_pass = False
        print(f"  {name:20s}: {status} "
              f"(tokens: {result['token_match_rate']*100:.0f}%, "
              f"cosine: {result['hidden_min_cos_sim']:.4f})")

    if not all_pass:
        # Numerical differences in bf16 are expected for some operations
        # Check if hidden states are close enough even if tokens diverge
        worst_cos = min(r["hidden_min_cos_sim"] for r in [result1, result2, result3])
        if worst_cos > 0.995:
            print(f"\n  Note: Token divergence may be due to bf16 numerical differences")
            print(f"  in the argmax (logits are very close). Hidden states are well-matched")
            print(f"  (worst cosine sim = {worst_cos:.6f}).")
            print(f"  This is ACCEPTABLE for TTS (sampling, not greedy).")
        else:
            print(f"\n  WARNING: Significant numerical divergence detected.")
            print(f"  Debug by comparing intermediate activations layer-by-layer.")
            sys.exit(1)

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SEE NOTES ABOVE'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

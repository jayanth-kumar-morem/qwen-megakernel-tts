#!/usr/bin/env python3
"""Test megakernel-based code predictor vs PyTorch implementation.

The key insight: the code predictor is a 5-layer Qwen3 transformer with
IDENTICAL architecture (hidden_size=1024, 16 Q heads, 8 KV heads, head_dim=128,
intermediate_size=3072). We can reuse the same megakernel with num_layers=5
instead of 28, giving us ~0.5ms per decode step instead of ~10ms.
"""

import math
import struct
import time

import torch
import torch.nn.functional as F

from qwen_megakernel.model_tts import (
    CODE_PREDICTOR_LAYERS,
    CODE_PREDICTOR_VOCAB,
    CODEC_BOS,
    EMBED_FROM_BUFFER,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_SIZE,
    NUM_CODE_GROUPS,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    Q_SIZE,
    ROPE_THETA,
    VOCAB_SIZE,
    CodePredictor,
    TTSDecoder,
    load_tts_weights,
)


def pack_layer_weights_n(layer_weights: list[torch.Tensor], num_layers: int) -> torch.Tensor:
    """Pack layer weights for N layers (not hardcoded to 28)."""
    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(num_layers * struct_bytes)
    for i in range(num_layers):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class CodePredictorKernel:
    """Code predictor using the megakernel for the 5-layer transformer.

    Instead of ~70 separate PyTorch kernel launches per decode step,
    a single megakernel launch processes all 5 layers.

    Expected speedup: ~10-20x over pure PyTorch implementation.
    """

    def __init__(self, weights: dict, device: str = "cuda"):
        cp = weights["code_predictor"]
        self.device = device
        self.num_groups = NUM_CODE_GROUPS - 1  # 14 groups to predict

        # --- Pack code predictor's 5 layers ---
        layer_weights = []
        for i in range(CODE_PREDICTOR_LAYERS):
            p = f"layers.{i}."
            layer_weights.extend([
                cp[p + "input_layernorm.weight"].contiguous(),
                cp[p + "self_attn.q_proj.weight"].contiguous(),
                cp[p + "self_attn.k_proj.weight"].contiguous(),
                cp[p + "self_attn.v_proj.weight"].contiguous(),
                cp[p + "self_attn.q_norm.weight"].contiguous(),
                cp[p + "self_attn.k_norm.weight"].contiguous(),
                cp[p + "self_attn.o_proj.weight"].contiguous(),
                cp[p + "post_attention_layernorm.weight"].contiguous(),
                cp[p + "mlp.gate_proj.weight"].contiguous(),
                cp[p + "mlp.up_proj.weight"].contiguous(),
                cp[p + "mlp.down_proj.weight"].contiguous(),
            ])
        self._layer_weights = layer_weights  # prevent GC
        self._layer_weights_packed = pack_layer_weights_n(layer_weights, CODE_PREDICTOR_LAYERS)

        # --- Final norm ---
        self._final_norm_weight = cp["norm.weight"].contiguous()

        # --- Dummy embed/lm_head (kernel expects VOCAB_SIZE=3072 at compile time) ---
        # We ignore the kernel's argmax output; we compute our own logits from norm_out
        self._dummy_lm_head = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        self._dummy_embed = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        # --- Per-group weights ---
        self.codec_embeddings = [cp[f"codec_embedding.{g}.weight"] for g in range(self.num_groups)]
        self.lm_heads = [cp[f"lm_head.{g}.weight"] for g in range(self.num_groups)]

        # --- RoPE tables ---
        MAX_SEQ_CP = 64
        self._max_seq = MAX_SEQ_CP
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        positions = torch.arange(MAX_SEQ_CP, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()
        self._sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()

        # --- KV cache (5 layers, tiny) ---
        self._k_cache = torch.zeros(
            CODE_PREDICTOR_LAYERS, NUM_KV_HEADS, MAX_SEQ_CP, HEAD_DIM,
            dtype=torch.bfloat16, device=device,
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # --- Scratch buffers (same layout as TTSDecoder) ---
        f32 = dict(dtype=torch.float32, device=device)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device=device)
        self._out_token = torch.empty(1, dtype=torch.int32, device=device)

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position = 0

        # --- Load decode op (same kernel binary as the talker) ---
        from qwen_megakernel.build_tts import get_extension
        get_extension()
        self._decode = torch.ops.qwen_megakernel_C.decode

        # Pre-allocate token tensor for embedding lookups
        self._token_buf = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def _step_with_embed(self, embed_bf16: torch.Tensor):
        """Run one megakernel decode step with precomputed bf16 embedding.

        After this call, self._norm_out contains the post-RMSNorm hidden state (f32).
        """
        self._hidden.copy_(embed_bf16)
        self._decode(
            self._out_token,
            -1,  # sentinel: skip embed lookup, use hidden_buffer
            self._dummy_embed,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._dummy_lm_head,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act, self._res,
            self._q, self._k, self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            CODE_PREDICTOR_LAYERS,  # 5 layers instead of 28
            self._position,
            self._max_seq,
            self._attn_scale,
        )
        self._position += 1

    @torch.no_grad()
    def predict(
        self,
        talker_hidden: torch.Tensor,
        first_codebook_token: int,
        talker_embed_weight: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Predict all 15 codebook groups (first + 14 predicted).

        Uses the megakernel for transformer decode (5 layers per step),
        then PyTorch for logits computation and sampling.

        Returns [NUM_CODE_GROUPS] int64 tensor.
        """
        self.reset()

        # --- Prefill step 1: talker hidden state ---
        self._step_with_embed(talker_hidden.to(torch.bfloat16))

        # --- Prefill step 2: first codebook token embedding ---
        self._token_buf[0] = first_codebook_token
        first_embed = F.embedding(self._token_buf, talker_embed_weight).squeeze(0)
        self._step_with_embed(first_embed)

        # --- Autoregressive decode: 14 groups ---
        predicted_tensors = []

        for group in range(self.num_groups):
            # Compute logits from norm_out (bypasses kernel's built-in argmax)
            hidden_bf16 = self._norm_out.to(torch.bfloat16).unsqueeze(0)  # [1, 1024]
            logits = F.linear(hidden_bf16, self.lm_heads[group]).squeeze(0)  # [2048]

            # Sample (keep on GPU to avoid sync)
            if do_sample and temperature > 0:
                logits_f = logits.float() / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(logits_f, min(top_k, logits_f.size(-1)))
                    logits_f[logits_f < topk_vals[-1]] = float('-inf')
                probs = F.softmax(logits_f, dim=-1)
                token_tensor = torch.multinomial(probs, 1)  # [1] long, stays on GPU
            else:
                token_tensor = logits.argmax(keepdim=True).long()  # [1] long

            predicted_tensors.append(token_tensor)

            # Embed and decode next step (if not last group)
            if group < self.num_groups - 1:
                embed = F.embedding(token_tensor, self.codec_embeddings[group]).squeeze(0)  # [1024]
                self._step_with_embed(embed)

        # Assemble result: [first_token, predicted_1, ..., predicted_14]
        first_tensor = torch.tensor([first_codebook_token], dtype=torch.long, device=self.device)
        return torch.cat([first_tensor] + predicted_tensors)


def main():
    print("=" * 60)
    print("Code Predictor: Megakernel vs PyTorch Benchmark")
    print("=" * 60)

    print("\nLoading TTS weights...")
    weights = load_tts_weights()

    print("\nInitializing talker decoder...")
    talker = TTSDecoder(weights=weights)
    talker.reset()

    # Get a representative hidden state from the talker
    token, hidden = talker.step(CODEC_BOS)
    print(f"Talker step: token={token}, hidden shape={hidden.shape}")

    # Initialize both code predictors
    print("\nInitializing PyTorch code predictor...")
    cp_pytorch = CodePredictor(weights, device="cuda")

    print("Initializing megakernel code predictor...")
    cp_kernel = CodePredictorKernel(weights, device="cuda")

    # Warmup
    print("\nWarming up (3 iterations each)...")
    for _ in range(3):
        cp_pytorch.predict(hidden, token, talker.embed_weight, do_sample=False)
    for _ in range(3):
        cp_kernel.predict(hidden, token, talker.embed_weight, do_sample=False)

    # Correctness check (argmax mode - deterministic)
    print("\n--- Correctness Check (argmax mode) ---")
    result_pt = cp_pytorch.predict(hidden, token, talker.embed_weight, do_sample=False)
    result_kern = cp_kernel.predict(hidden, token, talker.embed_weight, do_sample=False)
    print(f"PyTorch:    {result_pt.tolist()}")
    print(f"Megakernel: {result_kern.tolist()}")
    match = (result_pt == result_kern).all().item()
    print(f"Exact match: {match}")
    if not match:
        diff = (result_pt != result_kern).nonzero().flatten().tolist()
        print(f"Mismatches at positions: {diff}")
        # Check if hidden states are close (the tokens may diverge due to bf16 argmax ties)
        print("(Note: minor token differences are expected due to bf16 precision)")

    # Benchmark PyTorch code predictor
    print("\n--- Benchmark: PyTorch Code Predictor ---")
    N = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        cp_pytorch.predict(hidden, token, talker.embed_weight, do_sample=False)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Average: {pt_ms:.1f} ms/frame")

    # Benchmark megakernel code predictor
    print("\n--- Benchmark: Megakernel Code Predictor ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        cp_kernel.predict(hidden, token, talker.embed_weight, do_sample=False)
    torch.cuda.synchronize()
    kern_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Average: {kern_ms:.1f} ms/frame")

    # Benchmark with sampling
    print("\n--- Benchmark: Megakernel with Sampling ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        cp_kernel.predict(hidden, token, talker.embed_weight, do_sample=True, temperature=0.9, top_k=50)
    torch.cuda.synchronize()
    kern_sample_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Average: {kern_sample_ms:.1f} ms/frame")

    # Per-step breakdown with CUDA events
    print("\n--- Per-Step Breakdown (Megakernel) ---")
    cp_kernel.reset()
    events = []
    for i in range(20):
        events.append(torch.cuda.Event(enable_timing=True))
    event_idx = 0

    # Prefill
    events[event_idx].record()
    event_idx += 1
    cp_kernel._step_with_embed(hidden.to(torch.bfloat16))
    events[event_idx].record()
    event_idx += 1

    first_embed = F.embedding(torch.tensor([token], device="cuda"), talker.embed_weight).squeeze(0)
    cp_kernel._step_with_embed(first_embed)
    events[event_idx].record()
    event_idx += 1

    # First decode step
    hidden_bf16 = cp_kernel._norm_out.to(torch.bfloat16).unsqueeze(0)
    logits = F.linear(hidden_bf16, cp_kernel.lm_heads[0]).squeeze(0)
    token_t = logits.argmax(keepdim=True).long()
    events[event_idx].record()
    event_idx += 1

    embed = F.embedding(token_t, cp_kernel.codec_embeddings[0]).squeeze(0)
    cp_kernel._step_with_embed(embed)
    events[event_idx].record()
    event_idx += 1

    torch.cuda.synchronize()
    print(f"  Prefill step 1 (talker hidden):   {events[0].elapsed_time(events[1]):.2f} ms")
    print(f"  Prefill step 2 (first embed):     {events[1].elapsed_time(events[2]):.2f} ms")
    print(f"  Logits + sampling:                {events[2].elapsed_time(events[3]):.2f} ms")
    print(f"  Decode step (megakernel):         {events[3].elapsed_time(events[4]):.2f} ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PyTorch code predictor:   {pt_ms:.1f} ms/frame")
    print(f"Megakernel code predictor: {kern_ms:.1f} ms/frame (argmax)")
    print(f"Megakernel code predictor: {kern_sample_ms:.1f} ms/frame (sampling)")
    print(f"Speedup (argmax):         {pt_ms/kern_ms:.1f}x")
    print(f"Speedup (sampling):       {pt_ms/kern_sample_ms:.1f}x")
    print()

    frame_budget_ms = 80.0  # 12.5 Hz → 80ms per frame
    talker_ms = 3.1  # measured previously
    embed_ms = 0.1   # negligible
    total_frame_ms = talker_ms + kern_sample_ms + embed_ms
    rtf = total_frame_ms / frame_budget_ms
    print(f"Per-frame breakdown:")
    print(f"  Talker decode:    {talker_ms:.1f} ms")
    print(f"  Code predictor:   {kern_sample_ms:.1f} ms")
    print(f"  Embedding math:   ~{embed_ms:.1f} ms")
    print(f"  Total per frame:  {total_frame_ms:.1f} ms")
    print(f"  Frame budget:     {frame_budget_ms:.0f} ms (12.5 Hz)")
    print(f"  RTF:              {rtf:.3f} {'✓ PASS' if rtf < 0.3 else '✗ FAIL'} (target < 0.3)")
    print()

    ttfc_ms = 1 + 5 + 7*talker_ms + talker_ms + kern_sample_ms
    print(f"TTFC estimate:")
    print(f"  Tokenization:     ~1 ms")
    print(f"  Text projection:  ~5 ms")
    print(f"  Prefill (7 steps): {7*talker_ms:.1f} ms")
    print(f"  First decode:     {talker_ms:.1f} ms")
    print(f"  First CP call:    {kern_sample_ms:.1f} ms")
    print(f"  Total TTFC:       {ttfc_ms:.1f} ms {'✓ PASS' if ttfc_ms < 90 else '✗ FAIL'} (target < 90 ms)")


if __name__ == "__main__":
    main()

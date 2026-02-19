"""Weight loading and decode API for Qwen3-TTS talker decoder on the megakernel.

This module adapts the original Qwen3-0.6B megakernel to serve the talker
decoder of Qwen3-TTS-12Hz-0.6B-Base. The talker decoder has identical
transformer dimensions but differs in:
  - vocab_size: 3072 (codec tokens) vs 151936 (text tokens)
  - rope_theta: 1000000 vs 10000
  - lm_head: untied from embedding (separate codec_head weight)
  - Input: precomputed embedding (sum of codec embeddings + text) vs token ID
"""

import math
import struct
from typing import Optional

import torch

# ─── Talker decoder constants (identical to Qwen3-0.6B except where noted) ────
NUM_LAYERS = 28
NUM_KV_HEADS = 8
NUM_Q_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
VOCAB_SIZE = 3072       # codec vocab (was 151936)
MAX_SEQ_LEN = 8192      # practical decode limit (config allows 32768)
ROPE_THETA = 1000000.0  # (was 10000.0)

# ─── Code predictor constants ─────────────────────────────────────────────────
NUM_CODE_GROUPS = 16           # total codebook groups per audio frame (1 talker + 15 predicted)
CODE_PREDICTOR_LAYERS = 5
CODE_PREDICTOR_VOCAB = 2048

# ─── Special token IDs (from config.json) ─────────────────────────────────────
CODEC_BOS = 2149
CODEC_EOS = 2150
CODEC_PAD = 2148

# Thinking tokens (from talker_config)
CODEC_NOTHINK = 2155
CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157

# Text special tokens (from tokenizer)
TTS_BOS = 151672
TTS_EOS = 151673
TTS_PAD = 151671

# Sentinel value: when passed as token_id, kernel reads from hidden_buffer
# instead of doing an embedding lookup. Requires the one-line kernel patch.
EMBED_FROM_BUFFER = -1


def load_tts_weights(
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """Load Qwen3-TTS talker decoder weights into GPU tensors.

    Loads from safetensors directly (no transformers AutoModel needed).
    Returns a dict with all weights needed by TTSDecoder.

    Args:
        model_path: HuggingFace repo ID or local directory.
        device: Target device ("cuda" or "cuda:0" etc).
        verbose: Print progress.

    Returns:
        Dict with keys: embed_weight, lm_head_weight, final_norm_weight,
        layer_weights, cos_table, sin_table, and prefill/code_predictor weights.
    """
    if verbose:
        print(f"Loading TTS weights from {model_path}...")

    # ── Load safetensors ──────────────────────────────────────────────────
    import os
    if os.path.isdir(model_path):
        safetensors_path = os.path.join(model_path, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        safetensors_path = hf_hub_download(model_path, "model.safetensors")

    from safetensors.torch import load_file
    state = load_file(safetensors_path, device=device)

    # ── RoPE cos/sin tables ───────────────────────────────────────────────
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [MAX_SEQ_LEN, HEAD_DIM/2]
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()

    # ── Per-layer weights (11 tensors per layer, same order as LDGLayerWeights) ──
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"talker.model.layers.{i}."
        layer_weights.extend([
            state[p + "input_layernorm.weight"].contiguous(),
            state[p + "self_attn.q_proj.weight"].contiguous(),
            state[p + "self_attn.k_proj.weight"].contiguous(),
            state[p + "self_attn.v_proj.weight"].contiguous(),
            state[p + "self_attn.q_norm.weight"].contiguous(),
            state[p + "self_attn.k_norm.weight"].contiguous(),
            state[p + "self_attn.o_proj.weight"].contiguous(),
            state[p + "post_attention_layernorm.weight"].contiguous(),
            state[p + "mlp.gate_proj.weight"].contiguous(),
            state[p + "mlp.up_proj.weight"].contiguous(),
            state[p + "mlp.down_proj.weight"].contiguous(),
        ])

    # ── Global weights ────────────────────────────────────────────────────
    embed_weight = state["talker.model.codec_embedding.weight"].contiguous()  # [3072, 1024]
    lm_head_weight = state["talker.codec_head.weight"].contiguous()           # [3072, 1024] NOT tied
    final_norm_weight = state["talker.model.norm.weight"].contiguous()        # [1024]

    # ── Prefill weights (text embedding + projection) ─────────────────────
    text_embedding = state["talker.model.text_embedding.weight"].contiguous()  # [151936, 2048]
    text_proj_fc1_w = state["talker.text_projection.linear_fc1.weight"].contiguous()  # [2048, 2048]
    text_proj_fc1_b = state["talker.text_projection.linear_fc1.bias"].contiguous()    # [2048]
    text_proj_fc2_w = state["talker.text_projection.linear_fc2.weight"].contiguous()  # [1024, 2048]
    text_proj_fc2_b = state["talker.text_projection.linear_fc2.bias"].contiguous()    # [1024]

    # ── Code predictor weights ────────────────────────────────────────────
    code_predictor = {}
    for i in range(CODE_PREDICTOR_LAYERS):
        p = f"talker.code_predictor.model.layers.{i}."
        code_predictor[f"layers.{i}.input_layernorm.weight"] = state[p + "input_layernorm.weight"]
        code_predictor[f"layers.{i}.self_attn.q_proj.weight"] = state[p + "self_attn.q_proj.weight"]
        code_predictor[f"layers.{i}.self_attn.k_proj.weight"] = state[p + "self_attn.k_proj.weight"]
        code_predictor[f"layers.{i}.self_attn.v_proj.weight"] = state[p + "self_attn.v_proj.weight"]
        code_predictor[f"layers.{i}.self_attn.q_norm.weight"] = state[p + "self_attn.q_norm.weight"]
        code_predictor[f"layers.{i}.self_attn.k_norm.weight"] = state[p + "self_attn.k_norm.weight"]
        code_predictor[f"layers.{i}.self_attn.o_proj.weight"] = state[p + "self_attn.o_proj.weight"]
        code_predictor[f"layers.{i}.post_attention_layernorm.weight"] = state[p + "post_attention_layernorm.weight"]
        code_predictor[f"layers.{i}.mlp.gate_proj.weight"] = state[p + "mlp.gate_proj.weight"]
        code_predictor[f"layers.{i}.mlp.up_proj.weight"] = state[p + "mlp.up_proj.weight"]
        code_predictor[f"layers.{i}.mlp.down_proj.weight"] = state[p + "mlp.down_proj.weight"]
    code_predictor["norm.weight"] = state["talker.code_predictor.model.norm.weight"]
    for g in range(NUM_CODE_GROUPS - 1):  # 15 heads (talker handles group 0)
        code_predictor[f"lm_head.{g}.weight"] = state[f"talker.code_predictor.lm_head.{g}.weight"]
        code_predictor[f"codec_embedding.{g}.weight"] = state[f"talker.code_predictor.model.codec_embedding.{g}.weight"]

    # ── Speaker encoder weights ───────────────────────────────────────────
    speaker_encoder = {
        k: v for k, v in state.items() if k.startswith("speaker_encoder.")
    }

    weights = dict(
        # Megakernel decode weights
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
        final_norm_weight=final_norm_weight,
        layer_weights=layer_weights,
        cos_table=cos_table,
        sin_table=sin_table,
        # Prefill weights (PyTorch)
        text_embedding=text_embedding,
        text_proj_fc1_w=text_proj_fc1_w,
        text_proj_fc1_b=text_proj_fc1_b,
        text_proj_fc2_w=text_proj_fc2_w,
        text_proj_fc2_b=text_proj_fc2_b,
        # Code predictor weights (PyTorch)
        code_predictor=code_predictor,
        # Speaker encoder weights (PyTorch)
        speaker_encoder=speaker_encoder,
    )

    if verbose:
        n_params = sum(v.numel() for v in state.values()) / 1e6
        print(f"Loaded {len(state)} tensors ({n_params:.1f}M params)")

    del state
    torch.cuda.empty_cache()
    return weights


def _pack_layer_weights(layer_weights: list[torch.Tensor], num_layers: int = NUM_LAYERS) -> torch.Tensor:
    """Pack the 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8   # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(num_layers * struct_bytes)
    for i in range(num_layers):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class TTSDecoder:
    """Stateful talker decoder wrapping the Qwen TTS megakernel.

    Supports two input modes:
    - step(token_id): Embed a codec token via the kernel's embed_weight lookup
    - step_with_embed(embed_bf16): Use a precomputed embedding (e.g. sum of codec
      embeddings + text). Requires the kernel sentinel patch (token_id == -1).

    Both modes return (next_token_id, hidden_state_f32) where hidden_state is
    the post-RMSNorm output needed by the code predictor.
    """

    def __init__(self, weights: Optional[dict] = None, model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", verbose: bool = True):
        if weights is None:
            weights = load_tts_weights(model_path, verbose=verbose)
        self._position = 0

        # Keep references so tensors stay alive
        self._weights = weights

        # Model weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)       # hidden_buffer (bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)            # g_activations
        self._res = torch.empty(HIDDEN_SIZE, **f32)            # g_residual
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)       # post-RMSNorm hidden state
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

        # Build and import the decode op (triggers JIT compilation)
        from .build_tts import get_extension
        get_extension()
        self._decode = torch.ops.qwen_megakernel_C.decode

    def step(self, token_id: int) -> tuple[int, torch.Tensor]:
        """Decode one token via embedding lookup. Returns (next_token, hidden_state_f32)."""
        self._decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        # norm_out contains the post-RMSNorm hidden state (f32, [HIDDEN_SIZE])
        return self._out_token.item(), self._norm_out.clone()

    def step_with_embed(self, embed_bf16: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Decode with a precomputed bf16 embedding (e.g. sum of codec embeds + text).

        Copies the embedding into hidden_buffer, then calls decode with token_id=-1
        (sentinel) so the kernel skips the embedding lookup and reads from hidden_buffer.

        Args:
            embed_bf16: bf16 tensor of shape [HIDDEN_SIZE] on CUDA.

        Returns:
            (next_token_id, hidden_state_f32) where hidden_state is [HIDDEN_SIZE].
        """
        # Copy precomputed embedding into the kernel's hidden_buffer
        self._hidden.copy_(embed_bf16)

        self._decode(
            self._out_token,
            EMBED_FROM_BUFFER,   # sentinel: skip embed lookup, use hidden_buffer
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item(), self._norm_out.clone()

    def reset(self):
        """Reset decoder state for a new utterance."""
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    @property
    def embed_weight(self) -> torch.Tensor:
        """Codec embedding table [VOCAB_SIZE=3072, HIDDEN_SIZE=1024], bf16."""
        return self._embed_weight


class TextProjection:
    """Projects text embeddings from text_hidden_size (2048) to hidden_size (1024).

    text_embedding (151936 → 2048) → SiLU → fc1 (2048 → 2048) → fc2 (2048 → 1024)
    """

    def __init__(self, weights: dict, device: str = "cuda"):
        self.text_embedding = weights["text_embedding"]       # [151936, 2048]
        self.fc1_w = weights["text_proj_fc1_w"]               # [2048, 2048]
        self.fc1_b = weights["text_proj_fc1_b"]               # [2048]
        self.fc2_w = weights["text_proj_fc2_w"]               # [1024, 2048]
        self.fc2_b = weights["text_proj_fc2_b"]               # [1024]

    @torch.no_grad()
    def embed_text_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed text token IDs and project to hidden_size.

        Args:
            token_ids: [seq_len] or [batch, seq_len] long tensor.

        Returns:
            [seq_len, HIDDEN_SIZE] or [batch, seq_len, HIDDEN_SIZE] bf16 tensor.
        """
        x = torch.nn.functional.embedding(token_ids, self.text_embedding)  # → [*, 2048]
        x = torch.nn.functional.silu(torch.nn.functional.linear(x, self.fc1_w, self.fc1_b))
        x = torch.nn.functional.linear(x, self.fc2_w, self.fc2_b)  # → [*, 1024]
        return x


class CodePredictor:
    """Runs the code predictor (5-layer transformer) to generate codebook groups 1-14.

    Given the talker's hidden state and first codebook token, autoregressively
    generates the remaining NUM_CODE_GROUPS-1 codebook tokens.

    Uses KV caching for efficient autoregressive generation:
    - Prefill: 2 tokens ([hidden, first_embed]) through all 5 layers
    - Decode: 1 token per group through all 5 layers with cached KV
    """

    def __init__(self, weights: dict, device: str = "cuda"):
        cp = weights["code_predictor"]
        self.device = device
        self.num_groups = NUM_CODE_GROUPS - 1  # 14 groups to predict
        self._max_seq = 20  # max sequence length for KV cache

        # Per-group embedding and LM head
        self.codec_embeddings = []
        self.lm_heads = []
        for g in range(self.num_groups):
            self.codec_embeddings.append(cp[f"codec_embedding.{g}.weight"])  # [2048, 1024]
            self.lm_heads.append(cp[f"lm_head.{g}.weight"])                  # [2048, 1024]

        # Transformer layers
        self.layers = []
        for i in range(CODE_PREDICTOR_LAYERS):
            layer = {}
            for key in [
                "input_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            ]:
                layer[key] = cp[f"layers.{i}.{key}"]
            self.layers.append(layer)

        self.final_norm = cp["norm.weight"]  # [1024]

        # RoPE tables (same theta as talker)
        inv_freq = 1.0 / (
            ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
        positions = torch.arange(64, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).to(device)
        self._sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).to(device)

        # Pre-allocate KV cache: [num_layers, num_kv_heads, max_seq, head_dim]
        self._k_cache = torch.zeros(
            CODE_PREDICTOR_LAYERS, NUM_KV_HEADS, self._max_seq, HEAD_DIM,
            dtype=torch.bfloat16, device=device,
        )
        self._v_cache = torch.zeros_like(self._k_cache)

    def _reset_cache(self):
        self._k_cache.zero_()
        self._v_cache.zero_()

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
        """Predict codebook groups 1 through NUM_CODE_GROUPS-1.

        Uses KV caching: prefill 2 tokens, then decode 1 token per group.
        """
        self._reset_cache()

        first_embed = torch.nn.functional.embedding(
            torch.tensor([first_codebook_token], device=self.device),
            talker_embed_weight,
        ).squeeze(0)  # [1024], bf16

        hidden = talker_hidden.to(torch.bfloat16)  # [1024]

        # Prefill: [hidden, first_embed] → 2 tokens
        prefill = torch.stack([hidden, first_embed], dim=0).unsqueeze(0)  # [1, 2, 1024]
        h = prefill
        for layer_idx, layer_w in enumerate(self.layers):
            h = self._layer_prefill(h, layer_w, layer_idx, seq_len=2)
        h = self._rms_norm(h, self.final_norm)
        last_hidden = h[:, -1:, :]  # [1, 1, 1024]

        predicted_tokens = [first_codebook_token]
        pos = 2  # next position

        for group in range(self.num_groups):
            # LM head for this group
            logits = torch.nn.functional.linear(last_hidden, self.lm_heads[group])  # [1, 1, 2048]
            logits = logits.squeeze(0).squeeze(0).float()  # [2048]

            # Sample or argmax
            if do_sample and temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < topk_vals[-1]] = float('-inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1).item()
            else:
                token = logits.argmax().item()

            predicted_tokens.append(token)

            # Decode next token (if not last group)
            if group < self.num_groups - 1:
                new_embed = torch.nn.functional.embedding(
                    torch.tensor([token], device=self.device),
                    self.codec_embeddings[group],
                ).unsqueeze(0)  # [1, 1, 1024]

                h = new_embed
                for layer_idx, layer_w in enumerate(self.layers):
                    h = self._layer_decode(h, layer_w, layer_idx, pos)
                h = self._rms_norm(h, self.final_norm)
                last_hidden = h  # [1, 1, 1024]
                pos += 1

        return torch.tensor(predicted_tokens, dtype=torch.int64, device=self.device)

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
        return (x_f32 / rms * weight.float()).to(x.dtype)

    def _apply_rope_single(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """Apply RoPE to a single position. x: [B, H, 1, D]."""
        cos = self._cos[pos:pos+1].unsqueeze(0).unsqueeze(0).to(x.dtype)
        sin = self._sin[pos:pos+1].unsqueeze(0).unsqueeze(0).to(x.dtype)
        d2 = HEAD_DIM // 2
        x1 = x[..., :d2]
        x2 = x[..., d2:]
        cos1 = cos[..., :d2]
        sin1 = sin[..., :d2]
        return torch.cat([x1 * cos1 - x2 * sin1, x2 * cos1 + x1 * sin1], dim=-1)

    def _apply_rope_seq(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to a sequence. x: [B, H, L, D]."""
        cos = self._cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
        sin = self._sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
        d2 = HEAD_DIM // 2
        x1 = x[..., :d2]
        x2 = x[..., d2:]
        cos1 = cos[..., :d2]
        sin1 = sin[..., :d2]
        return torch.cat([x1 * cos1 - x2 * sin1, x2 * cos1 + x1 * sin1], dim=-1)

    def _layer_prefill(self, h: torch.Tensor, w: dict, layer_idx: int, seq_len: int) -> torch.Tensor:
        """Prefill: process full sequence, populate KV cache."""
        normed = self._rms_norm(h, w["input_layernorm.weight"])
        q = torch.nn.functional.linear(normed, w["self_attn.q_proj.weight"])
        k = torch.nn.functional.linear(normed, w["self_attn.k_proj.weight"])
        v = torch.nn.functional.linear(normed, w["self_attn.v_proj.weight"])

        B, L, _ = q.shape
        q = q.view(B, L, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q = self._rms_norm(q, w["self_attn.q_norm.weight"])
        k = self._rms_norm(k, w["self_attn.k_norm.weight"])

        q = self._apply_rope_seq(q, seq_len)
        k = self._apply_rope_seq(k, seq_len)

        # Store in KV cache
        self._k_cache[layer_idx, :, :seq_len, :] = k.squeeze(0)
        self._v_cache[layer_idx, :, :seq_len, :] = v.squeeze(0)

        # GQA expand
        if NUM_Q_HEADS != NUM_KV_HEADS:
            rep = NUM_Q_HEADS // NUM_KV_HEADS
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=True,
        ).to(h.dtype)

        attn = attn.transpose(1, 2).contiguous().view(B, L, Q_SIZE)
        attn_out = torch.nn.functional.linear(attn, w["self_attn.o_proj.weight"])
        h = h + attn_out

        normed = self._rms_norm(h, w["post_attention_layernorm.weight"])
        gate = torch.nn.functional.linear(normed, w["mlp.gate_proj.weight"])
        up = torch.nn.functional.linear(normed, w["mlp.up_proj.weight"])
        mlp_out = torch.nn.functional.silu(gate) * up
        mlp_out = torch.nn.functional.linear(mlp_out, w["mlp.down_proj.weight"])
        return h + mlp_out

    def _layer_decode(self, h: torch.Tensor, w: dict, layer_idx: int, pos: int) -> torch.Tensor:
        """Decode: process 1 new token using KV cache. h: [1, 1, 1024]."""
        normed = self._rms_norm(h, w["input_layernorm.weight"])
        q = torch.nn.functional.linear(normed, w["self_attn.q_proj.weight"])  # [1, 1, Q_SIZE]
        k_new = torch.nn.functional.linear(normed, w["self_attn.k_proj.weight"])  # [1, 1, KV_SIZE]
        v_new = torch.nn.functional.linear(normed, w["self_attn.v_proj.weight"])

        q = q.view(1, 1, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)      # [1, H, 1, D]
        k_new = k_new.view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [1, Hkv, 1, D]
        v_new = v_new.view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q = self._rms_norm(q, w["self_attn.q_norm.weight"])
        k_new = self._rms_norm(k_new, w["self_attn.k_norm.weight"])

        q = self._apply_rope_single(q, pos)
        k_new = self._apply_rope_single(k_new, pos)

        # Update KV cache
        self._k_cache[layer_idx, :, pos:pos+1, :] = k_new.squeeze(0)
        self._v_cache[layer_idx, :, pos:pos+1, :] = v_new.squeeze(0)

        # Attend to all cached KV (positions 0..pos inclusive)
        k_full = self._k_cache[layer_idx, :, :pos+1, :].unsqueeze(0)  # [1, Hkv, pos+1, D]
        v_full = self._v_cache[layer_idx, :, :pos+1, :].unsqueeze(0)

        if NUM_Q_HEADS != NUM_KV_HEADS:
            rep = NUM_Q_HEADS // NUM_KV_HEADS
            k_full = k_full.repeat_interleave(rep, dim=1)
            v_full = v_full.repeat_interleave(rep, dim=1)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k_full.float(), v_full.float(), is_causal=False,
        ).to(h.dtype)

        attn = attn.transpose(1, 2).contiguous().view(1, 1, Q_SIZE)
        attn_out = torch.nn.functional.linear(attn, w["self_attn.o_proj.weight"])
        h = h + attn_out

        normed = self._rms_norm(h, w["post_attention_layernorm.weight"])
        gate = torch.nn.functional.linear(normed, w["mlp.gate_proj.weight"])
        up = torch.nn.functional.linear(normed, w["mlp.up_proj.weight"])
        mlp_out = torch.nn.functional.silu(gate) * up
        mlp_out = torch.nn.functional.linear(mlp_out, w["mlp.down_proj.weight"])
        return h + mlp_out


class CodePredictorKernel:
    """Code predictor using the megakernel for the 5-layer transformer.

    Instead of ~70 separate PyTorch kernel launches per decode step,
    a single megakernel launch processes all 5 layers. ~18x speedup
    over the pure PyTorch CodePredictor.
    """

    def __init__(self, weights: dict, device: str = "cuda"):
        cp = weights["code_predictor"]
        self.device = device
        self.num_groups = NUM_CODE_GROUPS - 1  # 15 groups to predict

        # Pack code predictor's 5 layers (same format as talker)
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
        self._layer_weights = layer_weights
        self._layer_weights_packed = _pack_layer_weights(layer_weights, CODE_PREDICTOR_LAYERS)

        self._final_norm_weight = cp["norm.weight"].contiguous()

        # Dummy embed/lm_head (kernel expects VOCAB_SIZE=3072 compile-time)
        self._dummy_lm_head = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        self._dummy_embed = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        # Per-group weights
        self.codec_embeddings = [cp[f"codec_embedding.{g}.weight"] for g in range(self.num_groups)]
        self.lm_heads = [cp[f"lm_head.{g}.weight"] for g in range(self.num_groups)]

        # RoPE tables
        MAX_SEQ_CP = 64
        self._max_seq = MAX_SEQ_CP
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        positions = torch.arange(MAX_SEQ_CP, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()
        self._sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()

        # KV cache (5 layers, tiny)
        self._k_cache = torch.zeros(
            CODE_PREDICTOR_LAYERS, NUM_KV_HEADS, MAX_SEQ_CP, HEAD_DIM,
            dtype=torch.bfloat16, device=device,
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
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

        from .build_tts import get_extension
        get_extension()
        self._decode = torch.ops.qwen_megakernel_C.decode

        self._token_buf = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def _step_with_embed(self, embed_bf16: torch.Tensor):
        """Run one megakernel decode step. After call, norm_out has the hidden state."""
        self._hidden.copy_(embed_bf16)
        self._decode(
            self._out_token, -1,
            self._dummy_embed, self._layer_weights_packed,
            self._final_norm_weight, self._dummy_lm_head,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v,
            self._attn_out, self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            CODE_PREDICTOR_LAYERS, self._position, self._max_seq, self._attn_scale,
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
        """Predict all NUM_CODE_GROUPS codebook groups.

        Returns [NUM_CODE_GROUPS] int64 tensor (first token + 15 predicted).
        """
        self.reset()

        # Prefill: talker hidden → first token embedding
        self._step_with_embed(talker_hidden.to(torch.bfloat16))
        self._token_buf[0] = first_codebook_token
        first_embed = torch.nn.functional.embedding(self._token_buf, talker_embed_weight).squeeze(0)
        self._step_with_embed(first_embed)

        predicted_tensors = []

        for group in range(self.num_groups):
            hidden_bf16 = self._norm_out.to(torch.bfloat16).unsqueeze(0)
            logits = torch.nn.functional.linear(hidden_bf16, self.lm_heads[group]).squeeze(0)

            if do_sample and temperature > 0:
                logits_f = logits.float() / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(logits_f, min(top_k, logits_f.size(-1)))
                    logits_f[logits_f < topk_vals[-1]] = float('-inf')
                probs = torch.nn.functional.softmax(logits_f, dim=-1)
                token_tensor = torch.multinomial(probs, 1)
            else:
                token_tensor = logits.argmax(keepdim=True).long()

            predicted_tensors.append(token_tensor)

            if group < self.num_groups - 1:
                embed = torch.nn.functional.embedding(token_tensor, self.codec_embeddings[group]).squeeze(0)
                self._step_with_embed(embed)

        first_tensor = torch.tensor([first_codebook_token], dtype=torch.long, device=self.device)
        return torch.cat([first_tensor] + predicted_tensors)


def build_prefill_embeddings(
    text_token_ids: torch.Tensor,
    text_projection: TextProjection,
    codec_embed_weight: torch.Tensor,
    language: str = "Auto",
    device: str = "cuda",
    cached_tts_embeds: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the prefill embedding sequence for the talker decoder.

    Constructs the input sequence:
      [role_tokens] [codec_tags] [codec_bos + first_text] [trailing_text...]

    Args:
        text_token_ids: Tokenized text (including role tokens), shape [seq_len].
        text_projection: TextProjection instance for embedding text.
        codec_embed_weight: Talker's codec embedding table [3072, 1024].
        language: Language tag (not used for base model, but reserved).
        device: CUDA device.
        cached_tts_embeds: Optional dict with precomputed 'pad', 'bos', 'eos' embeddings.

    Returns:
        (prefill_embeds, trailing_text_embeds):
        - prefill_embeds: [prefill_len, HIDDEN_SIZE] bf16 tensor to feed through decoder
        - trailing_text_embeds: [trailing_len, HIDDEN_SIZE] bf16 tensor for decode phase
    """
    # Batch all text token embeddings in ONE call (role + content + special tokens)
    role_ids = text_token_ids[:3]
    content_ids = text_token_ids[3:]

    if cached_tts_embeds is not None:
        tts_pad_embed = cached_tts_embeds["pad"]
        tts_bos_embed = cached_tts_embeds["bos"]
        tts_eos_embed = cached_tts_embeds["eos"]
        # Batch role + content in one call
        all_text_ids = torch.cat([role_ids, content_ids]).to(device)
        all_text_embeds = text_projection.embed_text_ids(all_text_ids)
        role_embeds = all_text_embeds[:3]
        content_embeds = all_text_embeds[3:]
    else:
        # Batch ALL text IDs (role + content + special) in one call
        special_ids = torch.tensor([TTS_PAD, TTS_BOS, TTS_EOS], device=device)
        all_text_ids = torch.cat([role_ids.to(device), content_ids.to(device), special_ids])
        all_text_embeds = text_projection.embed_text_ids(all_text_ids)
        n_role = 3
        n_content = content_ids.shape[0]
        role_embeds = all_text_embeds[:n_role]
        content_embeds = all_text_embeds[n_role:n_role + n_content]
        tts_pad_embed = all_text_embeds[n_role + n_content:n_role + n_content + 1]
        tts_bos_embed = all_text_embeds[n_role + n_content + 1:n_role + n_content + 2]
        tts_eos_embed = all_text_embeds[n_role + n_content + 2:n_role + n_content + 3]

    # Codec input embeddings (matching official Qwen3-TTS format):
    # [nothink, think_bos, think_eos, codec_pad, codec_bos]
    codec_ids = torch.tensor([
        CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS,
        CODEC_PAD, CODEC_BOS,
    ], device=device)
    codec_embeds = torch.nn.functional.embedding(codec_ids, codec_embed_weight)  # [5, 1024]

    # Fuse TTS conditioning with codec tags (first 4 positions):
    # [(tts_pad + nothink), (tts_pad + think_bos), (tts_pad + think_eos), (tts_bos + codec_pad)]
    tts_prefix = torch.cat([
        tts_pad_embed.expand(3, -1),  # tts_pad × 3
        tts_bos_embed,                 # tts_bos × 1
    ], dim=0)  # [4, 1024]
    fused_tags = tts_prefix + codec_embeds[:4]  # [4, 1024]

    # First text token fused with codec_bos (last codec position)
    first_text_with_bos = content_embeds[:1] + codec_embeds[4:5]  # [1, 1024]

    # Prefill sequence: [role(3)] + [fused_tags(4)] + [first_text+bos(1)] = 8 steps
    prefill_embeds = torch.cat([
        role_embeds,
        fused_tags,
        first_text_with_bos,
    ], dim=0)

    # Trailing text: remaining content tokens (strip format markers at end)
    # input_id format: <|im_start|>assistant\n TEXT <|im_end|>\n<|im_start|>assistant\n
    # content_ids = text_token_ids[3:], content_embeds = projected content
    # Official uses input_id[4:-5] for trailing, which is text_token_ids[4:-5]
    # That means content_ids[1:-5] = text tokens without first and without last 5 format tokens
    trailing_text_embeds = torch.cat([
        content_embeds[1:-5],  # remaining text tokens (strip <|im_end|>\n<|im_start|>assistant\n)
        tts_eos_embed,
    ], dim=0)

    return prefill_embeds, trailing_text_embeds

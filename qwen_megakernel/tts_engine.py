"""Full TTS pipeline: text → talker (megakernel) → code predictor → vocoder → audio.

This module orchestrates all components:
  1. Text tokenization and embedding
  2. Prefill phase (via megakernel step-by-step)
  3. Autoregressive decode (megakernel talker + megakernel code predictor)
  4. Vocoder decode (codec tokens → waveform)

Streaming: yields audio chunks as codec frames accumulate.
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Generator, Optional

import numpy as np
import torch

from .model_tts import (
    CODEC_BOS,
    CODEC_EOS,
    CODEC_NOTHINK,
    CODEC_PAD,
    CODEC_THINK_BOS,
    CODEC_THINK_EOS,
    NUM_CODE_GROUPS,
    TTS_BOS,
    TTS_EOS,
    TTS_PAD,
    CodePredictorKernel,
    TextProjection,
    TTSDecoder,
    load_tts_weights,
)


@dataclass
class TTSConfig:
    """Configuration for the TTS engine."""
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    vocoder_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    sample_rate: int = 24000
    chunk_frames: int = 10       # ~0.8 sec per chunk at 12.5 Hz
    # Generation params
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    # Code predictor params
    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50


class MegakernelTTSEngine:
    """TTS engine using the megakernel for both talker and code predictor.

    Architecture:
        text → tokenizer → text_embedding + text_projection → prefill
        → megakernel decode loop:
            talker.step_with_embed() → first codebook + hidden state
            code_predictor.predict() → remaining 15 codebooks (megakernel-accelerated)
            sum(all_codec_embeds) + trailing_text → next input
        → vocoder.decode(codec_frames) → audio waveform
    """

    def __init__(self, config: Optional[TTSConfig] = None, device: str = "cuda"):
        self.config = config or TTSConfig()
        self.device = device
        self._initialized = False

    def initialize(self):
        """Load all model components. Call once before generation."""
        if self._initialized:
            return

        cfg = self.config
        print("Initializing MegakernelTTSEngine...")

        # Load weights
        weights = load_tts_weights(cfg.model_path, device=self.device, verbose=True)

        # Initialize components (TTSDecoder triggers JIT compilation)
        self.talker = TTSDecoder(weights=weights)
        self.text_projection = TextProjection(weights, device=self.device)
        # Use megakernel-accelerated code predictor (~18x faster than PyTorch)
        self.code_predictor = CodePredictorKernel(weights, device=self.device)

        # Codec embedding tables (for summing all codebook group embeddings)
        self._talker_embed = weights["embed_weight"]  # [3072, 1024] - group 0
        self._cp_embeds = []  # groups 1-15
        for g in range(NUM_CODE_GROUPS - 1):
            self._cp_embeds.append(
                weights["code_predictor"][f"codec_embedding.{g}.weight"]  # [2048, 1024]
            )

        # Load tokenizer (text)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

        # Load speech tokenizer (vocoder)
        self._load_vocoder(cfg.vocoder_path)

        # Precompute constant embeddings (TTS special tokens + role tokens + codec tags)
        with torch.no_grad():
            special_ids = torch.tensor([TTS_PAD, TTS_BOS, TTS_EOS], device=self.device)
            special_embeds = self.text_projection.embed_text_ids(special_ids)
            self._cached_tts_embeds = {
                "pad": special_embeds[0:1],  # [1, 1024]
                "bos": special_embeds[1:2],  # [1, 1024]
                "eos": special_embeds[2:3],  # [1, 1024]
            }
            self._tts_pad_embed = special_embeds[0].to(torch.bfloat16)  # [1024]

            # Precompute role token embeddings (<|im_start|> assistant \n)
            role_text = "<|im_start|>assistant\n"
            role_ids = self.tokenizer.encode(role_text, return_tensors="pt")[0][:3].to(self.device)
            self._cached_role_embeds = self.text_projection.embed_text_ids(role_ids)  # [3, 1024]

            # Precompute codec tag + TTS fused embeddings (official Qwen3-TTS format)
            # Codec: [nothink, think_bos, think_eos, codec_pad, codec_bos]
            codec_ids = torch.tensor([
                CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS,
                CODEC_PAD, CODEC_BOS,
            ], device=self.device)
            codec_embeds = torch.nn.functional.embedding(codec_ids, self._talker_embed)  # [5, 1024]

            # Fuse: [(tts_pad+nothink), (tts_pad+think_bos), (tts_pad+think_eos), (tts_bos+codec_pad)]
            tts_prefix = torch.cat([
                special_embeds[0:1].expand(3, -1),  # pad × 3
                special_embeds[1:2],                  # bos × 1
            ], dim=0)  # [4, 1024]
            self._cached_fused_tags = tts_prefix + codec_embeds[:4]  # [4, 1024]

            # Precompute codec BOS embedding (last in codec sequence)
            self._cached_codec_bos = codec_embeds[4:5]  # [1, 1024]

        # Warm up entire pipeline (first calls are slow due to CUDA JIT/cublas init)
        print("Warming up pipeline...")
        for do_sample in [False, False, True, True, True]:
            self.talker.reset()
            _, h = self.talker.step(CODEC_BOS)
            self.code_predictor.predict(
                h, 0, self._talker_embed,
                do_sample=do_sample, temperature=0.9, top_k=50,
            )
        self.talker.reset()
        if self.speech_tokenizer is not None:
            for n in [1, 1, 5]:
                dummy_codes = torch.randint(0, 2048, (n, NUM_CODE_GROUPS), dtype=torch.long, device=self.device)
                self.speech_tokenizer.decode([{"audio_codes": dummy_codes}])
        torch.cuda.synchronize()

        self._initialized = True
        print("MegakernelTTSEngine initialized.")

    def _load_vocoder(self, vocoder_path: str):
        """Load the speech tokenizer for codec → waveform decoding."""
        # Strategy: load the speech tokenizer model directly from the
        # speech_tokenizer/ subfolder, bypassing the broken AutoFeatureExtractor path.
        try:
            # Monkey-patch transformers if needed (qwen_tts compat with transformers 5.x)
            import transformers.utils.generic
            if not hasattr(transformers.utils.generic, 'check_model_inputs'):
                def _check_model_inputs(*args, **kwargs):
                    def decorator(func):
                        return func
                    return decorator
                transformers.utils.generic.check_model_inputs = _check_model_inputs

            from transformers import AutoConfig, AutoModel
            from qwen_tts.core import (
                Qwen3TTSTokenizerV2Config,
                Qwen3TTSTokenizerV2Model,
            )

            # Register speech tokenizer model type
            try:
                AutoConfig.register('qwen3_tts_tokenizer_12hz', Qwen3TTSTokenizerV2Config)
                AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
            except ValueError:
                pass  # Already registered

            # Load speech tokenizer from subfolder
            model = AutoModel.from_pretrained(
                vocoder_path,
                subfolder='speech_tokenizer',
                device_map=self.device,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Create Qwen3TTSTokenizer wrapper (skip feature_extractor — not needed for decode)
            from qwen_tts import Qwen3TTSTokenizer
            self.speech_tokenizer = Qwen3TTSTokenizer()
            self.speech_tokenizer.model = model
            self.speech_tokenizer.feature_extractor = None
            self.speech_tokenizer.config = model.config
            self.speech_tokenizer.device = model.device
            self.sample_rate = self.speech_tokenizer.get_output_sample_rate()
            print(f"Vocoder loaded (sample rate: {self.sample_rate} Hz)")
            return
        except Exception as e:
            print(f"Vocoder load failed: {e}")

        self.speech_tokenizer = None
        self.sample_rate = self.config.sample_rate
        print("Warning: Vocoder unavailable. Audio output will be silence.")

    @torch.no_grad()
    def synthesize(self, text: str, ref_audio: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
        """Non-streaming synthesis. Returns (waveform, sample_rate)."""
        self.initialize()
        codec_frames = list(self._generate_codec_frames(text))
        if not codec_frames:
            return np.array([], dtype=np.float32), self.sample_rate
        return self._decode_to_audio(codec_frames)

    async def synthesize_streaming(
        self,
        text: str,
        chunk_frames: Optional[int] = None,
    ) -> AsyncGenerator[tuple[np.ndarray, int], None]:
        """Streaming synthesis. Yields (audio_chunk, sample_rate) as frames accumulate."""
        self.initialize()
        chunk_size = chunk_frames or self.config.chunk_frames
        buffer = []
        first_chunk = True

        for frame in self._generate_codec_frames(text):
            buffer.append(frame)
            # Use smaller first chunk (1 frame) for fast TTFC, then normal chunk size
            target = 1 if first_chunk else chunk_size
            if len(buffer) >= target:
                audio, sr = self._decode_to_audio(buffer)
                buffer = []
                first_chunk = False
                yield audio, sr
                await asyncio.sleep(0)

        if buffer:
            audio, sr = self._decode_to_audio(buffer)
            yield audio, sr

    def _generate_codec_frames(self, text: str) -> Generator[torch.Tensor, None, None]:
        """Run the talker + code predictor to generate codec frames.

        Each frame is a tensor of shape [NUM_CODE_GROUPS] (int64).
        Yields frames one at a time for streaming support.
        """
        cfg = self.config
        self.talker.reset()

        # Tokenize only the content text (role tokens are precomputed)
        # Format: <|im_start|>assistant\n TEXT <|im_end|>\n<|im_start|>assistant\n
        # Tokens: [role(3)] [text...] [<|im_end|>(1) \n(1) <|im_start|>(1) assistant(1) \n(1)]
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = self.tokenizer.encode(formatted_text, return_tensors="pt")[0]
        content_ids = text_ids[3:].to(self.device)

        # Embed content tokens (single batched call — role/tags/special are precomputed)
        content_embeds = self.text_projection.embed_text_ids(content_ids)
        first_text_with_bos = content_embeds[:1] + self._cached_codec_bos

        # Build prefill: [role(3), fused_tags(4), first_text+bos(1)] = 8 steps
        prefill_embeds = torch.cat([
            self._cached_role_embeds,
            self._cached_fused_tags,
            first_text_with_bos,
        ], dim=0)  # [8, 1024]

        # Trailing text: content tokens[1:-5] + tts_eos
        # Strip last 5 format tokens: <|im_end|>\n<|im_start|>assistant\n
        trailing_text = torch.cat([
            content_embeds[1:-5],
            self._cached_tts_embeds["eos"],
        ], dim=0)

        # Phase 1: Prefill — feed all prefill embeddings through the talker
        for i in range(prefill_embeds.shape[0]):
            self.talker.step_with_embed(prefill_embeds[i])

        # Phase 2: Autoregressive decode
        trailing_idx = 0
        tts_pad_embed = self._tts_pad_embed

        # First decode step
        first_token, hidden = self.talker.step(CODEC_BOS)

        prev_token = first_token

        # Estimate max frames from text length
        # English speech: ~2.5 words/sec. At 12.5 codec frames/sec,
        # each word ≈ 5 frames. Use 2x margin since EOS is unreliable.
        word_count = max(len(text.split()), 1)
        estimated_speech_sec = word_count / 2.5
        max_frames = max(int(estimated_speech_sec * 12.5 * 2.0), 25)
        max_frames = min(max_frames, cfg.max_new_tokens)

        for step in range(max_frames):
            if prev_token == CODEC_EOS:
                break

            # Run code predictor (megakernel-accelerated)
            all_codes = self.code_predictor.predict(
                talker_hidden=hidden,
                first_codebook_token=prev_token,
                talker_embed_weight=self._talker_embed,
                do_sample=cfg.subtalker_do_sample,
                temperature=cfg.subtalker_temperature,
                top_k=cfg.subtalker_top_k,
            )  # [NUM_CODE_GROUPS] int64

            yield all_codes

            # Compute next input: sum of all codec group embeddings
            # Use slice indexing (all_codes[i:i+1]) to avoid GPU→CPU sync
            embed_sum = torch.nn.functional.embedding(
                all_codes[0:1], self._talker_embed,
            ).squeeze(0)

            for g in range(NUM_CODE_GROUPS - 1):
                embed_sum = embed_sum + torch.nn.functional.embedding(
                    all_codes[g + 1:g + 2], self._cp_embeds[g],
                ).squeeze(0)

            # Add trailing text embedding
            if trailing_idx < trailing_text.shape[0]:
                embed_sum = embed_sum + trailing_text[trailing_idx].to(torch.bfloat16)
                trailing_idx += 1
            else:
                embed_sum = embed_sum + tts_pad_embed

            prev_token, hidden = self.talker.step_with_embed(embed_sum)

    def _decode_to_audio(self, codec_frames: list[torch.Tensor]) -> tuple[np.ndarray, int]:
        """Decode codec frames to audio waveform."""
        if not codec_frames:
            return np.array([], dtype=np.float32), self.sample_rate

        audio_codes = torch.stack(codec_frames, dim=0)

        if self.speech_tokenizer is not None:
            wavs, sr = self.speech_tokenizer.decode([{"audio_codes": audio_codes}])
            return wavs[0], sr
        else:
            duration_sec = len(codec_frames) / 12.5
            num_samples = int(duration_sec * self.sample_rate)
            return np.zeros(num_samples, dtype=np.float32), self.sample_rate

    def get_metrics(self) -> dict:
        """Return performance metrics from the last generation."""
        return {
            "sample_rate": self.sample_rate,
            "position": self.talker.position if self._initialized else 0,
        }

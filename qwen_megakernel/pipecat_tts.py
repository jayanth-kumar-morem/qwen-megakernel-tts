"""Pipecat TTS service using the megakernel TTS engine.

This module provides a custom Pipecat TTS service that uses the megakernel
for both the talker decoder and code predictor, achieving real-time streaming
speech synthesis with TTFC < 90ms and RTF < 0.3.

Usage in a Pipecat pipeline:
    tts = MegakernelTTSService(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    pipeline = Pipeline([..., tts, ...])
"""

import asyncio
import logging
from typing import AsyncGenerator, AsyncIterator, Optional

import numpy as np
import torch

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from .tts_engine import MegakernelTTSEngine, TTSConfig

logger = logging.getLogger(__name__)


class MegakernelTTSService(TTSService):
    """Pipecat TTS service backed by the megakernel TTS engine.

    Streams audio chunks as they're generated — does NOT buffer the full
    utterance before sending.

    Performance targets:
        TTFC (time to first audio chunk): < 90 ms
        RTF (real-time factor): < 0.3
    """

    def __init__(
        self,
        *,
        model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        vocoder_path: Optional[str] = None,
        device: str = "cuda",
        chunk_frames: int = 10,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 2048,
        subtalker_do_sample: bool = True,
        subtalker_temperature: float = 0.9,
        subtalker_top_k: int = 50,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._config = TTSConfig(
            model_path=model_path,
            vocoder_path=vocoder_path or model_path,
            chunk_frames=chunk_frames,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            subtalker_do_sample=subtalker_do_sample,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
        )
        self._device = device
        self._engine: Optional[MegakernelTTSEngine] = None

    def can_generate_metrics(self) -> bool:
        return True

    def _ensure_engine(self):
        """Lazily initialize the TTS engine."""
        if self._engine is None:
            self._engine = MegakernelTTSEngine(config=self._config, device=self._device)
            self._engine.initialize()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate streaming speech from text using the megakernel.

        Yields audio frames as codec frames are generated and decoded,
        pushing audio to the pipeline chunk by chunk.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame(context_id=context_id)

            # Run streaming synthesis in a thread (megakernel is synchronous/GPU-bound)
            async def audio_chunk_iterator() -> AsyncIterator[bytes]:
                """Generate audio chunks as PCM16 bytes via streaming synthesis."""
                loop = asyncio.get_event_loop()

                # Ensure engine is initialized
                await loop.run_in_executor(None, self._ensure_engine)
                engine = self._engine

                # Run the streaming synthesis
                async for audio_chunk, sr in engine.synthesize_streaming(
                    text, chunk_frames=self._config.chunk_frames
                ):
                    # Convert float32 numpy array to PCM16 bytes
                    pcm16 = _float32_to_pcm16(audio_chunk)
                    yield pcm16

            async for frame in self._stream_audio_frames_from_iterator(
                audio_chunk_iterator(),
                in_sample_rate=self._engine.sample_rate if self._engine else 24000,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame

        except Exception as e:
            logger.error(f"{self} TTS exception: {e}")
            yield ErrorFrame(error=f"Megakernel TTS error: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio [-1, 1] to 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()

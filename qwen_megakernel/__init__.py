"""Qwen Megakernel — single-kernel Qwen3 decode for RTX 5090.

For TTS usage:
    from qwen_megakernel.tts_engine import MegakernelTTSEngine, TTSConfig

For Pipecat integration:
    from qwen_megakernel.pipecat_tts import MegakernelTTSService

Note: CUDA compilation is deferred until the kernel is actually used.
"""

__all__ = []

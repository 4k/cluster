"""
TTS Engine implementations.

Each engine implements the TTSEngine protocol and can be used
interchangeably with the TTS service.
"""

from .piper_engine import PiperTTSEngine

__all__ = [
    "PiperTTSEngine",
]

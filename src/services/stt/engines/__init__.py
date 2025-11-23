"""
STT Engine implementations.

Each engine implements the STTEngine protocol and can be used
interchangeably with the STT service.
"""

from .vosk_engine import VoskSTTEngine

__all__ = [
    "VoskSTTEngine",
]

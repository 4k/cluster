"""
Core audio abstractions for TTS and viseme extraction.

This module provides backend-agnostic interfaces for:
- Text-to-Speech engines (Piper, Azure, OpenAI, etc.)
- Viseme extraction providers (Rhubarb, text-based, Azure inline, etc.)
"""

from .types import (
    TTSResult,
    TTSCapabilities,
    AudioFormat,
    VisemeCue,
    VisemeSequence,
    VisemeShape,
)
from .tts_engine import TTSEngine
from .viseme_provider import VisemeProvider

__all__ = [
    # Types
    "TTSResult",
    "TTSCapabilities",
    "AudioFormat",
    "VisemeCue",
    "VisemeSequence",
    "VisemeShape",
    # Protocols
    "TTSEngine",
    "VisemeProvider",
]

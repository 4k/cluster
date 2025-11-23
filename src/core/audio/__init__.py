"""
Core audio abstractions for TTS, STT, and viseme extraction.

This module provides backend-agnostic interfaces for:
- Text-to-Speech engines (Piper, Azure, OpenAI, etc.)
- Speech-to-Text engines (Vosk, Whisper, Azure, Google, etc.)
- Wake word detection engines (openwakeword, Porcupine, etc.)
- Viseme extraction providers (Rhubarb, text-based, Azure inline, etc.)
"""

# TTS types
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

# STT types
from .stt_types import (
    STTResult,
    STTCapabilities,
    TranscriptionSegment,
    TranscriptionWord,
    TranscriptionState,
    WakeWordResult,
    WakeWordCapabilities,
)
from .stt_engine import STTEngine
from .wake_word_engine import WakeWordEngine, NoneWakeWordEngine

__all__ = [
    # TTS Types
    "TTSResult",
    "TTSCapabilities",
    "AudioFormat",
    "VisemeCue",
    "VisemeSequence",
    "VisemeShape",
    # TTS Protocols
    "TTSEngine",
    "VisemeProvider",
    # STT Types
    "STTResult",
    "STTCapabilities",
    "TranscriptionSegment",
    "TranscriptionWord",
    "TranscriptionState",
    "WakeWordResult",
    "WakeWordCapabilities",
    # STT Protocols
    "STTEngine",
    "WakeWordEngine",
    "NoneWakeWordEngine",
]

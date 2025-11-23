"""
STT Service Package.

Provides a modular, backend-agnostic speech-to-text service with
support for multiple STT engines and wake word detection engines.
"""

from .factory import STTEngineFactory, WakeWordEngineFactory, STTConfig
from .stt_service_v2 import STTServiceV2

__all__ = [
    "STTEngineFactory",
    "WakeWordEngineFactory",
    "STTConfig",
    "STTServiceV2",
]

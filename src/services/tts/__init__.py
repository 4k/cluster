"""
TTS Service Package.

Provides a modular, backend-agnostic text-to-speech service with
support for multiple TTS engines and viseme extraction providers.
"""

from .factory import TTSEngineFactory, VisemeProviderFactory
from .tts_service_v2 import TTSServiceV2

__all__ = [
    "TTSEngineFactory",
    "VisemeProviderFactory",
    "TTSServiceV2",
]

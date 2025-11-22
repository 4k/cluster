"""
Services Package

Independent event-driven services that communicate via the event bus:
- stt_service: Speech-to-Text with wake word detection
- llm_service: Language model integration (Ollama)
- tts_service: Text-to-Speech synthesis (Piper)
- animation_service: Animation coordination
- lip_sync_service: Rhubarb lip sync integration
"""

from .stt_service import STTService, list_audio_devices
from .llm_service import LLMService
from .tts_service import TTSService
from .animation_service import AnimationService
from .lip_sync_service import (
    RhubarbLipSyncService,
    RhubarbViseme,
    VisemeCue,
    LipSyncData,
    RHUBARB_TO_INTERNAL_VISEME
)

__all__ = [
    'STTService',
    'list_audio_devices',
    'LLMService',
    'TTSService',
    'AnimationService',
    'RhubarbLipSyncService',
    'RhubarbViseme',
    'VisemeCue',
    'LipSyncData',
    'RHUBARB_TO_INTERNAL_VISEME',
]

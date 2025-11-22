"""
Services Package

Core event-driven services that communicate via the event bus:
- stt_service: Speech-to-Text with wake word detection
- llm_service: Language model integration (Ollama)
- tts_service: Text-to-Speech synthesis (Piper)

Note: Animation and display are now in src/features/display/
"""

from .stt_service import STTService, list_audio_devices
from .llm_service import LLMService
from .tts_service import TTSService

__all__ = [
    'STTService',
    'list_audio_devices',
    'LLMService',
    'TTSService',
]

"""
Audio Package

Audio processing and management:
- audio_manager: Audio pipeline management
- ambient_stt: Continuous speech recognition
"""

from .audio_manager import AudioManager, AudioManagerConfig
from .ambient_stt import AmbientSTTConfig

__all__ = [
    'AudioManager',
    'AudioManagerConfig',
    'AmbientSTTConfig',
]

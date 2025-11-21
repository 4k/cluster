"""
Emotion Display Module
Two-window animated display system for eyes and mouth expressions.
Connects to the event bus to receive animation commands.

Features:
- Eye window with blinking, gaze tracking, and emotional expressions
- Mouth window with viseme-based lip-sync animation
- Pre-defined animation states: idle, waiting, listening, thinking, speaking
- Event bus integration for real-time control
- Platform independent (Windows, Linux, macOS)

Usage:
    # Standalone mode (combined window)
    python -m emotion_display.emotion_display_service

    # Multi-window mode (separate processes)
    python -m emotion_display.multi_window_launcher

    # Programmatic usage
    from emotion_display import EmotionDisplayService, EmotionDisplayConfig
    service = EmotionDisplayService()
    await service.initialize()
    await service.start()
"""

from .emotion_display_service import EmotionDisplayService, EmotionDisplayConfig, WindowMode
from .eye_window import EyeWindow, EyeConfig
from .mouth_window import MouthWindow, MouthConfig
from .animation_states import AnimationState, EmotionState, DisplayState
from .viseme_mapper import VisemeMapper, Viseme, VisemeData

__all__ = [
    # Main service
    'EmotionDisplayService',
    'EmotionDisplayConfig',
    'WindowMode',
    # Eye display
    'EyeWindow',
    'EyeConfig',
    # Mouth display
    'MouthWindow',
    'MouthConfig',
    # States
    'AnimationState',
    'EmotionState',
    'DisplayState',
    # Viseme/lip-sync
    'VisemeMapper',
    'Viseme',
    'VisemeData',
]

__version__ = '1.0.0'

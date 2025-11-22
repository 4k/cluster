"""
Core Package

Core infrastructure for the Cluster voice assistant:
- event_bus: Central pub/sub event system
- config: Configuration management
- types: Shared type definitions
"""

from .event_bus import EventBus, EventType, emit_event
from .config import ConfigManager
from .types import (
    EmotionType,
    GazeDirection,
    MouthShape,
    AnimationState,
    ConversationTurn,
    ConversationContext,
    AudioConfig,
    AudioFrame,
    TTSConfig,
    CameraConfig,
    DisplayConfig,
)

__all__ = [
    # Event bus
    'EventBus',
    'EventType',
    'emit_event',
    # Config
    'ConfigManager',
    # Types
    'EmotionType',
    'GazeDirection',
    'MouthShape',
    'AnimationState',
    'ConversationTurn',
    'ConversationContext',
    'AudioConfig',
    'AudioFrame',
    'TTSConfig',
    'CameraConfig',
    'DisplayConfig',
]

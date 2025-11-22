"""
Core Types for the Cluster Voice Assistant

Shared type definitions used across all modules.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import time


class EmotionType(Enum):
    """Emotion types for the voice assistant."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    CONFUSED = "confused"
    THINKING = "thinking"
    LISTENING = "listening"
    SPEAKING = "speaking"


class GazeDirection(Enum):
    """Gaze direction for eye animations."""
    CENTER = "center"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"


class MouthShape(Enum):
    """Mouth shapes for lip-sync animations."""
    CLOSED = "closed"
    OPEN = "open"
    SMILE = "smile"
    FROWN = "frown"
    NEUTRAL = "neutral"
    # Viseme shapes
    SILENCE = "silence"
    BMP = "bmp"  # B, M, P sounds
    LNT = "lnt"  # L, N, T sounds
    AH = "ah"    # Open vowel sounds
    EE = "ee"    # Wide mouth sounds
    OH = "oh"    # Rounded mouth sounds
    OO = "oo"    # Puckered mouth sounds
    FV = "fv"    # F, V sounds


class AnimationState(Enum):
    """Animation states for the display system."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Full conversation context."""
    turns: List[ConversationTurn] = field(default_factory=list)
    current_emotion: EmotionType = EmotionType.NEUTRAL
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, role: str, content: str) -> None:
        """Add a new turn to the conversation."""
        self.turns.append(ConversationTurn(role=role, content=content))
        self.last_activity = time.time()

    def get_recent_turns(self, count: int = 10) -> List[ConversationTurn]:
        """Get the most recent turns."""
        return self.turns[-count:] if self.turns else []

    def clear(self) -> None:
        """Clear the conversation history."""
        self.turns.clear()
        self.last_activity = time.time()


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 1024
    format: str = "int16"


@dataclass
class AudioFrame:
    """Audio frame with metadata."""
    data: bytes
    sample_rate: int
    channels: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    engine_type: str = "piper"
    model_path: Optional[str] = None
    emotion_support: bool = False
    phoneme_output: bool = False
    sample_rate: int = 22050


@dataclass
class CameraConfig:
    """Camera configuration."""
    enabled: bool = False
    device_id: int = 0
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30


@dataclass
class DisplayConfig:
    """Display configuration."""
    mode: str = "dual_display"
    resolution: Tuple[int, int] = (800, 480)
    fps: int = 30
    fullscreen: bool = False
    development_mode: bool = True


# Re-export common types
__all__ = [
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

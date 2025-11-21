"""
Base Renderer class for the multi-window display system.
All specific renderers (eyes, mouth, etc.) inherit from this.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class AnimationState(Enum):
    """Animation states for renderers."""
    IDLE = auto()
    WAITING = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()
    SLEEPING = auto()
    SURPRISED = auto()
    ACKNOWLEDGING = auto()


class EmotionState(Enum):
    """Emotion states for renderers."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    CONFUSED = auto()
    THINKING = auto()
    EXCITED = auto()
    TIRED = auto()
    SKEPTICAL = auto()


@dataclass
class RendererState:
    """Base state class for renderers."""
    animation_state: AnimationState = AnimationState.IDLE
    emotion_state: EmotionState = EmotionState.NEUTRAL
    timestamp: float = 0.0

    def copy(self) -> 'RendererState':
        """Create a copy of the state."""
        return RendererState(
            animation_state=self.animation_state,
            emotion_state=self.emotion_state,
            timestamp=self.timestamp
        )


# Emotion presets shared by all renderers
EMOTION_PRESETS: Dict[EmotionState, Dict[str, Any]] = {
    EmotionState.NEUTRAL: {
        'eyes': {'upper_lid': 1.0, 'squint': 0.0, 'wide': 0.0},
        'mouth': {'width': 0.5, 'left_corner': 0.0, 'right_corner': 0.0}
    },
    EmotionState.HAPPY: {
        'eyes': {'upper_lid': 0.9, 'squint': 0.2, 'wide': 0.0},
        'mouth': {'width': 0.7, 'left_corner': 0.3, 'right_corner': 0.3}
    },
    EmotionState.SAD: {
        'eyes': {'upper_lid': 0.7, 'squint': 0.0, 'wide': 0.0},
        'mouth': {'width': 0.4, 'left_corner': -0.3, 'right_corner': -0.3}
    },
    EmotionState.ANGRY: {
        'eyes': {'upper_lid': 0.8, 'squint': 0.4, 'wide': 0.0},
        'mouth': {'width': 0.6, 'left_corner': -0.2, 'right_corner': -0.2}
    },
    EmotionState.SURPRISED: {
        'eyes': {'upper_lid': 1.0, 'squint': 0.0, 'wide': 0.5},
        'mouth': {'width': 0.4, 'open_amount': 0.3}
    },
    EmotionState.CONFUSED: {
        'eyes': {'upper_lid': 0.9, 'squint': 0.1, 'wide': 0.0},
        'mouth': {'width': 0.45, 'left_corner': -0.1, 'right_corner': 0.1}
    },
    EmotionState.THINKING: {
        'eyes': {'upper_lid': 0.85, 'squint': 0.15, 'wide': 0.0},
        'mouth': {'width': 0.45, 'left_corner': 0.1, 'right_corner': -0.1}
    },
    EmotionState.EXCITED: {
        'eyes': {'upper_lid': 1.0, 'squint': 0.0, 'wide': 0.3, 'pupil_scale': 1.2},
        'mouth': {'width': 0.75, 'left_corner': 0.4, 'right_corner': 0.4, 'open_amount': 0.2}
    },
    EmotionState.TIRED: {
        'eyes': {'upper_lid': 0.6, 'squint': 0.1, 'wide': 0.0},
        'mouth': {'width': 0.5, 'left_corner': -0.1, 'right_corner': -0.1}
    },
    EmotionState.SKEPTICAL: {
        'eyes': {'upper_lid': 0.85, 'squint': 0.25, 'wide': 0.0},
        'mouth': {'width': 0.55, 'left_corner': 0.15, 'right_corner': -0.1}
    },
}


class BaseRenderer(ABC):
    """
    Abstract base class for all renderers.
    Provides common functionality for animation, state management, and rendering.
    """

    def __init__(self, width: int, height: int,
                 background_color: Tuple[int, int, int] = (20, 20, 25)):
        """
        Initialize the renderer.

        Args:
            width: Rendering width in pixels
            height: Rendering height in pixels
            background_color: RGB background color
        """
        self.width = width
        self.height = height
        self.background_color = background_color

        # State
        self.state = RendererState()
        self._target_state = RendererState()

        # Animation
        self.animation_speed = 0.15  # Interpolation speed

        # Rendering surface (set by set_surface)
        self.surface = None

        logger.debug(f"{self.__class__.__name__} initialized ({width}x{height})")

    def set_surface(self, surface) -> None:
        """
        Set the rendering surface.

        Args:
            surface: Pygame surface to render to
        """
        self.surface = surface

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Update animation state.

        Args:
            dt: Delta time in seconds since last update
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """Render to the surface."""
        pass

    @abstractmethod
    def handle_command(self, command: Dict[str, Any]) -> None:
        """
        Handle a command from the decision module.

        Args:
            command: Command dictionary with 'event', 'content_type', 'data'
        """
        pass

    def set_animation_state(self, state: AnimationState) -> None:
        """Set the animation state."""
        self.state.animation_state = state

    def set_emotion(self, emotion: EmotionState) -> None:
        """Set the emotion state."""
        self.state.emotion_state = emotion

    def get_emotion_preset(self, emotion: EmotionState) -> Dict[str, Any]:
        """Get preset values for an emotion."""
        return EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[EmotionState.NEUTRAL])

    def cleanup(self) -> None:
        """Clean up resources."""
        self.surface = None

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b."""
        return a + (b - a) * t

    @staticmethod
    def smooth_step(t: float) -> float:
        """Smooth step function for natural animation."""
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))

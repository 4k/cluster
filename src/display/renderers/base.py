"""
Base renderer class for content rendering.
All content renderers inherit from this class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple

import pygame

from core.types import ContentType, EmotionType, AnimationState

logger = logging.getLogger(__name__)


class BaseRenderer(ABC):
    """Abstract base class for content renderers.

    Each renderer is responsible for drawing a specific type of content
    to a pygame surface. Renderers are stateless - they receive state
    in the render() method and draw accordingly.

    Attributes:
        content_type: The type of content this renderer handles
        size: The (width, height) of the rendering area
    """

    def __init__(self, content_type: ContentType, size: Tuple[int, int]):
        """Initialize the renderer.

        Args:
            content_type: The type of content this renderer handles
            size: The (width, height) of the rendering area
        """
        self.content_type = content_type
        self.size = size
        self.width = size[0]
        self.height = size[1]

    @abstractmethod
    def render(self, surface: pygame.Surface, state: Any, dt: float) -> None:
        """Render content to the surface.

        Args:
            surface: The pygame surface to draw on
            state: The current state for this content type
            dt: Delta time since last frame in seconds
        """
        pass

    def get_color_for_emotion(self, emotion: EmotionType) -> Tuple[int, int, int]:
        """Get a color associated with an emotion.

        Args:
            emotion: The emotion to get a color for

        Returns:
            RGB tuple for the emotion
        """
        emotion_colors = {
            EmotionType.NEUTRAL: (200, 200, 200),
            EmotionType.HAPPY: (255, 220, 100),
            EmotionType.SAD: (100, 150, 200),
            EmotionType.ANGRY: (255, 100, 100),
            EmotionType.SURPRISED: (255, 200, 150),
            EmotionType.CONFUSED: (180, 150, 200),
            EmotionType.INTERESTED: (150, 220, 150),
            EmotionType.THINKING: (150, 180, 220),
            EmotionType.LISTENING: (100, 200, 255),
            EmotionType.SPEAKING: (150, 255, 150),
            EmotionType.ERROR: (255, 50, 50),
        }
        return emotion_colors.get(emotion, (200, 200, 200))

    def get_color_for_state(self, state: AnimationState) -> Tuple[int, int, int]:
        """Get a color associated with an animation state.

        Args:
            state: The animation state to get a color for

        Returns:
            RGB tuple for the state
        """
        state_colors = {
            AnimationState.IDLE: (200, 200, 200),
            AnimationState.LISTENING: (100, 200, 255),
            AnimationState.PROCESSING: (255, 200, 100),
            AnimationState.SPEAKING: (150, 255, 150),
            AnimationState.ERROR: (255, 50, 50),
            AnimationState.SLEEPING: (100, 100, 150),
        }
        return state_colors.get(state, (200, 200, 200))

    def resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize.

        Args:
            new_size: The new (width, height)
        """
        self.size = new_size
        self.width = new_size[0]
        self.height = new_size[1]

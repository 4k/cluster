"""
Eyes renderer for drawing animated eyes.
Supports gaze direction, blinking, and emotion-based expressions.
"""

import math
import logging
from typing import Tuple

import pygame

from core.types import (
    ContentType, EyeState, EmotionType, GazeDirection, AnimationState
)
from .base import BaseRenderer

logger = logging.getLogger(__name__)


class EyesRenderer(BaseRenderer):
    """Renderer for animated eyes.

    Draws two eyes with pupils that can move based on gaze direction,
    blink, and change appearance based on emotion.
    """

    def __init__(self, size: Tuple[int, int]):
        """Initialize eyes renderer.

        Args:
            size: The (width, height) of the rendering area
        """
        super().__init__(ContentType.EYES, size)

        # Eye geometry (will be calculated based on size)
        self._calculate_geometry()

        # Animation state
        self._blink_timer = 0.0
        self._blink_interval = 3.0  # Seconds between blinks
        self._blink_duration = 0.15  # Duration of blink

        # Smooth gaze interpolation
        self._current_gaze_offset = (0.0, 0.0)
        self._target_gaze_offset = (0.0, 0.0)
        self._gaze_smoothing = 8.0  # Higher = faster interpolation

    def _calculate_geometry(self) -> None:
        """Calculate eye geometry based on current size."""
        # Eye centers - positioned in left and right halves
        eye_spacing = self.width * 0.35
        center_y = self.height * 0.5

        self.left_eye_center = (self.width / 2 - eye_spacing / 2, center_y)
        self.right_eye_center = (self.width / 2 + eye_spacing / 2, center_y)

        # Eye dimensions
        self.eye_width = min(self.width * 0.25, self.height * 0.6)
        self.eye_height = self.eye_width * 0.7

        # Pupil dimensions
        self.pupil_radius = self.eye_width * 0.25
        self.iris_radius = self.eye_width * 0.4

        # Maximum gaze offset
        self.max_gaze_offset = self.eye_width * 0.2

    def resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize."""
        super().resize(new_size)
        self._calculate_geometry()

    def render(self, surface: pygame.Surface, state: EyeState, dt: float) -> None:
        """Render eyes to the surface.

        Args:
            surface: The pygame surface to draw on
            state: The current eye state
            dt: Delta time since last frame in seconds
        """
        # Clear surface with background
        surface.fill((0, 0, 0))

        # Update animations
        self._update_blink(state, dt)
        self._update_gaze(state, dt)

        # Calculate effective openness (including blink)
        openness = state.openness
        if state.is_blinking:
            # Smooth blink animation
            blink_curve = math.sin(state.blink_progress * math.pi)
            openness *= (1.0 - blink_curve)

        # Get colors based on emotion
        eye_color = self._get_eye_color(state.emotion)
        iris_color = self._get_iris_color(state.emotion)
        pupil_color = (20, 20, 30)

        # Draw both eyes
        self._draw_eye(
            surface,
            self.left_eye_center,
            openness,
            self._current_gaze_offset,
            state.pupil_dilation,
            eye_color,
            iris_color,
            pupil_color,
            state.emotion
        )
        self._draw_eye(
            surface,
            self.right_eye_center,
            openness,
            self._current_gaze_offset,
            state.pupil_dilation,
            eye_color,
            iris_color,
            pupil_color,
            state.emotion
        )

    def _update_blink(self, state: EyeState, dt: float) -> None:
        """Update automatic blink timer."""
        self._blink_timer += dt
        if self._blink_timer >= self._blink_interval and not state.is_blinking:
            # Trigger a blink (would be done via event bus in real system)
            self._blink_timer = 0.0

    def _update_gaze(self, state: EyeState, dt: float) -> None:
        """Update smooth gaze interpolation."""
        # Get target offset based on gaze direction
        self._target_gaze_offset = self._get_gaze_offset(state.gaze)

        # Smoothly interpolate current gaze to target
        lerp_factor = min(1.0, self._gaze_smoothing * dt)
        self._current_gaze_offset = (
            self._current_gaze_offset[0] + (self._target_gaze_offset[0] - self._current_gaze_offset[0]) * lerp_factor,
            self._current_gaze_offset[1] + (self._target_gaze_offset[1] - self._current_gaze_offset[1]) * lerp_factor
        )

    def _get_gaze_offset(self, gaze: GazeDirection) -> Tuple[float, float]:
        """Get pupil offset for a gaze direction.

        Args:
            gaze: The gaze direction

        Returns:
            (x_offset, y_offset) tuple
        """
        offsets = {
            GazeDirection.FORWARD: (0, 0),
            GazeDirection.UP: (0, -self.max_gaze_offset),
            GazeDirection.DOWN: (0, self.max_gaze_offset),
            GazeDirection.LEFT: (-self.max_gaze_offset, 0),
            GazeDirection.RIGHT: (self.max_gaze_offset, 0),
            GazeDirection.UP_LEFT: (-self.max_gaze_offset * 0.7, -self.max_gaze_offset * 0.7),
            GazeDirection.UP_RIGHT: (self.max_gaze_offset * 0.7, -self.max_gaze_offset * 0.7),
            GazeDirection.DOWN_LEFT: (-self.max_gaze_offset * 0.7, self.max_gaze_offset * 0.7),
            GazeDirection.DOWN_RIGHT: (self.max_gaze_offset * 0.7, self.max_gaze_offset * 0.7),
        }
        return offsets.get(gaze, (0, 0))

    def _get_eye_color(self, emotion: EmotionType) -> Tuple[int, int, int]:
        """Get eye white color based on emotion."""
        # Mostly white, but can tint based on emotion
        base = (240, 240, 245)
        if emotion == EmotionType.ANGRY:
            return (255, 220, 220)  # Slight red tint
        elif emotion == EmotionType.SAD:
            return (220, 230, 245)  # Slight blue tint
        return base

    def _get_iris_color(self, emotion: EmotionType) -> Tuple[int, int, int]:
        """Get iris color based on emotion."""
        base_color = (70, 130, 180)  # Steel blue
        emotion_tints = {
            EmotionType.HAPPY: (100, 160, 200),
            EmotionType.SAD: (60, 100, 150),
            EmotionType.ANGRY: (150, 80, 80),
            EmotionType.SURPRISED: (120, 150, 200),
            EmotionType.INTERESTED: (80, 160, 140),
            EmotionType.THINKING: (100, 120, 180),
            EmotionType.LISTENING: (80, 150, 200),
            EmotionType.SPEAKING: (90, 170, 160),
        }
        return emotion_tints.get(emotion, base_color)

    def _draw_eye(
        self,
        surface: pygame.Surface,
        center: Tuple[float, float],
        openness: float,
        gaze_offset: Tuple[float, float],
        pupil_dilation: float,
        eye_color: Tuple[int, int, int],
        iris_color: Tuple[int, int, int],
        pupil_color: Tuple[int, int, int],
        emotion: EmotionType
    ) -> None:
        """Draw a single eye.

        Args:
            surface: Surface to draw on
            center: Center position of the eye
            openness: 0.0 (closed) to 1.0 (open)
            gaze_offset: (x, y) offset for pupil position
            pupil_dilation: Pupil size multiplier
            eye_color: Color of the eye white
            iris_color: Color of the iris
            pupil_color: Color of the pupil
            emotion: Current emotion for expression
        """
        cx, cy = int(center[0]), int(center[1])

        # Calculate visible height based on openness
        visible_height = self.eye_height * openness

        if visible_height < 2:
            # Eye is closed - just draw a line
            pygame.draw.line(
                surface,
                (150, 150, 160),
                (cx - self.eye_width / 2, cy),
                (cx + self.eye_width / 2, cy),
                3
            )
            return

        # Draw eye white (ellipse)
        eye_rect = pygame.Rect(
            cx - self.eye_width / 2,
            cy - visible_height / 2,
            self.eye_width,
            visible_height
        )

        # Create a clipping mask for the eye shape
        pygame.draw.ellipse(surface, eye_color, eye_rect)

        # Draw iris and pupil (clipped to eye shape)
        iris_x = cx + gaze_offset[0]
        iris_y = cy + gaze_offset[1]

        # Draw iris
        iris_size = int(self.iris_radius)
        pygame.draw.circle(surface, iris_color, (int(iris_x), int(iris_y)), iris_size)

        # Draw iris detail (darker ring)
        pygame.draw.circle(
            surface,
            (iris_color[0] - 30, iris_color[1] - 30, iris_color[2] - 30),
            (int(iris_x), int(iris_y)),
            iris_size,
            3
        )

        # Draw pupil
        pupil_size = int(self.pupil_radius * pupil_dilation)
        pygame.draw.circle(surface, pupil_color, (int(iris_x), int(iris_y)), pupil_size)

        # Draw highlight (light reflection)
        highlight_x = iris_x - pupil_size * 0.3
        highlight_y = iris_y - pupil_size * 0.3
        highlight_size = max(2, int(pupil_size * 0.25))
        pygame.draw.circle(
            surface,
            (255, 255, 255),
            (int(highlight_x), int(highlight_y)),
            highlight_size
        )

        # Draw eyelid lines based on emotion
        self._draw_eyelids(surface, center, openness, emotion)

    def _draw_eyelids(
        self,
        surface: pygame.Surface,
        center: Tuple[float, float],
        openness: float,
        emotion: EmotionType
    ) -> None:
        """Draw eyelid lines for expression.

        Args:
            surface: Surface to draw on
            center: Center of the eye
            openness: Current eye openness
            emotion: Current emotion
        """
        cx, cy = center
        lid_color = (80, 80, 90)

        # Top eyelid
        top_y = cy - self.eye_height * openness / 2

        # Adjust eyelid curve based on emotion
        if emotion == EmotionType.ANGRY:
            # Angry: eyelids slope down toward center
            points = [
                (cx - self.eye_width / 2, top_y + 5),
                (cx, top_y - 10),
                (cx + self.eye_width / 2, top_y + 5)
            ]
        elif emotion == EmotionType.SAD:
            # Sad: eyelids slope up toward center
            points = [
                (cx - self.eye_width / 2, top_y - 5),
                (cx, top_y + 5),
                (cx + self.eye_width / 2, top_y - 5)
            ]
        elif emotion == EmotionType.SURPRISED:
            # Surprised: raised eyelids
            points = [
                (cx - self.eye_width / 2, top_y + 10),
                (cx, top_y - 5),
                (cx + self.eye_width / 2, top_y + 10)
            ]
        else:
            # Neutral
            points = [
                (cx - self.eye_width / 2, top_y),
                (cx, top_y - 3),
                (cx + self.eye_width / 2, top_y)
            ]

        # Draw eyelid line
        if openness > 0.1:
            pygame.draw.lines(surface, lid_color, False, [(int(p[0]), int(p[1])) for p in points], 2)

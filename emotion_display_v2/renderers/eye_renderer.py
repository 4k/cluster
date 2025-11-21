"""
Eye Renderer for the multi-window display system.
Handles eye rendering with blinking, gaze tracking, and emotional expressions.
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

from .base_renderer import (
    BaseRenderer, RendererState, AnimationState, EmotionState, EMOTION_PRESETS
)

logger = logging.getLogger(__name__)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


@dataclass
class EyeState:
    """State of a single eye."""
    x: float = 0.5  # Pupil position (0-1)
    y: float = 0.5
    upper_lid: float = 1.0  # 0=closed, 1=open
    lower_lid: float = 0.0
    pupil_scale: float = 1.0  # 0.5-1.5 multiplier
    squint: float = 0.0  # 0-1
    wide: float = 0.0  # 0-1

    def copy(self) -> 'EyeState':
        return EyeState(
            x=self.x, y=self.y,
            upper_lid=self.upper_lid, lower_lid=self.lower_lid,
            pupil_scale=self.pupil_scale, squint=self.squint, wide=self.wide
        )


@dataclass
class EyeRendererState(RendererState):
    """Extended state for eye renderer."""
    left_eye: EyeState = field(default_factory=EyeState)
    right_eye: EyeState = field(default_factory=EyeState)
    blink_progress: float = 0.0  # 0=open, 1=closed
    is_blinking: bool = False
    gaze_x: float = 0.5  # Target gaze (0-1)
    gaze_y: float = 0.5


class EyeRenderer(BaseRenderer):
    """
    Renderer for animated eyes with blinking, gaze tracking, and expressions.
    """

    def __init__(self, width: int, height: int,
                 background_color: Tuple[int, int, int] = (20, 20, 25),
                 **kwargs):
        super().__init__(width, height, background_color)

        # Eye state
        self.eye_state = EyeRendererState()

        # Colors
        self.iris_color = kwargs.get('iris_color', (100, 150, 200))
        self.pupil_color = kwargs.get('pupil_color', (20, 20, 30))
        self.sclera_color = kwargs.get('sclera_color', (240, 240, 245))
        self.highlight_color = (255, 255, 255)

        # Blink settings
        self.blink_duration = kwargs.get('blink_duration', 0.15)
        self.blink_interval_min = kwargs.get('blink_interval_min', 2.0)
        self.blink_interval_max = kwargs.get('blink_interval_max', 6.0)
        self._next_blink_time = time.time() + random.uniform(
            self.blink_interval_min, self.blink_interval_max
        )
        self._blink_start_time = 0.0

        # Gaze settings
        self.gaze_smoothing = kwargs.get('gaze_smoothing', 0.1)
        self.max_gaze_offset = kwargs.get('max_gaze_offset', 0.3)
        self._current_gaze_x = 0.5
        self._current_gaze_y = 0.5

        # Eye geometry
        self._calculate_geometry()

        logger.debug("EyeRenderer initialized")

    def _calculate_geometry(self) -> None:
        """Calculate eye geometry based on dimensions."""
        # Eye positions (two eyes)
        eye_spacing = self.width * 0.35
        center_x = self.width // 2
        center_y = self.height // 2

        self.left_eye_center = (center_x - int(eye_spacing / 2), center_y)
        self.right_eye_center = (center_x + int(eye_spacing / 2), center_y)

        # Eye sizes
        self.eye_width = int(self.width * 0.25)
        self.eye_height = int(self.height * 0.6)
        self.iris_radius = int(min(self.eye_width, self.eye_height) * 0.35)
        self.pupil_radius = int(self.iris_radius * 0.45)

    def update(self, dt: float) -> None:
        """Update eye animations."""
        current_time = time.time()

        # Update blink
        self._update_blink(current_time, dt)

        # Update gaze (smooth interpolation)
        self._update_gaze(dt)

        # Update emotion-based parameters
        self._update_emotion_params(dt)

        # Random idle movements
        if self.eye_state.animation_state == AnimationState.IDLE:
            self._update_idle_movement(dt)

    def _update_blink(self, current_time: float, dt: float) -> None:
        """Update blink animation."""
        if self.eye_state.is_blinking:
            # Calculate blink progress
            elapsed = current_time - self._blink_start_time
            if elapsed < self.blink_duration / 2:
                # Closing
                self.eye_state.blink_progress = elapsed / (self.blink_duration / 2)
            elif elapsed < self.blink_duration:
                # Opening
                self.eye_state.blink_progress = 1.0 - (
                    (elapsed - self.blink_duration / 2) / (self.blink_duration / 2)
                )
            else:
                # Blink complete
                self.eye_state.is_blinking = False
                self.eye_state.blink_progress = 0.0
                # Schedule next blink
                self._next_blink_time = current_time + random.uniform(
                    self.blink_interval_min, self.blink_interval_max
                )
        elif current_time >= self._next_blink_time:
            # Start blink
            self.trigger_blink()

    def _update_gaze(self, dt: float) -> None:
        """Update gaze with smooth interpolation."""
        # Smooth gaze movement
        self._current_gaze_x = self.lerp(
            self._current_gaze_x,
            self.eye_state.gaze_x,
            self.gaze_smoothing
        )
        self._current_gaze_y = self.lerp(
            self._current_gaze_y,
            self.eye_state.gaze_y,
            self.gaze_smoothing
        )

        # Apply gaze to eyes
        gaze_offset_x = (self._current_gaze_x - 0.5) * 2 * self.max_gaze_offset
        gaze_offset_y = (self._current_gaze_y - 0.5) * 2 * self.max_gaze_offset

        self.eye_state.left_eye.x = 0.5 + gaze_offset_x
        self.eye_state.left_eye.y = 0.5 + gaze_offset_y
        self.eye_state.right_eye.x = 0.5 + gaze_offset_x
        self.eye_state.right_eye.y = 0.5 + gaze_offset_y

    def _update_emotion_params(self, dt: float) -> None:
        """Update eye parameters based on emotion."""
        preset = self.get_emotion_preset(self.eye_state.emotion_state)
        eye_params = preset.get('eyes', {})

        # Interpolate to target values
        target_upper_lid = eye_params.get('upper_lid', 1.0)
        target_squint = eye_params.get('squint', 0.0)
        target_wide = eye_params.get('wide', 0.0)
        target_pupil_scale = eye_params.get('pupil_scale', 1.0)

        speed = self.animation_speed

        for eye in [self.eye_state.left_eye, self.eye_state.right_eye]:
            eye.upper_lid = self.lerp(eye.upper_lid, target_upper_lid, speed)
            eye.squint = self.lerp(eye.squint, target_squint, speed)
            eye.wide = self.lerp(eye.wide, target_wide, speed)
            eye.pupil_scale = self.lerp(eye.pupil_scale, target_pupil_scale, speed)

    def _update_idle_movement(self, dt: float) -> None:
        """Add subtle idle movements."""
        # Small random gaze drifts
        if random.random() < 0.01:  # 1% chance per frame
            self.eye_state.gaze_x = 0.5 + random.uniform(-0.1, 0.1)
            self.eye_state.gaze_y = 0.5 + random.uniform(-0.05, 0.05)

    def render(self) -> None:
        """Render the eyes."""
        if not self.surface or not PYGAME_AVAILABLE:
            return

        # Clear background
        self.surface.fill(self.background_color)

        # Render each eye
        self._render_eye(self.left_eye_center, self.eye_state.left_eye)
        self._render_eye(self.right_eye_center, self.eye_state.right_eye)

    def _render_eye(self, center: Tuple[int, int], eye: EyeState) -> None:
        """Render a single eye."""
        cx, cy = center

        # Calculate effective lid positions (affected by blink and emotion)
        effective_upper = eye.upper_lid * (1.0 - self.eye_state.blink_progress)
        effective_upper = max(0.0, effective_upper - eye.squint * 0.3)
        effective_upper = min(1.0, effective_upper + eye.wide * 0.2)

        effective_lower = eye.lower_lid + self.eye_state.blink_progress * 0.5

        # Calculate eye opening
        eye_open = effective_upper - effective_lower
        if eye_open <= 0.05:
            # Eye closed - just draw a line
            pygame.draw.line(
                self.surface,
                self.sclera_color,
                (cx - self.eye_width // 2, cy),
                (cx + self.eye_width // 2, cy),
                3
            )
            return

        # Draw eye socket (sclera)
        eye_rect = pygame.Rect(
            cx - self.eye_width // 2,
            cy - int(self.eye_height * eye_open / 2),
            self.eye_width,
            int(self.eye_height * eye_open)
        )
        pygame.draw.ellipse(self.surface, self.sclera_color, eye_rect)

        # Calculate pupil position within eye
        pupil_range_x = self.eye_width * 0.3
        pupil_range_y = self.eye_height * eye_open * 0.3

        pupil_x = cx + int((eye.x - 0.5) * 2 * pupil_range_x)
        pupil_y = cy + int((eye.y - 0.5) * 2 * pupil_range_y)

        # Draw iris
        scaled_iris = int(self.iris_radius * (0.8 + eye_open * 0.2))
        pygame.draw.circle(self.surface, self.iris_color, (pupil_x, pupil_y), scaled_iris)

        # Draw pupil
        scaled_pupil = int(self.pupil_radius * eye.pupil_scale)
        pygame.draw.circle(self.surface, self.pupil_color, (pupil_x, pupil_y), scaled_pupil)

        # Draw highlight
        highlight_offset = int(scaled_iris * 0.3)
        highlight_size = int(scaled_pupil * 0.4)
        pygame.draw.circle(
            self.surface,
            self.highlight_color,
            (pupil_x - highlight_offset, pupil_y - highlight_offset),
            highlight_size
        )

        # Draw eyelids if partially closed
        if effective_upper < 0.9:
            lid_height = int(self.eye_height * (1.0 - effective_upper) / 2)
            lid_rect = pygame.Rect(
                cx - self.eye_width // 2 - 5,
                cy - self.eye_height // 2,
                self.eye_width + 10,
                lid_height + cy - (cy - int(self.eye_height * eye_open / 2))
            )
            pygame.draw.rect(self.surface, self.background_color, lid_rect)

    def handle_command(self, command: Dict[str, Any]) -> None:
        """Handle commands from the decision module."""
        event = command.get('event', '')
        data = command.get('data', {})

        if event == 'blink':
            self.trigger_blink()
        elif event == 'gaze_update':
            self.set_gaze(data.get('x', 0.5), data.get('y', 0.5))
        elif event == 'pupil_dilate':
            scale = data.get('scale', 1.0)
            self.eye_state.left_eye.pupil_scale = scale
            self.eye_state.right_eye.pupil_scale = scale
        elif event == 'emotion_change':
            emotion_name = data.get('emotion', 'NEUTRAL')
            try:
                emotion = EmotionState[emotion_name.upper()]
                self.set_emotion(emotion)
            except KeyError:
                logger.warning(f"Unknown emotion: {emotion_name}")
        elif event == 'animation_state':
            state_name = data.get('state', 'IDLE')
            try:
                state = AnimationState[state_name.upper()]
                self.set_animation_state(state)
            except KeyError:
                logger.warning(f"Unknown animation state: {state_name}")

    def trigger_blink(self) -> None:
        """Trigger a blink animation."""
        if not self.eye_state.is_blinking:
            self.eye_state.is_blinking = True
            self._blink_start_time = time.time()

    def set_gaze(self, x: float, y: float) -> None:
        """Set gaze target position."""
        self.eye_state.gaze_x = self.clamp(x, 0.0, 1.0)
        self.eye_state.gaze_y = self.clamp(y, 0.0, 1.0)

    def set_emotion(self, emotion: EmotionState) -> None:
        """Set emotion state."""
        self.eye_state.emotion_state = emotion

    def set_animation_state(self, state: AnimationState) -> None:
        """Set animation state."""
        self.eye_state.animation_state = state

"""
Eye window for emotion display.
Renders animated eyes with blinking, gaze tracking, and emotional expressions.
Uses Pygame for cross-platform 2D rendering.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import pygame


from .animation_states import (
    AnimationState, EmotionState, EyeState, EyesState,
    EMOTION_PRESETS, get_emotion_preset
)


@dataclass
class EyeConfig:
    """Configuration for eye rendering."""
    # Window settings
    window_width: int = 800
    window_height: int = 400
    window_title: str = "Eyes"
    background_color: Tuple[int, int, int] = (20, 20, 25)

    # Eye dimensions
    eye_width: int = 160
    eye_height: int = 180
    eye_spacing: int = 100  # Space between eyes

    # Colors
    sclera_color: Tuple[int, int, int] = (255, 255, 255)
    iris_color: Tuple[int, int, int] = (70, 130, 180)  # Steel blue
    pupil_color: Tuple[int, int, int] = (10, 10, 15)
    eyelid_color: Tuple[int, int, int] = (40, 40, 50)
    highlight_color: Tuple[int, int, int] = (255, 255, 255)

    # Animation settings
    blink_interval_min: float = 2.0  # Minimum time between blinks
    blink_interval_max: float = 6.0  # Maximum time between blinks
    blink_duration: float = 0.15     # Duration of a single blink
    gaze_smoothing: float = 0.1      # Smoothing factor for gaze movement

    # Pupil settings
    pupil_radius_ratio: float = 0.35  # Pupil size relative to iris
    iris_radius_ratio: float = 0.45   # Iris size relative to eye

    # Movement limits
    max_gaze_offset: float = 0.3  # Maximum gaze offset from center


class EyeWindow:
    """
    Manages the eye display window with animations.
    """

    def __init__(self, config: Optional[EyeConfig] = None):
        """
        Initialize the eye window.

        Args:
            config: Eye configuration settings
        """
        self.config = config or EyeConfig()
        self.state = EyesState()
        self.animation_state = AnimationState.IDLE
        self.emotion_state = EmotionState.NEUTRAL

        # Window and rendering
        self.window: Optional[pygame.Surface] = None
        self.is_running = False

        # Blink timing
        self._next_blink_time = time.time() + random.uniform(
            self.config.blink_interval_min,
            self.config.blink_interval_max
        )
        self._blink_start_time = 0.0

        # Gaze animation
        self._current_gaze = (0.5, 0.5)
        self._target_gaze = (0.5, 0.5)

        # Idle animation
        self._idle_offset = 0.0
        self._idle_speed = 0.5

        # Emotion transition
        self._emotion_transition = 0.0
        self._prev_emotion = EmotionState.NEUTRAL

    def initialize(self) -> bool:
        """
        Initialize the pygame window.

        Returns:
            True if initialization succeeded
        """
        try:
            # Initialize pygame if not already done
            if not pygame.get_init():
                pygame.init()

            # Create window
            self.window = pygame.display.set_mode(
                (self.config.window_width, self.config.window_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption(self.config.window_title)

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to initialize eye window: {e}")
            return False

    def set_surface(self, surface: pygame.Surface) -> None:
        """
        Set an external surface to render to (for multi-window support).

        Args:
            surface: Pygame surface to render to
        """
        self.window = surface
        self.config.window_width = surface.get_width()
        self.config.window_height = surface.get_height()
        self.is_running = True

    def update(self, dt: float) -> None:
        """
        Update eye animations.

        Args:
            dt: Delta time since last update in seconds
        """
        current_time = time.time()

        # Update blink animation
        self._update_blink(current_time)

        # Update gaze position with smoothing
        self._update_gaze(dt)

        # Update idle animation
        self._update_idle(dt, current_time)

        # Update emotion transition
        self._update_emotion_transition(dt)

        # Apply animation state modifiers
        self._apply_animation_state(current_time)

    def _update_blink(self, current_time: float) -> None:
        """Update blink animation."""
        # Check if it's time to start a new blink
        if not self.state.is_blinking and current_time >= self._next_blink_time:
            self.state.is_blinking = True
            self._blink_start_time = current_time
            # Schedule next blink
            self._next_blink_time = current_time + random.uniform(
                self.config.blink_interval_min,
                self.config.blink_interval_max
            )

        # Update blink progress
        if self.state.is_blinking:
            blink_elapsed = current_time - self._blink_start_time
            blink_progress = blink_elapsed / self.config.blink_duration

            if blink_progress >= 1.0:
                # Blink complete
                self.state.is_blinking = False
                self.state.blink_progress = 0.0
            else:
                # Blink animation: quick close, slower open
                if blink_progress < 0.4:
                    # Closing (fast)
                    self.state.blink_progress = blink_progress / 0.4
                else:
                    # Opening (slower)
                    self.state.blink_progress = 1.0 - ((blink_progress - 0.4) / 0.6)

    def _update_gaze(self, dt: float) -> None:
        """Update gaze position with smoothing."""
        # Smooth interpolation towards target
        smoothing = 1.0 - pow(1.0 - self.config.gaze_smoothing, dt * 60)

        self._current_gaze = (
            self._current_gaze[0] + (self._target_gaze[0] - self._current_gaze[0]) * smoothing,
            self._current_gaze[1] + (self._target_gaze[1] - self._current_gaze[1]) * smoothing
        )

        # Update eye states with gaze
        max_offset = self.config.max_gaze_offset
        gaze_x = 0.5 + (self._current_gaze[0] - 0.5) * max_offset * 2
        gaze_y = 0.5 + (self._current_gaze[1] - 0.5) * max_offset * 2

        self.state.left.x = gaze_x
        self.state.left.y = gaze_y
        self.state.right.x = gaze_x
        self.state.right.y = gaze_y

    def _update_idle(self, dt: float, current_time: float) -> None:
        """Update idle animation (subtle movement)."""
        if self.animation_state == AnimationState.IDLE:
            # Subtle breathing/drifting movement
            self._idle_offset += dt * self._idle_speed
            drift_x = math.sin(self._idle_offset * 0.5) * 0.02
            drift_y = math.sin(self._idle_offset * 0.3) * 0.015

            # Apply drift to target gaze if no explicit target
            if self.state.gaze_target is None:
                self._target_gaze = (0.5 + drift_x, 0.5 + drift_y)

        elif self.animation_state == AnimationState.WAITING:
            # More pronounced movement when waiting
            self._idle_offset += dt * self._idle_speed * 1.5
            drift_x = math.sin(self._idle_offset * 0.7) * 0.05
            drift_y = math.sin(self._idle_offset * 0.4) * 0.03

            if self.state.gaze_target is None:
                self._target_gaze = (0.5 + drift_x, 0.5 + drift_y)

    def _update_emotion_transition(self, dt: float) -> None:
        """Update emotion state transition."""
        if self._emotion_transition < 1.0:
            self._emotion_transition = min(1.0, self._emotion_transition + dt * 3.0)

            # Get preset values
            prev_preset = get_emotion_preset(self._prev_emotion)
            curr_preset = get_emotion_preset(self.emotion_state)

            # Interpolate eye settings
            t = self._smooth_step(self._emotion_transition)

            for eye in [self.state.left, self.state.right]:
                prev_eyes = prev_preset.get('eyes', {})
                curr_eyes = curr_preset.get('eyes', {})

                eye.upper_lid = self._lerp(
                    prev_eyes.get('upper_lid', 1.0),
                    curr_eyes.get('upper_lid', 1.0), t
                )
                eye.squint = self._lerp(
                    prev_eyes.get('squint', 0.0),
                    curr_eyes.get('squint', 0.0), t
                )
                eye.wide = self._lerp(
                    prev_eyes.get('wide', 0.0),
                    curr_eyes.get('wide', 0.0), t
                )
                eye.pupil_scale = self._lerp(
                    prev_eyes.get('pupil_scale', 1.0),
                    curr_eyes.get('pupil_scale', 1.0), t
                )

    def _apply_animation_state(self, current_time: float) -> None:
        """Apply animation state specific modifications."""
        if self.animation_state == AnimationState.THINKING:
            # Look up and to the side when thinking
            think_offset = math.sin(current_time * 2) * 0.1
            self._target_gaze = (0.7 + think_offset, 0.3)

        elif self.animation_state == AnimationState.LISTENING:
            # Wide, attentive eyes
            for eye in [self.state.left, self.state.right]:
                eye.wide = max(eye.wide, 0.2)
                eye.pupil_scale = max(eye.pupil_scale, 1.1)

        elif self.animation_state == AnimationState.SPEAKING:
            # Slight squint, engaged expression
            for eye in [self.state.left, self.state.right]:
                eye.squint = max(eye.squint, 0.1)

        elif self.animation_state == AnimationState.ERROR:
            # Wide eyes, surprised look
            for eye in [self.state.left, self.state.right]:
                eye.wide = 0.4
                eye.pupil_scale = 0.8

        elif self.animation_state == AnimationState.SLEEPING:
            # Eyes mostly closed
            for eye in [self.state.left, self.state.right]:
                eye.upper_lid = 0.15

    def render(self) -> None:
        """Render the eyes to the window."""
        if self.window is None:
            return

        # Clear background
        self.window.fill(self.config.background_color)

        # Calculate eye positions
        center_x = self.config.window_width // 2
        center_y = self.config.window_height // 2
        half_spacing = self.config.eye_spacing // 2

        left_eye_center = (center_x - half_spacing - self.config.eye_width // 2, center_y)
        right_eye_center = (center_x + half_spacing + self.config.eye_width // 2, center_y)

        # Render each eye
        self._render_eye(left_eye_center, self.state.left, is_left=True)
        self._render_eye(right_eye_center, self.state.right, is_left=False)

    def _render_eye(self, center: Tuple[int, int], eye_state: EyeState,
                   is_left: bool) -> None:
        """Render a single eye."""
        cx, cy = center
        w = self.config.eye_width
        h = self.config.eye_height

        # Calculate effective lid positions (including blink)
        blink_amount = self.state.blink_progress
        upper_lid = eye_state.upper_lid * (1.0 - blink_amount)
        lower_lid = eye_state.lower_lid + blink_amount * 0.3

        # Apply squint/wide modifiers
        upper_lid *= (1.0 - eye_state.squint * 0.3)
        if eye_state.wide > 0:
            upper_lid = min(1.0, upper_lid + eye_state.wide * 0.2)

        # Draw sclera (white of eye) as ellipse
        sclera_rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
        pygame.draw.ellipse(self.window, self.config.sclera_color, sclera_rect)

        # Calculate iris position based on gaze
        iris_radius = int(min(w, h) * self.config.iris_radius_ratio)
        max_move_x = (w // 2 - iris_radius) * 0.7
        max_move_y = (h // 2 - iris_radius) * 0.7

        iris_x = cx + int((eye_state.x - 0.5) * max_move_x * 2)
        iris_y = cy + int((eye_state.y - 0.5) * max_move_y * 2)

        # Draw iris
        pygame.draw.circle(
            self.window, self.config.iris_color,
            (iris_x, iris_y), iris_radius
        )

        # Draw pupil
        pupil_radius = int(iris_radius * self.config.pupil_radius_ratio * eye_state.pupil_scale)
        pygame.draw.circle(
            self.window, self.config.pupil_color,
            (iris_x, iris_y), pupil_radius
        )

        # Draw highlight
        highlight_offset = iris_radius // 3
        highlight_radius = pupil_radius // 2
        pygame.draw.circle(
            self.window, self.config.highlight_color,
            (iris_x - highlight_offset, iris_y - highlight_offset),
            highlight_radius
        )

        # Draw eyelids
        self._render_eyelids(center, w, h, upper_lid, lower_lid)

    def _render_eyelids(self, center: Tuple[int, int], width: int, height: int,
                       upper_lid: float, lower_lid: float) -> None:
        """Render eyelids over the eye."""
        cx, cy = center
        half_w = width // 2
        half_h = height // 2

        # Upper eyelid
        if upper_lid < 1.0:
            # Calculate how much of the eye is covered
            lid_coverage = 1.0 - upper_lid
            lid_y = cy - half_h + int(height * lid_coverage * 0.6)

            # Draw upper lid as a curved shape
            points = [
                (cx - half_w - 10, cy - half_h - 20),
                (cx - half_w - 10, lid_y),
                (cx, lid_y + 10),
                (cx + half_w + 10, lid_y),
                (cx + half_w + 10, cy - half_h - 20),
            ]
            pygame.draw.polygon(self.window, self.config.eyelid_color, points)

        # Lower eyelid (usually minimal unless expressing emotion)
        if lower_lid > 0.0:
            lid_y = cy + half_h - int(height * lower_lid * 0.3)

            points = [
                (cx - half_w - 10, cy + half_h + 20),
                (cx - half_w - 10, lid_y),
                (cx, lid_y - 5),
                (cx + half_w + 10, lid_y),
                (cx + half_w + 10, cy + half_h + 20),
            ]
            pygame.draw.polygon(self.window, self.config.eyelid_color, points)

    def set_gaze(self, x: float, y: float) -> None:
        """
        Set the gaze target position.

        Args:
            x: Horizontal position (0=left, 1=right)
            y: Vertical position (0=top, 1=bottom)
        """
        self._target_gaze = (
            max(0.0, min(1.0, x)),
            max(0.0, min(1.0, y))
        )
        self.state.gaze_target = self._target_gaze

    def clear_gaze(self) -> None:
        """Clear the gaze target (return to idle movement)."""
        self.state.gaze_target = None

    def trigger_blink(self) -> None:
        """Trigger an immediate blink."""
        if not self.state.is_blinking:
            self.state.is_blinking = True
            self._blink_start_time = time.time()

    def set_emotion(self, emotion: EmotionState) -> None:
        """
        Set the emotional expression.

        Args:
            emotion: New emotion state
        """
        if emotion != self.emotion_state:
            self._prev_emotion = self.emotion_state
            self.emotion_state = emotion
            self._emotion_transition = 0.0

    def set_animation_state(self, state: AnimationState) -> None:
        """
        Set the animation state.

        Args:
            state: New animation state
        """
        self.animation_state = state

    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t

    @staticmethod
    def _smooth_step(t: float) -> float:
        """Smooth step function."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

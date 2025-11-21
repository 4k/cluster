"""
Mouth window for emotion display.
Renders animated mouth with viseme-based lip-sync and emotional expressions.
Uses Pygame for cross-platform 2D rendering.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pygame

from .animation_states import (
    AnimationState, EmotionState, MouthState,
    EMOTION_PRESETS, get_emotion_preset
)
from .viseme_mapper import Viseme, VisemeMapper, VisemeData, VISEME_SHAPES


@dataclass
class MouthConfig:
    """Configuration for mouth rendering."""
    # Window settings
    window_width: int = 800
    window_height: int = 300
    window_title: str = "Mouth"
    background_color: Tuple[int, int, int] = (20, 20, 25)

    # Mouth dimensions
    mouth_width: int = 300
    mouth_height: int = 150

    # Colors
    lip_color: Tuple[int, int, int] = (180, 80, 80)
    lip_outline_color: Tuple[int, int, int] = (140, 60, 60)
    inner_mouth_color: Tuple[int, int, int] = (60, 30, 30)
    teeth_color: Tuple[int, int, int] = (245, 245, 240)
    tongue_color: Tuple[int, int, int] = (200, 100, 100)

    # Animation settings
    transition_speed: float = 12.0  # Viseme transition speed
    idle_movement: float = 0.02     # Subtle movement when idle

    # Speaking settings
    min_speak_open: float = 0.1    # Minimum mouth opening when speaking
    max_speak_open: float = 0.8    # Maximum mouth opening when speaking


class MouthWindow:
    """
    Manages the mouth display window with animations.
    """

    def __init__(self, config: Optional[MouthConfig] = None):
        """
        Initialize the mouth window.

        Args:
            config: Mouth configuration settings
        """
        self.config = config or MouthConfig()
        self.state = MouthState()
        self.animation_state = AnimationState.IDLE
        self.emotion_state = EmotionState.NEUTRAL

        # Window and rendering
        self.window: Optional[pygame.Surface] = None
        self.is_running = False

        # Viseme animation
        self.viseme_mapper = VisemeMapper()
        self._current_viseme = Viseme.SILENCE
        self._target_viseme = Viseme.SILENCE
        self._current_viseme_data = VISEME_SHAPES[Viseme.SILENCE]
        self._target_viseme_data = VISEME_SHAPES[Viseme.SILENCE]
        self._viseme_transition = 1.0

        # Text animation queue
        self._viseme_queue: List[Tuple[Viseme, float]] = []
        self._current_viseme_time = 0.0
        self._viseme_duration = 0.0

        # Speaking animation (when no text available)
        self._speak_time = 0.0
        self._is_speaking = False

        # Emotion transition
        self._emotion_transition = 1.0
        self._prev_emotion = EmotionState.NEUTRAL

        # Idle animation
        self._idle_time = 0.0

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
            print(f"Failed to initialize mouth window: {e}")
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
        Update mouth animations.

        Args:
            dt: Delta time since last update in seconds
        """
        # Update viseme queue if speaking with text
        if self._viseme_queue:
            self._update_viseme_queue(dt)
        elif self._is_speaking:
            # Generic speaking animation
            self._update_speaking_animation(dt)

        # Update viseme transition
        self._update_viseme_transition(dt)

        # Update emotion transition
        self._update_emotion_transition(dt)

        # Update idle animation
        self._update_idle(dt)

        # Apply current viseme to mouth state
        self._apply_viseme_to_state()

    def _update_viseme_queue(self, dt: float) -> None:
        """Process the viseme animation queue."""
        self._current_viseme_time += dt

        # Check if current viseme is done
        if self._current_viseme_time >= self._viseme_duration:
            if self._viseme_queue:
                # Get next viseme
                next_viseme, duration = self._viseme_queue.pop(0)
                self._set_target_viseme(next_viseme)
                self._viseme_duration = duration
                self._current_viseme_time = 0.0
            else:
                # Queue empty, return to silence
                self._set_target_viseme(Viseme.SILENCE)
                self._is_speaking = False

    def _update_speaking_animation(self, dt: float) -> None:
        """Update generic speaking animation."""
        self._speak_time += dt

        # Create rhythmic mouth movement
        # Vary the frequency slightly for natural feel
        freq = 8.0 + math.sin(self._speak_time * 0.5) * 2.0
        amplitude = 0.3 + math.sin(self._speak_time * 1.3) * 0.15

        # Generate pseudo-random viseme changes
        if random.random() < dt * 10:  # About 10 changes per second
            speaking_visemes = [
                Viseme.AH, Viseme.EE, Viseme.OH, Viseme.EH,
                Viseme.LNT, Viseme.BMP, Viseme.SILENCE
            ]
            weights = [0.25, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1]
            new_viseme = random.choices(speaking_visemes, weights=weights)[0]
            self._set_target_viseme(new_viseme)

    def _update_viseme_transition(self, dt: float) -> None:
        """Update smooth transition between visemes."""
        if self._viseme_transition < 1.0:
            self._viseme_transition = min(
                1.0,
                self._viseme_transition + dt * self.config.transition_speed
            )

    def _update_emotion_transition(self, dt: float) -> None:
        """Update emotion state transition."""
        if self._emotion_transition < 1.0:
            self._emotion_transition = min(1.0, self._emotion_transition + dt * 3.0)

    def _update_idle(self, dt: float) -> None:
        """Update idle animation."""
        self._idle_time += dt

        if self.animation_state == AnimationState.IDLE and not self._is_speaking:
            # Very subtle mouth movement
            idle_offset = math.sin(self._idle_time * 0.5) * self.config.idle_movement
            self.state.open_amount = max(0, idle_offset)

        elif self.animation_state == AnimationState.WAITING:
            # Slight smile when waiting
            if not self._is_speaking:
                wait_smile = 0.05 + math.sin(self._idle_time * 0.3) * 0.02
                self.state.left_corner = wait_smile
                self.state.right_corner = wait_smile

    def _set_target_viseme(self, viseme: Viseme) -> None:
        """Set a new target viseme to transition to."""
        if viseme != self._target_viseme:
            self._current_viseme = self._target_viseme
            self._current_viseme_data = self._target_viseme_data
            self._target_viseme = viseme
            self._target_viseme_data = VISEME_SHAPES.get(viseme, VISEME_SHAPES[Viseme.SILENCE])
            self._viseme_transition = 0.0

    def _apply_viseme_to_state(self) -> None:
        """Apply the current viseme interpolation to the mouth state."""
        # Interpolate between current and target viseme
        t = self._smooth_step(self._viseme_transition)
        viseme_data = self._current_viseme_data.interpolate(self._target_viseme_data, t)

        # Apply viseme data to state (blend with emotion)
        emotion_preset = get_emotion_preset(self.emotion_state)
        emotion_mouth = emotion_preset.get('mouth', {})

        # Emotion-based corner positions
        base_left_corner = emotion_mouth.get('left_corner', 0.0)
        base_right_corner = emotion_mouth.get('right_corner', 0.0)
        base_width = emotion_mouth.get('width', 0.5)

        # Blend viseme with emotion
        self.state.open_amount = viseme_data.open_amount
        self.state.width = base_width + (viseme_data.width - 0.5) * 0.5
        self.state.pucker = viseme_data.pucker
        self.state.stretch = viseme_data.stretch
        self.state.upper_lip = viseme_data.upper_lip
        self.state.lower_lip = viseme_data.lower_lip

        # Add emotion-based corner offsets
        if not self._is_speaking:
            self.state.left_corner = base_left_corner
            self.state.right_corner = base_right_corner

        self.state.current_viseme = self._target_viseme.name

    def render(self) -> None:
        """Render the mouth to the window."""
        if self.window is None:
            return

        # Clear background
        self.window.fill(self.config.background_color)

        # Calculate mouth position
        center_x = self.config.window_width // 2
        center_y = self.config.window_height // 2

        # Render mouth
        self._render_mouth((center_x, center_y))

    def _render_mouth(self, center: Tuple[int, int]) -> None:
        """Render the mouth."""
        cx, cy = center
        base_w = self.config.mouth_width
        base_h = self.config.mouth_height

        # Calculate dimensions based on state
        width = int(base_w * (0.6 + self.state.width * 0.8))
        open_amount = self.state.open_amount

        # Pucker makes mouth narrower and rounder
        if self.state.pucker > 0:
            width = int(width * (1.0 - self.state.pucker * 0.4))

        # Stretch makes mouth wider
        if self.state.stretch > 0:
            width = int(width * (1.0 + self.state.stretch * 0.3))

        half_w = width // 2
        half_h = base_h // 2

        # Calculate corner positions
        left_corner_y = int(cy + self.state.left_corner * 30)
        right_corner_y = int(cy + self.state.right_corner * 30)

        # Calculate lip positions
        upper_lip_y = int(cy - 10 - open_amount * half_h * 0.7 + self.state.upper_lip * 10)
        lower_lip_y = int(cy + 10 + open_amount * half_h * 0.7 + self.state.lower_lip * 10)

        # Draw inner mouth (dark area)
        if open_amount > 0.05:
            inner_points = self._calculate_mouth_shape(
                cx, cy, half_w - 5, upper_lip_y + 8, lower_lip_y - 8,
                left_corner_y, right_corner_y, self.state.pucker
            )
            pygame.draw.polygon(self.window, self.config.inner_mouth_color, inner_points)

            # Draw teeth (when mouth is open enough)
            if open_amount > 0.15:
                teeth_y = upper_lip_y + 12
                teeth_w = int(half_w * 0.7)
                teeth_h = min(15, int(open_amount * 30))
                teeth_rect = pygame.Rect(cx - teeth_w, teeth_y, teeth_w * 2, teeth_h)
                pygame.draw.rect(self.window, self.config.teeth_color, teeth_rect, border_radius=3)

            # Draw tongue (when mouth is very open)
            if open_amount > 0.4:
                tongue_y = lower_lip_y - 15
                tongue_w = int(half_w * 0.4)
                tongue_h = int(open_amount * 20)
                pygame.draw.ellipse(
                    self.window, self.config.tongue_color,
                    pygame.Rect(cx - tongue_w, tongue_y - tongue_h//2, tongue_w * 2, tongue_h)
                )

        # Draw lips (upper)
        upper_points = self._calculate_upper_lip(
            cx, cy, half_w, upper_lip_y, left_corner_y, right_corner_y, self.state.pucker
        )
        pygame.draw.polygon(self.window, self.config.lip_color, upper_points)
        pygame.draw.lines(self.window, self.config.lip_outline_color, False, upper_points, 2)

        # Draw lips (lower)
        lower_points = self._calculate_lower_lip(
            cx, cy, half_w, lower_lip_y, left_corner_y, right_corner_y, self.state.pucker
        )
        pygame.draw.polygon(self.window, self.config.lip_color, lower_points)
        pygame.draw.lines(self.window, self.config.lip_outline_color, False, lower_points, 2)

    def _calculate_mouth_shape(self, cx: int, cy: int, half_w: int,
                              upper_y: int, lower_y: int,
                              left_corner_y: int, right_corner_y: int,
                              pucker: float) -> List[Tuple[int, int]]:
        """Calculate points for the inner mouth shape."""
        # Pucker affects the shape
        curve = 0.3 - pucker * 0.15

        points = [
            (cx - half_w, left_corner_y),
            (cx - int(half_w * 0.5), upper_y - int(curve * 20)),
            (cx, upper_y),
            (cx + int(half_w * 0.5), upper_y - int(curve * 20)),
            (cx + half_w, right_corner_y),
            (cx + int(half_w * 0.5), lower_y + int(curve * 20)),
            (cx, lower_y),
            (cx - int(half_w * 0.5), lower_y + int(curve * 20)),
        ]
        return points

    def _calculate_upper_lip(self, cx: int, cy: int, half_w: int,
                            lip_y: int, left_corner_y: int, right_corner_y: int,
                            pucker: float) -> List[Tuple[int, int]]:
        """Calculate points for the upper lip."""
        # Cupid's bow effect
        bow_depth = 5 + int(pucker * 8)

        points = [
            (cx - half_w, left_corner_y),
            (cx - int(half_w * 0.6), lip_y + 5),
            (cx - int(half_w * 0.2), lip_y - bow_depth),
            (cx, lip_y),
            (cx + int(half_w * 0.2), lip_y - bow_depth),
            (cx + int(half_w * 0.6), lip_y + 5),
            (cx + half_w, right_corner_y),
            # Inner edge
            (cx + int(half_w * 0.6), lip_y + 15),
            (cx, lip_y + 10 + int(pucker * 5)),
            (cx - int(half_w * 0.6), lip_y + 15),
        ]
        return points

    def _calculate_lower_lip(self, cx: int, cy: int, half_w: int,
                            lip_y: int, left_corner_y: int, right_corner_y: int,
                            pucker: float) -> List[Tuple[int, int]]:
        """Calculate points for the lower lip."""
        curve = 10 + int(pucker * 10)

        points = [
            (cx - half_w, left_corner_y),
            (cx - int(half_w * 0.6), lip_y - 10),
            (cx, lip_y - 5 - int(pucker * 5)),
            (cx + int(half_w * 0.6), lip_y - 10),
            (cx + half_w, right_corner_y),
            # Outer edge
            (cx + int(half_w * 0.5), lip_y + curve),
            (cx, lip_y + curve + 5),
            (cx - int(half_w * 0.5), lip_y + curve),
        ]
        return points

    def speak_text(self, text: str) -> None:
        """
        Start speaking animation with text-based lip sync.

        Args:
            text: Text to animate mouth to
        """
        # Convert text to viseme sequence
        self._viseme_queue = self.viseme_mapper.text_to_visemes(text)
        self._is_speaking = True
        self._current_viseme_time = 0.0

        if self._viseme_queue:
            viseme, duration = self._viseme_queue.pop(0)
            self._set_target_viseme(viseme)
            self._viseme_duration = duration

    def start_speaking(self) -> None:
        """Start generic speaking animation (when no text available)."""
        self._is_speaking = True
        self._speak_time = 0.0
        self._viseme_queue = []

    def stop_speaking(self) -> None:
        """Stop speaking animation."""
        self._is_speaking = False
        self._viseme_queue = []
        self._set_target_viseme(Viseme.SILENCE)

    def set_viseme(self, viseme: Viseme) -> None:
        """
        Set a specific viseme directly.

        Args:
            viseme: Viseme to display
        """
        self._set_target_viseme(viseme)

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
    def _smooth_step(t: float) -> float:
        """Smooth step function."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

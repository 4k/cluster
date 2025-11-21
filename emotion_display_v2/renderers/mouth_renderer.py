"""
Mouth Renderer for the multi-window display system.
Handles mouth rendering with lip-sync, visemes, and emotional expressions.
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple, List

from .base_renderer import (
    BaseRenderer, RendererState, AnimationState, EmotionState, EMOTION_PRESETS
)

logger = logging.getLogger(__name__)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class Viseme(Enum):
    """Viseme shapes for lip-sync animation."""
    SILENCE = auto()  # Closed mouth
    AH = auto()       # Open vowel (father)
    EE = auto()       # Smile vowel (see)
    EH = auto()       # Mid vowel (bed)
    OH = auto()       # Round vowel (go)
    OO = auto()       # Pucker vowel (too)
    BMP = auto()      # Bilabial (b, m, p)
    FV = auto()       # Labiodental (f, v)
    TH = auto()       # Dental (th)
    LNT = auto()      # Alveolar (l, n, t, d)
    KG = auto()       # Velar (k, g)
    SZ = auto()       # Sibilant (s, z)
    SH_CH = auto()    # Postalveolar (sh, ch)
    R = auto()        # R sound
    W = auto()        # W sound


# Viseme shape definitions
VISEME_SHAPES: Dict[Viseme, Dict[str, float]] = {
    Viseme.SILENCE: {'open': 0.0, 'width': 0.5, 'pucker': 0.0, 'stretch': 0.0},
    Viseme.AH: {'open': 0.8, 'width': 0.6, 'pucker': 0.0, 'stretch': 0.0},
    Viseme.EE: {'open': 0.3, 'width': 0.8, 'pucker': 0.0, 'stretch': 0.7},
    Viseme.EH: {'open': 0.5, 'width': 0.6, 'pucker': 0.0, 'stretch': 0.2},
    Viseme.OH: {'open': 0.6, 'width': 0.4, 'pucker': 0.5, 'stretch': 0.0},
    Viseme.OO: {'open': 0.4, 'width': 0.3, 'pucker': 0.8, 'stretch': 0.0},
    Viseme.BMP: {'open': 0.0, 'width': 0.5, 'pucker': 0.2, 'stretch': 0.0},
    Viseme.FV: {'open': 0.1, 'width': 0.6, 'pucker': 0.0, 'stretch': 0.1},
    Viseme.TH: {'open': 0.2, 'width': 0.5, 'pucker': 0.0, 'stretch': 0.1},
    Viseme.LNT: {'open': 0.3, 'width': 0.55, 'pucker': 0.0, 'stretch': 0.2},
    Viseme.KG: {'open': 0.4, 'width': 0.5, 'pucker': 0.0, 'stretch': 0.1},
    Viseme.SZ: {'open': 0.2, 'width': 0.65, 'pucker': 0.0, 'stretch': 0.4},
    Viseme.SH_CH: {'open': 0.25, 'width': 0.5, 'pucker': 0.3, 'stretch': 0.0},
    Viseme.R: {'open': 0.35, 'width': 0.45, 'pucker': 0.2, 'stretch': 0.0},
    Viseme.W: {'open': 0.3, 'width': 0.35, 'pucker': 0.6, 'stretch': 0.0},
}

# Character to viseme mapping
CHAR_TO_VISEME: Dict[str, Viseme] = {
    'a': Viseme.AH, 'e': Viseme.EE, 'i': Viseme.EE, 'o': Viseme.OH, 'u': Viseme.OO,
    'b': Viseme.BMP, 'm': Viseme.BMP, 'p': Viseme.BMP,
    'f': Viseme.FV, 'v': Viseme.FV,
    'l': Viseme.LNT, 'n': Viseme.LNT, 't': Viseme.LNT, 'd': Viseme.LNT,
    'k': Viseme.KG, 'g': Viseme.KG,
    's': Viseme.SZ, 'z': Viseme.SZ,
    'r': Viseme.R, 'w': Viseme.W,
    ' ': Viseme.SILENCE, '.': Viseme.SILENCE, ',': Viseme.SILENCE,
}


@dataclass
class MouthState:
    """State of the mouth."""
    open_amount: float = 0.0  # 0=closed, 1=fully open
    width: float = 0.5        # 0=narrow, 0.5=normal, 1=wide
    pucker: float = 0.0       # 0-1 for O sounds
    stretch: float = 0.0      # 0-1 for EE sounds
    left_corner: float = 0.0  # -1 to 1 (frown to smile)
    right_corner: float = 0.0
    current_viseme: Viseme = Viseme.SILENCE


@dataclass
class MouthRendererState(RendererState):
    """Extended state for mouth renderer."""
    mouth: MouthState = field(default_factory=MouthState)
    is_speaking: bool = False
    speak_text: str = ""
    speak_index: int = 0
    speak_start_time: float = 0.0


class MouthRenderer(BaseRenderer):
    """
    Renderer for animated mouth with lip-sync and expressions.
    """

    def __init__(self, width: int, height: int,
                 background_color: Tuple[int, int, int] = (20, 20, 25),
                 **kwargs):
        super().__init__(width, height, background_color)

        # Mouth state
        self.mouth_state = MouthRendererState()

        # Target shape for interpolation
        self._target_open = 0.0
        self._target_width = 0.5
        self._target_pucker = 0.0
        self._target_stretch = 0.0

        # Colors
        self.lip_color = kwargs.get('lip_color', (180, 100, 100))
        self.interior_color = kwargs.get('interior_color', (60, 30, 40))
        self.teeth_color = kwargs.get('teeth_color', (240, 240, 235))
        self.tongue_color = (160, 80, 90)

        # Animation settings
        self.transition_speed = kwargs.get('transition_speed', 12.0)
        self.idle_movement = kwargs.get('idle_movement', True)

        # Speaking settings
        self.chars_per_second = 15.0
        self._generic_speak_phase = 0.0

        # Geometry
        self._calculate_geometry()

        logger.debug("MouthRenderer initialized")

    def _calculate_geometry(self) -> None:
        """Calculate mouth geometry based on dimensions."""
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.mouth_width = int(self.width * 0.5)
        self.mouth_height = int(self.height * 0.4)
        self.lip_thickness = int(self.height * 0.06)

    def update(self, dt: float) -> None:
        """Update mouth animations."""
        # Update speaking animation
        if self.mouth_state.is_speaking:
            self._update_speaking(dt)

        # Interpolate to target shape
        self._update_shape_interpolation(dt)

        # Update emotion-based parameters
        self._update_emotion_params(dt)

        # Idle movements
        if not self.mouth_state.is_speaking and self.idle_movement:
            self._update_idle_movement(dt)

    def _update_speaking(self, dt: float) -> None:
        """Update speaking animation."""
        if self.mouth_state.speak_text:
            # Text-based speaking
            elapsed = time.time() - self.mouth_state.speak_start_time
            char_index = int(elapsed * self.chars_per_second)

            if char_index < len(self.mouth_state.speak_text):
                char = self.mouth_state.speak_text[char_index].lower()
                viseme = CHAR_TO_VISEME.get(char, Viseme.EH)
                self._set_viseme_target(viseme)
                self.mouth_state.mouth.current_viseme = viseme
            else:
                # Done speaking
                self.stop_speaking()
        else:
            # Generic speaking animation
            self._generic_speak_phase += dt * 8.0
            phase = math.sin(self._generic_speak_phase) * 0.5 + 0.5

            self._target_open = 0.2 + phase * 0.4
            self._target_width = 0.5 + math.sin(self._generic_speak_phase * 0.7) * 0.1

    def _update_shape_interpolation(self, dt: float) -> None:
        """Interpolate mouth shape towards target."""
        speed = self.transition_speed * dt
        mouth = self.mouth_state.mouth

        mouth.open_amount = self.lerp(mouth.open_amount, self._target_open, speed)
        mouth.width = self.lerp(mouth.width, self._target_width, speed)
        mouth.pucker = self.lerp(mouth.pucker, self._target_pucker, speed)
        mouth.stretch = self.lerp(mouth.stretch, self._target_stretch, speed)

    def _update_emotion_params(self, dt: float) -> None:
        """Update mouth corners based on emotion."""
        preset = self.get_emotion_preset(self.mouth_state.emotion_state)
        mouth_params = preset.get('mouth', {})

        target_left = mouth_params.get('left_corner', 0.0)
        target_right = mouth_params.get('right_corner', 0.0)
        target_width = mouth_params.get('width', 0.5)

        speed = self.animation_speed
        mouth = self.mouth_state.mouth

        mouth.left_corner = self.lerp(mouth.left_corner, target_left, speed)
        mouth.right_corner = self.lerp(mouth.right_corner, target_right, speed)

        if not self.mouth_state.is_speaking:
            mouth.width = self.lerp(mouth.width, target_width, speed)

    def _update_idle_movement(self, dt: float) -> None:
        """Add subtle idle movements."""
        # Occasional subtle movements
        if random.random() < 0.005:
            self._target_width = 0.5 + random.uniform(-0.05, 0.05)

    def _set_viseme_target(self, viseme: Viseme) -> None:
        """Set target shape based on viseme."""
        shape = VISEME_SHAPES.get(viseme, VISEME_SHAPES[Viseme.SILENCE])
        self._target_open = shape['open']
        self._target_width = shape['width']
        self._target_pucker = shape['pucker']
        self._target_stretch = shape['stretch']

    def render(self) -> None:
        """Render the mouth."""
        if not self.surface or not PYGAME_AVAILABLE:
            return

        # Clear background
        self.surface.fill(self.background_color)

        mouth = self.mouth_state.mouth

        # Calculate effective dimensions
        effective_width = int(self.mouth_width * mouth.width)
        effective_height = int(self.mouth_height * (0.1 + mouth.open_amount * 0.9))

        # Apply pucker/stretch
        if mouth.pucker > 0:
            effective_width = int(effective_width * (1.0 - mouth.pucker * 0.3))
        if mouth.stretch > 0:
            effective_width = int(effective_width * (1.0 + mouth.stretch * 0.2))

        # Calculate corner positions
        left_corner_y = self.center_y + int(mouth.left_corner * self.mouth_height * 0.3)
        right_corner_y = self.center_y + int(mouth.right_corner * self.mouth_height * 0.3)

        # Draw mouth interior if open
        if mouth.open_amount > 0.05:
            interior_rect = pygame.Rect(
                self.center_x - effective_width // 2,
                self.center_y - effective_height // 2,
                effective_width,
                effective_height
            )
            pygame.draw.ellipse(self.surface, self.interior_color, interior_rect)

            # Draw teeth if mouth is open enough
            if mouth.open_amount > 0.2:
                teeth_height = int(effective_height * 0.25)
                teeth_rect = pygame.Rect(
                    self.center_x - effective_width // 3,
                    self.center_y - effective_height // 2,
                    effective_width * 2 // 3,
                    teeth_height
                )
                pygame.draw.rect(self.surface, self.teeth_color, teeth_rect)

            # Draw tongue if mouth is very open
            if mouth.open_amount > 0.5:
                tongue_width = effective_width // 2
                tongue_height = int(effective_height * 0.3)
                tongue_y = self.center_y + effective_height // 4
                pygame.draw.ellipse(
                    self.surface,
                    self.tongue_color,
                    pygame.Rect(
                        self.center_x - tongue_width // 2,
                        tongue_y - tongue_height // 2,
                        tongue_width,
                        tongue_height
                    )
                )

        # Draw lips
        self._draw_lips(effective_width, effective_height, left_corner_y, right_corner_y)

    def _draw_lips(self, width: int, height: int,
                   left_corner_y: int, right_corner_y: int) -> None:
        """Draw the lips."""
        mouth = self.mouth_state.mouth

        # Calculate lip points
        left_x = self.center_x - width // 2
        right_x = self.center_x + width // 2

        # Upper lip
        upper_points = [
            (left_x, left_corner_y),
            (self.center_x - width // 4, self.center_y - height // 2 - self.lip_thickness),
            (self.center_x, self.center_y - height // 2 - self.lip_thickness - 5),
            (self.center_x + width // 4, self.center_y - height // 2 - self.lip_thickness),
            (right_x, right_corner_y),
            (self.center_x + width // 4, self.center_y - height // 2 + self.lip_thickness),
            (self.center_x, self.center_y - height // 2 + self.lip_thickness),
            (self.center_x - width // 4, self.center_y - height // 2 + self.lip_thickness),
        ]

        # Lower lip
        lower_points = [
            (left_x, left_corner_y),
            (self.center_x - width // 4, self.center_y + height // 2 - self.lip_thickness),
            (self.center_x, self.center_y + height // 2 - self.lip_thickness),
            (self.center_x + width // 4, self.center_y + height // 2 - self.lip_thickness),
            (right_x, right_corner_y),
            (self.center_x + width // 4, self.center_y + height // 2 + self.lip_thickness),
            (self.center_x, self.center_y + height // 2 + self.lip_thickness + 5),
            (self.center_x - width // 4, self.center_y + height // 2 + self.lip_thickness),
        ]

        if len(upper_points) >= 3:
            pygame.draw.polygon(self.surface, self.lip_color, upper_points)
        if len(lower_points) >= 3:
            pygame.draw.polygon(self.surface, self.lip_color, lower_points)

    def handle_command(self, command: Dict[str, Any]) -> None:
        """Handle commands from the decision module."""
        event = command.get('event', '')
        data = command.get('data', {})

        if event == 'speak_start':
            self.start_speaking()
        elif event == 'speak_stop':
            self.stop_speaking()
        elif event == 'speak_text':
            self.speak_text(data.get('text', ''))
        elif event == 'viseme_update':
            viseme_name = data.get('viseme', 'SILENCE')
            try:
                viseme = Viseme[viseme_name.upper()]
                self.set_viseme(viseme)
            except KeyError:
                logger.warning(f"Unknown viseme: {viseme_name}")
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

    def speak_text(self, text: str) -> None:
        """Start speaking with text-based lip sync."""
        self.mouth_state.is_speaking = True
        self.mouth_state.speak_text = text
        self.mouth_state.speak_index = 0
        self.mouth_state.speak_start_time = time.time()

    def start_speaking(self) -> None:
        """Start generic speaking animation."""
        self.mouth_state.is_speaking = True
        self.mouth_state.speak_text = ""
        self._generic_speak_phase = 0.0

    def stop_speaking(self) -> None:
        """Stop speaking animation."""
        self.mouth_state.is_speaking = False
        self.mouth_state.speak_text = ""
        self._target_open = 0.0
        self._target_pucker = 0.0
        self._target_stretch = 0.0
        self.mouth_state.mouth.current_viseme = Viseme.SILENCE

    def set_viseme(self, viseme: Viseme) -> None:
        """Set mouth shape to a specific viseme."""
        self._set_viseme_target(viseme)
        self.mouth_state.mouth.current_viseme = viseme

    def set_emotion(self, emotion: EmotionState) -> None:
        """Set emotion state."""
        self.mouth_state.emotion_state = emotion

    def set_animation_state(self, state: AnimationState) -> None:
        """Set animation state."""
        self.mouth_state.animation_state = state

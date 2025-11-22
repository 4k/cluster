"""
Mouth Renderer for the multi-window display system.
Handles mouth rendering with lip-sync, visemes, and emotional expressions.

Enhanced with Rhubarb lip sync integration for precise mouth animation:
- Support for Rhubarb's Preston Blair mouth shapes (A-F basic, G-H-X extended)
- Smooth viseme interpolation with easing functions
- Coarticulation support for natural transitions
- Detailed mouth parameters (teeth, tongue visibility)

Based on best practices from:
- Rhubarb Lip Sync documentation
- Traditional 2D animation (Hanna-Barbera, Disney)
- Industry-standard viseme interpolation techniques
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
    SILENCE = auto()  # Closed mouth (Rhubarb X)
    AH = auto()       # Open vowel - father (Rhubarb C/D)
    EE = auto()       # Smile vowel - see (Rhubarb B)
    EH = auto()       # Mid vowel - bed (Rhubarb C)
    OH = auto()       # Round vowel - go (Rhubarb E)
    OO = auto()       # Pucker vowel - too (Rhubarb F)
    BMP = auto()      # Bilabial - b, m, p (Rhubarb A)
    FV = auto()       # Labiodental - f, v (Rhubarb G)
    TH = auto()       # Dental - th
    LNT = auto()      # Alveolar - l, n, t, d (Rhubarb B/H)
    KG = auto()       # Velar - k, g
    SZ = auto()       # Sibilant - s, z
    SH_CH = auto()    # Postalveolar - sh, ch
    R = auto()        # R sound
    W = auto()        # W sound


class EasingType(Enum):
    """Easing types for viseme interpolation."""
    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()  # Smooth step - best for lip sync


def apply_easing(t: float, easing: EasingType) -> float:
    """Apply easing function to interpolation value."""
    t = max(0.0, min(1.0, t))

    if easing == EasingType.LINEAR:
        return t
    elif easing == EasingType.EASE_IN:
        return t * t
    elif easing == EasingType.EASE_OUT:
        return 1.0 - (1.0 - t) * (1.0 - t)
    elif easing == EasingType.EASE_IN_OUT:
        # Smooth step (hermite interpolation) - industry standard for animation
        return t * t * (3.0 - 2.0 * t)
    return t


# Viseme shape definitions - enhanced for Rhubarb compatibility
# Each shape includes additional parameters for detailed rendering
VISEME_SHAPES: Dict[Viseme, Dict[str, float]] = {
    Viseme.SILENCE: {
        'open': 0.0, 'width': 0.5, 'pucker': 0.0, 'stretch': 0.0,
        'teeth_visible': False, 'tongue_visible': False
    },
    Viseme.AH: {
        'open': 0.8, 'width': 0.6, 'pucker': 0.0, 'stretch': 0.0,
        'teeth_visible': True, 'tongue_visible': True
    },
    Viseme.EE: {
        'open': 0.3, 'width': 0.8, 'pucker': 0.0, 'stretch': 0.7,
        'teeth_visible': True, 'tongue_visible': False
    },
    Viseme.EH: {
        'open': 0.5, 'width': 0.6, 'pucker': 0.0, 'stretch': 0.2,
        'teeth_visible': True, 'tongue_visible': False
    },
    Viseme.OH: {
        'open': 0.6, 'width': 0.4, 'pucker': 0.5, 'stretch': 0.0,
        'teeth_visible': True, 'tongue_visible': False
    },
    Viseme.OO: {
        'open': 0.4, 'width': 0.3, 'pucker': 0.8, 'stretch': 0.0,
        'teeth_visible': False, 'tongue_visible': False
    },
    Viseme.BMP: {
        'open': 0.0, 'width': 0.5, 'pucker': 0.1, 'stretch': 0.0,
        'teeth_visible': False, 'tongue_visible': False
    },
    Viseme.FV: {
        'open': 0.15, 'width': 0.55, 'pucker': 0.0, 'stretch': 0.2,
        'teeth_visible': True, 'tongue_visible': False, 'lower_lip_tucked': True
    },
    Viseme.TH: {
        'open': 0.2, 'width': 0.5, 'pucker': 0.0, 'stretch': 0.1,
        'teeth_visible': True, 'tongue_visible': True, 'tongue_tip_out': True
    },
    Viseme.LNT: {
        'open': 0.3, 'width': 0.55, 'pucker': 0.0, 'stretch': 0.2,
        'teeth_visible': True, 'tongue_visible': True, 'tongue_raised': True
    },
    Viseme.KG: {
        'open': 0.4, 'width': 0.5, 'pucker': 0.0, 'stretch': 0.1,
        'teeth_visible': True, 'tongue_visible': False
    },
    Viseme.SZ: {
        'open': 0.2, 'width': 0.65, 'pucker': 0.0, 'stretch': 0.4,
        'teeth_visible': True, 'tongue_visible': False
    },
    Viseme.SH_CH: {
        'open': 0.25, 'width': 0.5, 'pucker': 0.3, 'stretch': 0.0,
        'teeth_visible': True, 'tongue_visible': False
    },
    Viseme.R: {
        'open': 0.35, 'width': 0.45, 'pucker': 0.2, 'stretch': 0.0,
        'teeth_visible': True, 'tongue_visible': False
    },
    Viseme.W: {
        'open': 0.3, 'width': 0.35, 'pucker': 0.6, 'stretch': 0.0,
        'teeth_visible': False, 'tongue_visible': False
    },
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

# Rhubarb shape to Viseme mapping for direct Rhubarb integration
RHUBARB_SHAPE_TO_VISEME: Dict[str, Viseme] = {
    'A': Viseme.BMP,      # Closed mouth (M, B, P)
    'B': Viseme.LNT,      # Slightly open (consonants, EE)
    'C': Viseme.EH,       # Open mouth (EH, AE vowels)
    'D': Viseme.AH,       # Wide open (AA vowel)
    'E': Viseme.OH,       # Rounded (AO, ER vowels)
    'F': Viseme.OO,       # Puckered (OO, UW, W)
    'G': Viseme.FV,       # F/V shape (upper teeth on lower lip)
    'H': Viseme.LNT,      # L shape (tongue visible)
    'X': Viseme.SILENCE,  # Idle/rest position
}


@dataclass
class MouthState:
    """State of the mouth with Rhubarb-compatible parameters."""
    # Core shape parameters
    open_amount: float = 0.0  # 0=closed, 1=fully open
    width: float = 0.5        # 0=narrow, 0.5=normal, 1=wide
    pucker: float = 0.0       # 0-1 for O sounds (Rhubarb F shape)
    stretch: float = 0.0      # 0-1 for EE sounds (Rhubarb B shape)

    # Emotion-driven parameters
    left_corner: float = 0.0  # -1 to 1 (frown to smile)
    right_corner: float = 0.0

    # Detailed parameters for Rhubarb integration
    teeth_visible: bool = False
    tongue_visible: bool = False
    lower_lip_tucked: bool = False  # For F/V sounds (Rhubarb G)
    tongue_raised: bool = False     # For L sounds (Rhubarb H)

    # Current viseme tracking
    current_viseme: Viseme = Viseme.SILENCE
    previous_viseme: Viseme = Viseme.SILENCE

    # Rhubarb-specific tracking
    rhubarb_shape: Optional[str] = None  # Current Rhubarb shape letter
    rhubarb_intensity: float = 1.0       # Viseme intensity (0-1)


@dataclass
class MouthRendererState(RendererState):
    """Extended state for mouth renderer with Rhubarb support."""
    mouth: MouthState = field(default_factory=MouthState)
    is_speaking: bool = False
    speak_text: str = ""
    speak_index: int = 0
    speak_start_time: float = 0.0
    speak_duration: float = 0.0  # Audio duration for timing sync

    # Rhubarb lip sync state
    rhubarb_active: bool = False  # True when Rhubarb is controlling mouth
    rhubarb_session_id: Optional[str] = None


class MouthRenderer(BaseRenderer):
    """
    Renderer for animated mouth with lip-sync and expressions.

    Enhanced with Rhubarb integration for precise lip-sync animation:
    - Smooth viseme interpolation with configurable easing
    - Support for Rhubarb's Preston Blair mouth shapes
    - Coarticulation for natural transitions between visemes
    - Detailed mouth rendering (teeth, tongue, lip positions)
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
        self._target_teeth_visible = False
        self._target_tongue_visible = False

        # Colors
        self.lip_color = kwargs.get('lip_color', (180, 100, 100))
        self.interior_color = kwargs.get('interior_color', (60, 30, 40))
        self.teeth_color = kwargs.get('teeth_color', (240, 240, 235))
        self.tongue_color = (160, 80, 90)

        # Animation settings
        self.transition_speed = kwargs.get('transition_speed', 12.0)
        self.idle_movement = kwargs.get('idle_movement', True)

        # Rhubarb integration settings
        self.rhubarb_transition_speed = kwargs.get('rhubarb_transition_speed', 18.0)
        self.viseme_easing = kwargs.get('viseme_easing', EasingType.EASE_IN_OUT)
        self.enable_coarticulation = kwargs.get('enable_coarticulation', True)

        # Speaking settings
        self.chars_per_second = 15.0
        self._generic_speak_phase = 0.0

        # Viseme transition tracking
        self._viseme_transition_progress = 1.0
        self._viseme_transition_start = 0.0
        self._previous_viseme_params: Dict[str, float] = {}
        self._target_viseme_params: Dict[str, float] = {}

        # Geometry
        self._calculate_geometry()

        # Statistics for debugging
        self._stats = {
            'visemes_processed': 0,
            'rhubarb_updates': 0,
            'text_updates': 0
        }

        logger.debug("MouthRenderer initialized with Rhubarb support")

    def _calculate_geometry(self) -> None:
        """Calculate mouth geometry based on dimensions."""
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.mouth_width = int(self.width * 0.5)
        self.mouth_height = int(self.height * 0.4)
        self.lip_thickness = int(self.height * 0.06)

    def update(self, dt: float) -> None:
        """Update mouth animations with Rhubarb integration."""
        # Handle different animation modes
        if self.mouth_state.rhubarb_active:
            # Rhubarb is controlling mouth - use viseme transitions
            self._interpolate_viseme_transition(dt)
        elif self.mouth_state.is_speaking:
            # Text-based or generic speaking animation
            self._update_speaking(dt)
            self._update_shape_interpolation(dt)
        else:
            # Standard interpolation for non-speaking states
            self._update_shape_interpolation(dt)

        # Update emotion-based parameters (always applies)
        self._update_emotion_params(dt)

        # Idle movements (only when not speaking/Rhubarb active)
        if not self.mouth_state.is_speaking and not self.mouth_state.rhubarb_active and self.idle_movement:
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

    def _set_viseme_target(self, viseme: Viseme, intensity: float = 1.0) -> None:
        """
        Set target shape based on viseme with Rhubarb-compatible parameters.

        Args:
            viseme: The target viseme
            intensity: Viseme intensity (0-1), affects mouth openness
        """
        shape = VISEME_SHAPES.get(viseme, VISEME_SHAPES[Viseme.SILENCE])

        # Store previous state for interpolation
        self._previous_viseme_params = {
            'open': self.mouth_state.mouth.open_amount,
            'width': self.mouth_state.mouth.width,
            'pucker': self.mouth_state.mouth.pucker,
            'stretch': self.mouth_state.mouth.stretch,
        }

        # Set targets with intensity scaling
        self._target_open = shape['open'] * intensity
        self._target_width = shape['width']
        self._target_pucker = shape['pucker']
        self._target_stretch = shape['stretch']

        # Extended Rhubarb parameters
        self._target_teeth_visible = shape.get('teeth_visible', False)
        self._target_tongue_visible = shape.get('tongue_visible', False)

        # Store target params for interpolation
        self._target_viseme_params = {
            'open': self._target_open,
            'width': self._target_width,
            'pucker': self._target_pucker,
            'stretch': self._target_stretch,
        }

        # Track previous viseme for coarticulation
        self.mouth_state.mouth.previous_viseme = self.mouth_state.mouth.current_viseme

        # Start transition
        self._viseme_transition_progress = 0.0
        self._viseme_transition_start = time.time()

        self._stats['visemes_processed'] += 1

    def set_rhubarb_shape(self, shape_letter: str, intensity: float = 1.0,
                          params: Optional[Dict[str, float]] = None) -> None:
        """
        Set mouth shape directly from Rhubarb output.

        This method provides direct integration with Rhubarb Lip Sync,
        accepting shape letters (A-F, G-H-X) and optional detailed parameters.

        Args:
            shape_letter: Rhubarb shape letter (A, B, C, D, E, F, G, H, X)
            intensity: Overall intensity of the viseme (0-1)
            params: Optional detailed mouth parameters from RhubarbVisemeController
        """
        # Convert Rhubarb shape to internal viseme
        viseme = RHUBARB_SHAPE_TO_VISEME.get(shape_letter.upper(), Viseme.SILENCE)

        # Update Rhubarb tracking state
        self.mouth_state.mouth.rhubarb_shape = shape_letter.upper()
        self.mouth_state.mouth.rhubarb_intensity = intensity
        self.mouth_state.rhubarb_active = True

        # If detailed params provided (from RhubarbVisemeController), use them directly
        if params:
            self._apply_rhubarb_params(params, intensity)
        else:
            # Use standard viseme target
            self._set_viseme_target(viseme, intensity)

        self._stats['rhubarb_updates'] += 1

    def _apply_rhubarb_params(self, params: Dict[str, float], intensity: float) -> None:
        """
        Apply detailed Rhubarb parameters directly to mouth.

        This allows the RhubarbVisemeController to provide blended/interpolated
        parameters for smoother animation with coarticulation.

        Args:
            params: Dictionary of mouth parameters
            intensity: Overall intensity multiplier
        """
        # Store previous state
        self._previous_viseme_params = {
            'open': self.mouth_state.mouth.open_amount,
            'width': self.mouth_state.mouth.width,
            'pucker': self.mouth_state.mouth.pucker,
            'stretch': self.mouth_state.mouth.stretch,
        }

        # Set targets from Rhubarb params
        self._target_open = params.get('open', 0.0) * intensity
        self._target_width = params.get('width', 0.5)
        self._target_pucker = params.get('pucker', 0.0)
        self._target_stretch = params.get('stretch', 0.0)
        self._target_teeth_visible = params.get('teeth_visible', False)
        self._target_tongue_visible = params.get('tongue_visible', False)

        # Store target params
        self._target_viseme_params = {
            'open': self._target_open,
            'width': self._target_width,
            'pucker': self._target_pucker,
            'stretch': self._target_stretch,
        }

        # Start transition with faster Rhubarb speed
        self._viseme_transition_progress = 0.0
        self._viseme_transition_start = time.time()

    def start_rhubarb_session(self, session_id: str) -> None:
        """
        Start a Rhubarb lip sync session.

        Marks the renderer as being controlled by Rhubarb, disabling
        text-based and generic speaking animations.

        Args:
            session_id: Unique session identifier
        """
        self.mouth_state.rhubarb_active = True
        self.mouth_state.rhubarb_session_id = session_id

        # Stop any text-based animation
        self.mouth_state.speak_text = ""
        self.mouth_state.is_speaking = True  # Keep speaking state for rendering

        logger.debug(f"Started Rhubarb session: {session_id}")

    def stop_rhubarb_session(self) -> None:
        """Stop the current Rhubarb lip sync session."""
        self.mouth_state.rhubarb_active = False
        self.mouth_state.rhubarb_session_id = None
        self.mouth_state.mouth.rhubarb_shape = None

        # Transition to rest position
        self._set_viseme_target(Viseme.SILENCE)

        logger.debug("Stopped Rhubarb session")

    def _interpolate_viseme_transition(self, dt: float) -> None:
        """
        Interpolate current mouth shape during viseme transition.

        Uses configurable easing for smooth, natural-looking animation.
        """
        if self._viseme_transition_progress >= 1.0:
            return

        # Calculate transition speed (faster for Rhubarb)
        speed = self.rhubarb_transition_speed if self.mouth_state.rhubarb_active else self.transition_speed
        progress_delta = dt * speed

        self._viseme_transition_progress = min(1.0, self._viseme_transition_progress + progress_delta)

        # Apply easing
        eased_t = apply_easing(self._viseme_transition_progress, self.viseme_easing)

        # Interpolate each parameter
        mouth = self.mouth_state.mouth

        if self._previous_viseme_params and self._target_viseme_params:
            mouth.open_amount = self._lerp_param('open', eased_t)
            mouth.width = self._lerp_param('width', eased_t)
            mouth.pucker = self._lerp_param('pucker', eased_t)
            mouth.stretch = self._lerp_param('stretch', eased_t)
        else:
            # Fallback to standard interpolation
            mouth.open_amount = self.lerp(mouth.open_amount, self._target_open, eased_t)
            mouth.width = self.lerp(mouth.width, self._target_width, eased_t)
            mouth.pucker = self.lerp(mouth.pucker, self._target_pucker, eased_t)
            mouth.stretch = self.lerp(mouth.stretch, self._target_stretch, eased_t)

        # Update visibility states (instant for these)
        mouth.teeth_visible = self._target_teeth_visible
        mouth.tongue_visible = self._target_tongue_visible

    def _lerp_param(self, param: str, t: float) -> float:
        """Interpolate a specific parameter using stored values."""
        prev = self._previous_viseme_params.get(param, 0.0)
        target = self._target_viseme_params.get(param, 0.0)
        return prev + (target - prev) * t

    def get_stats(self) -> Dict[str, Any]:
        """Get renderer statistics for debugging."""
        return {
            **self._stats,
            'rhubarb_active': self.mouth_state.rhubarb_active,
            'current_shape': self.mouth_state.mouth.rhubarb_shape,
            'current_viseme': self.mouth_state.mouth.current_viseme.name,
            'transition_progress': self._viseme_transition_progress
        }

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
        """
        Handle commands from the decision module.

        Supports both standard viseme commands and Rhubarb-specific events.
        """
        event = command.get('event', '')
        data = command.get('data', {})

        if event == 'speak_start':
            self.start_speaking()
        elif event == 'speak_stop':
            self.stop_speaking()
        elif event == 'speak_text':
            self.speak_text(data.get('text', ''), duration=data.get('duration'))
        elif event == 'viseme_update':
            viseme_name = data.get('viseme', 'SILENCE')
            intensity = data.get('intensity', 1.0)
            params = data.get('params')  # Optional detailed params from Rhubarb

            # Check if this is a Rhubarb shape letter
            if viseme_name.upper() in RHUBARB_SHAPE_TO_VISEME:
                self.set_rhubarb_shape(viseme_name, intensity, params)
            else:
                try:
                    viseme = Viseme[viseme_name.upper()]
                    self.set_viseme(viseme, intensity)
                except KeyError:
                    logger.warning(f"Unknown viseme: {viseme_name}")

        # Rhubarb-specific commands
        elif event == 'rhubarb_shape':
            shape_letter = data.get('shape', 'X')
            intensity = data.get('intensity', 1.0)
            params = data.get('params')
            self.set_rhubarb_shape(shape_letter, intensity, params)

        elif event == 'rhubarb_session_start':
            session_id = data.get('session_id', 'unknown')
            self.start_rhubarb_session(session_id)

        elif event == 'rhubarb_session_stop':
            self.stop_rhubarb_session()

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

                # Handle SPEAKING state from animation service
                if state == AnimationState.SPEAKING:
                    self.mouth_state.is_speaking = True
                elif state == AnimationState.IDLE:
                    if not self.mouth_state.rhubarb_active:
                        self.mouth_state.is_speaking = False
            except KeyError:
                logger.warning(f"Unknown animation state: {state_name}")

    def speak_text(self, text: str, duration: float = None) -> None:
        """
        Start speaking with text-based lip sync.

        Args:
            text: Text to animate
            duration: Audio duration in seconds (for timing sync).
                     If provided, animation speed is adjusted to match.
        """
        self.mouth_state.is_speaking = True
        self.mouth_state.speak_text = text
        self.mouth_state.speak_index = 0
        self.mouth_state.speak_start_time = time.time()
        self.mouth_state.speak_duration = duration or 0.0

        # Calculate chars_per_second based on audio duration for precise sync
        if duration and duration > 0 and len(text) > 0:
            self.chars_per_second = len(text) / duration
            logger.debug(f"Text animation: {len(text)} chars / {duration:.2f}s = {self.chars_per_second:.1f} chars/sec")
        else:
            self.chars_per_second = 15.0  # Default fallback

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

    def set_viseme(self, viseme: Viseme, intensity: float = 1.0) -> None:
        """
        Set mouth shape to a specific viseme.

        Args:
            viseme: The viseme to display
            intensity: Viseme intensity (0-1)
        """
        self._set_viseme_target(viseme, intensity)
        self.mouth_state.mouth.current_viseme = viseme
        self.mouth_state.mouth.rhubarb_intensity = intensity

    def set_emotion(self, emotion: EmotionState) -> None:
        """Set emotion state."""
        self.mouth_state.emotion_state = emotion

    def set_animation_state(self, state: AnimationState) -> None:
        """Set animation state."""
        self.mouth_state.animation_state = state

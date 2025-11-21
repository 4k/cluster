"""
Mouth renderer for drawing animated mouth.
Supports lip-sync visemes, expressions, and emotion-based shapes.
"""

import math
import logging
from typing import Tuple, List

import pygame

from core.types import (
    ContentType, MouthState, MouthShape, EmotionType, AnimationState
)
from .base import BaseRenderer

logger = logging.getLogger(__name__)


class MouthRenderer(BaseRenderer):
    """Renderer for animated mouth.

    Draws a mouth that can change shape for lip-sync,
    expressions, and emotions.
    """

    def __init__(self, size: Tuple[int, int]):
        """Initialize mouth renderer.

        Args:
            size: The (width, height) of the rendering area
        """
        super().__init__(ContentType.MOUTH, size)

        # Mouth geometry (will be calculated based on size)
        self._calculate_geometry()

        # Animation state for smooth transitions
        self._current_openness = 0.0
        self._current_smile = 0.0
        self._smoothing = 12.0  # Higher = faster transitions

        # Speaking animation
        self._speak_phase = 0.0
        self._speak_frequency = 8.0  # Hz

    def _calculate_geometry(self) -> None:
        """Calculate mouth geometry based on current size."""
        # Mouth center
        self.center = (self.width / 2, self.height / 2)

        # Mouth dimensions
        self.mouth_width = min(self.width * 0.6, self.height * 1.5)
        self.mouth_height = self.mouth_width * 0.4

        # Lip thickness
        self.lip_thickness = self.mouth_height * 0.15

    def resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize."""
        super().resize(new_size)
        self._calculate_geometry()

    def render(self, surface: pygame.Surface, state: MouthState, dt: float) -> None:
        """Render mouth to the surface.

        Args:
            surface: The pygame surface to draw on
            state: The current mouth state
            dt: Delta time since last frame in seconds
        """
        # Clear surface with background
        surface.fill((0, 0, 0))

        # Update smooth transitions
        self._update_animations(state, dt)

        # Get colors based on emotion
        lip_color = self._get_lip_color(state.emotion)
        mouth_interior_color = (40, 20, 30)
        teeth_color = (250, 250, 255)
        tongue_color = (180, 100, 100)

        # Determine mouth shape parameters
        openness = self._current_openness
        smile = self._current_smile

        # If speaking, add variation
        if state.is_speaking:
            self._speak_phase += dt * self._speak_frequency * 2 * math.pi
            speak_variation = math.sin(self._speak_phase) * 0.3 + 0.5
            openness = max(openness, speak_variation * 0.6)

        # Apply viseme if present
        if state.viseme:
            viseme_params = self._get_viseme_params(state.viseme)
            openness = viseme_params['openness']
            smile = viseme_params.get('smile', smile)

        # Draw the mouth
        self._draw_mouth(
            surface,
            openness,
            smile,
            lip_color,
            mouth_interior_color,
            teeth_color,
            tongue_color,
            state.emotion
        )

    def _update_animations(self, state: MouthState, dt: float) -> None:
        """Update smooth animation transitions."""
        lerp_factor = min(1.0, self._smoothing * dt)

        # Smooth openness
        target_openness = state.openness
        self._current_openness += (target_openness - self._current_openness) * lerp_factor

        # Smooth smile
        target_smile = state.smile_amount
        self._current_smile += (target_smile - self._current_smile) * lerp_factor

    def _get_viseme_params(self, viseme: MouthShape) -> dict:
        """Get mouth parameters for a viseme.

        Args:
            viseme: The viseme shape

        Returns:
            Dictionary with 'openness' and optionally 'smile'
        """
        viseme_params = {
            MouthShape.CLOSED: {'openness': 0.0, 'smile': 0.0},
            MouthShape.SLIGHTLY_OPEN: {'openness': 0.2, 'smile': 0.0},
            MouthShape.OPEN: {'openness': 0.5, 'smile': 0.0},
            MouthShape.WIDE_OPEN: {'openness': 1.0, 'smile': 0.0},
            MouthShape.SMILE: {'openness': 0.1, 'smile': 0.8},
            MouthShape.FROWN: {'openness': 0.1, 'smile': -0.8},
            MouthShape.O_SHAPE: {'openness': 0.6, 'smile': -0.3},
            # Visemes for phonemes
            MouthShape.VISEME_AA: {'openness': 0.8, 'smile': 0.0},
            MouthShape.VISEME_EE: {'openness': 0.3, 'smile': 0.5},
            MouthShape.VISEME_OO: {'openness': 0.5, 'smile': -0.4},
            MouthShape.VISEME_OH: {'openness': 0.6, 'smile': -0.2},
            MouthShape.VISEME_TH: {'openness': 0.2, 'smile': 0.2},
            MouthShape.VISEME_FF: {'openness': 0.1, 'smile': 0.1},
            MouthShape.VISEME_MM: {'openness': 0.0, 'smile': 0.0},
        }
        return viseme_params.get(viseme, {'openness': 0.0, 'smile': 0.0})

    def _get_lip_color(self, emotion: EmotionType) -> Tuple[int, int, int]:
        """Get lip color based on emotion."""
        base_color = (180, 100, 110)
        emotion_colors = {
            EmotionType.HAPPY: (200, 120, 120),
            EmotionType.SAD: (150, 90, 100),
            EmotionType.ANGRY: (200, 80, 80),
            EmotionType.SURPRISED: (190, 110, 115),
            EmotionType.SPEAKING: (190, 115, 120),
        }
        return emotion_colors.get(emotion, base_color)

    def _draw_mouth(
        self,
        surface: pygame.Surface,
        openness: float,
        smile: float,
        lip_color: Tuple[int, int, int],
        interior_color: Tuple[int, int, int],
        teeth_color: Tuple[int, int, int],
        tongue_color: Tuple[int, int, int],
        emotion: EmotionType
    ) -> None:
        """Draw the mouth.

        Args:
            surface: Surface to draw on
            openness: 0.0 (closed) to 1.0 (wide open)
            smile: -1.0 (frown) to 1.0 (smile)
            lip_color: Color of the lips
            interior_color: Color of mouth interior
            teeth_color: Color of teeth
            tongue_color: Color of tongue
            emotion: Current emotion
        """
        cx, cy = int(self.center[0]), int(self.center[1])

        # Calculate mouth opening height
        open_height = self.mouth_height * openness

        # Calculate corner positions based on smile
        # Smile pulls corners up, frown pulls them down
        corner_offset_y = -smile * self.mouth_height * 0.3

        # Mouth corner positions
        left_corner = (cx - self.mouth_width / 2, cy + corner_offset_y)
        right_corner = (cx + self.mouth_width / 2, cy + corner_offset_y)

        # Control points for bezier curves
        # Top lip center is pulled up slightly for smile
        top_center_y = cy - open_height / 2 - smile * 5

        # Bottom lip center
        bottom_center_y = cy + open_height / 2 + abs(smile) * 3

        if openness < 0.05:
            # Closed mouth - just draw a curved line
            self._draw_closed_mouth(surface, lip_color, smile)
        else:
            # Open mouth
            self._draw_open_mouth(
                surface,
                openness,
                smile,
                lip_color,
                interior_color,
                teeth_color,
                tongue_color
            )

    def _draw_closed_mouth(
        self,
        surface: pygame.Surface,
        lip_color: Tuple[int, int, int],
        smile: float
    ) -> None:
        """Draw a closed mouth line.

        Args:
            surface: Surface to draw on
            lip_color: Color of the lips
            smile: -1.0 (frown) to 1.0 (smile)
        """
        cx, cy = self.center

        # Create points for a curved line
        num_points = 20
        points = []

        for i in range(num_points + 1):
            t = i / num_points
            x = cx - self.mouth_width / 2 + self.mouth_width * t

            # Parabolic curve for smile/frown
            # Highest/lowest at center
            curve_factor = 4 * t * (1 - t)  # 0 at ends, 1 at center
            y_offset = -smile * self.mouth_height * 0.2 * curve_factor

            points.append((int(x), int(cy + y_offset)))

        # Draw the mouth line
        if len(points) > 1:
            pygame.draw.lines(surface, lip_color, False, points, int(self.lip_thickness))

    def _draw_open_mouth(
        self,
        surface: pygame.Surface,
        openness: float,
        smile: float,
        lip_color: Tuple[int, int, int],
        interior_color: Tuple[int, int, int],
        teeth_color: Tuple[int, int, int],
        tongue_color: Tuple[int, int, int]
    ) -> None:
        """Draw an open mouth with interior, teeth, and lips.

        Args:
            surface: Surface to draw on
            openness: How open the mouth is
            smile: Smile amount
            lip_color: Lip color
            interior_color: Mouth interior color
            teeth_color: Teeth color
            tongue_color: Tongue color
        """
        cx, cy = self.center
        open_height = self.mouth_height * openness

        # Corner offset for smile
        corner_y_offset = -smile * self.mouth_height * 0.25

        # Generate mouth outline points
        top_lip_points = self._generate_lip_curve(
            cx, cy - open_height / 2,
            self.mouth_width,
            corner_y_offset,
            is_top=True,
            smile=smile
        )

        bottom_lip_points = self._generate_lip_curve(
            cx, cy + open_height / 2,
            self.mouth_width,
            corner_y_offset,
            is_top=False,
            smile=smile
        )

        # Combine into mouth shape
        mouth_outline = top_lip_points + list(reversed(bottom_lip_points))

        if len(mouth_outline) >= 3:
            # Draw mouth interior
            pygame.draw.polygon(surface, interior_color, mouth_outline)

            # Draw teeth if mouth is open enough
            if openness > 0.2:
                self._draw_teeth(surface, cx, cy, open_height, teeth_color, smile)

            # Draw tongue if mouth is wide open
            if openness > 0.5:
                self._draw_tongue(surface, cx, cy + open_height * 0.3, tongue_color, openness)

            # Draw lips (outline)
            pygame.draw.polygon(surface, lip_color, mouth_outline, int(self.lip_thickness))

    def _generate_lip_curve(
        self,
        cx: float,
        cy: float,
        width: float,
        corner_offset: float,
        is_top: bool,
        smile: float
    ) -> List[Tuple[int, int]]:
        """Generate points for a lip curve.

        Args:
            cx: Center x
            cy: Center y at lip
            width: Mouth width
            corner_offset: Y offset for corners
            is_top: Whether this is the top lip
            smile: Smile amount

        Returns:
            List of (x, y) points
        """
        points = []
        num_points = 15

        for i in range(num_points + 1):
            t = i / num_points
            x = cx - width / 2 + width * t

            # Base curve
            curve_factor = 4 * t * (1 - t)

            if is_top:
                # Top lip: cupid's bow shape
                cupid_bow = math.sin(t * math.pi * 2) * 5 if 0.3 < t < 0.7 else 0
                y = cy + corner_offset * (1 - curve_factor) - cupid_bow
            else:
                # Bottom lip: simple curve
                y = cy + corner_offset * (1 - curve_factor)

            points.append((int(x), int(y)))

        return points

    def _draw_teeth(
        self,
        surface: pygame.Surface,
        cx: float,
        cy: float,
        open_height: float,
        teeth_color: Tuple[int, int, int],
        smile: float
    ) -> None:
        """Draw teeth inside the mouth.

        Args:
            surface: Surface to draw on
            cx: Center x
            cy: Center y
            open_height: How open the mouth is
            teeth_color: Color for teeth
            smile: Smile amount
        """
        teeth_width = self.mouth_width * 0.6
        teeth_height = min(open_height * 0.3, self.mouth_height * 0.2)

        # Top teeth
        teeth_rect = pygame.Rect(
            cx - teeth_width / 2,
            cy - open_height / 2 + self.lip_thickness,
            teeth_width,
            teeth_height
        )
        pygame.draw.rect(surface, teeth_color, teeth_rect, border_radius=3)

        # Draw tooth separations
        num_teeth = 6
        tooth_width = teeth_width / num_teeth
        for i in range(1, num_teeth):
            tooth_x = int(cx - teeth_width / 2 + i * tooth_width)
            pygame.draw.line(
                surface,
                (200, 200, 210),
                (tooth_x, int(cy - open_height / 2 + self.lip_thickness)),
                (tooth_x, int(cy - open_height / 2 + self.lip_thickness + teeth_height)),
                1
            )

    def _draw_tongue(
        self,
        surface: pygame.Surface,
        cx: float,
        cy: float,
        tongue_color: Tuple[int, int, int],
        openness: float
    ) -> None:
        """Draw tongue inside the mouth.

        Args:
            surface: Surface to draw on
            cx: Center x
            cy: Center y (adjusted for tongue position)
            tongue_color: Color for tongue
            openness: How open the mouth is
        """
        tongue_width = self.mouth_width * 0.4
        tongue_height = self.mouth_height * 0.2 * openness

        tongue_rect = pygame.Rect(
            cx - tongue_width / 2,
            cy,
            tongue_width,
            tongue_height
        )

        pygame.draw.ellipse(surface, tongue_color, tongue_rect)

        # Tongue highlight
        highlight_color = (tongue_color[0] + 20, tongue_color[1] + 10, tongue_color[2] + 10)
        highlight_rect = pygame.Rect(
            cx - tongue_width / 4,
            cy + 2,
            tongue_width / 2,
            tongue_height / 2
        )
        pygame.draw.ellipse(surface, highlight_color, highlight_rect)

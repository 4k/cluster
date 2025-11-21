"""
Content renderers for the display system.
Each renderer handles drawing a specific type of content.
"""

from .base import BaseRenderer
from .eyes_renderer import EyesRenderer
from .mouth_renderer import MouthRenderer

__all__ = [
    'BaseRenderer',
    'EyesRenderer',
    'MouthRenderer',
]

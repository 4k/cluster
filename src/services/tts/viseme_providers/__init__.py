"""
Viseme Provider implementations.

Each provider implements the VisemeProvider protocol and can be used
interchangeably with the animation service.
"""

from .rhubarb_provider import RhubarbVisemeProvider
from .text_based_provider import TextBasedVisemeProvider

__all__ = [
    "RhubarbVisemeProvider",
    "TextBasedVisemeProvider",
]

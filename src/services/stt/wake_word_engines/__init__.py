"""
Wake Word Engine implementations.

Each engine implements the WakeWordEngine protocol and can be used
interchangeably with the STT service.
"""

from .openwakeword_engine import OpenWakeWordEngine

__all__ = [
    "OpenWakeWordEngine",
]

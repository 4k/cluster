"""
Display system for the voice assistant.

This module provides a flexible, multi-window display system where:
- A centralized DisplayDecisionModule manages what content is displayed
- Windows subscribe to specific ContentTypes (eyes, mouth, etc.)
- Window sizes and positions are configurable in settings
- Multiple windows can be created and destroyed dynamically

Architecture:
    DisplayManager (main interface)
        -> WindowManager (orchestrates windows)
            -> DisplayDecisionModule (central content routing)
            -> WindowProcess (per-window subprocess)
                -> Renderer (eyes, mouth, etc.)

Usage:
    config = DisplayConfig(
        mode="dual_display",
        resolution=(800, 480),
        window_positions={
            'eyes': (100, 100),
            'mouth': (100, 600),
        }
    )
    manager = DisplayManager(config)
    await manager.initialize()
    await manager.start()
"""

from .simple_display_manager import DisplayManager, DisplayConfig
from .window_manager import WindowManager
from .decision_module import DisplayDecisionModule
from .window import DisplayWindow

__all__ = [
    'DisplayManager',
    'DisplayConfig',
    'WindowManager',
    'DisplayDecisionModule',
    'DisplayWindow',
]

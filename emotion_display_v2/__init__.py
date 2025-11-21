"""
Emotion Display V2 - Multi-Window Facial Animation System

A modular, event-driven display system for animated facial expressions
with support for multiple synchronized windows.

Architecture:
    DisplayManager (main interface)
        -> WindowManager (orchestrates windows)
            -> DisplayDecisionModule (central content routing)
            -> WindowProcess (per-window subprocess)
                -> Renderer (eyes, mouth, etc.)

Features:
- Multiple synchronized windows (eyes, mouth, or custom)
- Centralized content routing via DisplayDecisionModule
- Event bus integration for real-time control
- Configurable window sizes and positions
- Cross-platform support (Windows, Linux, macOS)
- Extensible renderer architecture

Usage:
    # Basic dual-window setup
    python -m emotion_display_v2.demo

    # Programmatic usage
    from emotion_display_v2 import DisplayManager, DisplaySettings

    settings = DisplaySettings()
    manager = DisplayManager(settings)

    async def main():
        await manager.initialize()
        await manager.start()

        # Control the display
        manager.set_emotion('HAPPY')
        manager.speak_text("Hello, world!")

        # Wait for windows to close
        while manager.any_window_alive():
            await asyncio.sleep(0.1)

        await manager.stop()

    asyncio.run(main())
"""

from .settings import (
    DisplaySettings,
    WindowSettings,
    RendererSettings,
    ContentType,
    WindowType
)
from .decision_module import (
    DisplayDecisionModule,
    DisplayEvent,
    DisplayCommand
)
from .window_manager import WindowManager
from .display_manager import DisplayManager, run_display_manager
from .renderers import (
    BaseRenderer,
    RendererState,
    EyeRenderer,
    MouthRenderer
)
from .renderers.base_renderer import AnimationState, EmotionState

__all__ = [
    # Main interface
    'DisplayManager',
    'run_display_manager',

    # Window management
    'WindowManager',

    # Decision module
    'DisplayDecisionModule',
    'DisplayEvent',
    'DisplayCommand',

    # Settings
    'DisplaySettings',
    'WindowSettings',
    'RendererSettings',
    'ContentType',
    'WindowType',

    # Renderers
    'BaseRenderer',
    'RendererState',
    'EyeRenderer',
    'MouthRenderer',

    # States
    'AnimationState',
    'EmotionState'
]

__version__ = '2.0.0'

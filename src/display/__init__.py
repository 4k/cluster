"""
Display Package

Multi-window facial animation system with Rhubarb lip sync integration.
"""

from .v2 import (
    DisplayManager,
    run_display_manager,
    WindowManager,
    DisplayDecisionModule,
    DisplayEvent,
    DisplayCommand,
    DisplaySettings,
    WindowSettings,
    RendererSettings,
    ContentType,
    WindowType,
    BaseRenderer,
    RendererState,
    EyeRenderer,
    MouthRenderer,
    AnimationState,
    EmotionState,
)

__all__ = [
    'DisplayManager',
    'run_display_manager',
    'WindowManager',
    'DisplayDecisionModule',
    'DisplayEvent',
    'DisplayCommand',
    'DisplaySettings',
    'WindowSettings',
    'RendererSettings',
    'ContentType',
    'WindowType',
    'BaseRenderer',
    'RendererState',
    'EyeRenderer',
    'MouthRenderer',
    'AnimationState',
    'EmotionState',
]

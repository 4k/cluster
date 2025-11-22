"""
Display Package

Visual display systems for the voice assistant:
- v1: Original emotion display system
- v2: Refactored multi-window display system with Rhubarb integration
"""

# Re-export from v1
from .v1 import (
    EmotionDisplayService,
    EmotionDisplayConfig,
    WindowMode,
    EyeWindow,
    EyeConfig,
    MouthWindow,
    MouthConfig,
    AnimationState as V1AnimationState,
    EmotionState as V1EmotionState,
    DisplayState,
    VisemeMapper,
    Viseme,
    VisemeData,
)

# Re-export from v2
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
    AnimationState as V2AnimationState,
    EmotionState as V2EmotionState,
)

__all__ = [
    # V1 exports
    'EmotionDisplayService',
    'EmotionDisplayConfig',
    'WindowMode',
    'EyeWindow',
    'EyeConfig',
    'MouthWindow',
    'MouthConfig',
    'V1AnimationState',
    'V1EmotionState',
    'DisplayState',
    'VisemeMapper',
    'Viseme',
    'VisemeData',
    # V2 exports
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
    'V2AnimationState',
    'V2EmotionState',
]

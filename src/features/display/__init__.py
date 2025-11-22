"""
Display Feature - Multi-Window Facial Animation System

A pluggable feature that provides animated facial expressions with
lip sync support. Subscribes to TTS events and renders synchronized
mouth animations.

Components:
- DisplayManager: Main display coordination
- AnimationService: Coordinates lip sync with audio playback
- LipSync: Rhubarb integration for viseme generation
- Renderers: Eye and mouth rendering

Usage:
    # As a feature (recommended)
    from src.features import FeatureLoader
    loader = FeatureLoader()
    loader.load_feature('display')

    # Direct usage
    from src.features.display import DisplayManager, DisplaySettings
    manager = DisplayManager(DisplaySettings())
    await manager.initialize()
    await manager.start()
"""

import logging
from src.features import Feature, register_feature

logger = logging.getLogger(__name__)

# Display components
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

# Animation components
from .animation import AnimationService
from .lip_sync import (
    RhubarbLipSyncService,
    RhubarbViseme,
    VisemeCue,
    LipSyncData,
    RHUBARB_TO_INTERNAL_VISEME
)


@register_feature('display')
class DisplayFeature(Feature):
    """
    Display feature - animated face with lip sync.

    Provides:
    - Dual-window face display (eyes + mouth)
    - Lip sync animation synchronized with TTS
    - Emotion expressions
    - Gaze tracking
    """

    name = "display"
    description = "Animated face display with lip sync"

    def __init__(self):
        super().__init__()
        self.display_manager = None
        self.animation_service = None
        self.settings = None

    async def initialize(self) -> bool:
        """Initialize the display feature."""
        try:
            from src.core.event_bus import EventBus

            self.event_bus = await EventBus.get_instance()
            self.settings = DisplaySettings()

            # Initialize display manager
            self.display_manager = DisplayManager(self.settings)
            await self.display_manager.initialize()

            # Initialize animation service (handles lip sync)
            self.animation_service = AnimationService()
            await self.animation_service.initialize()

            self.is_initialized = True
            logger.info("Display feature initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize display feature: {e}")
            return False

    async def start(self) -> None:
        """Start the display feature."""
        if not self.is_initialized:
            raise RuntimeError("Display feature not initialized")

        # Start display manager (spawns windows)
        await self.display_manager.start()
        self.is_running = True
        logger.info("Display feature started")

    async def stop(self) -> None:
        """Stop the display feature."""
        if self.animation_service:
            await self.animation_service.stop()

        if self.display_manager:
            await self.display_manager.stop()

        self.is_running = False
        logger.info("Display feature stopped")


__all__ = [
    # Feature class
    'DisplayFeature',

    # Display components
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

    # Animation components
    'AnimationService',
    'RhubarbLipSyncService',
    'RhubarbViseme',
    'VisemeCue',
    'LipSyncData',
    'RHUBARB_TO_INTERNAL_VISEME',
]

__version__ = '2.0.0'

"""
Simple Display Manager - Main interface for the display system.

This module provides the DisplayManager class expected by main.py,
wrapping the WindowManager and providing a simple interface for
display control.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.event_bus import EventBus
from core.types import (
    ContentType, WindowConfig, DisplaySettings,
    EmotionType, AnimationState, GazeDirection
)
from .window_manager import WindowManager
from .decision_module import DisplayDecisionModule

logger = logging.getLogger(__name__)


@dataclass
class DisplayConfig:
    """Configuration for the display system.

    This is the configuration interface expected by main.py.
    It gets converted to DisplaySettings internally.

    Attributes:
        mode: Display mode ('dual_display', 'single', 'multi')
        resolution: Default resolution for windows
        fps: Target frames per second
        development_mode: Enable debug features
        window_positions: Optional dict of window positions
        fullscreen: Default fullscreen setting
        borderless: Default borderless setting
        always_on_top: Default always-on-top setting
        led_count: Number of NeoPixel LEDs
        led_pin: GPIO pin for LEDs
        touch_enabled: Enable touch input
        calibration_file: Path to touch calibration
        windows: Optional list of WindowConfig for custom setup
    """
    mode: str = "dual_display"
    resolution: Tuple[int, int] = (800, 480)
    fps: int = 30
    development_mode: bool = True
    window_positions: Optional[Dict[str, Tuple[int, int]]] = None
    fullscreen: bool = False
    borderless: bool = False
    always_on_top: bool = False
    led_count: int = 60
    led_pin: int = 18
    touch_enabled: bool = False
    calibration_file: Optional[str] = None
    windows: Optional[List[WindowConfig]] = None

    def to_display_settings(self) -> DisplaySettings:
        """Convert to DisplaySettings.

        Returns:
            DisplaySettings with window configurations
        """
        if self.windows:
            # Use custom window configuration
            return DisplaySettings(
                windows=self.windows,
                fps=self.fps,
                development_mode=self.development_mode,
                led_count=self.led_count,
                led_pin=self.led_pin,
                touch_enabled=self.touch_enabled,
                calibration_file=self.calibration_file,
            )

        # Generate windows based on mode
        windows = []

        if self.mode == "dual_display":
            # Default dual window: eyes on top, mouth on bottom
            eyes_pos = (100, 100)
            mouth_pos = (100, 100 + self.resolution[1] + 20)

            if self.window_positions:
                eyes_pos = self.window_positions.get('eyes', eyes_pos)
                mouth_pos = self.window_positions.get('mouth', mouth_pos)

            windows = [
                WindowConfig(
                    name="eyes",
                    title="Eyes Display",
                    content_type=ContentType.EYES,
                    position=eyes_pos,
                    size=self.resolution,
                    fullscreen=self.fullscreen,
                    borderless=self.borderless,
                    always_on_top=self.always_on_top,
                ),
                WindowConfig(
                    name="mouth",
                    title="Mouth Display",
                    content_type=ContentType.MOUTH,
                    position=mouth_pos,
                    size=(self.resolution[0], int(self.resolution[1] * 0.6)),
                    fullscreen=self.fullscreen,
                    borderless=self.borderless,
                    always_on_top=self.always_on_top,
                ),
            ]

        elif self.mode == "single":
            # Single window with full face
            windows = [
                WindowConfig(
                    name="face",
                    title="Face Display",
                    content_type=ContentType.FULL_FACE,
                    position=(100, 100),
                    size=self.resolution,
                    fullscreen=self.fullscreen,
                    borderless=self.borderless,
                    always_on_top=self.always_on_top,
                ),
            ]

        elif self.mode == "multi":
            # Multi-window mode - use window_positions if provided
            if self.window_positions:
                for name, pos in self.window_positions.items():
                    content_type = ContentType.EYES if 'eye' in name.lower() else \
                                   ContentType.MOUTH if 'mouth' in name.lower() else \
                                   ContentType.STATUS
                    windows.append(WindowConfig(
                        name=name,
                        title=f"{name.title()} Display",
                        content_type=content_type,
                        position=pos,
                        size=self.resolution,
                        fullscreen=self.fullscreen,
                        borderless=self.borderless,
                        always_on_top=self.always_on_top,
                    ))

        return DisplaySettings(
            windows=windows,
            fps=self.fps,
            development_mode=self.development_mode,
            led_count=self.led_count,
            led_pin=self.led_pin,
            touch_enabled=self.touch_enabled,
            calibration_file=self.calibration_file,
        )


class DisplayManager:
    """Main display manager interface.

    This class provides the interface expected by main.py and
    wraps the WindowManager for multi-window support.

    The DisplayManager:
    - Creates and manages display windows based on configuration
    - Provides methods for controlling display state
    - Interfaces with the event bus for state synchronization
    """

    def __init__(self, config: DisplayConfig):
        """Initialize the display manager.

        Args:
            config: Display configuration
        """
        self.config = config
        self.settings = config.to_display_settings()

        # Components (initialized in initialize())
        self.event_bus: Optional[EventBus] = None
        self.window_manager: Optional[WindowManager] = None

        # State
        self._running = False
        self._mock_mode = False

    async def initialize(self) -> None:
        """Initialize the display system."""
        logger.info(f"Initializing DisplayManager in '{self.config.mode}' mode")

        # Get event bus
        self.event_bus = await EventBus.get_instance()

        # Create window manager
        self.window_manager = WindowManager(self.settings, self.event_bus)
        await self.window_manager.initialize()

        logger.info(f"DisplayManager initialized with {len(self.settings.windows)} windows")

    async def start(self) -> None:
        """Start the display system."""
        if self._running:
            return

        logger.info("Starting DisplayManager")

        if self.window_manager:
            await self.window_manager.start()

        self._running = True
        logger.info("DisplayManager started")

    async def stop(self) -> None:
        """Stop the display system."""
        if not self._running:
            return

        logger.info("Stopping DisplayManager")

        if self.window_manager:
            await self.window_manager.stop()

        self._running = False
        logger.info("DisplayManager stopped")

    async def cleanup(self) -> None:
        """Clean up display resources."""
        await self.stop()

        if self.window_manager:
            await self.window_manager.cleanup()

        logger.info("DisplayManager cleaned up")

    # Control methods

    def set_emotion(self, emotion: EmotionType) -> None:
        """Set the displayed emotion.

        Args:
            emotion: The emotion to display
        """
        if self.window_manager and self.window_manager.decision_module:
            self.window_manager.decision_module.set_emotion(emotion)

    def set_gaze(self, direction: GazeDirection) -> None:
        """Set the gaze direction.

        Args:
            direction: The direction to look
        """
        if self.window_manager and self.window_manager.decision_module:
            self.window_manager.decision_module.set_gaze(direction)

    def trigger_blink(self) -> None:
        """Trigger an eye blink."""
        if self.window_manager and self.window_manager.decision_module:
            self.window_manager.decision_module.trigger_blink()

    def set_animation_state(self, state: AnimationState) -> None:
        """Set the animation state.

        Args:
            state: The animation state
        """
        if self.window_manager and self.window_manager.decision_module:
            self.window_manager.decision_module.set_animation_state(state)

    # Window management

    def add_window(self, config: WindowConfig) -> bool:
        """Add a new display window.

        Args:
            config: Configuration for the new window

        Returns:
            True if window was added successfully
        """
        if self.window_manager:
            return self.window_manager.add_window(config)
        return False

    def remove_window(self, name: str) -> bool:
        """Remove a display window.

        Args:
            name: Name of the window to remove

        Returns:
            True if window was removed
        """
        if self.window_manager:
            return self.window_manager.remove_window(name)
        return False

    def get_windows(self) -> List[str]:
        """Get list of current window names.

        Returns:
            List of window names
        """
        if self.window_manager:
            return list(self.window_manager.windows.keys())
        return []

    # State methods

    def get_state(self) -> Dict[str, Any]:
        """Get current display state.

        Returns:
            Dictionary with display state
        """
        state = {
            "running": self._running,
            "mode": self.config.mode,
            "mock_mode": self._mock_mode,
        }

        if self.window_manager:
            state["window_manager"] = self.window_manager.get_state()

        return state

    def is_running(self) -> bool:
        """Check if display is running.

        Returns:
            True if running
        """
        return self._running

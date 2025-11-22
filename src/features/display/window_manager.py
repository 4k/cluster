"""
WindowManager - Orchestrates multiple display windows for the emotion display system.
Manages window creation, lifecycle, and communication through the DisplayDecisionModule.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set

from .settings import DisplaySettings, WindowSettings, ContentType, WindowType, RendererSettings
from .decision_module import DisplayDecisionModule, DisplayEvent
from .window_process import WindowProcessFactory, WindowProcessHandle

logger = logging.getLogger(__name__)


class WindowManager:
    """
    Orchestrates multiple display windows.
    Creates and manages window processes, routes events through the decision module.
    """

    def __init__(self, settings: Optional[DisplaySettings] = None):
        """
        Initialize the window manager.

        Args:
            settings: Display settings (uses defaults if not provided)
        """
        self.settings = settings or DisplaySettings()

        # Window processes
        self._windows: Dict[str, WindowProcessHandle] = {}

        # Decision module for content routing
        self.decision_module = DisplayDecisionModule()

        # State
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Initialize multiprocessing
        WindowProcessFactory.set_multiprocessing_start_method()

        logger.info("WindowManager initialized")

    def start(self) -> bool:
        """
        Start all configured windows.

        Returns:
            True if at least one window started successfully
        """
        if self.is_running:
            logger.warning("WindowManager is already running")
            return True

        enabled_windows = self.settings.get_enabled_windows()
        if not enabled_windows:
            logger.error("No enabled windows configured")
            return False

        # Build renderer settings from display settings
        renderer_settings = self._build_renderer_settings()

        # Create windows
        started_count = 0
        for window_id, window_settings in enabled_windows.items():
            try:
                handle = WindowProcessFactory.create_window(
                    window_id=window_id,
                    settings=window_settings,
                    renderer_settings=renderer_settings
                )
                self._windows[window_id] = handle

                # Register with decision module
                self.decision_module.register_window(
                    window_id=window_id,
                    subscriptions=list(window_settings.subscriptions),
                    command_queue=handle.command_queue
                )

                started_count += 1
                logger.info(f"Started window '{window_id}'")

            except Exception as e:
                logger.error(f"Failed to start window '{window_id}': {e}")

        if started_count == 0:
            logger.error("Failed to start any windows")
            return False

        self.is_running = True
        logger.info(f"WindowManager started with {started_count} windows")
        return True

    def _build_renderer_settings(self) -> Dict[str, Any]:
        """Build renderer settings dictionary from display settings."""
        rs = self.settings.renderer
        return {
            # Eye settings
            'iris_color': rs.eye_iris_color,
            'pupil_color': rs.eye_pupil_color,
            'sclera_color': rs.eye_sclera_color,
            'blink_duration': rs.eye_blink_duration,
            'blink_interval_min': rs.eye_blink_interval_min,
            'blink_interval_max': rs.eye_blink_interval_max,
            'gaze_smoothing': rs.eye_gaze_smoothing,
            'max_gaze_offset': rs.eye_max_gaze_offset,
            # Mouth settings
            'lip_color': rs.mouth_lip_color,
            'interior_color': rs.mouth_interior_color,
            'teeth_color': rs.mouth_teeth_color,
            'transition_speed': rs.mouth_transition_speed,
            'idle_movement': rs.mouth_idle_movement
        }

    def stop(self) -> None:
        """Stop all windows."""
        if not self.is_running:
            return

        logger.info("Stopping WindowManager...")

        # Cancel monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

        # Stop all windows
        for window_id, handle in self._windows.items():
            try:
                self.decision_module.unregister_window(window_id)
                handle.stop()
                logger.info(f"Stopped window '{window_id}'")
            except Exception as e:
                logger.error(f"Error stopping window '{window_id}': {e}")

        self._windows.clear()
        self.is_running = False
        logger.info("WindowManager stopped")

    async def start_async(self) -> bool:
        """
        Start windows asynchronously and begin monitoring.

        Returns:
            True if started successfully
        """
        if not self.start():
            return False

        # Start monitor task
        self._monitor_task = asyncio.create_task(self._monitor_windows())
        return True

    async def stop_async(self) -> None:
        """Stop windows asynchronously."""
        self.stop()

    async def _monitor_windows(self) -> None:
        """Monitor window processes and handle crashes."""
        try:
            while self.is_running:
                # Check each window
                for window_id, handle in list(self._windows.items()):
                    if not handle.is_alive():
                        logger.warning(f"Window '{window_id}' died unexpectedly")
                        self.decision_module.unregister_window(window_id)
                        del self._windows[window_id]

                        # Optionally restart the window
                        # await self._restart_window(window_id)

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass

    def add_window(self, window_id: str, settings: WindowSettings) -> bool:
        """
        Add and start a new window dynamically.

        Args:
            window_id: Unique identifier for the window
            settings: Window settings

        Returns:
            True if window was added successfully
        """
        if window_id in self._windows:
            logger.warning(f"Window '{window_id}' already exists")
            return False

        try:
            renderer_settings = self._build_renderer_settings()
            handle = WindowProcessFactory.create_window(
                window_id=window_id,
                settings=settings,
                renderer_settings=renderer_settings
            )
            self._windows[window_id] = handle

            self.decision_module.register_window(
                window_id=window_id,
                subscriptions=list(settings.subscriptions),
                command_queue=handle.command_queue
            )

            # Also add to settings for persistence
            self.settings.windows[window_id] = settings

            logger.info(f"Added window '{window_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to add window '{window_id}': {e}")
            return False

    def remove_window(self, window_id: str) -> bool:
        """
        Remove and stop a window.

        Args:
            window_id: Window identifier to remove

        Returns:
            True if window was removed successfully
        """
        if window_id not in self._windows:
            logger.warning(f"Window '{window_id}' not found")
            return False

        try:
            handle = self._windows[window_id]
            self.decision_module.unregister_window(window_id)
            handle.stop()
            del self._windows[window_id]

            if window_id in self.settings.windows:
                del self.settings.windows[window_id]

            logger.info(f"Removed window '{window_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to remove window '{window_id}': {e}")
            return False

    def get_window_ids(self) -> List[str]:
        """Get list of active window IDs."""
        return list(self._windows.keys())

    def is_window_alive(self, window_id: str) -> bool:
        """Check if a specific window is alive."""
        if window_id not in self._windows:
            return False
        return self._windows[window_id].is_alive()

    def all_windows_alive(self) -> bool:
        """Check if all windows are still alive."""
        return all(handle.is_alive() for handle in self._windows.values())

    def any_window_alive(self) -> bool:
        """Check if any window is still alive."""
        return any(handle.is_alive() for handle in self._windows.values())

    # Convenience methods that delegate to decision module

    def trigger_blink(self) -> int:
        """Trigger an eye blink."""
        return self.decision_module.trigger_blink()

    def set_gaze(self, x: float, y: float) -> int:
        """Set gaze position."""
        return self.decision_module.set_gaze(x, y)

    def set_emotion(self, emotion: str) -> int:
        """Set emotion for all windows."""
        return self.decision_module.set_emotion(emotion)

    def set_animation_state(self, state: str) -> int:
        """Set animation state for all windows."""
        return self.decision_module.set_animation_state(state)

    def start_speaking(self, text: Optional[str] = None, duration: float = None) -> int:
        """Start speaking animation."""
        return self.decision_module.start_speaking(text, duration=duration)

    def stop_speaking(self) -> int:
        """Stop speaking animation."""
        return self.decision_module.stop_speaking()

    def set_viseme(self, viseme: str) -> int:
        """Set mouth viseme directly."""
        return self.decision_module.set_viseme(viseme)

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            'is_running': self.is_running,
            'window_count': len(self._windows),
            'windows': {
                window_id: {
                    'alive': handle.is_alive(),
                    'pid': handle.process.pid if handle.process else None
                }
                for window_id, handle in self._windows.items()
            },
            'decision_module': self.decision_module.get_statistics()
        }

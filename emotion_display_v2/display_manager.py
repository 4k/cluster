"""
DisplayManager - Main interface for the multi-window emotion display system.
Provides high-level API for managing display windows and connecting to the event bus.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .settings import DisplaySettings, WindowSettings, ContentType, WindowType
from .window_manager import WindowManager
from .decision_module import DisplayDecisionModule, DisplayEvent

logger = logging.getLogger(__name__)


class DisplayManager:
    """
    Main interface for the multi-window emotion display system.
    Provides high-level API and event bus integration.

    Architecture:
        DisplayManager (this class - main interface)
            -> WindowManager (orchestrates windows)
                -> DisplayDecisionModule (central content routing)
                -> WindowProcess (per-window subprocess)
                    -> Renderer (eyes, mouth, etc.)
    """

    def __init__(self, settings: Optional[DisplaySettings] = None):
        """
        Initialize the display manager.

        Args:
            settings: Display settings (uses defaults if not provided)
        """
        self.settings = settings or DisplaySettings()
        self.window_manager = WindowManager(self.settings)

        # Event bus connection
        self.event_bus = None
        self._event_bus_connected = False

        # State
        self.is_running = False

        logger.info("DisplayManager initialized")

    @property
    def decision_module(self) -> DisplayDecisionModule:
        """Get the decision module for direct access."""
        return self.window_manager.decision_module

    async def initialize(self) -> bool:
        """
        Initialize the display manager and optionally connect to event bus.

        Returns:
            True if initialization succeeded
        """
        try:
            # Connect to event bus if configured
            if self.settings.connect_event_bus:
                await self._connect_event_bus()

            logger.info("DisplayManager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DisplayManager: {e}")
            return False

    async def start(self) -> bool:
        """
        Start the display manager and all windows.

        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("DisplayManager is already running")
            return True

        # Start window manager
        if not await self.window_manager.start_async():
            return False

        self.is_running = True
        logger.info("DisplayManager started")
        return True

    async def stop(self) -> None:
        """Stop the display manager and all windows."""
        if not self.is_running:
            return

        await self.window_manager.stop_async()
        self.is_running = False
        logger.info("DisplayManager stopped")

    async def _connect_event_bus(self) -> bool:
        """Connect to the event bus and subscribe to relevant events."""
        try:
            from src.core.event_bus import EventBus, EventType

            self.event_bus = await EventBus.get_instance()

            # Subscribe to animation-related events
            self.event_bus.subscribe(EventType.TTS_STARTED, self._on_tts_started)
            self.event_bus.subscribe(EventType.TTS_COMPLETED, self._on_tts_completed)
            self.event_bus.subscribe(EventType.PHONEME_EVENT, self._on_phoneme_event)
            self.event_bus.subscribe(EventType.EXPRESSION_CHANGE, self._on_expression_change)
            self.event_bus.subscribe(EventType.EMOTION_CHANGED, self._on_emotion_changed)
            self.event_bus.subscribe(EventType.GAZE_UPDATE, self._on_gaze_update)
            self.event_bus.subscribe(EventType.MOUTH_SHAPE_UPDATE, self._on_mouth_shape_update)
            self.event_bus.subscribe(EventType.BLINK_TRIGGERED, self._on_blink_triggered)
            self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word)
            self.event_bus.subscribe(EventType.SPEECH_DETECTED, self._on_speech_detected)
            self.event_bus.subscribe(EventType.RESPONSE_GENERATING, self._on_response_generating)
            self.event_bus.subscribe(EventType.RESPONSE_GENERATED, self._on_response_generated)
            self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)

            self._event_bus_connected = True
            logger.info("Connected to event bus")
            return True

        except ImportError:
            logger.warning("Event bus not available, running in standalone mode")
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to event bus: {e}")
            return False

    # Event handlers
    async def _on_tts_started(self, event) -> None:
        """Handle TTS start - begin speaking animation."""
        text = event.data.get('text', '')
        self.set_animation_state('SPEAKING')
        if text:
            self.speak_text(text)
        else:
            self.start_speaking()

    async def _on_tts_completed(self, event) -> None:
        """Handle TTS completion - stop speaking animation."""
        self.stop_speaking()
        self.set_animation_state('IDLE')

    async def _on_phoneme_event(self, event) -> None:
        """Handle phoneme event for lip-sync."""
        phoneme = event.data.get('phoneme', '')
        # Map phoneme to viseme
        from emotion_display.viseme_mapper import PHONEME_TO_VISEME
        viseme = PHONEME_TO_VISEME.get(phoneme.upper())
        if viseme:
            self.set_viseme(viseme.name)

    async def _on_expression_change(self, event) -> None:
        """Handle expression change."""
        emotion = event.data.get('emotion', 'NEUTRAL')
        self.set_emotion(emotion)

    async def _on_emotion_changed(self, event) -> None:
        """Handle emotion change."""
        await self._on_expression_change(event)

    async def _on_gaze_update(self, event) -> None:
        """Handle gaze update."""
        x = event.data.get('x', 0.5)
        y = event.data.get('y', 0.5)
        self.set_gaze(x, y)

    async def _on_mouth_shape_update(self, event) -> None:
        """Handle direct mouth shape update."""
        viseme = event.data.get('viseme', 'SILENCE')
        self.set_viseme(viseme)

    async def _on_blink_triggered(self, event) -> None:
        """Handle blink trigger."""
        self.trigger_blink()

    async def _on_wake_word(self, event) -> None:
        """Handle wake word detection."""
        self.set_animation_state('LISTENING')
        self.set_emotion('SURPRISED')
        self.trigger_blink()

    async def _on_speech_detected(self, event) -> None:
        """Handle speech detection."""
        self.set_animation_state('LISTENING')

    async def _on_response_generating(self, event) -> None:
        """Handle response generation start."""
        self.set_animation_state('THINKING')
        self.set_emotion('THINKING')

    async def _on_response_generated(self, event) -> None:
        """Handle response generation complete."""
        text = event.data.get('response', '')
        if text:
            self.set_animation_state('SPEAKING')
            self.speak_text(text)

    async def _on_error(self, event) -> None:
        """Handle error event."""
        self.set_animation_state('ERROR')
        self.set_emotion('CONFUSED')

    # Public API methods

    def trigger_blink(self) -> None:
        """Trigger an eye blink."""
        self.window_manager.trigger_blink()

    def set_gaze(self, x: float, y: float) -> None:
        """
        Set gaze position.

        Args:
            x: Horizontal position (0=left, 0.5=center, 1=right)
            y: Vertical position (0=up, 0.5=center, 1=down)
        """
        self.window_manager.set_gaze(x, y)

    def set_emotion(self, emotion: str) -> None:
        """
        Set emotion for all windows.

        Args:
            emotion: Emotion name (NEUTRAL, HAPPY, SAD, ANGRY, etc.)
        """
        self.window_manager.set_emotion(emotion)

    def set_animation_state(self, state: str) -> None:
        """
        Set animation state for all windows.

        Args:
            state: Animation state (IDLE, LISTENING, THINKING, SPEAKING, etc.)
        """
        self.window_manager.set_animation_state(state)

    def start_speaking(self) -> None:
        """Start generic speaking animation."""
        self.window_manager.start_speaking()

    def speak_text(self, text: str) -> None:
        """
        Start speaking with text-based lip sync.

        Args:
            text: Text to animate
        """
        self.window_manager.start_speaking(text)

    def stop_speaking(self) -> None:
        """Stop speaking animation."""
        self.window_manager.stop_speaking()

    def set_viseme(self, viseme: str) -> None:
        """
        Set mouth viseme directly.

        Args:
            viseme: Viseme name (SILENCE, AH, EE, OH, etc.)
        """
        self.window_manager.set_viseme(viseme)

    # Window management

    def add_window(self, window_id: str, settings: WindowSettings) -> bool:
        """
        Add a new window dynamically.

        Args:
            window_id: Unique identifier for the window
            settings: Window settings

        Returns:
            True if window was added successfully
        """
        return self.window_manager.add_window(window_id, settings)

    def remove_window(self, window_id: str) -> bool:
        """
        Remove a window.

        Args:
            window_id: Window identifier to remove

        Returns:
            True if window was removed successfully
        """
        return self.window_manager.remove_window(window_id)

    def get_window_ids(self) -> List[str]:
        """Get list of active window IDs."""
        return self.window_manager.get_window_ids()

    def is_window_alive(self, window_id: str) -> bool:
        """Check if a specific window is alive."""
        return self.window_manager.is_window_alive(window_id)

    def all_windows_alive(self) -> bool:
        """Check if all windows are still alive."""
        return self.window_manager.all_windows_alive()

    def any_window_alive(self) -> bool:
        """Check if any window is still alive."""
        return self.window_manager.any_window_alive()

    # Status and statistics

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the display manager."""
        return {
            'is_running': self.is_running,
            'event_bus_connected': self._event_bus_connected,
            'current_state': self.decision_module.get_current_state(),
            'statistics': self.window_manager.get_statistics()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get display statistics."""
        return self.window_manager.get_statistics()

    # Async context manager support

    async def __aenter__(self) -> 'DisplayManager':
        """Async context manager entry."""
        await self.initialize()
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


async def run_display_manager(settings: Optional[DisplaySettings] = None,
                             connect_event_bus: bool = True) -> None:
    """
    Run the display manager until all windows are closed.

    Args:
        settings: Display settings
        connect_event_bus: Whether to connect to the event bus
    """
    if settings is None:
        settings = DisplaySettings(connect_event_bus=connect_event_bus)

    manager = DisplayManager(settings)

    try:
        await manager.initialize()
        await manager.start()

        print("\nMulti-Window Emotion Display Running")
        print("=" * 40)
        print("Controls (in any window):")
        print("  SPACE - Trigger blink")
        print("  S     - Start speaking")
        print("  X     - Stop speaking")
        print("  T     - Test text animation")
        print("  H     - Happy emotion")
        print("  N     - Neutral emotion")
        print("  ESC   - Quit window")
        print("=" * 40)

        # Run until all windows are closed
        while manager.any_window_alive():
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await manager.stop()


# Entry point for module execution
def main():
    """Main entry point."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run with default dual-window configuration
    asyncio.run(run_display_manager(connect_event_bus=False))


if __name__ == "__main__":
    main()

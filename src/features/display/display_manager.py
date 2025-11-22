"""
DisplayManager - Main interface for the multi-window emotion display system.
Provides high-level API for managing display windows and connecting to the event bus.

Enhanced with Rhubarb lip sync integration:
- Direct Rhubarb viseme control with smooth interpolation
- Coordinated switching between text-based and Rhubarb lip sync
- Integration with RhubarbVisemeController for advanced features
- Support for coarticulation and viseme lookahead

Based on best practices from:
- Rhubarb Lip Sync official documentation
- Industry-standard animation techniques
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
from .rhubarb_controller import (
    RhubarbVisemeController,
    RhubarbControllerConfig,
    RhubarbLipSyncIntegration
)

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

    def __init__(self, settings: Optional[DisplaySettings] = None,
                 rhubarb_config: Optional[RhubarbControllerConfig] = None):
        """
        Initialize the display manager with Rhubarb integration.

        Args:
            settings: Display settings (uses defaults if not provided)
            rhubarb_config: Rhubarb controller configuration
        """
        self.settings = settings or DisplaySettings()
        self.window_manager = WindowManager(self.settings)

        # Event bus connection
        self.event_bus = None
        self._event_bus_connected = False

        # State
        self.is_running = False

        # Rhubarb lip sync integration
        self._rhubarb_lip_sync_active = False
        self._rhubarb_session_id: Optional[str] = None
        self._rhubarb_config = rhubarb_config or RhubarbControllerConfig()

        # Initialize Rhubarb controller for direct viseme control
        self._rhubarb_controller = RhubarbVisemeController(self._rhubarb_config)
        self._rhubarb_controller.set_viseme_callback(self._on_rhubarb_viseme)

        # Statistics for Rhubarb integration
        self._rhubarb_stats = {
            'sessions_started': 0,
            'sessions_completed': 0,
            'visemes_received': 0,
            'fallback_used': 0
        }

        # Pending text for animation sync (set on TTS_STARTED, used on AUDIO_PLAYBACK_STARTED)
        self._pending_speak_text: Optional[str] = None

        logger.info("DisplayManager initialized with Rhubarb integration")

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

            # Subscribe to lip sync events (Rhubarb integration)
            self.event_bus.subscribe(EventType.LIP_SYNC_STARTED, self._on_lip_sync_started)
            self.event_bus.subscribe(EventType.LIP_SYNC_COMPLETED, self._on_lip_sync_completed)

            # Subscribe to audio playback events for proper timing
            self.event_bus.subscribe(EventType.AUDIO_PLAYBACK_STARTED, self._on_audio_playback_started)

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
        """Handle TTS start - prepare for speaking animation."""
        text = event.data.get('text', '')
        audio_file = event.data.get('audio_file')

        # Store text for later use when audio actually starts
        self._pending_speak_text = text

        # Set animation state but don't start mouth animation yet
        # Wait for AUDIO_PLAYBACK_STARTED for proper timing sync
        self.set_animation_state('SPEAKING')

        logger.debug(f"TTS started, waiting for audio playback (rhubarb_active={self._rhubarb_lip_sync_active})")

    async def _on_audio_playback_started(self, event) -> None:
        """Handle audio playback start - Rhubarb handles animation, text-based is fallback only."""
        audio_file = event.data.get('audio_file')

        # If there's an audio file, Rhubarb should handle lip sync
        # (AnimationService will emit LIP_SYNC_STARTED shortly after this event)
        # Only use text-based fallback when there's no audio file
        if not audio_file and not self._rhubarb_lip_sync_active:
            text = getattr(self, '_pending_speak_text', '')
            duration = event.data.get('duration')
            if text:
                self.speak_text(text, duration=duration)
            else:
                self.start_speaking()
            logger.debug("Using text-based animation (no audio file)")
        else:
            logger.debug(f"Audio playback started, waiting for Rhubarb lip sync")

    async def _on_lip_sync_started(self, event) -> None:
        """Handle Rhubarb lip sync started - configure for Rhubarb control."""
        self._rhubarb_lip_sync_active = True
        self._rhubarb_session_id = event.data.get('session_id')
        self._rhubarb_stats['sessions_started'] += 1

        # Stop any text-based animation that might be running
        self.stop_speaking()

        # Set speaking state without text animation
        self.set_animation_state('SPEAKING')

        # Notify mouth renderer of Rhubarb session start
        self.window_manager.decision_module.route_event(
            DisplayEvent.SPEAK_START,
            {'session_id': self._rhubarb_session_id, 'rhubarb': True}
        )

        # Load lip sync data into controller if provided
        cues = event.data.get('cues', [])
        duration = event.data.get('duration', 0.0)
        if cues:
            # Convert cues from animation_service format
            formatted_cues = []
            for cue in cues:
                if hasattr(cue, 'viseme'):
                    # VisemeCue object from RhubarbLipSyncService
                    formatted_cues.append({
                        'start': cue.start_time,
                        'value': cue.viseme.value if hasattr(cue.viseme, 'value') else str(cue.viseme)
                    })
                elif isinstance(cue, dict):
                    formatted_cues.append(cue)

            if formatted_cues:
                self._rhubarb_controller.load_lip_sync_data(
                    formatted_cues,
                    self._rhubarb_session_id or 'unknown',
                    duration
                )
                self._rhubarb_controller.start_session()

        logger.info(f"Rhubarb lip sync started: {event.data.get('cue_count', len(cues))} cues, "
                   f"session={self._rhubarb_session_id}")

    async def _on_lip_sync_completed(self, event) -> None:
        """Handle Rhubarb lip sync completed - transition back to idle."""
        self._rhubarb_lip_sync_active = False
        self._rhubarb_stats['sessions_completed'] += 1

        # Stop Rhubarb controller session
        if self._rhubarb_controller.is_active():
            self._rhubarb_controller.stop_session()

        # Notify mouth renderer
        self.window_manager.decision_module.route_event(
            DisplayEvent.SPEAK_STOP,
            {'session_id': self._rhubarb_session_id, 'rhubarb': True}
        )

        self._rhubarb_session_id = None
        logger.debug("Rhubarb lip sync completed")

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
        """Handle direct mouth shape update from animation service or Rhubarb."""
        viseme = event.data.get('viseme', 'SILENCE')
        params = event.data.get('params')  # Optional detailed params from Rhubarb
        source = event.data.get('source', 'unknown')

        self._rhubarb_stats['visemes_received'] += 1

        # Route viseme with optional parameters for enhanced animation
        if params:
            self.set_viseme_with_params(viseme, params)
        else:
            self.set_viseme(viseme)

    def _on_rhubarb_viseme(self, viseme_name: str, mouth_params: Dict[str, float]) -> None:
        """
        Callback from RhubarbVisemeController for viseme updates.

        This provides direct, high-fidelity control from the Rhubarb controller
        with interpolated parameters for smooth animation.
        """
        if not self.is_running:
            return

        # Route to mouth renderer with detailed parameters
        self.window_manager.decision_module.route_event(
            DisplayEvent.VISEME_UPDATE,
            {
                'viseme': viseme_name,
                'params': mouth_params,
                'source': 'rhubarb_controller'
            }
        )

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

    def speak_text(self, text: str, duration: float = None) -> None:
        """
        Start speaking with text-based lip sync.

        Args:
            text: Text to animate
            duration: Audio duration in seconds (for animation timing sync)
        """
        self.window_manager.start_speaking(text, duration=duration)

    def stop_speaking(self) -> None:
        """Stop speaking animation."""
        self.window_manager.stop_speaking()

    def set_viseme(self, viseme: str) -> None:
        """
        Set mouth viseme directly.

        Args:
            viseme: Viseme name (SILENCE, AH, EE, OH, etc.) or Rhubarb shape (A-X)
        """
        self.window_manager.set_viseme(viseme)

    def set_viseme_with_params(self, viseme: str, params: Dict[str, float]) -> None:
        """
        Set mouth viseme with detailed parameters from Rhubarb controller.

        This method provides enhanced control with interpolated parameters
        for smoother animation with coarticulation support.

        Args:
            viseme: Viseme name or Rhubarb shape letter
            params: Detailed mouth parameters (open, width, pucker, etc.)
        """
        self.window_manager.decision_module.route_event(
            DisplayEvent.VISEME_UPDATE,
            {
                'viseme': viseme,
                'params': params,
                'source': 'display_manager'
            }
        )

    def set_rhubarb_shape(self, shape: str, intensity: float = 1.0) -> None:
        """
        Set mouth to a specific Rhubarb shape.

        Args:
            shape: Rhubarb shape letter (A, B, C, D, E, F, G, H, X)
            intensity: Viseme intensity (0-1)
        """
        self.window_manager.decision_module.route_event(
            DisplayEvent.VISEME_UPDATE,
            {
                'viseme': shape,
                'intensity': intensity,
                'source': 'rhubarb_direct'
            }
        )

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
        """Get current state of the display manager including Rhubarb status."""
        return {
            'is_running': self.is_running,
            'event_bus_connected': self._event_bus_connected,
            'rhubarb_lip_sync_active': self._rhubarb_lip_sync_active,
            'rhubarb_session_id': self._rhubarb_session_id,
            'current_state': self.decision_module.get_current_state(),
            'statistics': self.window_manager.get_statistics(),
            'rhubarb_stats': self._rhubarb_stats,
            'rhubarb_controller_stats': self._rhubarb_controller.get_stats()
        }

    def get_rhubarb_config(self) -> RhubarbControllerConfig:
        """Get the current Rhubarb controller configuration."""
        return self._rhubarb_config

    def update_rhubarb_config(self, **kwargs) -> None:
        """
        Update Rhubarb controller configuration.

        Args:
            **kwargs: Configuration parameters to update
                - lookahead_ms: Viseme lookahead in milliseconds
                - transition_duration_ms: Transition time between visemes
                - enable_coarticulation: Whether to blend adjacent visemes
                - coarticulation_strength: Blend strength (0-1)
                - intensity_scale: Overall mouth movement intensity
        """
        for key, value in kwargs.items():
            if hasattr(self._rhubarb_config, key):
                setattr(self._rhubarb_config, key, value)

        # Recreate controller with new config
        self._rhubarb_controller = RhubarbVisemeController(self._rhubarb_config)
        self._rhubarb_controller.set_viseme_callback(self._on_rhubarb_viseme)

        logger.info(f"Updated Rhubarb config: {kwargs}")

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

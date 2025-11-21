"""
Emotion Display Service
Main service that manages the eye and mouth display windows.
Connects to the event bus to receive animation commands.
Supports both single-process (SDL2 multi-window) and multi-process modes.
"""

import asyncio
import logging
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Install with: pip install pygame")

from .animation_states import AnimationState, EmotionState, DisplayState
from .eye_window import EyeWindow, EyeConfig
from .mouth_window import MouthWindow, MouthConfig
from .viseme_mapper import Viseme

logger = logging.getLogger(__name__)


class WindowMode(Enum):
    """Display window mode."""
    SINGLE_WINDOW = "single"      # Combined view in one window
    DUAL_WINDOW = "dual"          # Separate eye and mouth windows
    SDL2_MULTI = "sdl2_multi"     # SDL2 native multi-window


@dataclass
class EmotionDisplayConfig:
    """Configuration for the emotion display service."""
    # Window mode
    mode: WindowMode = WindowMode.DUAL_WINDOW

    # Eye window settings
    eye_width: int = 800
    eye_height: int = 400
    eye_position: tuple = (100, 100)

    # Mouth window settings
    mouth_width: int = 800
    mouth_height: int = 300
    mouth_position: tuple = (100, 550)

    # General settings
    fps: int = 60
    background_color: tuple = (20, 20, 25)

    # Auto-connect to event bus
    connect_event_bus: bool = True


class EmotionDisplayService:
    """
    Main service for emotion display.
    Creates and manages eye and mouth display windows.
    Connects to event bus for receiving animation commands.
    """

    def __init__(self, config: Optional[EmotionDisplayConfig] = None):
        """
        Initialize the emotion display service.

        Args:
            config: Service configuration
        """
        self.config = config or EmotionDisplayConfig()

        # Display components
        self.eye_window: Optional[EyeWindow] = None
        self.mouth_window: Optional[MouthWindow] = None

        # State
        self.state = DisplayState()
        self.is_running = False
        self._should_stop = False

        # Event bus connection
        self.event_bus = None
        self._event_handlers: Dict[str, Callable] = {}

        # Pygame windows (for SDL2 multi-window mode)
        self._pygame_windows: List[Any] = []

        # Threading
        self._render_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Performance tracking
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._fps_update_time = 0.0
        self._current_fps = 0.0

    async def initialize(self) -> bool:
        """
        Initialize the display service.

        Returns:
            True if initialization succeeded
        """
        if not PYGAME_AVAILABLE:
            logger.error("Pygame is not available. Cannot initialize display.")
            return False

        try:
            # Initialize pygame
            pygame.init()

            # Create eye window config
            eye_config = EyeConfig(
                window_width=self.config.eye_width,
                window_height=self.config.eye_height,
                window_title="Emotion Display - Eyes",
                background_color=self.config.background_color
            )
            self.eye_window = EyeWindow(eye_config)

            # Create mouth window config
            mouth_config = MouthConfig(
                window_width=self.config.mouth_width,
                window_height=self.config.mouth_height,
                window_title="Emotion Display - Mouth",
                background_color=self.config.background_color
            )
            self.mouth_window = MouthWindow(mouth_config)

            # Initialize based on mode
            if self.config.mode == WindowMode.SDL2_MULTI:
                success = self._initialize_sdl2_multi()
            elif self.config.mode == WindowMode.DUAL_WINDOW:
                success = self._initialize_dual_process()
            else:
                success = self._initialize_single_window()

            if not success:
                return False

            # Connect to event bus if configured
            if self.config.connect_event_bus:
                await self._connect_event_bus()

            logger.info("Emotion display service initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize emotion display: {e}")
            return False

    def _initialize_sdl2_multi(self) -> bool:
        """Initialize using SDL2 multi-window support (Pygame 2.x)."""
        try:
            # Check if SDL2 multi-window is available
            if not hasattr(pygame, '_sdl2'):
                logger.warning("SDL2 multi-window not available, falling back to dual process")
                return self._initialize_dual_process()

            from pygame._sdl2.video import Window, Renderer

            # Create eye window
            eye_win = Window(
                "Emotion Display - Eyes",
                size=(self.config.eye_width, self.config.eye_height),
                position=self.config.eye_position
            )
            eye_renderer = Renderer(eye_win)
            eye_surface = pygame.Surface((self.config.eye_width, self.config.eye_height))
            self.eye_window.set_surface(eye_surface)
            self._pygame_windows.append((eye_win, eye_renderer, eye_surface, self.eye_window))

            # Create mouth window
            mouth_win = Window(
                "Emotion Display - Mouth",
                size=(self.config.mouth_width, self.config.mouth_height),
                position=self.config.mouth_position
            )
            mouth_renderer = Renderer(mouth_win)
            mouth_surface = pygame.Surface((self.config.mouth_width, self.config.mouth_height))
            self.mouth_window.set_surface(mouth_surface)
            self._pygame_windows.append((mouth_win, mouth_renderer, mouth_surface, self.mouth_window))

            logger.info("Initialized SDL2 multi-window mode")
            return True

        except Exception as e:
            logger.warning(f"SDL2 multi-window failed: {e}, falling back to dual process")
            return self._initialize_dual_process()

    def _initialize_dual_process(self) -> bool:
        """Initialize using separate windows (standard pygame)."""
        try:
            # For standard pygame, we'll use a single display but manage
            # two logical windows by splitting or using separate processes

            # Create main display large enough for both
            total_height = self.config.eye_height + self.config.mouth_height + 20
            total_width = max(self.config.eye_width, self.config.mouth_width)

            screen = pygame.display.set_mode(
                (total_width, total_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption("Emotion Display")

            # Create surfaces for each component
            eye_surface = pygame.Surface((self.config.eye_width, self.config.eye_height))
            mouth_surface = pygame.Surface((self.config.mouth_width, self.config.mouth_height))

            self.eye_window.set_surface(eye_surface)
            self.mouth_window.set_surface(mouth_surface)

            # Store for rendering
            self._main_screen = screen
            self._eye_surface = eye_surface
            self._mouth_surface = mouth_surface

            logger.info("Initialized dual-window mode (combined display)")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize dual process mode: {e}")
            return False

    def _initialize_single_window(self) -> bool:
        """Initialize with a single combined window."""
        return self._initialize_dual_process()

    async def _connect_event_bus(self) -> None:
        """Connect to the event bus and subscribe to relevant events."""
        try:
            from src.core.event_bus import EventBus, EventType

            self.event_bus = await EventBus.get_instance()

            # Subscribe to animation-related events
            self.event_bus.subscribe(EventType.TTS_STARTED, self._on_tts_started)
            self.event_bus.subscribe(EventType.TTS_COMPLETED, self._on_tts_completed)
            self.event_bus.subscribe(EventType.PHONEME_EVENT, self._on_phoneme_event)
            self.event_bus.subscribe(EventType.EXPRESSION_CHANGE, self._on_expression_change)
            self.event_bus.subscribe(EventType.GAZE_UPDATE, self._on_gaze_update)
            self.event_bus.subscribe(EventType.MOUTH_SHAPE_UPDATE, self._on_mouth_shape_update)
            self.event_bus.subscribe(EventType.BLINK_TRIGGERED, self._on_blink_triggered)
            self.event_bus.subscribe(EventType.EMOTION_CHANGED, self._on_emotion_changed)
            self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word)
            self.event_bus.subscribe(EventType.SPEECH_DETECTED, self._on_speech_detected)
            self.event_bus.subscribe(EventType.RESPONSE_GENERATING, self._on_response_generating)
            self.event_bus.subscribe(EventType.RESPONSE_GENERATED, self._on_response_generated)
            self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)

            logger.info("Connected to event bus")

        except ImportError:
            logger.warning("Event bus not available, running in standalone mode")
        except Exception as e:
            logger.warning(f"Failed to connect to event bus: {e}")

    # Event handlers
    async def _on_tts_started(self, event) -> None:
        """Handle TTS start - begin speaking animation."""
        text = event.data.get('text', '')
        self.set_animation_state(AnimationState.SPEAKING)
        if text:
            self.mouth_window.speak_text(text)
        else:
            self.mouth_window.start_speaking()

    async def _on_tts_completed(self, event) -> None:
        """Handle TTS completion - stop speaking animation."""
        self.mouth_window.stop_speaking()
        self.set_animation_state(AnimationState.IDLE)

    async def _on_phoneme_event(self, event) -> None:
        """Handle phoneme event for lip-sync."""
        phoneme = event.data.get('phoneme', '')
        duration = event.data.get('duration', 0.1)
        # Convert phoneme to viseme and update mouth
        from .viseme_mapper import PHONEME_TO_VISEME, Viseme
        viseme = PHONEME_TO_VISEME.get(phoneme.upper(), Viseme.SILENCE)
        self.mouth_window.set_viseme(viseme)

    async def _on_expression_change(self, event) -> None:
        """Handle expression change."""
        emotion_name = event.data.get('emotion', 'NEUTRAL')
        try:
            emotion = EmotionState[emotion_name.upper()]
            self.set_emotion(emotion)
        except KeyError:
            logger.warning(f"Unknown emotion: {emotion_name}")

    async def _on_gaze_update(self, event) -> None:
        """Handle gaze update."""
        x = event.data.get('x', 0.5)
        y = event.data.get('y', 0.5)
        self.eye_window.set_gaze(x, y)

    async def _on_mouth_shape_update(self, event) -> None:
        """Handle direct mouth shape update."""
        viseme_name = event.data.get('viseme', 'SILENCE')
        try:
            viseme = Viseme[viseme_name.upper()]
            self.mouth_window.set_viseme(viseme)
        except KeyError:
            logger.warning(f"Unknown viseme: {viseme_name}")

    async def _on_blink_triggered(self, event) -> None:
        """Handle blink trigger."""
        self.eye_window.trigger_blink()

    async def _on_emotion_changed(self, event) -> None:
        """Handle emotion change."""
        await self._on_expression_change(event)

    async def _on_wake_word(self, event) -> None:
        """Handle wake word detection."""
        self.set_animation_state(AnimationState.LISTENING)
        self.set_emotion(EmotionState.SURPRISED)
        self.eye_window.trigger_blink()

    async def _on_speech_detected(self, event) -> None:
        """Handle speech detection."""
        self.set_animation_state(AnimationState.LISTENING)

    async def _on_response_generating(self, event) -> None:
        """Handle response generation start."""
        self.set_animation_state(AnimationState.THINKING)
        self.set_emotion(EmotionState.THINKING)

    async def _on_response_generated(self, event) -> None:
        """Handle response generation complete."""
        text = event.data.get('response', '')
        if text:
            # Start speaking animation with the response text
            self.set_animation_state(AnimationState.SPEAKING)
            self.mouth_window.speak_text(text)

    async def _on_error(self, event) -> None:
        """Handle error event."""
        self.set_animation_state(AnimationState.ERROR)
        self.set_emotion(EmotionState.CONFUSED)

    def set_animation_state(self, state: AnimationState) -> None:
        """
        Set the animation state for both windows.

        Args:
            state: New animation state
        """
        self.state.animation_state = state
        self.eye_window.set_animation_state(state)
        self.mouth_window.set_animation_state(state)

    def set_emotion(self, emotion: EmotionState) -> None:
        """
        Set the emotion state for both windows.

        Args:
            emotion: New emotion state
        """
        self.state.emotion_state = emotion
        self.eye_window.set_emotion(emotion)
        self.mouth_window.set_emotion(emotion)

    async def start(self) -> None:
        """Start the display service."""
        if self.is_running:
            return

        self.is_running = True
        self._should_stop = False
        self._last_frame_time = time.time()

        logger.info("Emotion display service started")

    async def stop(self) -> None:
        """Stop the display service."""
        self._should_stop = True
        self.is_running = False

        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)

        logger.info("Emotion display service stopped")

    def update(self, dt: float) -> None:
        """
        Update all animations.

        Args:
            dt: Delta time in seconds
        """
        if not self.is_running:
            return

        self.eye_window.update(dt)
        self.mouth_window.update(dt)

    def render(self) -> None:
        """Render all windows."""
        if not self.is_running:
            return

        if self.config.mode == WindowMode.SDL2_MULTI:
            self._render_sdl2_multi()
        else:
            self._render_combined()

    def _render_sdl2_multi(self) -> None:
        """Render using SDL2 multi-window."""
        try:
            from pygame._sdl2.video import Texture

            for window, renderer, surface, component in self._pygame_windows:
                # Render component to its surface
                component.render()

                # Create texture from surface and render
                texture = Texture.from_surface(renderer, surface)
                renderer.clear()
                texture.draw()
                renderer.present()

        except Exception as e:
            logger.error(f"SDL2 render error: {e}")

    def _render_combined(self) -> None:
        """Render to combined window."""
        if not hasattr(self, '_main_screen'):
            return

        # Clear main screen
        self._main_screen.fill(self.config.background_color)

        # Render eyes
        self.eye_window.render()

        # Render mouth
        self.mouth_window.render()

        # Blit surfaces to main screen
        eye_x = (self._main_screen.get_width() - self._eye_surface.get_width()) // 2
        mouth_x = (self._main_screen.get_width() - self._mouth_surface.get_width()) // 2

        self._main_screen.blit(self._eye_surface, (eye_x, 0))
        self._main_screen.blit(self._mouth_surface, (mouth_x, self.config.eye_height + 20))

        # Update display
        pygame.display.flip()

    def process_events(self) -> bool:
        """
        Process pygame events.

        Returns:
            False if window was closed, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                # Debug keys
                elif event.key == pygame.K_SPACE:
                    self.eye_window.trigger_blink()
                elif event.key == pygame.K_s:
                    self.mouth_window.start_speaking()
                elif event.key == pygame.K_x:
                    self.mouth_window.stop_speaking()
                elif event.key == pygame.K_h:
                    self.set_emotion(EmotionState.HAPPY)
                elif event.key == pygame.K_n:
                    self.set_emotion(EmotionState.NEUTRAL)
                elif event.key == pygame.K_t:
                    self.mouth_window.speak_text("Hello, how are you today?")
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                pass

        return True

    def run_blocking(self) -> None:
        """Run the display service in blocking mode (for standalone use)."""
        if not self.is_running:
            asyncio.run(self.initialize())
            asyncio.run(self.start())

        clock = pygame.time.Clock()
        self._last_frame_time = time.time()

        try:
            while self.is_running and not self._should_stop:
                # Calculate delta time
                current_time = time.time()
                dt = current_time - self._last_frame_time
                self._last_frame_time = current_time

                # Process events
                if not self.process_events():
                    break

                # Update animations
                self.update(dt)

                # Render
                self.render()

                # Track FPS
                self._frame_count += 1
                if current_time - self._fps_update_time >= 1.0:
                    self._current_fps = self._frame_count / (current_time - self._fps_update_time)
                    self._frame_count = 0
                    self._fps_update_time = current_time

                # Cap frame rate
                clock.tick(self.config.fps)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    async def run_async(self) -> None:
        """Run the display service asynchronously."""
        clock = pygame.time.Clock()
        self._last_frame_time = time.time()

        try:
            while self.is_running and not self._should_stop:
                # Calculate delta time
                current_time = time.time()
                dt = current_time - self._last_frame_time
                self._last_frame_time = current_time

                # Process events
                if not self.process_events():
                    break

                # Update animations
                self.update(dt)

                # Render
                self.render()

                # Cap frame rate and yield to event loop
                clock.tick(self.config.fps)
                await asyncio.sleep(0)  # Yield to event loop

        except asyncio.CancelledError:
            pass
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False

        if self.eye_window:
            self.eye_window.cleanup()
        if self.mouth_window:
            self.mouth_window.cleanup()

        pygame.quit()
        logger.info("Emotion display service cleaned up")

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the display service."""
        return {
            'is_running': self.is_running,
            'animation_state': self.state.animation_state.name,
            'emotion_state': self.state.emotion_state.name,
            'fps': self._current_fps,
            'mode': self.config.mode.value
        }


# Standalone entry point
def main():
    """Run the emotion display service standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Emotion Display Service")
    print("=" * 40)
    print("Controls:")
    print("  SPACE - Trigger blink")
    print("  S     - Start speaking animation")
    print("  X     - Stop speaking animation")
    print("  H     - Happy emotion")
    print("  N     - Neutral emotion")
    print("  T     - Test text animation")
    print("  ESC   - Quit")
    print("=" * 40)

    # Create and run service
    config = EmotionDisplayConfig(
        mode=WindowMode.DUAL_WINDOW,
        connect_event_bus=False  # Standalone mode
    )

    service = EmotionDisplayService(config)

    # Initialize and run
    asyncio.run(service.initialize())
    asyncio.run(service.start())
    service.run_blocking()


if __name__ == "__main__":
    main()

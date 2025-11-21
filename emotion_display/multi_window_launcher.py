"""
Multi-Window Launcher for Emotion Display
Launches eye and mouth windows as separate processes for true multi-window support.
Uses multiprocessing for cross-platform compatibility.
"""

import asyncio
import logging
import multiprocessing as mp
import queue
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for a display window."""
    window_type: str  # 'eye' or 'mouth'
    width: int = 800
    height: int = 400
    position_x: int = 100
    position_y: int = 100
    fps: int = 60


def run_eye_window(command_queue: mp.Queue, config: Dict[str, Any]):
    """Run the eye window in a separate process."""
    import pygame
    from emotion_display.eye_window import EyeWindow, EyeConfig
    from emotion_display.animation_states import AnimationState, EmotionState

    # Initialize pygame in this process
    pygame.init()

    # Set window position
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{config['position_x']},{config['position_y']}"

    # Create display
    screen = pygame.display.set_mode((config['width'], config['height']))
    pygame.display.set_caption("Emotion Display - Eyes")

    # Create eye window
    eye_config = EyeConfig(
        window_width=config['width'],
        window_height=config['height']
    )
    eye_window = EyeWindow(eye_config)
    eye_window.set_surface(screen)

    # Main loop
    clock = pygame.time.Clock()
    last_time = time.time()
    running = True

    while running:
        # Calculate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    eye_window.trigger_blink()

        # Process commands from queue
        try:
            while True:
                cmd = command_queue.get_nowait()
                if cmd['type'] == 'quit':
                    running = False
                elif cmd['type'] == 'blink':
                    eye_window.trigger_blink()
                elif cmd['type'] == 'gaze':
                    eye_window.set_gaze(cmd.get('x', 0.5), cmd.get('y', 0.5))
                elif cmd['type'] == 'emotion':
                    try:
                        emotion = EmotionState[cmd.get('emotion', 'NEUTRAL').upper()]
                        eye_window.set_emotion(emotion)
                    except KeyError:
                        pass
                elif cmd['type'] == 'state':
                    try:
                        state = AnimationState[cmd.get('state', 'IDLE').upper()]
                        eye_window.set_animation_state(state)
                    except KeyError:
                        pass
        except queue.Empty:
            pass

        # Update and render
        eye_window.update(dt)
        eye_window.render()
        pygame.display.flip()

        clock.tick(config['fps'])

    pygame.quit()


def run_mouth_window(command_queue: mp.Queue, config: Dict[str, Any]):
    """Run the mouth window in a separate process."""
    import pygame
    from emotion_display.mouth_window import MouthWindow, MouthConfig
    from emotion_display.animation_states import AnimationState, EmotionState
    from emotion_display.viseme_mapper import Viseme

    # Initialize pygame in this process
    pygame.init()

    # Set window position
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{config['position_x']},{config['position_y']}"

    # Create display
    screen = pygame.display.set_mode((config['width'], config['height']))
    pygame.display.set_caption("Emotion Display - Mouth")

    # Create mouth window
    mouth_config = MouthConfig(
        window_width=config['width'],
        window_height=config['height']
    )
    mouth_window = MouthWindow(mouth_config)
    mouth_window.set_surface(screen)

    # Main loop
    clock = pygame.time.Clock()
    last_time = time.time()
    running = True

    while running:
        # Calculate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    mouth_window.start_speaking()
                elif event.key == pygame.K_x:
                    mouth_window.stop_speaking()
                elif event.key == pygame.K_t:
                    mouth_window.speak_text("Hello, how are you today?")

        # Process commands from queue
        try:
            while True:
                cmd = command_queue.get_nowait()
                if cmd['type'] == 'quit':
                    running = False
                elif cmd['type'] == 'speak_text':
                    mouth_window.speak_text(cmd.get('text', ''))
                elif cmd['type'] == 'start_speaking':
                    mouth_window.start_speaking()
                elif cmd['type'] == 'stop_speaking':
                    mouth_window.stop_speaking()
                elif cmd['type'] == 'viseme':
                    try:
                        viseme = Viseme[cmd.get('viseme', 'SILENCE').upper()]
                        mouth_window.set_viseme(viseme)
                    except KeyError:
                        pass
                elif cmd['type'] == 'emotion':
                    try:
                        emotion = EmotionState[cmd.get('emotion', 'NEUTRAL').upper()]
                        mouth_window.set_emotion(emotion)
                    except KeyError:
                        pass
                elif cmd['type'] == 'state':
                    try:
                        state = AnimationState[cmd.get('state', 'IDLE').upper()]
                        mouth_window.set_animation_state(state)
                    except KeyError:
                        pass
        except queue.Empty:
            pass

        # Update and render
        mouth_window.update(dt)
        mouth_window.render()
        pygame.display.flip()

        clock.tick(config['fps'])

    pygame.quit()


class MultiWindowEmotionDisplay:
    """
    Manages multiple display windows using separate processes.
    Provides communication between the main application and display windows.
    """

    def __init__(self,
                 eye_config: Optional[Dict[str, Any]] = None,
                 mouth_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-window display manager.

        Args:
            eye_config: Configuration for eye window
            mouth_config: Configuration for mouth window
        """
        self.eye_config = eye_config or {
            'width': 800,
            'height': 400,
            'position_x': 100,
            'position_y': 100,
            'fps': 60
        }

        self.mouth_config = mouth_config or {
            'width': 800,
            'height': 300,
            'position_x': 100,
            'position_y': 550,
            'fps': 60
        }

        # Process management
        self.eye_process: Optional[mp.Process] = None
        self.mouth_process: Optional[mp.Process] = None

        # Command queues for inter-process communication
        self.eye_queue: Optional[mp.Queue] = None
        self.mouth_queue: Optional[mp.Queue] = None

        self.is_running = False

    def start(self) -> None:
        """Start the display windows."""
        if self.is_running:
            return

        # Create command queues
        self.eye_queue = mp.Queue()
        self.mouth_queue = mp.Queue()

        # Start eye window process
        self.eye_process = mp.Process(
            target=run_eye_window,
            args=(self.eye_queue, self.eye_config)
        )
        self.eye_process.start()

        # Start mouth window process
        self.mouth_process = mp.Process(
            target=run_mouth_window,
            args=(self.mouth_queue, self.mouth_config)
        )
        self.mouth_process.start()

        self.is_running = True
        logger.info("Multi-window emotion display started")

    def stop(self) -> None:
        """Stop the display windows."""
        if not self.is_running:
            return

        # Send quit commands
        if self.eye_queue:
            self.eye_queue.put({'type': 'quit'})
        if self.mouth_queue:
            self.mouth_queue.put({'type': 'quit'})

        # Wait for processes to finish
        if self.eye_process and self.eye_process.is_alive():
            self.eye_process.join(timeout=2.0)
            if self.eye_process.is_alive():
                self.eye_process.terminate()

        if self.mouth_process and self.mouth_process.is_alive():
            self.mouth_process.join(timeout=2.0)
            if self.mouth_process.is_alive():
                self.mouth_process.terminate()

        self.is_running = False
        logger.info("Multi-window emotion display stopped")

    # Eye commands
    def trigger_blink(self) -> None:
        """Trigger an eye blink."""
        if self.eye_queue:
            self.eye_queue.put({'type': 'blink'})

    def set_gaze(self, x: float, y: float) -> None:
        """Set eye gaze position."""
        if self.eye_queue:
            self.eye_queue.put({'type': 'gaze', 'x': x, 'y': y})

    # Mouth commands
    def speak_text(self, text: str) -> None:
        """Start speaking with text-based lip sync."""
        if self.mouth_queue:
            self.mouth_queue.put({'type': 'speak_text', 'text': text})

    def start_speaking(self) -> None:
        """Start generic speaking animation."""
        if self.mouth_queue:
            self.mouth_queue.put({'type': 'start_speaking'})

    def stop_speaking(self) -> None:
        """Stop speaking animation."""
        if self.mouth_queue:
            self.mouth_queue.put({'type': 'stop_speaking'})

    def set_viseme(self, viseme: str) -> None:
        """Set mouth viseme directly."""
        if self.mouth_queue:
            self.mouth_queue.put({'type': 'viseme', 'viseme': viseme})

    # Both windows
    def set_emotion(self, emotion: str) -> None:
        """Set emotion for both windows."""
        if self.eye_queue:
            self.eye_queue.put({'type': 'emotion', 'emotion': emotion})
        if self.mouth_queue:
            self.mouth_queue.put({'type': 'emotion', 'emotion': emotion})

    def set_animation_state(self, state: str) -> None:
        """Set animation state for both windows."""
        if self.eye_queue:
            self.eye_queue.put({'type': 'state', 'state': state})
        if self.mouth_queue:
            self.mouth_queue.put({'type': 'state', 'state': state})

    def is_alive(self) -> bool:
        """Check if both windows are still running."""
        eye_alive = self.eye_process and self.eye_process.is_alive()
        mouth_alive = self.mouth_process and self.mouth_process.is_alive()
        return eye_alive and mouth_alive


class EventBusIntegration:
    """
    Integrates the multi-window display with the event bus.
    """

    def __init__(self, display: MultiWindowEmotionDisplay):
        """
        Initialize event bus integration.

        Args:
            display: Multi-window display instance
        """
        self.display = display
        self.event_bus = None

    async def connect(self) -> bool:
        """Connect to the event bus."""
        try:
            from src.core.event_bus import EventBus, EventType

            self.event_bus = await EventBus.get_instance()

            # Subscribe to events
            self.event_bus.subscribe(EventType.TTS_STARTED, self._on_tts_started)
            self.event_bus.subscribe(EventType.TTS_COMPLETED, self._on_tts_completed)
            self.event_bus.subscribe(EventType.PHONEME_EVENT, self._on_phoneme)
            self.event_bus.subscribe(EventType.EXPRESSION_CHANGE, self._on_expression)
            self.event_bus.subscribe(EventType.EMOTION_CHANGED, self._on_expression)
            self.event_bus.subscribe(EventType.GAZE_UPDATE, self._on_gaze)
            self.event_bus.subscribe(EventType.BLINK_TRIGGERED, self._on_blink)
            self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word)
            self.event_bus.subscribe(EventType.SPEECH_DETECTED, self._on_speech)
            self.event_bus.subscribe(EventType.RESPONSE_GENERATING, self._on_thinking)
            self.event_bus.subscribe(EventType.RESPONSE_GENERATED, self._on_response)
            self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)

            logger.info("Event bus integration connected")
            return True

        except Exception as e:
            logger.warning(f"Could not connect to event bus: {e}")
            return False

    async def _on_tts_started(self, event) -> None:
        text = event.data.get('text', '')
        self.display.set_animation_state('SPEAKING')
        if text:
            self.display.speak_text(text)
        else:
            self.display.start_speaking()

    async def _on_tts_completed(self, event) -> None:
        self.display.stop_speaking()
        self.display.set_animation_state('IDLE')

    async def _on_phoneme(self, event) -> None:
        phoneme = event.data.get('phoneme', '')
        # Map phoneme to viseme
        from emotion_display.viseme_mapper import PHONEME_TO_VISEME
        viseme = PHONEME_TO_VISEME.get(phoneme.upper())
        if viseme:
            self.display.set_viseme(viseme.name)

    async def _on_expression(self, event) -> None:
        emotion = event.data.get('emotion', 'NEUTRAL')
        self.display.set_emotion(emotion)

    async def _on_gaze(self, event) -> None:
        x = event.data.get('x', 0.5)
        y = event.data.get('y', 0.5)
        self.display.set_gaze(x, y)

    async def _on_blink(self, event) -> None:
        self.display.trigger_blink()

    async def _on_wake_word(self, event) -> None:
        self.display.set_animation_state('LISTENING')
        self.display.set_emotion('SURPRISED')
        self.display.trigger_blink()

    async def _on_speech(self, event) -> None:
        self.display.set_animation_state('LISTENING')

    async def _on_thinking(self, event) -> None:
        self.display.set_animation_state('THINKING')
        self.display.set_emotion('THINKING')

    async def _on_response(self, event) -> None:
        text = event.data.get('response', '')
        if text:
            self.display.set_animation_state('SPEAKING')
            self.display.speak_text(text)

    async def _on_error(self, event) -> None:
        self.display.set_animation_state('ERROR')
        self.display.set_emotion('CONFUSED')


async def main_async():
    """Run multi-window display with event bus integration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create display
    display = MultiWindowEmotionDisplay()
    display.start()

    # Try to connect to event bus
    integration = EventBusIntegration(display)
    await integration.connect()

    print("\nMulti-Window Emotion Display Running")
    print("=" * 40)
    print("Controls (in eye window):")
    print("  SPACE - Trigger blink")
    print("  ESC   - Quit")
    print("\nControls (in mouth window):")
    print("  S     - Start speaking")
    print("  X     - Stop speaking")
    print("  T     - Test text animation")
    print("  ESC   - Quit")
    print("=" * 40)

    # Run until windows are closed
    try:
        while display.is_alive():
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        display.stop()


def main():
    """Main entry point."""
    # Use spawn method for cross-platform compatibility
    mp.set_start_method('spawn', force=True)
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

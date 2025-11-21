"""
WindowProcess - Per-window subprocess management for the multi-window display system.
Each window runs in its own process for true multi-window support.
"""

import logging
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable

from .settings import WindowSettings, WindowType, ContentType

logger = logging.getLogger(__name__)


def run_window_process(command_queue: mp.Queue, config: Dict[str, Any]) -> None:
    """
    Main function for a window subprocess.
    This runs in a separate process and manages its own pygame instance.

    Args:
        command_queue: Queue for receiving commands from the main process
        config: Window configuration dictionary
    """
    try:
        import pygame
    except ImportError:
        logger.error("pygame not available in subprocess")
        return

    from .settings import WindowType, ContentType
    from .renderers import EyeRenderer, MouthRenderer

    # Initialize pygame in this process
    pygame.init()

    # Set window position via SDL environment variable
    position_x = config.get('position_x', 100)
    position_y = config.get('position_y', 100)
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{position_x},{position_y}"

    # Create display
    width = config.get('width', 800)
    height = config.get('height', 400)
    title = config.get('title', 'Display Window')

    flags = 0
    if config.get('borderless', False):
        flags |= pygame.NOFRAME

    screen = pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption(title)

    # Determine renderer type based on subscriptions
    subscriptions = config.get('subscriptions', [])
    background_color = tuple(config.get('background_color', (20, 20, 25)))

    # Check if this is a combined window (both eyes and mouth)
    has_eyes = ContentType.EYES.value in subscriptions or 'eyes' in subscriptions
    has_mouth = ContentType.MOUTH.value in subscriptions or 'mouth' in subscriptions
    is_combined = has_eyes and has_mouth

    # Create appropriate renderer(s) with sub-surfaces for combined mode
    renderers = []
    surfaces = {}  # Store surfaces and their blit positions

    if is_combined:
        # Combined mode: split the window vertically
        # Eyes take ~55% of height, mouth takes ~40%, with spacing
        eye_height = int(height * 0.55)
        mouth_height = int(height * 0.40)
        spacing = int(height * 0.02)

        # Create sub-surfaces
        eye_surface = pygame.Surface((width, eye_height))
        mouth_surface = pygame.Surface((width, mouth_height))

        # Eye renderer
        eye_renderer = EyeRenderer(
            width, eye_height, background_color,
            iris_color=tuple(config.get('iris_color', (100, 150, 200))),
            pupil_color=tuple(config.get('pupil_color', (20, 20, 30))),
            sclera_color=tuple(config.get('sclera_color', (240, 240, 245))),
            blink_duration=config.get('blink_duration', 0.15),
            blink_interval_min=config.get('blink_interval_min', 2.0),
            blink_interval_max=config.get('blink_interval_max', 6.0),
            gaze_smoothing=config.get('gaze_smoothing', 0.1),
            max_gaze_offset=config.get('max_gaze_offset', 0.3)
        )
        eye_renderer.set_surface(eye_surface)
        renderers.append(('eyes', eye_renderer))
        surfaces['eyes'] = (eye_surface, (0, 0))

        # Mouth renderer
        mouth_renderer = MouthRenderer(
            width, mouth_height, background_color,
            lip_color=tuple(config.get('lip_color', (180, 100, 100))),
            interior_color=tuple(config.get('interior_color', (60, 30, 40))),
            teeth_color=tuple(config.get('teeth_color', (240, 240, 235))),
            transition_speed=config.get('transition_speed', 12.0),
            idle_movement=config.get('idle_movement', True)
        )
        mouth_renderer.set_surface(mouth_surface)
        renderers.append(('mouth', mouth_renderer))
        surfaces['mouth'] = (mouth_surface, (0, eye_height + spacing))

    else:
        # Single renderer mode: use full screen
        if has_eyes:
            eye_renderer = EyeRenderer(
                width, height, background_color,
                iris_color=tuple(config.get('iris_color', (100, 150, 200))),
                pupil_color=tuple(config.get('pupil_color', (20, 20, 30))),
                sclera_color=tuple(config.get('sclera_color', (240, 240, 245))),
                blink_duration=config.get('blink_duration', 0.15),
                blink_interval_min=config.get('blink_interval_min', 2.0),
                blink_interval_max=config.get('blink_interval_max', 6.0),
                gaze_smoothing=config.get('gaze_smoothing', 0.1),
                max_gaze_offset=config.get('max_gaze_offset', 0.3)
            )
            eye_renderer.set_surface(screen)
            renderers.append(('eyes', eye_renderer))

        if has_mouth:
            mouth_renderer = MouthRenderer(
                width, height, background_color,
                lip_color=tuple(config.get('lip_color', (180, 100, 100))),
                interior_color=tuple(config.get('interior_color', (60, 30, 40))),
                teeth_color=tuple(config.get('teeth_color', (240, 240, 235))),
                transition_speed=config.get('transition_speed', 12.0),
                idle_movement=config.get('idle_movement', True)
            )
            mouth_renderer.set_surface(screen)
            renderers.append(('mouth', mouth_renderer))

    if not renderers:
        logger.warning(f"No renderers created for subscriptions: {subscriptions}")
        pygame.quit()
        return

    # Main loop
    clock = pygame.time.Clock()
    last_time = time.time()
    running = True
    fps = config.get('fps', 60)

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
                # Debug keys
                elif event.key == pygame.K_SPACE:
                    for name, renderer in renderers:
                        if hasattr(renderer, 'trigger_blink'):
                            renderer.trigger_blink()
                elif event.key == pygame.K_s:
                    for name, renderer in renderers:
                        if hasattr(renderer, 'start_speaking'):
                            renderer.start_speaking()
                elif event.key == pygame.K_x:
                    for name, renderer in renderers:
                        if hasattr(renderer, 'stop_speaking'):
                            renderer.stop_speaking()
                elif event.key == pygame.K_t:
                    for name, renderer in renderers:
                        if hasattr(renderer, 'speak_text'):
                            renderer.speak_text("Hello, how are you today?")
                elif event.key == pygame.K_h:
                    from .renderers.base_renderer import EmotionState
                    for name, renderer in renderers:
                        renderer.set_emotion(EmotionState.HAPPY)
                elif event.key == pygame.K_n:
                    from .renderers.base_renderer import EmotionState
                    for name, renderer in renderers:
                        renderer.set_emotion(EmotionState.NEUTRAL)

        # Process commands from queue
        try:
            while True:
                cmd = command_queue.get_nowait()
                if cmd.get('event') == 'quit' or cmd.get('type') == 'quit':
                    running = False
                    break
                else:
                    # Route command to appropriate renderers
                    for name, renderer in renderers:
                        renderer.handle_command(cmd)
        except queue.Empty:
            pass

        # Update renderers
        for name, renderer in renderers:
            renderer.update(dt)

        # Clear main screen
        screen.fill(background_color)

        # Render
        if is_combined:
            # Render each to its sub-surface and blit to main screen
            for name, renderer in renderers:
                renderer.render()
            for name, (surface, pos) in surfaces.items():
                screen.blit(surface, pos)
        else:
            # Single renderer mode: render directly to screen
            for name, renderer in renderers:
                renderer.render()

        pygame.display.flip()
        clock.tick(fps)

    # Cleanup
    for name, renderer in renderers:
        renderer.cleanup()
    pygame.quit()


@dataclass
class WindowProcessHandle:
    """Handle for a window subprocess."""
    window_id: str
    process: mp.Process
    command_queue: mp.Queue
    config: Dict[str, Any]

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process.is_alive()

    def send_command(self, command: Dict[str, Any]) -> bool:
        """
        Send a command to the window.

        Args:
            command: Command dictionary

        Returns:
            True if command was sent successfully
        """
        try:
            self.command_queue.put_nowait(command)
            return True
        except Exception as e:
            logger.warning(f"Failed to send command to window '{self.window_id}': {e}")
            return False

    def stop(self, timeout: float = 2.0) -> None:
        """
        Stop the window process gracefully.

        Args:
            timeout: Time to wait for graceful shutdown
        """
        try:
            self.command_queue.put_nowait({'event': 'quit'})
        except Exception:
            pass

        if self.process.is_alive():
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                logger.warning(f"Force terminating window '{self.window_id}'")
                self.process.terminate()
                self.process.join(timeout=1.0)


class WindowProcessFactory:
    """Factory for creating window subprocess instances."""

    @staticmethod
    def create_window(window_id: str, settings: WindowSettings,
                     renderer_settings: Optional[Dict[str, Any]] = None) -> WindowProcessHandle:
        """
        Create a new window subprocess.

        Args:
            window_id: Unique identifier for the window
            settings: Window settings
            renderer_settings: Optional renderer-specific settings

        Returns:
            Handle for the created window process
        """
        # Build configuration dictionary
        config = {
            'window_id': window_id,
            'window_type': settings.window_type.value,
            'title': settings.title,
            'width': settings.width,
            'height': settings.height,
            'position_x': settings.position_x,
            'position_y': settings.position_y,
            'fps': settings.fps,
            'background_color': settings.background_color,
            'subscriptions': [s.value for s in settings.subscriptions],
            'borderless': settings.borderless,
            'always_on_top': settings.always_on_top
        }

        # Add renderer settings
        if renderer_settings:
            config.update(renderer_settings)

        # Create command queue
        command_queue = mp.Queue()

        # Create and start process
        process = mp.Process(
            target=run_window_process,
            args=(command_queue, config),
            name=f"window_{window_id}"
        )
        process.start()

        logger.info(f"Started window process '{window_id}' (PID: {process.pid})")

        return WindowProcessHandle(
            window_id=window_id,
            process=process,
            command_queue=command_queue,
            config=config
        )

    @staticmethod
    def set_multiprocessing_start_method() -> None:
        """Set the multiprocessing start method for cross-platform compatibility."""
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set
            pass

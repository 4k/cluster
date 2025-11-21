"""
Window Manager for orchestrating multiple display windows.

Supports multiple windows through subprocess-based rendering,
where each window runs in its own process for true OS-level
window independence.
"""

import asyncio
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
import os
import sys
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from core.event_bus import EventBus, EventType
from core.types import (
    ContentType, WindowConfig, DisplaySettings,
    EyeState, MouthState, FaceState, AnimationState, EmotionType, GazeDirection
)
from .decision_module import DisplayDecisionModule

logger = logging.getLogger(__name__)


def _run_window_process(
    config_dict: dict,
    state_queue: Queue,
    command_queue: Queue,
    ready_event: mp.Event
):
    """Run a display window in a separate process.

    This function is the entry point for each window subprocess.

    Args:
        config_dict: Window configuration as dictionary
        state_queue: Queue for receiving state updates
        command_queue: Queue for receiving commands (stop, etc.)
        ready_event: Event to signal when window is ready
    """
    import pygame

    # Reconstruct config
    content_type = ContentType(config_dict['content_type'])
    config = WindowConfig(
        name=config_dict['name'],
        title=config_dict['title'],
        content_type=content_type,
        position=tuple(config_dict['position']),
        size=tuple(config_dict['size']),
        fullscreen=config_dict.get('fullscreen', False),
        borderless=config_dict.get('borderless', False),
        always_on_top=config_dict.get('always_on_top', False),
        background_color=tuple(config_dict.get('background_color', (0, 0, 0))),
        monitor=config_dict.get('monitor', 0),
    )

    # Set window position
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{config.position[0]},{config.position[1]}"

    # Initialize pygame
    pygame.init()

    # Create window
    flags = 0
    if config.fullscreen:
        flags |= pygame.FULLSCREEN
    if config.borderless:
        flags |= pygame.NOFRAME

    screen = pygame.display.set_mode(config.size, flags)
    pygame.display.set_caption(config.title)
    clock = pygame.time.Clock()

    # Create renderer based on content type
    renderer = None
    if content_type == ContentType.EYES:
        from display.renderers.eyes_renderer import EyesRenderer
        renderer = EyesRenderer(config.size)
    elif content_type == ContentType.MOUTH:
        from display.renderers.mouth_renderer import MouthRenderer
        renderer = MouthRenderer(config.size)

    # Current state
    current_state = None
    if content_type == ContentType.EYES:
        current_state = EyeState()
    elif content_type == ContentType.MOUTH:
        current_state = MouthState()

    # Signal ready
    ready_event.set()

    # Main loop
    running = True
    while running:
        # Check for commands
        try:
            while not command_queue.empty():
                cmd = command_queue.get_nowait()
                if cmd == 'stop':
                    running = False
                    break
        except Exception:
            pass

        # Check for state updates
        try:
            while not state_queue.empty():
                state_data = state_queue.get_nowait()
                if content_type == ContentType.EYES:
                    current_state = EyeState(
                        gaze=GazeDirection(state_data.get('gaze', 'forward')),
                        openness=state_data.get('openness', 1.0),
                        pupil_dilation=state_data.get('pupil_dilation', 1.0),
                        is_blinking=state_data.get('is_blinking', False),
                        blink_progress=state_data.get('blink_progress', 0.0),
                        emotion=EmotionType(state_data.get('emotion', 'neutral')),
                    )
                elif content_type == ContentType.MOUTH:
                    from core.types import MouthShape
                    viseme_str = state_data.get('viseme')
                    viseme = MouthShape(viseme_str) if viseme_str else None
                    current_state = MouthState(
                        shape=MouthShape(state_data.get('shape', 'closed')),
                        openness=state_data.get('openness', 0.0),
                        smile_amount=state_data.get('smile_amount', 0.0),
                        emotion=EmotionType(state_data.get('emotion', 'neutral')),
                        is_speaking=state_data.get('is_speaking', False),
                        viseme=viseme,
                    )
        except Exception as e:
            pass

        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Render
        dt = clock.tick(60) / 1000.0
        screen.fill(config.background_color)

        if renderer and current_state:
            renderer.render(screen, current_state, dt)

        pygame.display.flip()

    pygame.quit()


class WindowProcess:
    """Wrapper for a window running in a separate process."""

    def __init__(self, config: WindowConfig):
        """Initialize window process wrapper.

        Args:
            config: Window configuration
        """
        self.config = config
        self.process: Optional[Process] = None
        self.state_queue: Optional[Queue] = None
        self.command_queue: Optional[Queue] = None
        self.ready_event: Optional[mp.Event] = None

    def start(self) -> bool:
        """Start the window process.

        Returns:
            True if started successfully
        """
        try:
            # Create queues for communication
            self.state_queue = Queue()
            self.command_queue = Queue()
            self.ready_event = mp.Event()

            # Convert config to dict for pickling
            config_dict = {
                'name': self.config.name,
                'title': self.config.title,
                'content_type': self.config.content_type.value,
                'position': list(self.config.position),
                'size': list(self.config.size),
                'fullscreen': self.config.fullscreen,
                'borderless': self.config.borderless,
                'always_on_top': self.config.always_on_top,
                'background_color': list(self.config.background_color),
                'monitor': self.config.monitor,
            }

            # Start process
            self.process = Process(
                target=_run_window_process,
                args=(config_dict, self.state_queue, self.command_queue, self.ready_event),
                daemon=True
            )
            self.process.start()

            # Wait for ready signal
            if not self.ready_event.wait(timeout=10.0):
                logger.error(f"Window '{self.config.name}' failed to start")
                return False

            logger.info(f"Window process '{self.config.name}' started (PID: {self.process.pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to start window process '{self.config.name}': {e}")
            return False

    def stop(self) -> None:
        """Stop the window process."""
        if self.command_queue:
            try:
                self.command_queue.put('stop')
            except Exception:
                pass

        if self.process and self.process.is_alive():
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)

        logger.info(f"Window process '{self.config.name}' stopped")

    def send_state(self, state_dict: dict) -> None:
        """Send state update to the window.

        Args:
            state_dict: State as dictionary
        """
        if self.state_queue:
            try:
                # Clear old states and send new one
                while not self.state_queue.empty():
                    try:
                        self.state_queue.get_nowait()
                    except Exception:
                        break
                self.state_queue.put(state_dict)
            except Exception as e:
                logger.debug(f"Failed to send state to '{self.config.name}': {e}")

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process is not None and self.process.is_alive()


class WindowManager:
    """Manages multiple display windows.

    Creates and coordinates windows based on DisplaySettings.
    Each window runs in its own process for true multi-window support.
    The WindowManager connects to the DisplayDecisionModule to receive
    content updates and routes them to the appropriate windows.
    """

    def __init__(self, settings: DisplaySettings, event_bus: EventBus):
        """Initialize the window manager.

        Args:
            settings: Display settings with window configurations
            event_bus: Event bus for communication
        """
        self.settings = settings
        self.event_bus = event_bus

        # Window processes
        self.windows: Dict[str, WindowProcess] = {}

        # Decision module for content routing
        self.decision_module: Optional[DisplayDecisionModule] = None

        # Running state
        self._running = False

    async def initialize(self) -> None:
        """Initialize the window manager and decision module."""
        logger.info("Initializing WindowManager")

        # Create decision module
        self.decision_module = DisplayDecisionModule(self.event_bus)
        await self.decision_module.initialize()

        # Subscribe to decision module updates for each window's content type
        for window_config in self.settings.windows:
            self.decision_module.subscribe(
                window_config.content_type,
                lambda update, dt, wc=window_config: self._on_content_update(wc.name, update, dt)
            )

        logger.info(f"WindowManager initialized with {len(self.settings.windows)} window configs")

    async def start(self) -> None:
        """Start all windows and the decision module."""
        if self._running:
            return

        logger.info("Starting WindowManager")

        # Start decision module
        if self.decision_module:
            await self.decision_module.start()

        # Start window processes
        for window_config in self.settings.windows:
            window_proc = WindowProcess(window_config)
            if window_proc.start():
                self.windows[window_config.name] = window_proc
            else:
                logger.error(f"Failed to start window '{window_config.name}'")

        self._running = True
        logger.info(f"WindowManager started with {len(self.windows)} windows")

    async def stop(self) -> None:
        """Stop all windows and the decision module."""
        if not self._running:
            return

        logger.info("Stopping WindowManager")

        # Stop decision module
        if self.decision_module:
            await self.decision_module.stop()

        # Stop all window processes
        for name, window_proc in self.windows.items():
            window_proc.stop()

        self.windows.clear()
        self._running = False
        logger.info("WindowManager stopped")

    async def cleanup(self) -> None:
        """Clean up all resources."""
        await self.stop()
        logger.info("WindowManager cleaned up")

    def _on_content_update(self, window_name: str, update: Any, dt: float) -> None:
        """Handle content update from decision module.

        Routes the update to the appropriate window process.

        Args:
            window_name: Name of the window to update
            update: The content update
            dt: Delta time
        """
        if window_name not in self.windows:
            return

        window_proc = self.windows[window_name]
        if not window_proc.is_alive():
            return

        # Convert update to dictionary for IPC
        state_dict = {}

        if hasattr(update, 'state'):
            state = update.state
            if hasattr(state, 'gaze'):  # EyeState
                state_dict = {
                    'gaze': state.gaze.value,
                    'openness': state.openness,
                    'pupil_dilation': state.pupil_dilation,
                    'is_blinking': state.is_blinking,
                    'blink_progress': state.blink_progress,
                    'emotion': state.emotion.value,
                }
            elif hasattr(state, 'smile_amount'):  # MouthState
                state_dict = {
                    'shape': state.shape.value,
                    'openness': state.openness,
                    'smile_amount': state.smile_amount,
                    'emotion': state.emotion.value,
                    'is_speaking': state.is_speaking,
                    'viseme': state.viseme.value if state.viseme else None,
                }

        if state_dict:
            window_proc.send_state(state_dict)

    def add_window(self, config: WindowConfig) -> bool:
        """Add and start a new window.

        Args:
            config: Window configuration

        Returns:
            True if window was added successfully
        """
        if config.name in self.windows:
            logger.warning(f"Window '{config.name}' already exists")
            return False

        # Add to settings
        self.settings.windows.append(config)

        # Subscribe to content updates
        if self.decision_module:
            self.decision_module.subscribe(
                config.content_type,
                lambda update, dt, wc=config: self._on_content_update(wc.name, update, dt)
            )

        # Start window process
        window_proc = WindowProcess(config)
        if window_proc.start():
            self.windows[config.name] = window_proc
            return True

        return False

    def remove_window(self, name: str) -> bool:
        """Remove and stop a window.

        Args:
            name: Name of window to remove

        Returns:
            True if window was removed
        """
        if name not in self.windows:
            return False

        # Stop window process
        self.windows[name].stop()
        del self.windows[name]

        # Remove from settings
        self.settings.windows = [w for w in self.settings.windows if w.name != name]

        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current state for debugging.

        Returns:
            Dictionary with manager state
        """
        return {
            "running": self._running,
            "window_count": len(self.windows),
            "windows": {
                name: {
                    "alive": proc.is_alive(),
                    "content_type": proc.config.content_type.value,
                }
                for name, proc in self.windows.items()
            },
            "decision_module": self.decision_module.get_state() if self.decision_module else None,
        }

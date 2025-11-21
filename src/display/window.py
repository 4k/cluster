"""
Generic display window that subscribes to content from the DisplayDecisionModule.

Each window can display any content type by subscribing to that type.
Window configuration (size, position, content) is defined in settings.
"""

import logging
import asyncio
from typing import Any, Optional, Dict, Type

import pygame

from core.types import (
    ContentType, WindowConfig, EyeState, MouthState, FaceState,
    EyesContentUpdate, MouthContentUpdate, FullFaceContentUpdate, StatusContentUpdate
)
from .decision_module import DisplayDecisionModule
from .renderers.base import BaseRenderer
from .renderers.eyes_renderer import EyesRenderer
from .renderers.mouth_renderer import MouthRenderer

logger = logging.getLogger(__name__)


# Registry of renderers for each content type
RENDERER_REGISTRY: Dict[ContentType, Type[BaseRenderer]] = {
    ContentType.EYES: EyesRenderer,
    ContentType.MOUTH: MouthRenderer,
    # Additional renderers can be registered here
}


class DisplayWindow:
    """A display window that subscribes to content from the decision module.

    Each window:
    - Has its own pygame display/surface
    - Subscribes to a specific ContentType
    - Uses the appropriate renderer for that content type
    - Receives updates from the DisplayDecisionModule via callbacks

    Attributes:
        config: Window configuration
        decision_module: The central decision module for content updates
        renderer: The renderer for this window's content type
    """

    def __init__(self, config: WindowConfig, decision_module: DisplayDecisionModule):
        """Initialize the display window.

        Args:
            config: Configuration for this window
            decision_module: The central decision module to subscribe to
        """
        self.config = config
        self.decision_module = decision_module

        # Pygame resources (initialized in initialize())
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        # Renderer for this content type
        self.renderer: Optional[BaseRenderer] = None

        # Current content state (received from decision module)
        self._current_state: Optional[Any] = None
        self._last_update_time = 0.0

        # Running state
        self._running = False
        self._render_task: Optional[asyncio.Task] = None

        # Window handle for multi-window support
        self._window_id: Optional[int] = None

    async def initialize(self) -> None:
        """Initialize the window and renderer."""
        logger.info(f"Initializing window '{self.config.name}' for {self.config.content_type.value}")

        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()

        # Set window position before creating
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.config.position[0]},{self.config.position[1]}"

        # Create window flags
        flags = 0
        if self.config.fullscreen:
            flags |= pygame.FULLSCREEN
        if self.config.borderless:
            flags |= pygame.NOFRAME

        # Create the display window
        # For multiple windows, we use pygame's windowed mode
        self.screen = pygame.display.set_mode(
            self.config.size,
            flags,
            display=self.config.monitor
        )
        pygame.display.set_caption(self.config.title)

        self.clock = pygame.time.Clock()

        # Create the appropriate renderer
        renderer_class = RENDERER_REGISTRY.get(self.config.content_type)
        if renderer_class:
            self.renderer = renderer_class(self.config.size)
            logger.info(f"Created {renderer_class.__name__} for window '{self.config.name}'")
        else:
            logger.warning(f"No renderer for content type {self.config.content_type.value}")

        # Subscribe to content updates from decision module
        self.decision_module.subscribe(
            self.config.content_type,
            self._on_content_update
        )

        logger.info(f"Window '{self.config.name}' initialized at {self.config.position} size {self.config.size}")

    async def start(self) -> None:
        """Start the window render loop."""
        if self._running:
            return

        self._running = True
        self._render_task = asyncio.create_task(self._render_loop())
        logger.info(f"Window '{self.config.name}' started")

    async def stop(self) -> None:
        """Stop the window render loop."""
        self._running = False
        if self._render_task:
            self._render_task.cancel()
            try:
                await self._render_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Window '{self.config.name}' stopped")

    async def cleanup(self) -> None:
        """Clean up window resources."""
        await self.stop()

        # Unsubscribe from decision module
        self.decision_module.unsubscribe(
            self.config.content_type,
            self._on_content_update
        )

        # Note: pygame.quit() should be called by WindowManager
        logger.info(f"Window '{self.config.name}' cleaned up")

    def _on_content_update(self, update: Any, dt: float) -> None:
        """Handle content updates from the decision module.

        Args:
            update: The content update (type depends on content_type)
            dt: Delta time since last update
        """
        # Extract state from update based on content type
        if isinstance(update, EyesContentUpdate):
            self._current_state = update.state
        elif isinstance(update, MouthContentUpdate):
            self._current_state = update.state
        elif isinstance(update, FullFaceContentUpdate):
            self._current_state = update.state
        elif isinstance(update, StatusContentUpdate):
            self._current_state = update
        else:
            self._current_state = update

        self._last_update_time = update.timestamp if hasattr(update, 'timestamp') else 0.0

    async def _render_loop(self) -> None:
        """Main render loop for the window."""
        target_fps = 60  # Use config fps if available
        frame_time = 1.0 / target_fps

        while self._running:
            try:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self._running = False
                            break

                # Calculate delta time
                dt = self.clock.tick(target_fps) / 1000.0

                # Render if we have a renderer and state
                if self.renderer and self._current_state is not None:
                    self.renderer.render(self.screen, self._current_state, dt)
                else:
                    # Clear with background color if no content
                    self.screen.fill(self.config.background_color)

                # Update display
                pygame.display.flip()

                # Yield to other tasks
                await asyncio.sleep(0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in render loop for '{self.config.name}': {e}")
                await asyncio.sleep(0.1)

    def handle_resize(self, new_size: tuple) -> None:
        """Handle window resize.

        Args:
            new_size: New (width, height)
        """
        self.config.size = new_size
        if self.renderer:
            self.renderer.resize(new_size)
        logger.debug(f"Window '{self.config.name}' resized to {new_size}")

    def get_state(self) -> Dict[str, Any]:
        """Get the current window state for debugging.

        Returns:
            Dictionary with window state info
        """
        return {
            "name": self.config.name,
            "content_type": self.config.content_type.value,
            "position": self.config.position,
            "size": self.config.size,
            "running": self._running,
            "has_state": self._current_state is not None,
        }


class MultiWindowDisplay:
    """Helper class for managing multiple pygame windows.

    Pygame traditionally supports only one window, but we can use
    multiple surfaces and blit them to different areas, or use
    OS-level window management.

    For true multi-window support, consider using pygame_sdl2 or
    implementing with separate processes.
    """

    def __init__(self):
        """Initialize multi-window support."""
        self.windows: Dict[str, DisplayWindow] = {}
        self._surfaces: Dict[str, pygame.Surface] = {}

    def add_window(self, window: DisplayWindow) -> None:
        """Add a window to manage.

        Args:
            window: The window to add
        """
        self.windows[window.config.name] = window

    def remove_window(self, name: str) -> None:
        """Remove a window.

        Args:
            name: Name of window to remove
        """
        if name in self.windows:
            del self.windows[name]
            if name in self._surfaces:
                del self._surfaces[name]

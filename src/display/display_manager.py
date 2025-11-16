"""
Display Manager for generic display output.
Manages only the physical display of frames using pygame windows.

Does NOT decide what to display - receives already-rendered frames from AnimationEngine.

Both development and production modes use pygame windows:
- Development: Flexible window positioning, resizable, on development monitor
- Production: Fixed positions/sizes for physical displays (DSI, HDMI, etc. on Raspberry Pi)

Supports multiple display modes:
- dual_display: Two separate displays (requires multiple processes or combined window)
- single_display: One display window
- led_stripe: LED stripe output (console logging for now)
- disabled: No display output
"""

import logging
import asyncio
import pygame
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import time
from pathlib import Path
import threading
from enum import Enum

from core.event_bus import EventBus, EventType, emit_event

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display output modes."""
    DUAL_DISPLAY = "dual_display"      # Two separate displays
    SINGLE_DISPLAY = "single_display"  # One combined display
    LED_STRIPE = "led_stripe"          # LED stripe output
    DISABLED = "disabled"              # No display output


@dataclass
class DisplayConfig:
    """Configuration for display management."""
    mode: str = "dual_display"  # Display mode: dual_display, single_display, led_stripe, disabled
    resolution: Tuple[int, int] = (800, 480)
    fps: int = 30
    development_mode: bool = True
    window_positions: Optional[List[Tuple[int, int]]] = field(default=None)
    fullscreen: bool = False
    borderless: bool = False
    always_on_top: bool = False
    led_count: int = 60
    led_pin: int = 18
    touch_enabled: bool = True
    calibration_file: Optional[str] = field(default=None)


class DisplayManager:
    """Manages generic display output.
    
    Responsibilities:
    - Display already-rendered frames (from AnimationEngine)
    - Support multiple display modes (dual, single, LED stripe, disabled)
    - Show pygame windows during development
    - Send frames to DSI displays on Raspberry Pi in production
    - Handle touch events
    """
    
    def __init__(self, config: DisplayConfig):
        self.config = config
        self.is_running = False
        self.is_initialized = False
        
        # Validate and parse display mode
        self.display_mode = DisplayMode(config.mode)
        
        # Calculate number of displays based on mode
        self.num_displays = self._get_num_displays()
        
        # Display windows (pygame windows for both dev and production)
        self.display_windows: List[Optional[pygame.Surface]] = [None] * self.num_displays
        
        # Threading
        self._display_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self.on_touch_event: Optional[Callable] = None
        
        # Frame buffers (generic, indexed by display number)
        self._current_frames: List[Optional[pygame.Surface]] = [None] * self.num_displays
        self._frame_lock = threading.Lock()
    
    def _get_num_displays(self) -> int:
        """Get the number of displays based on mode."""
        if self.display_mode == DisplayMode.DUAL_DISPLAY:
            return 2
        elif self.display_mode == DisplayMode.SINGLE_DISPLAY:
            return 1
        elif self.display_mode == DisplayMode.LED_STRIPE:
            return 0  # LED stripe doesn't use pygame surfaces
        elif self.display_mode == DisplayMode.DISABLED:
            return 0  # No displays
        return 0
    
    async def initialize(self) -> None:
        """Initialize display manager."""
        try:
            logger.info(f"Initializing display manager in {self.display_mode.value} mode")
            
            # Skip initialization for disabled mode
            if self.display_mode == DisplayMode.DISABLED:
                self.is_initialized = True
                logger.info("Display manager initialized (disabled mode - no output)")
                return
            
            # Skip pygame initialization for LED stripe mode
            if self.display_mode != DisplayMode.LED_STRIPE:
                pygame.init()
            
            # Initialize display surfaces/windows based on mode
            await self._initialize_displays()
            
            self.is_initialized = True
            logger.info(f"Display manager initialized ({self.display_mode.value} mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize display manager: {e}")
            raise
    
    async def _initialize_displays(self) -> None:
        """Initialize pygame windows based on mode."""
        try:
            if self.display_mode == DisplayMode.LED_STRIPE:
                # LED stripe mode: No pygame initialization needed
                logger.info("LED stripe mode initialized (console output only)")
                return
            
            if self.display_mode == DisplayMode.DISABLED:
                # Disabled mode: No initialization needed
                return
            
            # Both development and production use pygame windows
            # Difference is in window positioning and flags
            await self._initialize_pygame_windows()
            
        except Exception as e:
            logger.error(f"Failed to initialize displays: {e}")
            raise
    
    async def _initialize_pygame_windows(self) -> None:
        """Initialize pygame windows for displays.
        
        Both development and production use pygame windows.
        Development: Flexible positioning, resizable
        Production: Fixed positions for physical displays (DSI, HDMI, etc.)
        
        Note: pygame limitation - can only create one window per process.
        For multiple displays, need multiple processes or combined window.
        """
        if self.display_mode == DisplayMode.SINGLE_DISPLAY:
            # Single display: One window
            window = self._create_window(0)
            self.display_windows[0] = window
            logger.info(f"Initialized single display window ({self.config.resolution[0]}x{self.config.resolution[1]})")
            
        elif self.display_mode == DisplayMode.DUAL_DISPLAY:
            # Dual display: pygame limitation - only one window per process
            # For true dual display, would need:
            # - Option 1: Two separate processes (recommended for production)
            # - Option 2: One wide window spanning both displays
            # For now: Create window for display 0
            window = self._create_window(0)
            self.display_windows[0] = window
            
            if self.config.development_mode:
                logger.info(f"Initialized dual display window (showing display 0 only)")
                logger.warning("Dual display: pygame limitation - only display 0 shown. Use separate processes for true dual display.")
            else:
                logger.info(f"Initialized production window for display 0 at position {self.config.window_positions[0] if self.config.window_positions else 'auto'}")
                logger.warning("For dual physical displays, run two separate processes with different window positions")
    
    def _create_window(self, display_index: int) -> pygame.Surface:
        """Create a pygame window with appropriate flags.
        
        Args:
            display_index: Index of the display (for positioning)
            
        Returns:
            pygame.Surface: The created window surface
        """
        # Build window flags
        flags = 0
        
        if self.config.fullscreen:
            flags |= pygame.FULLSCREEN
        if self.config.borderless:
            flags |= pygame.NOFRAME
        if self.config.development_mode:
            flags |= pygame.RESIZABLE  # Only resizable in development
        
        # Set window position (production mode)
        if not self.config.development_mode and self.config.window_positions:
            if display_index < len(self.config.window_positions):
                pos = self.config.window_positions[display_index]
                import os
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{pos[0]},{pos[1]}"
                logger.debug(f"Setting window position to {pos}")
        
        # Create window
        window = pygame.display.set_mode(self.config.resolution, flags)
        
        # Set caption
        mode_str = "Dev" if self.config.development_mode else "Prod"
        pygame.display.set_caption(f"Display {display_index} ({mode_str})")
        
        # Set always on top (if requested)
        if self.config.always_on_top and not self.config.development_mode:
            # Note: pygame doesn't have built-in always-on-top
            # Would need platform-specific code
            pass
        
        return window
    
    async def start(self) -> None:
        """Start display manager."""
        if self.is_running:
            return
        
        # Skip starting for disabled mode
        if self.display_mode == DisplayMode.DISABLED:
            self.is_running = True
            logger.info("Display manager started (disabled mode - no output)")
            return
        
        try:
            # Start display thread (not needed for LED stripe in current implementation)
            if self.display_mode != DisplayMode.LED_STRIPE:
                self._stop_event.clear()
                self._display_thread = threading.Thread(target=self._display_loop)
                self._display_thread.start()
            
            self.is_running = True
            logger.info(f"Display manager started ({self.display_mode.value} mode)")
            
        except Exception as e:
            logger.error(f"Failed to start display manager: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop display manager."""
        if not self.is_running:
            return
        
        try:
            # Stop display thread
            self._stop_event.set()
            if self._display_thread:
                self._display_thread.join(timeout=1.0)
            
            self.is_running = False
            logger.info("Display manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping display manager: {e}")
    
    def _display_loop(self) -> None:
        """Main display rendering loop."""
        clock = pygame.time.Clock()
        
        while not self._stop_event.is_set():
            try:
                # Display current frames
                self._display_frames()
                
                # Handle events
                self._handle_events()
                
                # Maintain FPS
                clock.tick(self.config.fps)
                
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
    
    def _display_frames(self) -> None:
        """Display the current frames to pygame windows.
        
        Works the same in both development and production modes.
        The difference is in window positioning and flags, not in rendering.
        """
        if not self.is_initialized:
            return
        
        try:
            with self._frame_lock:
                # Display frames to pygame windows (same for dev and prod)
                for i in range(self.num_displays):
                    if self._current_frames[i] and i < len(self.display_windows) and self.display_windows[i]:
                        self.display_windows[i].blit(self._current_frames[i], (0, 0))
                
                # Update display (only once for all windows in this process)
                if any(self.display_windows):
                    pygame.display.flip()
            
        except Exception as e:
            logger.error(f"Error displaying frames: {e}")
    
    def _handle_events(self) -> None:
        """Handle PyGame events."""
        if not self.config.touch_enabled:
            return
        
        try:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Touch event
                    if self.on_touch_event:
                        try:
                            self.on_touch_event('touch_down', event.pos)
                        except Exception as e:
                            logger.error(f"Error in touch event callback: {e}")
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    # Touch release
                    if self.on_touch_event:
                        try:
                            self.on_touch_event('touch_up', event.pos)
                        except Exception as e:
                            logger.error(f"Error in touch event callback: {e}")
                
                elif event.type == pygame.QUIT:
                    # Quit event
                    self._stop_event.set()
                    
        except Exception as e:
            logger.error(f"Error handling display events: {e}")
    
    def display_frame(self, display_index: int, frame: pygame.Surface) -> None:
        """Display a frame on a specific display (called by AnimationEngine).
        
        Args:
            display_index: Index of the display (0-based)
            frame: Already-rendered pygame surface to display
        """
        # Handle disabled mode
        if self.display_mode == DisplayMode.DISABLED:
            return
        
        # Handle LED stripe mode
        if self.display_mode == DisplayMode.LED_STRIPE:
            self._display_led_stripe(display_index, frame)
            return
        
        # Validate display index
        if display_index < 0 or display_index >= self.num_displays:
            logger.warning(f"Invalid display index: {display_index} (have {self.num_displays} displays)")
            return
        
        # Update frame buffer
        with self._frame_lock:
            self._current_frames[display_index] = frame
        
        logger.debug(f"Received frame for display {display_index}")
    
    def _display_led_stripe(self, display_index: int, frame: pygame.Surface) -> None:
        """Display frame on LED stripe (console output for now).
        
        Args:
            display_index: Index of the display (for logging purposes)
            frame: Pygame surface (will extract average color for LED)
        """
        try:
            # Extract average color from frame
            if frame:
                # Get average color of the frame
                avg_color = pygame.transform.average_color(frame)
                logger.info(f"LED Stripe [{display_index}]: RGB=({avg_color[0]}, {avg_color[1]}, {avg_color[2]})")
            else:
                logger.debug(f"LED Stripe [{display_index}]: No frame")
        except Exception as e:
            logger.error(f"Error processing LED stripe output: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get display manager state."""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "mode": self.display_mode.value,
            "num_displays": self.num_displays,
            "development_mode": self.config.development_mode,
            "resolution": self.config.resolution,
            "fps": self.config.fps,
            "active_displays": [i for i in range(self.num_displays) if self._current_frames[i] is not None]
        }
    
    def set_touch_callback(self, on_touch_event: Optional[Callable] = None) -> None:
        """Set callback function for touch events."""
        self.on_touch_event = on_touch_event
    
    async def cleanup(self) -> None:
        """Cleanup display manager resources."""
        await self.stop()
        
        # Clean up pygame resources
        if self.display_mode not in [DisplayMode.DISABLED, DisplayMode.LED_STRIPE]:
            if any(self.display_windows):
                pygame.display.quit()
            
            pygame.quit()
        
        # Clear all resources
        self.display_windows = [None] * self.num_displays
        self._current_frames = [None] * self.num_displays
        self.is_initialized = False
        
        logger.info(f"Display manager cleaned up ({self.display_mode.value} mode)")

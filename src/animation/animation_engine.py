"""
Animation Engine for coordinating visual expressions and lip-sync.
Controls animations based on emotion state, speech audio, and TTS output.
Loads and renders static images from data/displays/ folder.
"""

import logging
import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pygame

from core.types import EmotionType, GazeDirection, MouthShape
from core.event_bus import EventBus, EventType, emit_event

logger = logging.getLogger(__name__)


class AnimationState(Enum):
    """States of the animation system."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class AnimationConfig:
    """Configuration for animation engine."""
    # Display paths
    displays_dir: str = "data/displays"
    eyes_dir: str = "data/displays/eyes"
    mouth_dir: str = "data/displays/mouth"
    
    # Display settings
    resolution: tuple = (800, 480)
    
    # Display mapping (which physical display shows which animation)
    # For dual_display mode: display_0 = eyes, display_1 = mouth
    # For single_display mode: display_0 = combined (both eyes and mouth)
    eyes_display_index: int = 0  # Which display shows eyes
    mouth_display_index: int = 1  # Which display shows mouth
    
    # Timing settings
    blink_interval: float = 3.0  # Seconds between blinks
    gaze_shift_interval: float = 2.0  # Seconds between gaze shifts
    mouth_update_rate: float = 1.0 / 30.0  # 30 FPS for mouth updates
    
    # Emotion settings
    default_emotion: EmotionType = EmotionType.NEUTRAL
    processing_emotion: EmotionType = EmotionType.FOCUSED
    speaking_emotion: EmotionType = EmotionType.HAPPY
    error_emotion: EmotionType = EmotionType.CONFUSED


class AnimationEngine:
    """Manages visual animations and expressions based on system state.
    
    Responsibilities:
    - Load static images from data/displays/
    - Decide what to display based on state (emotion, mouth shape)
    - Render static images and pass to DisplayManager
    """
    
    def __init__(self, display_manager, config: Optional[Dict[str, Any]] = None):
        self.config = AnimationConfig(**(config or {}))
        self.display_manager = display_manager
        self.is_initialized = False
        self.current_state = AnimationState.IDLE
        self.last_blink_time = time.time()
        self.last_gaze_shift = time.time()
        
        # Image caches
        self.eyes_images: Dict[str, pygame.Surface] = {}
        self.mouth_images: Dict[str, pygame.Surface] = {}
        
        # Current display state
        self.current_emotion = EmotionType.NEUTRAL
        self.current_gaze = GazeDirection.CENTER
        self.current_mouth = MouthShape.CLOSED
        
    async def initialize(self) -> None:
        """Initialize the animation engine."""
        try:
            # Initialize pygame
            pygame.init()
            
            # Load static images
            await self._load_eyes_images()
            await self._load_mouth_images()
            
            self.is_initialized = True
            
            # Set initial state
            await self._set_animation_state(AnimationState.IDLE)
            
            logger.info("Animation engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize animation engine: {e}")
            raise
    
    async def _load_eyes_images(self) -> None:
        """Load static eye images from data/displays/eyes/."""
        eyes_dir = Path(self.config.eyes_dir)
        
        if not eyes_dir.exists():
            logger.warning(f"Eyes directory not found: {eyes_dir}")
            return
        
        # Load each emotion image
        for emotion in EmotionType:
            image_path = eyes_dir / f"{emotion.value}.png"
            if image_path.exists():
                try:
                    image = pygame.image.load(str(image_path))
                    # Scale to display resolution
                    scaled_image = pygame.transform.scale(image, self.config.resolution)
                    self.eyes_images[emotion.value] = scaled_image
                    logger.debug(f"Loaded eye image: {emotion.value}")
                except Exception as e:
                    logger.warning(f"Failed to load eye image {image_path}: {e}")
            else:
                logger.warning(f"Eye image not found: {image_path}")
        
        logger.info(f"Loaded {len(self.eyes_images)} eye images")
    
    async def _load_mouth_images(self) -> None:
        """Load static mouth images from data/displays/mouth/."""
        mouth_dir = Path(self.config.mouth_dir)
        
        if not mouth_dir.exists():
            logger.warning(f"Mouth directory not found: {mouth_dir}")
            return
        
        # Load each mouth shape image
        for shape in MouthShape:
            image_path = mouth_dir / f"{shape.value}.png"
            if image_path.exists():
                try:
                    image = pygame.image.load(str(image_path))
                    # Scale to display resolution
                    scaled_image = pygame.transform.scale(image, self.config.resolution)
                    self.mouth_images[shape.value] = scaled_image
                    logger.debug(f"Loaded mouth image: {shape.value}")
                except Exception as e:
                    logger.warning(f"Failed to load mouth image {image_path}: {e}")
            else:
                logger.warning(f"Mouth image not found: {image_path}")
        
        logger.info(f"Loaded {len(self.mouth_images)} mouth images")
    
    async def _set_animation_state(self, new_state: AnimationState) -> None:
        """Update the animation state."""
        if self.current_state == new_state:
            return
        
        self.current_state = new_state
        logger.debug(f"Animation state changed to: {new_state.value}")
        
        # Update display based on new state
        if new_state == AnimationState.IDLE:
            await self._render_idle()
        elif new_state == AnimationState.LISTENING:
            await self._render_listening()
        elif new_state == AnimationState.PROCESSING:
            await self._render_processing()
        elif new_state == AnimationState.SPEAKING:
            await self._render_speaking()
        elif new_state == AnimationState.ERROR:
            await self._render_error()
    
    async def _render_idle(self) -> None:
        """Render idle state."""
        await self._update_eyes(self.config.default_emotion, GazeDirection.CENTER)
        await self._update_mouth(MouthShape.CLOSED)
    
    async def _render_listening(self) -> None:
        """Render listening state."""
        await self._update_eyes(EmotionType.CURIOUS, GazeDirection.CENTER)
        await self._update_mouth(MouthShape.CLOSED)
    
    async def _render_processing(self) -> None:
        """Render processing state."""
        await self._update_eyes(self.config.processing_emotion, GazeDirection.CENTER)
        await self._update_mouth(MouthShape.CLOSED)
    
    async def _render_speaking(self) -> None:
        """Render speaking state."""
        await self._update_eyes(self.config.speaking_emotion, GazeDirection.CENTER)
        await self._update_mouth(MouthShape.SMILE)
    
    async def _render_error(self) -> None:
        """Render error state."""
        await self._update_eyes(self.config.error_emotion, GazeDirection.CENTER)
        await self._update_mouth(MouthShape.FROWN)
    
    async def _update_eyes(self, emotion: EmotionType, gaze: GazeDirection = GazeDirection.CENTER) -> None:
        """Update eyes display by rendering static image and passing to display manager."""
        self.current_emotion = emotion
        self.current_gaze = gaze
        
        # Get the static image for this emotion
        eyes_surface = self.eyes_images.get(emotion.value)
        
        if eyes_surface is None:
            logger.warning(f"No eye image found for emotion: {emotion.value}")
            return
        
        # TODO: Apply gaze transformation in the future
        # For now, just pass the static image as-is
        
        # Pass rendered frame to display manager (generic API)
        self.display_manager.display_frame(self.config.eyes_display_index, eyes_surface)
        logger.debug(f"Updated eyes: {emotion.value}, {gaze.value} on display {self.config.eyes_display_index}")
    
    async def _update_mouth(self, shape: MouthShape) -> None:
        """Update mouth display by rendering static image and passing to display manager."""
        self.current_mouth = shape
        
        # Get the static image for this mouth shape
        mouth_surface = self.mouth_images.get(shape.value)
        
        if mouth_surface is None:
            logger.warning(f"No mouth image found for shape: {shape.value}")
            return
        
        # Pass rendered frame to display manager (generic API)
        self.display_manager.display_frame(self.config.mouth_display_index, mouth_surface)
        logger.debug(f"Updated mouth: {shape.value} on display {self.config.mouth_display_index}")
    
    async def on_wake_word_detected(self) -> None:
        """Handle wake word detection."""
        await self._set_animation_state(AnimationState.LISTENING)
    
    async def on_speech_detected(self) -> None:
        """Handle speech detection - enter processing state."""
        await self._set_animation_state(AnimationState.PROCESSING)
    
    async def on_speech_processed(self, success: bool) -> None:
        """Handle speech processing completion."""
        if success:
            await self._set_animation_state(AnimationState.SPEAKING)
        else:
            await self._set_animation_state(AnimationState.ERROR)
    
    async def on_tts_started(self) -> None:
        """Handle TTS start - animate mouth for lip-sync."""
        await self._set_animation_state(AnimationState.SPEAKING)
    
    async def on_tts_completed(self) -> None:
        """Handle TTS completion - return to idle."""
        await self._set_animation_state(AnimationState.IDLE)
    
    async def on_error_occurred(self) -> None:
        """Handle error state."""
        await self._set_animation_state(AnimationState.ERROR)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current animation state."""
        return {
            "is_initialized": self.is_initialized,
            "current_state": self.current_state.value,
            "blink_interval": self.config.blink_interval,
            "gaze_shift_interval": self.config.gaze_shift_interval
        }
    
    def cleanup(self) -> None:
        """Cleanup animation engine."""
        self.is_initialized = False
        self.current_state = AnimationState.IDLE
        logger.info("Animation engine cleaned up")


# Global instance
_animation_engine: Optional[AnimationEngine] = None


async def get_animation_engine(display_manager, config: Optional[Dict[str, Any]] = None) -> AnimationEngine:
    """Get or create the global animation engine instance."""
    global _animation_engine
    if _animation_engine is None:
        _animation_engine = AnimationEngine(display_manager, config)
        await _animation_engine.initialize()
    return _animation_engine


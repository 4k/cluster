"""
Animation Engine - Coordinates animation state and events.

The AnimationEngine manages the high-level animation state machine
and emits events that the DisplayDecisionModule uses to update
the face state.
"""

import asyncio
import logging
from typing import Optional, Any

from core.event_bus import EventBus, EventType
from core.types import EmotionType, AnimationState, GazeDirection

logger = logging.getLogger(__name__)


class AnimationEngine:
    """Coordinates animation state for the display system.

    The AnimationEngine:
    - Manages the high-level animation state machine (idle, listening, etc.)
    - Emits events for state changes
    - Provides methods for external control (wake word, speech, etc.)
    - Optionally interfaces directly with DisplayManager

    State Machine:
        IDLE -> (wake word) -> LISTENING
        LISTENING -> (speech detected) -> PROCESSING
        PROCESSING -> (response ready) -> SPEAKING
        SPEAKING -> (TTS complete) -> IDLE
        Any state -> (error) -> ERROR -> IDLE
    """

    def __init__(self, event_bus: EventBus, display_manager=None):
        """Initialize the animation engine.

        Args:
            event_bus: Event bus for emitting state changes
            display_manager: Optional display manager for direct control
        """
        self.event_bus = event_bus
        self.display_manager = display_manager

        # Current state
        self.state = AnimationState.IDLE
        self.emotion = EmotionType.NEUTRAL

        # State timing
        self._state_start_time = 0.0
        self._error_duration = 2.0  # Seconds to show error state

    async def initialize(self) -> None:
        """Initialize the animation engine."""
        logger.info("Initializing AnimationEngine")

        # Set initial state
        await self._set_state(AnimationState.IDLE)
        await self._set_emotion(EmotionType.NEUTRAL)

        logger.info("AnimationEngine initialized")

    async def _set_state(self, state: AnimationState) -> None:
        """Set the animation state and emit event.

        Args:
            state: New animation state
        """
        old_state = self.state
        self.state = state

        # Emit state change event
        await self.event_bus.emit(
            EventType.EXPRESSION_CHANGE,
            {
                "state": state.value,
                "previous_state": old_state.value,
            },
            source="animation_engine"
        )

        # Update display manager directly if available
        if self.display_manager:
            self.display_manager.set_animation_state(state)

        logger.debug(f"Animation state: {old_state.value} -> {state.value}")

    async def _set_emotion(self, emotion: EmotionType) -> None:
        """Set the emotion and emit event.

        Args:
            emotion: New emotion
        """
        old_emotion = self.emotion
        self.emotion = emotion

        # Emit emotion change event
        await self.event_bus.emit(
            EventType.EMOTION_CHANGED,
            {
                "emotion": emotion.value,
                "previous_emotion": old_emotion.value,
            },
            source="animation_engine"
        )

        # Update display manager directly if available
        if self.display_manager:
            self.display_manager.set_emotion(emotion)

        logger.debug(f"Emotion: {old_emotion.value} -> {emotion.value}")

    async def _set_gaze(self, direction: GazeDirection) -> None:
        """Set the gaze direction and emit event.

        Args:
            direction: New gaze direction
        """
        await self.event_bus.emit(
            EventType.GAZE_UPDATE,
            {"direction": direction.value},
            source="animation_engine"
        )

        if self.display_manager:
            self.display_manager.set_gaze(direction)

    async def _trigger_blink(self) -> None:
        """Trigger an eye blink."""
        await self.event_bus.emit(
            EventType.BLINK_TRIGGERED,
            {},
            source="animation_engine"
        )

        if self.display_manager:
            self.display_manager.trigger_blink()

    # Public methods called by main.py

    async def on_wake_word_detected(self) -> None:
        """Handle wake word detection - transition to listening state."""
        logger.info("Animation: Wake word detected")

        await self._set_state(AnimationState.LISTENING)
        await self._set_emotion(EmotionType.LISTENING)
        await self._set_gaze(GazeDirection.FORWARD)

        # Quick blink to acknowledge
        await self._trigger_blink()

    async def on_speech_detected(self) -> None:
        """Handle speech detection - transition to processing state."""
        logger.info("Animation: Speech detected")

        await self._set_state(AnimationState.PROCESSING)
        await self._set_emotion(EmotionType.THINKING)

        # Look up-left when thinking
        await self._set_gaze(GazeDirection.UP_LEFT)

    async def on_speech_processed(self, success: bool) -> None:
        """Handle speech processing completion.

        Args:
            success: Whether processing was successful
        """
        if success:
            logger.info("Animation: Speech processed successfully")
            # State will transition to SPEAKING when TTS starts
        else:
            logger.info("Animation: Speech processing failed")
            await self._set_state(AnimationState.IDLE)
            await self._set_emotion(EmotionType.CONFUSED)

            # Return to neutral after a moment
            await asyncio.sleep(1.5)
            await self._set_emotion(EmotionType.NEUTRAL)

    async def on_tts_started(self) -> None:
        """Handle TTS start - transition to speaking state."""
        logger.info("Animation: TTS started")

        await self._set_state(AnimationState.SPEAKING)
        await self._set_emotion(EmotionType.SPEAKING)
        await self._set_gaze(GazeDirection.FORWARD)

    async def on_tts_completed(self) -> None:
        """Handle TTS completion - return to idle state."""
        logger.info("Animation: TTS completed")

        await self._set_state(AnimationState.IDLE)
        await self._set_emotion(EmotionType.NEUTRAL)
        await self._set_gaze(GazeDirection.FORWARD)

    async def on_error_occurred(self) -> None:
        """Handle error - show error state temporarily."""
        logger.info("Animation: Error occurred")

        await self._set_state(AnimationState.ERROR)
        await self._set_emotion(EmotionType.ERROR)

        # Return to idle after error duration
        await asyncio.sleep(self._error_duration)

        if self.state == AnimationState.ERROR:
            await self._set_state(AnimationState.IDLE)
            await self._set_emotion(EmotionType.NEUTRAL)

    # Additional control methods

    async def set_idle(self) -> None:
        """Set to idle state."""
        await self._set_state(AnimationState.IDLE)
        await self._set_emotion(EmotionType.NEUTRAL)
        await self._set_gaze(GazeDirection.FORWARD)

    async def set_sleeping(self) -> None:
        """Set to sleeping state."""
        await self._set_state(AnimationState.SLEEPING)
        await self._set_emotion(EmotionType.NEUTRAL)

    async def express_emotion(self, emotion: EmotionType, duration: float = 0.0) -> None:
        """Express an emotion, optionally for a duration.

        Args:
            emotion: Emotion to express
            duration: If > 0, return to neutral after this many seconds
        """
        await self._set_emotion(emotion)

        if duration > 0:
            await asyncio.sleep(duration)
            await self._set_emotion(EmotionType.NEUTRAL)

    async def look_at(self, direction: GazeDirection) -> None:
        """Look in a direction.

        Args:
            direction: Direction to look
        """
        await self._set_gaze(direction)

    async def blink(self) -> None:
        """Trigger a blink."""
        await self._trigger_blink()

    def get_state(self) -> dict:
        """Get current animation state.

        Returns:
            Dictionary with current state info
        """
        return {
            "state": self.state.value,
            "emotion": self.emotion.value,
        }

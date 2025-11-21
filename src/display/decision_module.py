"""
Display Decision Module - Centralized content routing for the display system.

This module manages what content is displayed in which window by:
1. Maintaining the current face/animation state
2. Generating content updates for each content type
3. Publishing updates via the event bus for subscribed windows

Windows subscribe to specific ContentTypes and receive updates
through the event bus.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Callable, Any

from core.event_bus import EventBus, EventType, Event
from core.types import (
    ContentType, EmotionType, AnimationState, GazeDirection, MouthShape,
    EyeState, MouthState, FaceState,
    EyesContentUpdate, MouthContentUpdate, FullFaceContentUpdate, StatusContentUpdate
)

logger = logging.getLogger(__name__)


# Custom event type for content updates (extend EventType in event_bus.py)
CONTENT_UPDATE_EVENT = "content_update"


class DisplayDecisionModule:
    """Centralized decision module for display content routing.

    This module:
    - Maintains the current face/animation state
    - Listens to relevant events (emotion changes, TTS, etc.)
    - Generates and publishes content updates for each ContentType
    - Allows windows to subscribe to specific content types

    The module uses the event bus to publish updates, allowing
    any number of windows to subscribe to any content type.
    """

    def __init__(self, event_bus: EventBus):
        """Initialize the decision module.

        Args:
            event_bus: The event bus for pub/sub communication
        """
        self.event_bus = event_bus

        # Current state
        self.face_state = FaceState()
        self.animation_state = AnimationState.IDLE

        # Content subscribers (callback functions by content type)
        self._subscribers: Dict[ContentType, list] = {ct: [] for ct in ContentType}

        # Update timing
        self._last_update = time.time()
        self._update_interval = 1.0 / 60  # 60 FPS updates

        # Automatic behaviors
        self._blink_timer = 0.0
        self._blink_interval = 3.5  # Seconds between auto-blinks
        self._gaze_timer = 0.0
        self._gaze_interval = 4.0  # Seconds between random gaze changes
        self._idle_movement_timer = 0.0

        # Running state
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the decision module and subscribe to events."""
        logger.info("Initializing DisplayDecisionModule")

        # Subscribe to relevant events
        self._setup_event_handlers()

        logger.info("DisplayDecisionModule initialized")

    def _setup_event_handlers(self) -> None:
        """Subscribe to events that affect display state."""
        # Emotion and expression events
        self.event_bus.subscribe(EventType.EMOTION_CHANGED, self._on_emotion_changed)
        self.event_bus.subscribe(EventType.EXPRESSION_CHANGE, self._on_expression_change)

        # Animation events
        self.event_bus.subscribe(EventType.GAZE_UPDATE, self._on_gaze_update)
        self.event_bus.subscribe(EventType.MOUTH_SHAPE_UPDATE, self._on_mouth_shape_update)
        self.event_bus.subscribe(EventType.BLINK_TRIGGERED, self._on_blink_triggered)

        # TTS events for lip-sync
        self.event_bus.subscribe(EventType.TTS_STARTED, self._on_tts_started)
        self.event_bus.subscribe(EventType.TTS_COMPLETED, self._on_tts_completed)
        self.event_bus.subscribe(EventType.PHONEME_EVENT, self._on_phoneme_event)

        # Audio events for listening state
        self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word)
        self.event_bus.subscribe(EventType.SPEECH_DETECTED, self._on_speech_detected)
        self.event_bus.subscribe(EventType.SPEECH_ENDED, self._on_speech_ended)

        # System events
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)

    async def start(self) -> None:
        """Start the update loop."""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("DisplayDecisionModule started")

    async def stop(self) -> None:
        """Stop the update loop."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("DisplayDecisionModule stopped")

    def subscribe(self, content_type: ContentType, callback: Callable) -> None:
        """Subscribe to content updates for a specific type.

        Args:
            content_type: The type of content to subscribe to
            callback: Function called with (content_update, dt) on each update
        """
        if callback not in self._subscribers[content_type]:
            self._subscribers[content_type].append(callback)
            logger.debug(f"Added subscriber for {content_type.value}")

    def unsubscribe(self, content_type: ContentType, callback: Callable) -> None:
        """Unsubscribe from content updates.

        Args:
            content_type: The content type to unsubscribe from
            callback: The callback to remove
        """
        if callback in self._subscribers[content_type]:
            self._subscribers[content_type].remove(callback)
            logger.debug(f"Removed subscriber for {content_type.value}")

    async def _update_loop(self) -> None:
        """Main update loop for generating content updates."""
        while self._running:
            try:
                current_time = time.time()
                dt = current_time - self._last_update
                self._last_update = current_time

                # Update automatic behaviors
                self._update_automatic_behaviors(dt)

                # Generate and publish content updates
                await self._publish_content_updates(dt)

                # Wait for next frame
                await asyncio.sleep(self._update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(0.1)

    def _update_automatic_behaviors(self, dt: float) -> None:
        """Update automatic behaviors like blinking and gaze shifts."""
        # Auto-blink
        self._blink_timer += dt
        if self._blink_timer >= self._blink_interval:
            self._blink_timer = 0.0
            self._trigger_blink()

        # Random gaze shifts when idle
        if self.animation_state == AnimationState.IDLE:
            self._gaze_timer += dt
            if self._gaze_timer >= self._gaze_interval:
                self._gaze_timer = 0.0
                self._random_gaze_shift()

        # Idle mouth movements (subtle)
        if self.animation_state == AnimationState.IDLE:
            self._idle_movement_timer += dt

    def _trigger_blink(self) -> None:
        """Trigger an eye blink."""
        self.face_state.eyes.is_blinking = True
        self.face_state.eyes.blink_progress = 0.0
        # Blink animation is handled in the renderer

    def _random_gaze_shift(self) -> None:
        """Shift gaze to a random direction."""
        import random
        directions = list(GazeDirection)
        # Weight toward forward
        weights = [3 if d == GazeDirection.FORWARD else 1 for d in directions]
        new_gaze = random.choices(directions, weights=weights)[0]
        self.face_state.eyes.gaze = new_gaze

    async def _publish_content_updates(self, dt: float) -> None:
        """Generate and publish content updates for all content types."""
        timestamp = time.time()

        # Update blink animation progress
        if self.face_state.eyes.is_blinking:
            self.face_state.eyes.blink_progress += dt / 0.15  # 150ms blink
            if self.face_state.eyes.blink_progress >= 1.0:
                self.face_state.eyes.is_blinking = False
                self.face_state.eyes.blink_progress = 0.0

        # Generate eyes update
        eyes_update = EyesContentUpdate(
            timestamp=timestamp,
            state=EyeState(
                gaze=self.face_state.eyes.gaze,
                openness=self.face_state.eyes.openness,
                pupil_dilation=self.face_state.eyes.pupil_dilation,
                is_blinking=self.face_state.eyes.is_blinking,
                blink_progress=self.face_state.eyes.blink_progress,
                emotion=self.face_state.emotion
            )
        )

        # Generate mouth update
        mouth_update = MouthContentUpdate(
            timestamp=timestamp,
            state=MouthState(
                shape=self.face_state.mouth.shape,
                openness=self.face_state.mouth.openness,
                smile_amount=self.face_state.mouth.smile_amount,
                emotion=self.face_state.emotion,
                is_speaking=self.face_state.mouth.is_speaking,
                viseme=self.face_state.mouth.viseme
            )
        )

        # Generate full face update
        face_update = FullFaceContentUpdate(
            timestamp=timestamp,
            state=FaceState(
                eyes=self.face_state.eyes,
                mouth=self.face_state.mouth,
                emotion=self.face_state.emotion,
                animation_state=self.animation_state
            )
        )

        # Generate status update
        status_update = StatusContentUpdate(
            timestamp=timestamp,
            animation_state=self.animation_state,
            emotion=self.face_state.emotion,
            message=self._get_status_message()
        )

        # Notify subscribers
        updates = {
            ContentType.EYES: eyes_update,
            ContentType.MOUTH: mouth_update,
            ContentType.FULL_FACE: face_update,
            ContentType.STATUS: status_update,
        }

        for content_type, update in updates.items():
            for callback in self._subscribers[content_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update, dt)
                    else:
                        callback(update, dt)
                except Exception as e:
                    logger.error(f"Error in content subscriber: {e}")

    def _get_status_message(self) -> str:
        """Get a status message based on current state."""
        messages = {
            AnimationState.IDLE: "Ready",
            AnimationState.LISTENING: "Listening...",
            AnimationState.PROCESSING: "Thinking...",
            AnimationState.SPEAKING: "Speaking",
            AnimationState.ERROR: "Error occurred",
            AnimationState.SLEEPING: "Sleeping",
        }
        return messages.get(self.animation_state, "")

    # Event handlers

    async def _on_emotion_changed(self, event: Event) -> None:
        """Handle emotion change events."""
        emotion_str = event.data.get('emotion', 'neutral')
        try:
            emotion = EmotionType(emotion_str)
            self.face_state.emotion = emotion
            self._apply_emotion_to_face(emotion)
            logger.debug(f"Emotion changed to: {emotion.value}")
        except ValueError:
            logger.warning(f"Unknown emotion: {emotion_str}")

    def _apply_emotion_to_face(self, emotion: EmotionType) -> None:
        """Apply emotion to face state (eyes and mouth)."""
        # Eye adjustments
        if emotion == EmotionType.HAPPY:
            self.face_state.eyes.openness = 0.9
            self.face_state.mouth.smile_amount = 0.6
        elif emotion == EmotionType.SAD:
            self.face_state.eyes.openness = 0.7
            self.face_state.mouth.smile_amount = -0.4
        elif emotion == EmotionType.ANGRY:
            self.face_state.eyes.openness = 0.8
            self.face_state.mouth.smile_amount = -0.3
        elif emotion == EmotionType.SURPRISED:
            self.face_state.eyes.openness = 1.0
            self.face_state.eyes.pupil_dilation = 1.2
            self.face_state.mouth.openness = 0.4
        elif emotion == EmotionType.CONFUSED:
            self.face_state.eyes.openness = 0.85
            self.face_state.mouth.smile_amount = -0.1
        elif emotion == EmotionType.INTERESTED:
            self.face_state.eyes.openness = 0.95
            self.face_state.eyes.pupil_dilation = 1.1
        elif emotion == EmotionType.THINKING:
            self.face_state.eyes.gaze = GazeDirection.UP_LEFT
            self.face_state.mouth.smile_amount = 0.0
        elif emotion == EmotionType.LISTENING:
            self.face_state.eyes.openness = 0.95
            self.face_state.mouth.openness = 0.1
        elif emotion == EmotionType.SPEAKING:
            self.face_state.mouth.is_speaking = True
        elif emotion == EmotionType.ERROR:
            self.face_state.eyes.openness = 0.7
            self.face_state.mouth.smile_amount = -0.5
        else:  # NEUTRAL
            self.face_state.eyes.openness = 1.0
            self.face_state.eyes.pupil_dilation = 1.0
            self.face_state.mouth.openness = 0.0
            self.face_state.mouth.smile_amount = 0.0

    async def _on_expression_change(self, event: Event) -> None:
        """Handle expression change events."""
        # Expression can set multiple face parameters at once
        if 'eyes' in event.data:
            eyes_data = event.data['eyes']
            if 'openness' in eyes_data:
                self.face_state.eyes.openness = eyes_data['openness']
            if 'gaze' in eyes_data:
                try:
                    self.face_state.eyes.gaze = GazeDirection(eyes_data['gaze'])
                except ValueError:
                    pass

        if 'mouth' in event.data:
            mouth_data = event.data['mouth']
            if 'openness' in mouth_data:
                self.face_state.mouth.openness = mouth_data['openness']
            if 'smile' in mouth_data:
                self.face_state.mouth.smile_amount = mouth_data['smile']

    async def _on_gaze_update(self, event: Event) -> None:
        """Handle gaze update events."""
        direction_str = event.data.get('direction', 'forward')
        try:
            direction = GazeDirection(direction_str)
            self.face_state.eyes.gaze = direction
        except ValueError:
            logger.warning(f"Unknown gaze direction: {direction_str}")

    async def _on_mouth_shape_update(self, event: Event) -> None:
        """Handle mouth shape update events."""
        shape_str = event.data.get('shape', 'closed')
        try:
            shape = MouthShape(shape_str)
            self.face_state.mouth.shape = shape
            self.face_state.mouth.viseme = shape
        except ValueError:
            logger.warning(f"Unknown mouth shape: {shape_str}")

    async def _on_blink_triggered(self, event: Event) -> None:
        """Handle blink trigger events."""
        self._trigger_blink()

    async def _on_tts_started(self, event: Event) -> None:
        """Handle TTS start - begin speaking animation."""
        self.animation_state = AnimationState.SPEAKING
        self.face_state.mouth.is_speaking = True
        self.face_state.emotion = EmotionType.SPEAKING
        logger.debug("TTS started - speaking animation")

    async def _on_tts_completed(self, event: Event) -> None:
        """Handle TTS completion - return to idle."""
        self.animation_state = AnimationState.IDLE
        self.face_state.mouth.is_speaking = False
        self.face_state.mouth.viseme = None
        self.face_state.emotion = EmotionType.NEUTRAL
        logger.debug("TTS completed - returning to idle")

    async def _on_phoneme_event(self, event: Event) -> None:
        """Handle phoneme events for lip-sync."""
        viseme_str = event.data.get('viseme')
        if viseme_str:
            try:
                viseme = MouthShape(viseme_str)
                self.face_state.mouth.viseme = viseme
            except ValueError:
                pass

    async def _on_wake_word(self, event: Event) -> None:
        """Handle wake word detection."""
        self.animation_state = AnimationState.LISTENING
        self.face_state.emotion = EmotionType.LISTENING
        self.face_state.eyes.openness = 1.0
        self.face_state.eyes.pupil_dilation = 1.1
        logger.debug("Wake word detected - listening state")

    async def _on_speech_detected(self, event: Event) -> None:
        """Handle speech detection - processing state."""
        self.animation_state = AnimationState.PROCESSING
        self.face_state.emotion = EmotionType.THINKING
        self.face_state.eyes.gaze = GazeDirection.UP_LEFT
        logger.debug("Speech detected - processing state")

    async def _on_speech_ended(self, event: Event) -> None:
        """Handle speech end."""
        # Don't change state here - wait for response
        pass

    async def _on_error(self, event: Event) -> None:
        """Handle error events."""
        self.animation_state = AnimationState.ERROR
        self.face_state.emotion = EmotionType.ERROR
        logger.debug("Error state triggered")

        # Return to idle after delay
        await asyncio.sleep(2.0)
        if self.animation_state == AnimationState.ERROR:
            self.animation_state = AnimationState.IDLE
            self.face_state.emotion = EmotionType.NEUTRAL

    # Public methods for external control

    def set_emotion(self, emotion: EmotionType) -> None:
        """Set the current emotion.

        Args:
            emotion: The emotion to set
        """
        self.face_state.emotion = emotion
        self._apply_emotion_to_face(emotion)

    def set_gaze(self, direction: GazeDirection) -> None:
        """Set the gaze direction.

        Args:
            direction: The direction to look
        """
        self.face_state.eyes.gaze = direction

    def trigger_blink(self) -> None:
        """Trigger an eye blink."""
        self._trigger_blink()

    def set_animation_state(self, state: AnimationState) -> None:
        """Set the animation state.

        Args:
            state: The animation state to set
        """
        self.animation_state = state
        self.face_state.animation_state = state

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for debugging.

        Returns:
            Dictionary with current state info
        """
        return {
            "animation_state": self.animation_state.value,
            "emotion": self.face_state.emotion.value,
            "eyes": {
                "gaze": self.face_state.eyes.gaze.value,
                "openness": self.face_state.eyes.openness,
                "is_blinking": self.face_state.eyes.is_blinking,
            },
            "mouth": {
                "openness": self.face_state.mouth.openness,
                "smile": self.face_state.mouth.smile_amount,
                "is_speaking": self.face_state.mouth.is_speaking,
            },
            "subscribers": {ct.value: len(subs) for ct, subs in self._subscribers.items()},
        }

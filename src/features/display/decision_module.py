"""
DisplayDecisionModule - Central content routing for the multi-window display system.
Manages what content is displayed in which window based on subscriptions and events.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Set
from collections import defaultdict

from .settings import ContentType, WindowType

logger = logging.getLogger(__name__)


class DisplayEvent(Enum):
    """Events that trigger display updates."""
    # Eye events
    BLINK = "blink"
    GAZE_UPDATE = "gaze_update"
    PUPIL_DILATE = "pupil_dilate"
    EYE_EXPRESSION = "eye_expression"

    # Mouth events
    SPEAK_START = "speak_start"
    SPEAK_STOP = "speak_stop"
    SPEAK_TEXT = "speak_text"
    VISEME_UPDATE = "viseme_update"
    MOUTH_SHAPE = "mouth_shape"

    # Combined events
    EMOTION_CHANGE = "emotion_change"
    ANIMATION_STATE = "animation_state"

    # System events
    WINDOW_READY = "window_ready"
    WINDOW_CLOSED = "window_closed"
    SYNC_REQUEST = "sync_request"


@dataclass
class DisplayCommand:
    """Command to be sent to a window."""
    event: DisplayEvent
    content_type: ContentType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for queue serialization."""
        return {
            'event': self.event.value,
            'content_type': self.content_type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'priority': self.priority,
            'correlation_id': self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DisplayCommand':
        """Create from dictionary."""
        return cls(
            event=DisplayEvent(data['event']),
            content_type=ContentType(data['content_type']),
            data=data['data'],
            timestamp=data.get('timestamp', time.time()),
            priority=data.get('priority', 0),
            correlation_id=data.get('correlation_id')
        )


class DisplayDecisionModule:
    """
    Central decision module for routing content to windows.
    Manages subscriptions, event routing, and synchronization.
    """

    def __init__(self):
        # Window subscriptions: window_id -> set of ContentTypes
        self._subscriptions: Dict[str, Set[ContentType]] = defaultdict(set)

        # Content routing: ContentType -> list of window_ids
        self._content_routes: Dict[ContentType, List[str]] = defaultdict(list)

        # Window command queues: window_id -> queue
        self._window_queues: Dict[str, Any] = {}

        # Event handlers for internal processing
        self._event_handlers: Dict[DisplayEvent, List[Callable]] = defaultdict(list)

        # Current state for synchronization
        self._current_state: Dict[str, Any] = {
            'emotion': 'NEUTRAL',
            'animation_state': 'IDLE',
            'gaze_x': 0.5,
            'gaze_y': 0.5,
            'is_speaking': False,
            'current_viseme': 'SILENCE'
        }

        # Event history for debugging
        self._event_history: List[DisplayCommand] = []
        self._max_history = 100

        # Statistics
        self._stats = {
            'events_routed': 0,
            'events_by_type': defaultdict(int)
        }

        logger.info("DisplayDecisionModule initialized")

    def register_window(self, window_id: str, subscriptions: List[ContentType],
                       command_queue: Any) -> None:
        """
        Register a window with its subscriptions and command queue.

        Args:
            window_id: Unique identifier for the window
            subscriptions: List of content types this window subscribes to
            command_queue: Queue for sending commands to this window
        """
        self._subscriptions[window_id] = set(subscriptions)
        self._window_queues[window_id] = command_queue

        # Update content routes
        for content_type in subscriptions:
            if window_id not in self._content_routes[content_type]:
                self._content_routes[content_type].append(window_id)

        logger.info(f"Window '{window_id}' registered with subscriptions: "
                   f"{[s.value for s in subscriptions]}")

    def unregister_window(self, window_id: str) -> None:
        """
        Unregister a window.

        Args:
            window_id: Window identifier to remove
        """
        if window_id in self._subscriptions:
            # Remove from content routes
            for content_type in self._subscriptions[window_id]:
                if window_id in self._content_routes[content_type]:
                    self._content_routes[content_type].remove(window_id)

            del self._subscriptions[window_id]

        if window_id in self._window_queues:
            del self._window_queues[window_id]

        logger.info(f"Window '{window_id}' unregistered")

    def add_subscription(self, window_id: str, content_type: ContentType) -> None:
        """Add a content subscription to a window."""
        if window_id in self._subscriptions:
            self._subscriptions[window_id].add(content_type)
            if window_id not in self._content_routes[content_type]:
                self._content_routes[content_type].append(window_id)

    def remove_subscription(self, window_id: str, content_type: ContentType) -> None:
        """Remove a content subscription from a window."""
        if window_id in self._subscriptions:
            self._subscriptions[window_id].discard(content_type)
            if window_id in self._content_routes[content_type]:
                self._content_routes[content_type].remove(window_id)

    def route_event(self, event: DisplayEvent, data: Dict[str, Any],
                    priority: int = 0, correlation_id: Optional[str] = None) -> int:
        """
        Route an event to appropriate windows based on content type.

        Args:
            event: The display event type
            data: Event data
            priority: Event priority (higher = more urgent)
            correlation_id: Optional correlation ID for tracking

        Returns:
            Number of windows the event was routed to
        """
        # Determine content type from event
        content_type = self._get_content_type_for_event(event)

        # Create command
        command = DisplayCommand(
            event=event,
            content_type=content_type,
            data=data,
            priority=priority,
            correlation_id=correlation_id
        )

        # Add to history
        self._event_history.append(command)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Update current state
        self._update_state(event, data)

        # Route to subscribed windows
        routed_count = 0
        target_windows = self._content_routes.get(content_type, [])

        for window_id in target_windows:
            if window_id in self._window_queues:
                try:
                    queue = self._window_queues[window_id]
                    queue.put_nowait(command.to_dict())
                    routed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to route event to window '{window_id}': {e}")

        # Update statistics
        self._stats['events_routed'] += routed_count
        self._stats['events_by_type'][event.value] += 1

        # Call internal handlers
        for handler in self._event_handlers.get(event, []):
            try:
                handler(command)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

        return routed_count

    def broadcast_event(self, event: DisplayEvent, data: Dict[str, Any],
                       priority: int = 0) -> int:
        """
        Broadcast an event to all registered windows regardless of subscription.

        Args:
            event: The display event type
            data: Event data
            priority: Event priority

        Returns:
            Number of windows the event was sent to
        """
        content_type = self._get_content_type_for_event(event)
        command = DisplayCommand(
            event=event,
            content_type=content_type,
            data=data,
            priority=priority
        )

        sent_count = 0
        for window_id, queue in self._window_queues.items():
            try:
                queue.put_nowait(command.to_dict())
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to broadcast to window '{window_id}': {e}")

        return sent_count

    def _get_content_type_for_event(self, event: DisplayEvent) -> ContentType:
        """Map display event to content type."""
        eye_events = {
            DisplayEvent.BLINK,
            DisplayEvent.GAZE_UPDATE,
            DisplayEvent.PUPIL_DILATE,
            DisplayEvent.EYE_EXPRESSION
        }

        mouth_events = {
            DisplayEvent.SPEAK_START,
            DisplayEvent.SPEAK_STOP,
            DisplayEvent.SPEAK_TEXT,
            DisplayEvent.VISEME_UPDATE,
            DisplayEvent.MOUTH_SHAPE
        }

        if event in eye_events:
            return ContentType.EYES
        elif event in mouth_events:
            return ContentType.MOUTH
        else:
            # Combined/system events go to both
            return ContentType.FACE_FULL

    def _update_state(self, event: DisplayEvent, data: Dict[str, Any]) -> None:
        """Update internal state based on event."""
        if event == DisplayEvent.EMOTION_CHANGE:
            self._current_state['emotion'] = data.get('emotion', 'NEUTRAL')
        elif event == DisplayEvent.ANIMATION_STATE:
            self._current_state['animation_state'] = data.get('state', 'IDLE')
        elif event == DisplayEvent.GAZE_UPDATE:
            self._current_state['gaze_x'] = data.get('x', 0.5)
            self._current_state['gaze_y'] = data.get('y', 0.5)
        elif event == DisplayEvent.SPEAK_START:
            self._current_state['is_speaking'] = True
        elif event == DisplayEvent.SPEAK_STOP:
            self._current_state['is_speaking'] = False
        elif event == DisplayEvent.VISEME_UPDATE:
            self._current_state['current_viseme'] = data.get('viseme', 'SILENCE')

    def get_current_state(self) -> Dict[str, Any]:
        """Get current display state."""
        return self._current_state.copy()

    def sync_window(self, window_id: str) -> None:
        """
        Send current state to a specific window for synchronization.
        Useful when a window reconnects or needs state refresh.
        """
        if window_id not in self._window_queues:
            return

        # Send current emotion
        self.route_event(
            DisplayEvent.EMOTION_CHANGE,
            {'emotion': self._current_state['emotion']},
            priority=10
        )

        # Send current animation state
        self.route_event(
            DisplayEvent.ANIMATION_STATE,
            {'state': self._current_state['animation_state']},
            priority=10
        )

        # Send gaze if subscribed to eyes
        if ContentType.EYES in self._subscriptions.get(window_id, set()):
            self.route_event(
                DisplayEvent.GAZE_UPDATE,
                {
                    'x': self._current_state['gaze_x'],
                    'y': self._current_state['gaze_y']
                }
            )

        logger.info(f"Synchronized state for window '{window_id}'")

    def on_event(self, event: DisplayEvent, handler: Callable) -> None:
        """Register an internal event handler."""
        self._event_handlers[event].append(handler)

    def off_event(self, event: DisplayEvent, handler: Callable) -> None:
        """Unregister an internal event handler."""
        if handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'events_routed': self._stats['events_routed'],
            'events_by_type': dict(self._stats['events_by_type']),
            'registered_windows': list(self._subscriptions.keys()),
            'content_routes': {
                k.value: v for k, v in self._content_routes.items()
            }
        }

    def get_event_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent event history."""
        return [cmd.to_dict() for cmd in self._event_history[-limit:]]

    # Convenience methods for common operations

    def trigger_blink(self) -> int:
        """Trigger an eye blink."""
        return self.route_event(DisplayEvent.BLINK, {})

    def set_gaze(self, x: float, y: float) -> int:
        """Set gaze position."""
        return self.route_event(DisplayEvent.GAZE_UPDATE, {'x': x, 'y': y})

    def set_emotion(self, emotion: str) -> int:
        """Set emotion for all windows."""
        return self.broadcast_event(DisplayEvent.EMOTION_CHANGE, {'emotion': emotion})

    def set_animation_state(self, state: str) -> int:
        """Set animation state for all windows."""
        return self.broadcast_event(DisplayEvent.ANIMATION_STATE, {'state': state})

    def start_speaking(self, text: Optional[str] = None, duration: float = None) -> int:
        """Start speaking animation."""
        if text:
            data = {'text': text}
            if duration is not None:
                data['duration'] = duration
            return self.route_event(DisplayEvent.SPEAK_TEXT, data)
        else:
            return self.route_event(DisplayEvent.SPEAK_START, {})

    def stop_speaking(self) -> int:
        """Stop speaking animation."""
        return self.route_event(DisplayEvent.SPEAK_STOP, {})

    def set_viseme(self, viseme: str) -> int:
        """Set mouth viseme directly."""
        return self.route_event(DisplayEvent.VISEME_UPDATE, {'viseme': viseme})

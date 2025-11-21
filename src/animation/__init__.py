"""
Animation system for the voice assistant.

Provides the AnimationEngine that coordinates animation state
and emits events for the display system.
"""

from .engine import AnimationEngine

# Singleton instance
_animation_engine = None


async def get_animation_engine(display_manager=None) -> AnimationEngine:
    """Get or create the animation engine singleton.

    Args:
        display_manager: Optional display manager for direct control

    Returns:
        The AnimationEngine instance
    """
    global _animation_engine

    if _animation_engine is None:
        from core.event_bus import EventBus
        event_bus = await EventBus.get_instance()
        _animation_engine = AnimationEngine(event_bus, display_manager)
        await _animation_engine.initialize()

    return _animation_engine


__all__ = [
    'AnimationEngine',
    'get_animation_engine',
]

"""
Features Package - Plugin Architecture

Features are optional, pluggable modules that extend the voice assistant.
Each feature subscribes to relevant events from the event bus and provides
additional functionality.

Available features:
- display: Animated face display with lip sync

Usage:
    from src.features import FeatureLoader

    loader = FeatureLoader()
    await loader.load_features(['display'])
    await loader.initialize_all()
    await loader.start_all()
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class Feature(ABC):
    """Base class for all features."""

    name: str = "base"
    description: str = "Base feature"

    def __init__(self):
        self.is_initialized = False
        self.is_running = False
        self.event_bus = None

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the feature. Returns True if successful."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the feature."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the feature."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources. Override if needed."""
        await self.stop()


# Registry of available features
_FEATURE_REGISTRY: Dict[str, Type[Feature]] = {}


def register_feature(name: str):
    """Decorator to register a feature class."""
    def decorator(cls: Type[Feature]):
        _FEATURE_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_available_features() -> List[str]:
    """Get list of available feature names."""
    return list(_FEATURE_REGISTRY.keys())


class FeatureLoader:
    """Loads and manages features based on configuration."""

    def __init__(self):
        self.features: Dict[str, Feature] = {}
        self._discover_features()

    def _discover_features(self) -> None:
        """Discover available features by importing feature modules."""
        # Import feature modules to trigger registration
        try:
            from . import display
        except ImportError as e:
            logger.warning(f"Could not import display feature: {e}")

    def load_feature(self, name: str) -> Optional[Feature]:
        """Load a single feature by name."""
        if name in self.features:
            return self.features[name]

        if name not in _FEATURE_REGISTRY:
            logger.error(f"Unknown feature: {name}")
            return None

        feature_class = _FEATURE_REGISTRY[name]
        feature = feature_class()
        self.features[name] = feature
        logger.info(f"Loaded feature: {name}")
        return feature

    def load_features(self, names: List[str]) -> List[Feature]:
        """Load multiple features by name."""
        loaded = []
        for name in names:
            feature = self.load_feature(name)
            if feature:
                loaded.append(feature)
        return loaded

    async def initialize_all(self) -> bool:
        """Initialize all loaded features."""
        success = True
        for name, feature in self.features.items():
            try:
                if await feature.initialize():
                    logger.info(f"Feature '{name}' initialized")
                else:
                    logger.error(f"Feature '{name}' failed to initialize")
                    success = False
            except Exception as e:
                logger.error(f"Error initializing feature '{name}': {e}")
                success = False
        return success

    async def start_all(self) -> None:
        """Start all loaded features."""
        for name, feature in self.features.items():
            if feature.is_initialized:
                try:
                    await feature.start()
                    logger.info(f"Feature '{name}' started")
                except Exception as e:
                    logger.error(f"Error starting feature '{name}': {e}")

    async def stop_all(self) -> None:
        """Stop all loaded features."""
        for name, feature in reversed(list(self.features.items())):
            if feature.is_running:
                try:
                    await feature.stop()
                    logger.info(f"Feature '{name}' stopped")
                except Exception as e:
                    logger.error(f"Error stopping feature '{name}': {e}")

    async def cleanup_all(self) -> None:
        """Cleanup all loaded features."""
        for name, feature in reversed(list(self.features.items())):
            try:
                await feature.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up feature '{name}': {e}")
        self.features.clear()

    def get_feature(self, name: str) -> Optional[Feature]:
        """Get a loaded feature by name."""
        return self.features.get(name)


__all__ = [
    'Feature',
    'FeatureLoader',
    'register_feature',
    'get_available_features',
]

"""
STT Engine and Wake Word Engine Factories.

Creates engine instances based on configuration.
Implements the factory pattern for flexible backend selection.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from src.core.audio.stt_engine import STTEngine
from src.core.audio.wake_word_engine import WakeWordEngine, NoneWakeWordEngine

from .engines.vosk_engine import VoskSTTEngine, VoskEngineConfig
from .wake_word_engines.openwakeword_engine import OpenWakeWordEngine, OpenWakeWordConfig

logger = logging.getLogger(__name__)


# Registry of available STT engines
STT_ENGINE_REGISTRY: Dict[str, Type[STTEngine]] = {
    "vosk": VoskSTTEngine,
    # Future engines:
    # "whisper": WhisperSTTEngine,
    # "azure": AzureSTTEngine,
    # "google": GoogleSTTEngine,
}

# Registry of available wake word engines
WAKE_WORD_ENGINE_REGISTRY: Dict[str, Type[WakeWordEngine]] = {
    "openwakeword": OpenWakeWordEngine,
    "none": NoneWakeWordEngine,
    # Future engines:
    # "porcupine": PorcupineEngine,
}


@dataclass
class STTConfig:
    """
    Complete STT configuration.

    Supports engine selection and per-engine configuration.
    """
    # STT engine selection
    engine: str = "vosk"

    # Wake word engine selection
    wake_word_engine: str = "openwakeword"

    # Wake word settings
    wake_word: str = "jarvis"
    threshold: float = 0.5

    # Voice activity detection
    silence_threshold_rms: int = 500
    max_recording_seconds: float = 5.0
    silence_duration_seconds: float = 1.5

    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1280

    # Device settings
    device_index: Optional[int] = None
    verbose: bool = False

    # Engine-specific configurations
    engines: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Wake word engine-specific configurations
    wake_word_engines: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "STTConfig":
        """Create from dictionary (e.g., from YAML)."""
        return cls(
            engine=config.get("engine", cls.engine),
            wake_word_engine=config.get("wake_word_engine", cls.wake_word_engine),
            wake_word=config.get("wake_word", cls.wake_word),
            threshold=config.get("threshold", cls.threshold),
            silence_threshold_rms=config.get("silence_threshold_rms", cls.silence_threshold_rms),
            max_recording_seconds=config.get("max_recording_seconds", cls.max_recording_seconds),
            silence_duration_seconds=config.get("silence_duration_seconds", cls.silence_duration_seconds),
            sample_rate=config.get("sample_rate", cls.sample_rate),
            chunk_size=config.get("chunk_size", cls.chunk_size),
            device_index=config.get("device_index"),
            verbose=config.get("verbose", cls.verbose),
            engines=config.get("engines", {}),
            wake_word_engines=config.get("wake_word_engines", {}),
        )

    @classmethod
    def load(cls) -> "STTConfig":
        """Load configuration from file and environment."""
        from src.core.service_config import load_yaml_config, apply_env_overrides

        # Try new config path first, fall back to legacy
        file_config = load_yaml_config("audio/stt")
        if not file_config:
            # Fall back to legacy config and migrate
            legacy_config = load_yaml_config("stt")
            file_config = cls._migrate_legacy_config(legacy_config)

        # Apply environment overrides
        config = apply_env_overrides(file_config, "STT")

        return cls.from_dict(config)

    @staticmethod
    def _migrate_legacy_config(legacy: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy STT config to new format."""
        if not legacy:
            return {}

        return {
            "engine": "vosk",
            "wake_word_engine": "openwakeword",
            "wake_word": legacy.get("wake_word", "jarvis"),
            "threshold": legacy.get("threshold", 0.5),
            "silence_threshold_rms": legacy.get("silence_threshold_rms", 500),
            "max_recording_seconds": legacy.get("max_recording_seconds", 5.0),
            "silence_duration_seconds": legacy.get("silence_duration_seconds", 1.5),
            "sample_rate": legacy.get("sample_rate", 16000),
            "chunk_size": legacy.get("chunk_size", 1280),
            "device_index": legacy.get("device_index"),
            "verbose": legacy.get("verbose", False),
            "engines": {
                "vosk": {
                    "model_path": legacy.get("vosk_model_path"),
                    "model_search_paths": legacy.get("vosk_model_search_paths", []),
                }
            },
            "wake_word_engines": {
                "openwakeword": {
                    "wake_word": legacy.get("wake_word", "jarvis"),
                    "threshold": legacy.get("threshold", 0.5),
                }
            },
        }


class STTEngineFactory:
    """Factory for creating STT engine instances."""

    @staticmethod
    def create(config: STTConfig) -> STTEngine:
        """
        Create an STT engine based on configuration.

        Args:
            config: STT configuration

        Returns:
            Configured STTEngine instance

        Raises:
            ValueError: If engine type is not supported
        """
        engine_name = config.engine.lower()

        if engine_name not in STT_ENGINE_REGISTRY:
            available = ", ".join(STT_ENGINE_REGISTRY.keys())
            raise ValueError(
                f"Unknown STT engine: {engine_name}. "
                f"Available engines: {available}"
            )

        engine_class = STT_ENGINE_REGISTRY[engine_name]
        engine_config = config.engines.get(engine_name, {})

        logger.info(f"Creating STT engine: {engine_name}")

        if engine_name == "vosk":
            vosk_config = VoskEngineConfig.from_dict(engine_config)
            return VoskSTTEngine(vosk_config)

        # Generic fallback
        return engine_class()

    @staticmethod
    def get_available_engines() -> Dict[str, bool]:
        """Get list of available engines and their status."""
        result = {}
        for name, engine_class in STT_ENGINE_REGISTRY.items():
            try:
                engine = engine_class()
                result[name] = engine.is_available()
            except Exception:
                result[name] = False
        return result


class WakeWordEngineFactory:
    """Factory for creating wake word engine instances."""

    @staticmethod
    def create(config: STTConfig) -> WakeWordEngine:
        """
        Create a wake word engine based on configuration.

        Args:
            config: STT configuration

        Returns:
            Configured WakeWordEngine instance

        Raises:
            ValueError: If engine type is not supported
        """
        engine_name = config.wake_word_engine.lower()

        if engine_name not in WAKE_WORD_ENGINE_REGISTRY:
            available = ", ".join(WAKE_WORD_ENGINE_REGISTRY.keys())
            raise ValueError(
                f"Unknown wake word engine: {engine_name}. "
                f"Available engines: {available}"
            )

        engine_class = WAKE_WORD_ENGINE_REGISTRY[engine_name]
        engine_config = config.wake_word_engines.get(engine_name, {})

        # Merge global wake word settings into engine config
        if "wake_word" not in engine_config:
            engine_config["wake_word"] = config.wake_word
        if "threshold" not in engine_config:
            engine_config["threshold"] = config.threshold

        logger.info(f"Creating wake word engine: {engine_name}")

        if engine_name == "openwakeword":
            oww_config = OpenWakeWordConfig.from_dict(engine_config)
            return OpenWakeWordEngine(oww_config)
        elif engine_name == "none":
            return NoneWakeWordEngine()

        # Generic fallback
        return engine_class()

    @staticmethod
    def get_available_engines() -> Dict[str, bool]:
        """Get list of available engines and their status."""
        result = {}
        for name, engine_class in WAKE_WORD_ENGINE_REGISTRY.items():
            try:
                engine = engine_class()
                result[name] = engine.is_available()
            except Exception:
                result[name] = False
        return result

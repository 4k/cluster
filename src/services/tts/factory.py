"""
TTS Engine and Viseme Provider Factories.

Creates engine and provider instances based on configuration.
Implements the factory pattern for flexible backend selection.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from src.core.audio.tts_engine import TTSEngine
from src.core.audio.viseme_provider import VisemeProvider, NoneVisemeProvider

from .engines.piper_engine import PiperTTSEngine, PiperEngineConfig
from .viseme_providers.rhubarb_provider import RhubarbVisemeProvider, RhubarbProviderConfig
from .viseme_providers.text_based_provider import TextBasedVisemeProvider, TextBasedProviderConfig

logger = logging.getLogger(__name__)


# Registry of available TTS engines
TTS_ENGINE_REGISTRY: Dict[str, Type[TTSEngine]] = {
    "piper": PiperTTSEngine,
    # Future engines:
    # "azure": AzureTTSEngine,
    # "openai": OpenAITTSEngine,
    # "elevenlabs": ElevenLabsTTSEngine,
}

# Registry of available viseme providers
VISEME_PROVIDER_REGISTRY: Dict[str, Type[VisemeProvider]] = {
    "rhubarb": RhubarbVisemeProvider,
    "text_based": TextBasedVisemeProvider,
    "none": NoneVisemeProvider,
}


@dataclass
class TTSConfig:
    """
    Complete TTS configuration.

    Supports engine selection and per-engine configuration.
    """
    # Engine selection
    engine: str = "piper"

    # Viseme provider selection
    viseme_provider: str = "rhubarb"

    # Whether to wait for visemes before audio playback
    wait_for_visemes: bool = True

    # Fallback provider if primary fails
    viseme_fallback: str = "text_based"

    # Audio settings
    keep_audio_files: bool = True
    max_audio_files: int = 10
    audio_output_dir: Optional[str] = None

    # Engine-specific configurations
    engines: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Viseme provider-specific configurations
    viseme_providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TTSConfig":
        """Create from dictionary (e.g., from YAML)."""
        return cls(
            engine=config.get("engine", cls.engine),
            viseme_provider=config.get("viseme_provider", cls.viseme_provider),
            wait_for_visemes=config.get("wait_for_visemes", cls.wait_for_visemes),
            viseme_fallback=config.get("viseme_fallback", cls.viseme_fallback),
            keep_audio_files=config.get("keep_audio_files", cls.keep_audio_files),
            max_audio_files=config.get("max_audio_files", cls.max_audio_files),
            audio_output_dir=config.get("audio_output_dir"),
            engines=config.get("engines", {}),
            viseme_providers=config.get("viseme_providers", {}),
        )

    @classmethod
    def load(cls) -> "TTSConfig":
        """Load configuration from file and environment."""
        from src.core.service_config import load_yaml_config, apply_env_overrides

        # Try new config path first, fall back to legacy
        file_config = load_yaml_config("audio/tts")
        if not file_config:
            # Fall back to legacy config and migrate
            legacy_config = load_yaml_config("tts")
            file_config = cls._migrate_legacy_config(legacy_config)

        # Apply environment overrides
        config = apply_env_overrides(file_config, "TTS")

        return cls.from_dict(config)

    @staticmethod
    def _migrate_legacy_config(legacy: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy TTS config to new format."""
        if not legacy:
            return {}

        return {
            "engine": "piper",  # Legacy only supported Piper
            "viseme_provider": "rhubarb",
            "wait_for_visemes": True,
            "keep_audio_files": legacy.get("keep_audio_files", True),
            "max_audio_files": legacy.get("max_audio_files", 10),
            "audio_output_dir": legacy.get("audio_output_dir"),
            "engines": {
                "piper": {
                    "model_path": legacy.get("model_path"),
                    "model_search_dir": legacy.get("model_search_dir", "~/.local/share/piper/voices"),
                }
            },
            "viseme_providers": {
                "rhubarb": {
                    "recognizer": "pocketSphinx",
                    "extended_shapes": True,
                }
            },
        }


class TTSEngineFactory:
    """
    Factory for creating TTS engine instances.

    Uses configuration to instantiate the appropriate engine
    with the correct settings.
    """

    @staticmethod
    def create(config: TTSConfig) -> TTSEngine:
        """
        Create a TTS engine based on configuration.

        Args:
            config: TTS configuration

        Returns:
            Configured TTSEngine instance

        Raises:
            ValueError: If engine type is not supported
        """
        engine_name = config.engine.lower()

        if engine_name not in TTS_ENGINE_REGISTRY:
            available = ", ".join(TTS_ENGINE_REGISTRY.keys())
            raise ValueError(
                f"Unknown TTS engine: {engine_name}. "
                f"Available engines: {available}"
            )

        engine_class = TTS_ENGINE_REGISTRY[engine_name]
        engine_config = config.engines.get(engine_name, {})

        logger.info(f"Creating TTS engine: {engine_name}")

        # Create engine with appropriate config
        if engine_name == "piper":
            piper_config = PiperEngineConfig.from_dict(engine_config)
            return PiperTTSEngine(piper_config)

        # Future engines would be handled here
        # elif engine_name == "azure":
        #     azure_config = AzureEngineConfig.from_dict(engine_config)
        #     return AzureTTSEngine(azure_config)

        # Generic fallback (shouldn't reach here if registry is correct)
        return engine_class()

    @staticmethod
    def get_available_engines() -> Dict[str, bool]:
        """
        Get list of available engines and their status.

        Returns:
            Dictionary of engine name -> is_available
        """
        result = {}
        for name, engine_class in TTS_ENGINE_REGISTRY.items():
            try:
                engine = engine_class()
                result[name] = engine.is_available()
            except Exception:
                result[name] = False
        return result


class VisemeProviderFactory:
    """
    Factory for creating viseme provider instances.

    Uses configuration to instantiate the appropriate provider
    with the correct settings.
    """

    @staticmethod
    def create(config: TTSConfig) -> VisemeProvider:
        """
        Create a viseme provider based on configuration.

        Args:
            config: TTS configuration

        Returns:
            Configured VisemeProvider instance

        Raises:
            ValueError: If provider type is not supported
        """
        provider_name = config.viseme_provider.lower()

        if provider_name not in VISEME_PROVIDER_REGISTRY:
            available = ", ".join(VISEME_PROVIDER_REGISTRY.keys())
            raise ValueError(
                f"Unknown viseme provider: {provider_name}. "
                f"Available providers: {available}"
            )

        provider_class = VISEME_PROVIDER_REGISTRY[provider_name]
        provider_config = config.viseme_providers.get(provider_name, {})

        logger.info(f"Creating viseme provider: {provider_name}")

        # Create provider with appropriate config
        if provider_name == "rhubarb":
            rhubarb_config = RhubarbProviderConfig.from_dict(provider_config)
            return RhubarbVisemeProvider(rhubarb_config)
        elif provider_name == "text_based":
            text_config = TextBasedProviderConfig.from_dict(provider_config)
            return TextBasedVisemeProvider(text_config)
        elif provider_name == "none":
            return NoneVisemeProvider()

        # Generic fallback
        return provider_class()

    @staticmethod
    def create_fallback(config: TTSConfig) -> Optional[VisemeProvider]:
        """
        Create a fallback viseme provider.

        Args:
            config: TTS configuration

        Returns:
            Fallback VisemeProvider or None if no fallback configured
        """
        if not config.viseme_fallback:
            return None

        fallback_name = config.viseme_fallback.lower()
        if fallback_name not in VISEME_PROVIDER_REGISTRY:
            logger.warning(f"Unknown fallback provider: {fallback_name}")
            return None

        fallback_config = config.viseme_providers.get(fallback_name, {})

        if fallback_name == "text_based":
            text_config = TextBasedProviderConfig.from_dict(fallback_config)
            return TextBasedVisemeProvider(text_config)
        elif fallback_name == "none":
            return NoneVisemeProvider()

        return VISEME_PROVIDER_REGISTRY[fallback_name]()

    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """
        Get list of available providers and their status.

        Returns:
            Dictionary of provider name -> is_available
        """
        result = {}
        for name, provider_class in VISEME_PROVIDER_REGISTRY.items():
            try:
                provider = provider_class()
                result[name] = provider.is_available()
            except Exception:
                result[name] = False
        return result

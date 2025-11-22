"""
Service Configuration Loader.

Provides utilities for loading service-specific configuration from YAML files.
Follows best practices:
- Separate config file per service/feature
- Config files in config/ directory
- Merges file config with code defaults
- Environment variables can override config file values
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Type
from dataclasses import dataclass, field, fields, is_dataclass
import yaml

logger = logging.getLogger(__name__)

# Type variable for generic config loading
T = TypeVar('T')


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    # Check environment variable first
    config_dir = os.getenv('CONFIG_DIR', 'config')

    # Make path absolute if relative
    config_path = Path(config_dir)
    if not config_path.is_absolute():
        # Relative to project root (parent of src/)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_dir

    return config_path


def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_name: Name of the config file (without .yaml extension)

    Returns:
        Dictionary with configuration values, empty dict if file not found
    """
    config_dir = get_config_dir()
    config_file = config_dir / f"{config_name}.yaml"

    if not config_file.exists():
        logger.debug(f"Config file not found: {config_file}, using defaults")
        return {}

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_file}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_file}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file {config_file}: {e}")
        return {}


def merge_configs(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        defaults: Default configuration values
        overrides: Override values (from file or environment)

    Returns:
        Merged configuration dictionary
    """
    result = defaults.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def apply_env_overrides(config: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables should be named: {PREFIX}_{KEY} (uppercase)
    Nested keys use double underscore: {PREFIX}_{SECTION}__{KEY}

    Args:
        config: Configuration dictionary to update
        prefix: Environment variable prefix (e.g., 'STT', 'LLM')

    Returns:
        Updated configuration dictionary
    """
    result = config.copy()
    prefix_upper = prefix.upper()

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(f"{prefix_upper}_"):
            continue

        # Remove prefix and convert to config key
        key_part = env_key[len(prefix_upper) + 1:]

        # Handle nested keys (double underscore)
        if '__' in key_part:
            parts = key_part.lower().split('__')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = _parse_env_value(env_value)
        else:
            result[key_part.lower()] = _parse_env_value(env_value)

    return result


def _parse_env_value(value: str) -> Any:
    """Parse an environment variable value to the appropriate type."""
    # Boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'

    # None/null
    if value.lower() in ('null', 'none', ''):
        return None

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # List (comma-separated)
    if ',' in value and not value.startswith('['):
        return [v.strip() for v in value.split(',')]

    # String
    return value


@dataclass
class STTServiceConfig:
    """Configuration for the Speech-to-Text service."""
    # Wake word settings
    wake_word: str = "jarvis"
    threshold: float = 0.5

    # Audio settings
    chunk_size: int = 1280  # 80ms at 16kHz
    sample_rate: int = 16000

    # Vosk model settings
    vosk_model_path: Optional[str] = None
    vosk_model_search_paths: list = field(default_factory=lambda: [
        "models/vosk/vosk-model-small-en-us-0.15",
        "models/vosk/vosk-model-en-us-0.22",
        "models/vosk-model-small-en-us-0.15",
    ])

    # Voice activity detection
    silence_threshold_rms: int = 500
    max_recording_seconds: float = 5.0
    silence_duration_seconds: float = 1.5

    # Device settings
    device_index: Optional[int] = None
    verbose: bool = False

    @classmethod
    def load(cls) -> 'STTServiceConfig':
        """Load STT configuration from file and environment."""
        # Load from YAML file
        file_config = load_yaml_config('stt')

        # Apply environment overrides
        config = apply_env_overrides(file_config, 'STT')

        # Create instance with merged config
        return cls(
            wake_word=config.get('wake_word', cls.wake_word),
            threshold=config.get('threshold', cls.threshold),
            chunk_size=config.get('chunk_size', cls.chunk_size),
            sample_rate=config.get('sample_rate', cls.sample_rate),
            vosk_model_path=config.get('vosk_model_path', cls.vosk_model_path),
            vosk_model_search_paths=config.get('vosk_model_search_paths', cls.__dataclass_fields__['vosk_model_search_paths'].default_factory()),
            silence_threshold_rms=config.get('silence_threshold_rms', cls.silence_threshold_rms),
            max_recording_seconds=config.get('max_recording_seconds', cls.max_recording_seconds),
            silence_duration_seconds=config.get('silence_duration_seconds', cls.silence_duration_seconds),
            device_index=config.get('device_index', cls.device_index),
            verbose=config.get('verbose', cls.verbose),
        )


@dataclass
class LLMServiceConfig:
    """Configuration for the LLM service."""
    # Connection settings
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    api_type: str = "auto"  # auto, ollama-chat, ollama-generate, openai

    # Model settings
    default_model: str = "llama3.2:3b"
    system_prompt: str = "You are a helpful voice assistant. Provide concise, natural responses suitable for voice output."

    # Request settings
    timeout_seconds: int = 120
    stream: bool = False

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 512

    @classmethod
    def load(cls) -> 'LLMServiceConfig':
        """Load LLM configuration from file and environment."""
        # Load from YAML file
        file_config = load_yaml_config('llm')

        # Apply environment overrides
        config = apply_env_overrides(file_config, 'LLM')

        # Create instance with merged config
        return cls(
            base_url=config.get('base_url', cls.base_url),
            api_key=config.get('api_key', cls.api_key),
            api_type=config.get('api_type', cls.api_type),
            default_model=config.get('default_model', cls.default_model),
            system_prompt=config.get('system_prompt', cls.system_prompt),
            timeout_seconds=config.get('timeout_seconds', cls.timeout_seconds),
            stream=config.get('stream', cls.stream),
            temperature=config.get('temperature', cls.temperature),
            max_tokens=config.get('max_tokens', cls.max_tokens),
        )


@dataclass
class TTSServiceConfig:
    """Configuration for the Text-to-Speech service."""
    # Model settings
    model_path: Optional[str] = None
    model_search_dir: str = "~/.local/share/piper/voices"

    # Audio settings
    keep_audio_files: bool = True
    max_audio_files: int = 10
    audio_chunk_size: int = 4096
    frames_per_buffer: int = 1024

    # Output settings
    audio_output_dir: Optional[str] = None  # None = system temp dir

    @classmethod
    def load(cls) -> 'TTSServiceConfig':
        """Load TTS configuration from file and environment."""
        # Load from YAML file
        file_config = load_yaml_config('tts')

        # Apply environment overrides
        config = apply_env_overrides(file_config, 'TTS')

        # Create instance with merged config
        return cls(
            model_path=config.get('model_path', cls.model_path),
            model_search_dir=config.get('model_search_dir', cls.model_search_dir),
            keep_audio_files=config.get('keep_audio_files', cls.keep_audio_files),
            max_audio_files=config.get('max_audio_files', cls.max_audio_files),
            audio_chunk_size=config.get('audio_chunk_size', cls.audio_chunk_size),
            frames_per_buffer=config.get('frames_per_buffer', cls.frames_per_buffer),
            audio_output_dir=config.get('audio_output_dir', cls.audio_output_dir),
        )


def load_display_config() -> Dict[str, Any]:
    """
    Load display configuration from file.

    Returns a dictionary that can be used to create DisplaySettings.
    """
    # Load from YAML file
    file_config = load_yaml_config('display')

    # Apply environment overrides
    config = apply_env_overrides(file_config, 'DISPLAY')

    return config

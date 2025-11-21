"""
Configuration management for the voice assistant system.
Supports environment variable configuration only.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

from .types import (
    TTSConfig, AudioConfig, CameraConfig,
    EmotionType, GazeDirection, MouthShape
)

logger = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    """Main configuration for the voice assistant.
    
    Configuration hierarchy:
    1. .env file - Environment overrides
    2. Environment variables - Runtime overrides
    
    For LLM models:
    - Only model_id needs to be specified in env
    - All model details (repo, paths, patterns) are in models.json
    - This ensures single source of truth for model configuration
    """
    
    # Core settings
    name: str = "Voice Assistant"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    tts: TTSConfig = field(default_factory=lambda: TTSConfig(
        engine_type="piper",
        model_path="models/piper/en_US-lessac-medium/en_US-lessac-medium.onnx",
        emotion_support=False,
        phoneme_output=False
    ))
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: Dict[str, Any] = field(default_factory=dict)
    camera: CameraConfig = field(default_factory=CameraConfig)
    
    # Audio processing configurations
    vad: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model_path": "models/vad",
        "threshold": 0.5,
        "min_silence_duration": 0.5,
        "min_speech_duration": 0.3
    })
    
    wake_word: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model_path": "models/wake_word",
        "threshold": 0.5,
        "keywords": ["hey assistant", "hey pi"]
    })
    
    stt: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model_path": "models/stt",
        "confidence_threshold": 0.5,
        "language": "en"
    })

    ambient_stt: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model_path": "models/vosk-model-small-en-us-0.15",
        "confidence_threshold": 0.3,
        "language": "en",
        "wake_word_timeout": 5.0,
        "frame_skip": 1,
        "min_confidence": 0.3
    })
    
    # AI settings
    # Note: model_id references models.json, not a path
    llm: Dict[str, Any] = field(default_factory=lambda: {
        "provider_type": "local",
        "model_id": "llama32-1b-q6k",  # References models.json
        "temperature": 0.7,
        "max_tokens": 512,
        "context_window": 2048,
        "n_gpu_layers": 33,
        "n_threads": 8,
        "use_mmap": True,
        "use_mlock": True,
        "ngl": 33,
        "main_gpu": 0,
        "tensor_split": None
    })
    
    # Mock mode settings
    mock_llm: bool = False
    mock_tts: bool = False
    mock_stt: bool = False
    mock_ambient_stt: bool = False
    mock_display: bool = False
    mock_audio: bool = False
    
    # Conversation settings
    conversation: Dict[str, Any] = field(default_factory=lambda: {
        "max_history": 10,
        "response_threshold": 0.3,
        "enable_memory": True,
        "memory_weight": 0.2,
        "memory_config": {
            "max_tokens": 1000,
            "memory_file": "data/memory.yaml",
            "max_memory_entries": 100,
            "importance_threshold": 0.5,
            "summary_length": 1,
            "auto_save": True,
            "save_interval": 300.0
        }
    })
    
    # Development settings
    development: Dict[str, Any] = field(default_factory=lambda: {
        "enable_debug_logging": True,
        "enable_mock_mode": False,
        "mock_llm": False,
        "mock_tts": False,
        "mock_stt": False,
        "mock_display": False,
        "mock_audio": False
    })
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    voices_dir: str = "voices"
    config_dir: str = "config"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)
        self.voices_dir = Path(self.voices_dir)
        self.config_dir = Path(self.config_dir)
        self.logs_dir = Path(self.logs_dir)
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.voices_dir, 
                        self.config_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self):
        # Load environment variables from .env file
        self._load_env_file()
        
        self.config: Optional[AssistantConfig] = None
        self._env_overrides: Dict[str, str] = {}
    
    def _load_env_file(self):
        """Load environment variables from .env file."""
        # Try to load .env file from project root
        env_file = Path.cwd() / '.env'
        
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            # Also try looking one level up (if running from src/)
            env_file_alt = Path.cwd().parent / '.env'
            if env_file_alt.exists():
                load_dotenv(env_file_alt)
                logger.info(f"Loaded environment variables from {env_file_alt}")
            else:
                logger.info("No .env file found. Using system environment variables or defaults.")
    
    def load_config(self) -> AssistantConfig:
        """Load configuration from environment variables."""
        # Start with default configuration
        config_data = {}
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Apply mock mode logic BEFORE creating config object
        config_data = self._apply_mock_mode_logic_early(config_data)
        
        # Create configuration object
        self.config = self._create_config_object(config_data)
        
        # Validate configuration
        self._validate_config(self.config)
        
        # Apply mock mode logic again (in case validation needs it)
        self._apply_mock_mode_logic(self.config)
        
        logger.info("Configuration loaded from environment variables")
        return self.config
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            "ASSISTANT_DEBUG": "debug",
            "ASSISTANT_LOG_LEVEL": "log_level",
            "ASSISTANT_NAME": "name",
            "ASSISTANT_VERSION": "version",
            "TTS_MODEL_PATH": "tts.model_path",
            "TTS_SPEED": "tts.speed",
            "TTS_PITCH": "tts.pitch",
            "TTS_VOLUME": "tts.volume",
            "TTS_ENABLE_MOCK": "tts.enable_mock",
            "AUDIO_SAMPLE_RATE": "audio.sample_rate",
            "AUDIO_CHANNELS": "audio.channels",
            "AUDIO_BUFFER_SIZE": "audio.buffer_size",
            "AUDIO_INPUT_DEVICE": "audio.input_device",
            "AUDIO_OUTPUT_DEVICE": "audio.output_device",
            "AUDIO_ENABLE_MOCK": "audio.enable_mock",
            "DISPLAY_MODE": "display.mode",
            "DISPLAY_RESOLUTION": "display.resolution",
            "DISPLAY_FPS": "display.fps",
            "DISPLAY_DEVELOPMENT_MODE": "display.development_mode",
            "DISPLAY_WINDOW_POSITIONS": "display.window_positions",
            "DISPLAY_FULLSCREEN": "display.fullscreen",
            "DISPLAY_BORDERLESS": "display.borderless",
            "DISPLAY_ALWAYS_ON_TOP": "display.always_on_top",
            "DISPLAY_LED_COUNT": "display.led_count",
            "DISPLAY_LED_PIN": "display.led_pin",
            "DISPLAY_TOUCH_ENABLED": "display.touch_enabled",
            "DISPLAY_CALIBRATION_FILE": "display.calibration_file",
            # Window-specific settings
            "DISPLAY_EYES_POSITION": "display.eyes_position",
            "DISPLAY_EYES_SIZE": "display.eyes_size",
            "DISPLAY_MOUTH_POSITION": "display.mouth_position",
            "DISPLAY_MOUTH_SIZE": "display.mouth_size",
            "DISPLAY_WINDOWS_CONFIG": "display.windows_config",
            "CAMERA_ENABLED": "camera.enabled",
            "CAMERA_DEVICE_ID": "camera.device_id",
            "CAMERA_RESOLUTION": "camera.resolution",
            "CAMERA_FPS": "camera.fps",
            # LLM configuration - model_id is the single source of truth
            "LLM_PROVIDER_TYPE": "llm.provider_type",
            "LLM_MODEL_ID": "llm.model_id",
            "LLM_TEMPERATURE": "llm.temperature",
            "LLM_MAX_TOKENS": "llm.max_tokens",
            "LLM_TOP_P": "llm.top_p",
            "LLM_FREQUENCY_PENALTY": "llm.frequency_penalty",
            "LLM_PRESENCE_PENALTY": "llm.presence_penalty",
            "LLM_STOP": "llm.stop",
            "LLM_STREAM": "llm.stream",
            "LLM_TIMEOUT": "llm.timeout",
            "LLM_RETRIES": "llm.retries",
            "LLM_API_KEY": "llm.api_key",
            "LLM_BASE_URL": "llm.base_url",
            "LLM_ORGANIZATION": "llm.organization",
            "LLM_CONTEXT_WINDOW": "llm.context_window",
            "LLM_GPU_LAYERS": "llm.n_gpu_layers",
            "LLM_THREADS": "llm.n_threads",
            "LLM_USE_MMAP": "llm.use_mmap",
            "LLM_USE_MLOCK": "llm.use_mlock",
            "LLM_NGL": "llm.ngl",
            "LLM_MAIN_GPU": "llm.main_gpu",
            "LLM_TENSOR_SPLIT": "llm.tensor_split",
            "LLM_ENABLE_MEMORY": "llm.enable_memory",
            "LLM_ENABLE_CHAT_LOG": "llm.enable_chat_log",
            "LLM_MEMORY_MAX_TOKENS": "llm.memory_max_tokens",
            "LLM_MEMORY_FILE": "llm.memory_file",
            "LLM_MEMORY_MAX_ENTRIES": "llm.memory_max_entries",
            "MOCK_LLM": "mock_llm",
            "MOCK_TTS": "mock_tts",
            "MOCK_STT": "mock_stt",
            "MOCK_AMBIENT_STT": "mock_ambient_stt",
            "MOCK_DISPLAY": "mock_display",
            "MOCK_AUDIO": "mock_audio",
            # Ambient STT configuration
            "AMBIENT_STT_ENABLED": "ambient_stt.enabled",
            "AMBIENT_STT_MODEL_PATH": "ambient_stt.model_path",
            "AMBIENT_STT_CONFIDENCE_THRESHOLD": "ambient_stt.confidence_threshold",
            "AMBIENT_STT_LANGUAGE": "ambient_stt.language",
            "AMBIENT_STT_WAKE_WORD_TIMEOUT": "ambient_stt.wake_word_timeout",
            "AMBIENT_STT_FRAME_SKIP": "ambient_stt.frame_skip",
            "AMBIENT_STT_MIN_CONFIDENCE": "ambient_stt.min_confidence",
            "CONVERSATION_MAX_HISTORY": "conversation.max_history",
            "CONVERSATION_RESPONSE_THRESHOLD": "conversation.response_threshold",
            "CONVERSATION_ENABLE_MEMORY": "conversation.enable_memory",
            "CONVERSATION_MEMORY_WEIGHT": "conversation.memory_weight",
            "DEVELOPMENT_ENABLE_DEBUG_LOGGING": "development.enable_debug_logging",
            "DEVELOPMENT_ENABLE_MOCK_MODE": "development.enable_mock_mode",
            "DEVELOPMENT_MOCK_LLM": "development.mock_llm",
            "DEVELOPMENT_MOCK_TTS": "development.mock_tts",
            "DEVELOPMENT_MOCK_STT": "development.mock_stt",
            "DEVELOPMENT_MOCK_DISPLAY": "development.mock_display",
            "DEVELOPMENT_MOCK_AUDIO": "development.mock_audio",
            "DATA_DIR": "data_dir",
            "MODELS_DIR": "models_dir",
            "VOICES_DIR": "voices_dir",
            "CONFIG_DIR": "config_dir",
            "LOGS_DIR": "logs_dir"
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_data, config_path, env_value)
                self._env_overrides[env_var] = env_value
        
        return config_data
    
    def _apply_mock_mode_logic_early(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mock mode logic early in the config loading process."""
        # Check individual mock flags from environment variables
        mock_llm = os.getenv('MOCK_LLM', '').lower() == 'true'
        mock_tts = os.getenv('MOCK_TTS', '').lower() == 'true'
        mock_stt = os.getenv('MOCK_STT', '').lower() == 'true'
        mock_ambient_stt = os.getenv('MOCK_AMBIENT_STT', '').lower() == 'true'
        mock_display = os.getenv('MOCK_DISPLAY', '').lower() == 'true'
        mock_audio = os.getenv('MOCK_AUDIO', '').lower() == 'true'
        
        # Also check config data (in case passed via config file)
        if not mock_llm:
            mock_llm = config_data.get('mock_llm', False)
        if not mock_tts:
            mock_tts = config_data.get('mock_tts', False)
        if not mock_stt:
            mock_stt = config_data.get('mock_stt', False)
        if not mock_ambient_stt:
            mock_ambient_stt = config_data.get('mock_ambient_stt', False)
        if not mock_display:
            mock_display = config_data.get('mock_display', False)
        if not mock_audio:
            mock_audio = config_data.get('mock_audio', False)

        logger.info(f"Mock mode check: MOCK_LLM={mock_llm}, MOCK_TTS={mock_tts}, MOCK_STT={mock_stt}, MOCK_AMBIENT_STT={mock_ambient_stt}, MOCK_DISPLAY={mock_display}, MOCK_AUDIO={mock_audio}")
        
        # Apply mock LLM if enabled
        if mock_llm:
            config_data['mock_llm'] = True
            if 'llm' not in config_data:
                config_data['llm'] = {}
            config_data['llm']['provider_type'] = 'mock'
            config_data['llm']['model'] = 'mock-llm-v1.0'
            config_data['llm']['model_id'] = 'mock-llm-v1.0'
            logger.info("MOCK_LLM=true - using mock LLM provider")
        
        # Apply mock TTS if enabled
        if mock_tts:
            config_data['mock_tts'] = True
            if 'tts' not in config_data:
                config_data['tts'] = {}
            config_data['tts']['engine_type'] = 'mock'
            config_data['tts']['model_path'] = 'mock'
            logger.info("MOCK_TTS=true - using mock TTS engine")
        
        # Apply mock STT if enabled
        if mock_stt:
            config_data['mock_stt'] = True
            logger.info("MOCK_STT=true - using mock STT")

        # Apply mock ambient STT if enabled
        if mock_ambient_stt:
            config_data['mock_ambient_stt'] = True
            logger.info("MOCK_AMBIENT_STT=true - using mock ambient STT")

        # Apply mock display if enabled
        if mock_display:
            config_data['mock_display'] = True
            logger.info("MOCK_DISPLAY=true - using mock display")
        
        # Apply mock audio if enabled
        if mock_audio:
            config_data['mock_audio'] = True
            logger.info("MOCK_AUDIO=true - using mock audio")
        
        return config_data
    
    def _apply_mock_mode_logic(self, config: AssistantConfig) -> None:
        """Apply mock mode logic based on MOCK_* flags."""
        # Override LLM provider if mock LLM is enabled
        if config.mock_llm:
            config.llm["provider_type"] = "mock"
            config.llm["model"] = "mock-llm-v1.0"
            config.llm["model_id"] = "mock-llm-v1.0"
            logger.info("Mock LLM enabled - using mock LLM provider")
        
        # Override TTS engine if mock TTS is enabled
        if config.mock_tts:
            config.tts.engine_type = "mock"
            config.tts.model_path = "mock"
            logger.info("Mock TTS enabled - using mock TTS engine")
        
        # Note: STT mock is handled in the STT creation logic
        if config.mock_stt:
            logger.info("Mock STT enabled - using mock STT")

        # Note: Ambient STT mock is handled in the ambient STT creation logic
        if config.mock_ambient_stt:
            logger.info("Mock Ambient STT enabled - using mock ambient STT")

        if config.mock_display:
            logger.info("Mock display enabled - using mock display")
        
        if config.mock_audio:
            logger.info("Mock audio enabled - using mock audio")
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in a dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.lower() == 'null':
                value = None
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.count('.') == 1 and value.replace('.', '').isdigit():
                # Only convert to float if there's exactly one dot (to avoid version strings like '1.0.0')
                value = float(value)
            elif ',' in value and not value.startswith('[') and not value.startswith('('):
                # Handle comma-separated values (like resolution)
                try:
                    # Try to parse as tuple of integers
                    parts = [int(x.strip()) for x in value.split(',')]
                    value = tuple(parts)
                except ValueError:
                    # If not integers, keep as string
                    pass
        
        current[keys[-1]] = value
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> AssistantConfig:
        """Create AssistantConfig object from dictionary data."""
        # Extract component configurations
        tts_config = TTSConfig(**config_data.get('tts', {}))
        audio_config = AudioConfig(**config_data.get('audio', {}))
        display_config = config_data.get('display', {})
        camera_config = CameraConfig(**config_data.get('camera', {}))
        
        # Create main config
        config = AssistantConfig(
            name=config_data.get('name', 'Voice Assistant'),
            version=config_data.get('version', '1.0.0'),
            debug=config_data.get('debug', False),
            log_level=config_data.get('log_level', 'INFO'),
            tts=tts_config,
            audio=audio_config,
            display=display_config,
            camera=camera_config,
            llm=config_data.get('llm', {}),
            ambient_stt=config_data.get('ambient_stt', {}),
            mock_llm=config_data.get('mock_llm', False),
            mock_tts=config_data.get('mock_tts', False),
            mock_stt=config_data.get('mock_stt', False),
            mock_ambient_stt=config_data.get('mock_ambient_stt', False),
            mock_display=config_data.get('mock_display', False),
            mock_audio=config_data.get('mock_audio', False),
            conversation=config_data.get('conversation', {}),
            development=config_data.get('development', {}),
            data_dir=config_data.get('data_dir', 'data'),
            models_dir=config_data.get('models_dir', 'models'),
            voices_dir=config_data.get('voices_dir', 'voices'),
            config_dir=config_data.get('config_dir', 'config'),
            logs_dir=config_data.get('logs_dir', 'logs')
        )
        
        return config
    
    def _validate_config(self, config: AssistantConfig) -> None:
        """Validate configuration values."""
        errors = []
        
        # Validate TTS configuration (skip for mock mode)
        if not config.mock_tts and not config.tts.model_path:
            errors.append("TTS model path is required")
        
        # Validate audio configuration
        if config.audio.sample_rate <= 0:
            errors.append("Audio sample rate must be positive")
        
        if config.audio.buffer_size <= 0:
            errors.append("Audio buffer size must be positive")
        
        # Validate display configuration
        display_resolution = config.display.get('resolution', [800, 480])
        if display_resolution[0] <= 0 or display_resolution[1] <= 0:
            errors.append("Display resolution must be positive")
        
        if config.display.get('fps', 30) <= 0:
            errors.append("Display FPS must be positive")
        
        # Validate LLM configuration (skip for mock mode)
        if not config.mock_llm and not config.llm.get('model_id'):
            errors.append("LLM model_id is required (must match a model in models.json)")
        
        if not (0 <= config.llm.get('temperature', 0.7) <= 2.0):
            errors.append("LLM temperature must be between 0 and 2")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_env_overrides(self) -> Dict[str, str]:
        """Get environment variable overrides that were applied."""
        return self._env_overrides.copy()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config() -> AssistantConfig:
    """Load the global configuration."""
    return get_config_manager().load_config()


def get_config() -> Optional[AssistantConfig]:
    """Get the current configuration."""
    return get_config_manager().config


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config_manager = ConfigManager()
    
    # Load configuration
    config = config_manager.load_config()
    
    print(f"Loaded configuration: {config.name} v{config.version}")
    print(f"TTS Engine: {config.tts.engine_type}")
    print(f"Audio Sample Rate: {config.audio.sample_rate}")
    print(f"Display Resolution: {config.display.get('resolution', 'default')}")
    print(f"Camera Enabled: {config.camera.enabled}")
    print(f"Environment Overrides: {config_manager.get_env_overrides()}")
# config.py - Configuration Management Analysis

## Overview

`config.py` provides centralized configuration management for the voice assistant system. It supports environment variable configuration with dataclass-based type safety and validation.

## File Location
`/home/user/cluster/src/core/config.py`

## Classes

### AssistantConfig (Dataclass)

Main configuration container with hierarchical settings.

**Core Settings**:
- `name: str` - Assistant name
- `version: str` - Version string
- `debug: bool` - Debug mode
- `log_level: str` - Logging level

**Component Configurations**:
- `tts: TTSConfig` - Text-to-speech settings
- `audio: AudioConfig` - Audio processing settings
- `display: Dict` - Display settings
- `camera: CameraConfig` - Camera settings

**AI Settings**:
- `llm: Dict` - LLM provider configuration
- `ambient_stt: Dict` - Ambient speech recognition
- `vad: Dict` - Voice activity detection
- `wake_word: Dict` - Wake word detection

**Mock Mode Flags**:
- `mock_llm`, `mock_tts`, `mock_stt`, `mock_ambient_stt`, `mock_display`, `mock_audio`

**Path Settings**:
- `data_dir`, `models_dir`, `voices_dir`, `config_dir`, `logs_dir`

### ConfigManager

Manages configuration loading, validation, and environment overrides.

**Methods**:

| Method | Purpose |
|--------|---------|
| `load_config()` | Load configuration from environment |
| `_load_env_file()` | Load .env file |
| `_apply_env_overrides()` | Apply environment variables |
| `_apply_mock_mode_logic_early()` | Process mock flags early |
| `_set_nested_value()` | Set nested config values |
| `_create_config_object()` | Create AssistantConfig instance |
| `_validate_config()` | Validate configuration values |
| `get_env_overrides()` | Get applied overrides |

## Configuration Hierarchy

```
1. Default values (in dataclass)
       ↓
2. .env file (loaded via dotenv)
       ↓
3. Environment variables (runtime overrides)
```

## Environment Variable Mapping

```python
# Examples of environment variable to config path mapping:
"LLM_MODEL_ID" → "llm.model_id"
"TTS_MODEL_PATH" → "tts.model_path"
"MOCK_LLM" → "mock_llm"
"DISPLAY_RESOLUTION" → "display.resolution"
"CONVERSATION_MAX_HISTORY" → "conversation.max_history"
```

## Validation Rules

- TTS model path required (unless mock mode)
- Audio sample rate must be positive
- Audio buffer size must be positive
- Display resolution must be positive
- LLM model_id required (unless mock mode)
- LLM temperature must be 0-2

## Global Functions

```python
get_config_manager() -> ConfigManager  # Get singleton manager
load_config() -> AssistantConfig       # Load configuration
get_config() -> Optional[AssistantConfig]  # Get current config
```

## Improvements Suggested

### 1. Configuration Schema
Add JSON Schema or Pydantic validation:
```python
from pydantic import BaseModel, validator

class LLMConfig(BaseModel):
    model_id: str
    temperature: float = 0.7

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be 0-2')
        return v
```

### 2. Config Hot Reloading
Watch for config file changes:
```python
async def watch_config_changes(self):
    """Watch for .env file changes and reload."""
    import watchdog
    # Implement file watching
```

### 3. Config Export
Export current config to file:
```python
def export_config(self, path: str) -> None:
    """Export current configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(asdict(self.config), f)
```

### 4. Secrets Management
Separate sensitive config:
```python
SENSITIVE_KEYS = {'api_key', 'password', 'secret'}

def redact_secrets(config: Dict) -> Dict:
    """Redact sensitive values for logging."""
    return {k: '***' if k in SENSITIVE_KEYS else v for k, v in config.items()}
```

### 5. Configuration Profiles
Support named configuration profiles:
```python
def load_profile(self, profile_name: str) -> AssistantConfig:
    """Load a named configuration profile."""
    profile_path = Path(f"config/profiles/{profile_name}.yaml")
    # Load and merge with defaults
```

# service_config.py - Service Configuration Loader Analysis

## Overview

`service_config.py` provides utilities for loading service-specific configuration from YAML files. It follows the pattern of separate config files per service with environment variable overrides.

## File Location
`/home/user/cluster/src/core/service_config.py`

## Key Functions

### get_config_dir() -> Path
Returns the configuration directory path, checking `CONFIG_DIR` environment variable first.

### load_yaml_config(config_name: str) -> Dict
Loads a YAML configuration file by name (without .yaml extension).

### merge_configs(defaults: Dict, overrides: Dict) -> Dict
Deep merges two configuration dictionaries recursively.

### apply_env_overrides(config: Dict, prefix: str) -> Dict
Applies environment variable overrides with prefix (e.g., `STT_`, `LLM_`, `TTS_`).

### _parse_env_value(value: str) -> Any
Parses environment variable strings to appropriate Python types.

## Service Configuration Classes

### STTServiceConfig
```python
@dataclass
class STTServiceConfig:
    wake_word: str = "jarvis"
    threshold: float = 0.5
    chunk_size: int = 1280
    sample_rate: int = 16000
    vosk_model_path: Optional[str] = None
    vosk_model_search_paths: list
    silence_threshold_rms: int = 500
    max_recording_seconds: float = 5.0
    silence_duration_seconds: float = 1.5
    device_index: Optional[int] = None
    verbose: bool = False
```

### LLMServiceConfig
```python
@dataclass
class LLMServiceConfig:
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    api_type: str = "auto"
    default_model: str = "llama3.2:3b"
    system_prompt: str
    timeout_seconds: int = 120
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 512
```

### TTSServiceConfig
```python
@dataclass
class TTSServiceConfig:
    model_path: Optional[str] = None
    model_search_dir: str = "~/.local/share/piper/voices"
    keep_audio_files: bool = True
    max_audio_files: int = 10
    audio_chunk_size: int = 4096
    frames_per_buffer: int = 1024
    audio_output_dir: Optional[str] = None
```

## Configuration Loading Pattern

```
Service.load()
    ↓
load_yaml_config("service_name")
    ↓
apply_env_overrides(config, "PREFIX")
    ↓
Create dataclass instance
```

## Environment Variable Format

- Simple keys: `{PREFIX}_{KEY}` → `config[key]`
- Nested keys: `{PREFIX}_{SECTION}__{KEY}` → `config[section][key]`

## Type Parsing

| Input | Parsed Type |
|-------|-------------|
| `"true"`, `"false"` | bool |
| `"null"`, `"none"` | None |
| `"123"` | int |
| `"1.23"` | float |
| `"a,b,c"` | list |
| Other | str |

## Improvements Suggested

### 1. Schema Validation
Add YAML schema validation:
```python
def validate_schema(config: Dict, schema_name: str) -> bool:
    """Validate config against JSON schema."""
    schema = load_schema(schema_name)
    jsonschema.validate(config, schema)
```

### 2. Config Caching
Cache loaded configs:
```python
_config_cache: Dict[str, Tuple[float, Dict]] = {}

def load_yaml_config_cached(config_name: str, ttl: float = 60.0) -> Dict:
    """Load config with caching."""
    if config_name in _config_cache:
        cached_time, config = _config_cache[config_name]
        if time.time() - cached_time < ttl:
            return config
    config = load_yaml_config(config_name)
    _config_cache[config_name] = (time.time(), config)
    return config
```

### 3. Default Config Generation
Generate default config files:
```python
def generate_default_config(service_name: str) -> None:
    """Generate default YAML config file for a service."""
    config_class = SERVICE_CONFIGS[service_name]
    defaults = {f.name: f.default for f in fields(config_class)}
    with open(f"config/{service_name}.yaml", 'w') as f:
        yaml.dump(defaults, f, default_flow_style=False)
```

### 4. Config Diff
Show differences from defaults:
```python
def get_config_diff(config: Dict, defaults: Dict) -> Dict:
    """Get non-default configuration values."""
    diff = {}
    for key, value in config.items():
        if key not in defaults or defaults[key] != value:
            diff[key] = value
    return diff
```

# Configuration Architecture

## Overview

The Voice Assistant uses a clean, hierarchical configuration system with **single source of truth** for model information.

## Configuration Hierarchy

1. **`config/assistant_config.yaml`** - Base configuration with defaults
2. **`.env` file** - Environment-specific overrides
3. **Environment variables** - Runtime overrides (highest priority)

## LLM Model Configuration

### Single Source of Truth: `models.json`

All model-specific details (repository, file patterns, download URLs) are stored in `models.json`:

```json
{
  "models": [
    {
      "id": "llama32-1b-q6k",
      "name": "Llama 3.2 1B Q6_K",
      "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
      "local_path": "bartowski/Llama-3.2-1B-Instruct-GGUF",
      "file_patterns": ["Llama-3.2-1B-Instruct-Q6_K.gguf"],
      "is_default": true,
      ...
    }
  ]
}
```

### Configuration Files Only Need `model_id`

**In `config/assistant_config.yaml`:**
```yaml
llm:
  provider_type: "local"
  model_id: "llama32-1b-q6k"  # References models.json
  temperature: 0.7
  max_tokens: 512
  ...
```

**In `.env` file:**
```bash
LLM_PROVIDER_TYPE=local
LLM_MODEL_ID=llama32-1b-q6k  # References models.json
LLM_TEMPERATURE=0.7
...
```

### Why This Architecture?

1. **Single Source of Truth**: Model details are only in `models.json`
2. **No Duplication**: Avoid inconsistencies between config files
3. **Easy Updates**: Change default model by updating `is_default` flag in `models.json`
4. **Clean Separation**: 
   - `models.json` = What models exist and how to download them
   - Config files = Which model to use and how to configure it

### Adding New Models

1. Add model definition to `models.json`:
```json
{
  "id": "new-model-id",
  "name": "New Model Name",
  "repo_id": "huggingface/repo",
  "local_path": "huggingface/repo",
  "file_patterns": ["*.gguf"],
  "is_default": false,
  ...
}
```

2. Update config to use it:
```bash
LLM_MODEL_ID=new-model-id
```

3. Download the model:
```bash
python -c "from ai.model_manager import get_model_manager; import asyncio; asyncio.run(get_model_manager().download_model('new-model-id'))"
```

## Docker Compose

`docker-compose.yml` **does NOT** contain default values for LLM configuration. It only passes through environment variables:

```yaml
environment:
  - LLM_PROVIDER_TYPE=${LLM_PROVIDER_TYPE}
  - LLM_MODEL_ID=${LLM_MODEL_ID}
  # No defaults! Values must be in .env file
```

This ensures the `.env` file is the single source for environment configuration.

## Best Practices

1. **Never hardcode model paths or URLs in config files**
2. **Always use `model_id` to reference models**
3. **Keep `models.json` as the authoritative source for model metadata**
4. **Use `.env` file for environment-specific settings**
5. **Use `config/assistant_config.yaml` for application defaults**

## Configuration Priority

```
Environment Variables (highest)
    ↓
.env file
    ↓
config/assistant_config.yaml
    ↓
Code defaults (lowest)
```

## Example: Changing the Default Model

**Old way (multiple places to update):**
- ❌ Update `models.json`
- ❌ Update `config/assistant_config.yaml`
- ❌ Update `.env`
- ❌ Update `docker-compose.yml` defaults
- ❌ Update install scripts

**New way (single source of truth):**
1. ✅ Update `models.json`:
```json
{"id": "old-model", "is_default": false}
{"id": "new-model", "is_default": true}
```

2. ✅ Update `.env`:
```bash
LLM_MODEL_ID=new-model
```

Done! All other files reference these two sources.


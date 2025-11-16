# Configuration Guide

## Quick Configuration

### Understanding Mock Mode

The application supports **Mock Mode** for testing without real hardware. Each component can be mocked independently:

- **MOCK_LLM**: Use fake LLM responses (no AI model needed)
- **MOCK_TTS**: Use fake text-to-speech (no audio output)
- **MOCK_STT**: Use fake speech-to-text (no microphone input)

## Configuration Files

### Primary Configuration: `.env` file

The `.env` file in the project root controls the application behavior:

```bash
# Copy example to create your .env
cp env.example .env

# Edit the .env file
nano .env  # or use your favorite editor
```

### Key Settings

#### Mock Mode Configuration

```bash
# In .env file

# Enable/disable mock providers
MOCK_LLM=false      # Set to false to use real llama.cpp
MOCK_TTS=false      # Set to false to use real TTS engine
MOCK_STT=false      # Set to false to use real speech recognition
```

#### Model Selection

```bash
# Choose your LLM provider
LLM_PROVIDER_TYPE=local    # Options: local, openai, anthropic

# Model configuration (for local provider)
LLM_MODEL=gemma-3n-q4-xl
LLM_MODEL_ID=gemma-3n-q4-xl
LLM_CONTEXT_WINDOW=8192
LLM_TEMPERATURE=0.7
```

## Applying Configuration Changes

After editing `.env` file, you **must restart the container** for changes to take effect:

```bash
# Stop the container
docker-compose down

# Start the container (will load new .env values)
docker-compose up -d

# View logs to verify
docker-compose logs -f
```

## Verifying Configuration

### Check Environment Variables in Container

```bash
# Check all MOCK_* variables
docker-compose exec voice-assistant env | grep MOCK_

# Should show (when disabled):
# MOCK_LLM=false
# MOCK_TTS=false
# MOCK_STT=false

# Check LLM configuration
docker-compose exec voice-assistant env | grep LLM_
```

### Check Application Logs

Look for these log messages:

**When Mock Mode is ON:**
```
[INFO] MOCK_LLM=true - using mock LLM provider
[INFO] MOCK_TTS=true - using mock TTS engine
[INFO] Mock mode check: MOCK_LLM=True, MOCK_TTS=True, MOCK_STT=True
```

**When Mock Mode is OFF:**
```
[INFO] Mock mode check: MOCK_LLM=False, MOCK_TTS=False, MOCK_STT=False
[INFO] Initializing llama.cpp provider
[INFO] Using model: gemma-3n-q4-xl
```

## Common Issues

### Issue: Mock LLM Still Active After Changing Configuration

**Symptoms:**
- You see "Thanks for testing the mock LLM provider" messages
- MOCK_LLM is set to `false` in .env

**Solution:**
```bash
# 1. Verify .env file has the correct values
cat .env | grep MOCK_LLM

# 2. Restart the container to load new values
docker-compose down
docker-compose up -d

# 3. Check if values are loaded
docker-compose exec voice-assistant env | grep MOCK_LLM
```

### Issue: Container Not Reading .env Changes

**Symptoms:**
- Edited .env but changes don't apply
- Old values still in effect

**Solution:**
The `env_file` directive loads .env, but you need to ensure:
1. `.env` file exists in project root (same directory as docker-compose.yml)
2. Values have correct format:
   ```bash
   MOCK_LLM=false    # ✅ Correct
   MOCK_LLM="false"  # ✅ Also correct
   MOCK_LLM = false  # ❌ Don't use spaces
   MOCK_LLM=         # ❌ Empty sets to empty string
   ```
3. Restart container after changes

### Issue: Model Not Loading

**Symptoms:**
- "Model file not found" errors
- Container exits with model errors

**Solutions:**
```bash
# 1. Check if model is downloaded
docker-compose exec voice-assistant ls -lh /app/models/

# 2. Check logs for download errors
docker-compose logs voice-assistant | grep -i model

# 3. Manually trigger model download
docker-compose exec voice-assistant python scripts/install.py
```

## Configuration Priority

Environment variables are loaded in this order:

1. **System environment variables** (host OS)
2. **`.env` file** (loaded by `env_file` directive)
3. **`environment:` section** in docker-compose.yml
4. **YAML configuration files** (`config/assistant_config.yaml`)

**Priority:** Last loaded wins (environment section → .env → system)

## Best Practices

### 1. Development vs Production

**Development (.env):**
```bash
MOCK_LLM=true       # Test without downloading models
MOCK_TTS=true       # Test without audio hardware
MOCK_STT=true       # Test without microphone
ASSISTANT_DEBUG=true
ASSISTANT_LOG_LEVEL=DEBUG
```

**Production (.env):**
```bash
MOCK_LLM=false      # Use real AI
MOCK_TTS=false      # Use real TTS
MOCK_STT=false      # Use real speech recognition
ASSISTANT_DEBUG=false
ASSISTANT_LOG_LEVEL=INFO
```

### 2. Model Configuration

For **local provider** (llama.cpp):
```bash
LLM_PROVIDER_TYPE=local
LLM_MODEL=gemma-3n-q4-xl
LLM_CONTEXT_WINDOW=8192
LLM_TEMPERATURE=0.7
LLM_THREADS=4
LLM_GPU_LAYERS=0    # CPU only by default
```

For **cloud provider** (OpenAI):
```bash
LLM_PROVIDER_TYPE=openai
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4
LLM_MAX_TOKENS=1000
```

### 3. Testing Configuration

Always test with mock mode first:
```bash
# 1. Start with mocks enabled
MOCK_LLM=true
MOCK_TTS=true
MOCK_STT=true

# 2. Verify basic functionality

# 3. Disable mocks one by one
MOCK_LLM=false    # Test real LLM
MOCK_TTS=false    # Test real TTS
MOCK_STT=false    # Test real STT
```

## Troubleshooting

### Debug Mode

Enable detailed logging:
```bash
ASSISTANT_DEBUG=true
ASSISTANT_LOG_LEVEL=DEBUG
```

View logs:
```bash
docker-compose logs -f voice-assistant
```

### Environment Variable Not Taking Effect

**Check:**
1. .env file exists in correct location
2. No syntax errors in .env
3. Container was restarted after .env changes
4. Variable name is correct (case-sensitive)

```bash
# Test by checking container environment
docker-compose exec voice-assistant env | grep YOUR_VARIABLE
```

### Configuration Precedence Issues

If you're seeing unexpected values, check:
1. What's in .env file
2. What's in docker-compose.yml `environment:` section
3. System environment variables

```bash
# Check all sources
echo $MOCK_LLM                    # System
docker-compose config | grep MOCK_LLM  # Compose
docker-compose exec voice-assistant env | grep MOCK_LLM  # Container
```

## Quick Reference

### Editing Configuration

```bash
# Edit .env file
nano .env

# Or on Windows:
notepad .env

# Restart to apply changes
docker-compose restart
```

### Checking Current Configuration

```bash
# View environment in container
docker-compose exec voice-assistant env

# View configuration object
docker-compose exec voice-assistant python -c "from core.config import ConfigManager; cm = ConfigManager(); c = cm.load_config(); print(c.mock_llm, c.mock_tts, c.mock_stt)"
```

### Resetting to Defaults

```bash
# Reset to example configuration
cp env.example .env

# Restart
docker-compose restart
```

## Summary

✅ **Always edit `.env` file to change configuration**  
✅ **Restart container after changes**  
✅ **Check logs to verify configuration loaded**  
✅ **Use mock mode for initial testing**  
✅ **Enable debug logging for troubleshooting**


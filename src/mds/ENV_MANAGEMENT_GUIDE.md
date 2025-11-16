# Environment Variable Management Guide

## Overview
This application uses `.env` files for managing environment variables. The setup follows best practices for local development and deployment.

## How It Works

### Current Implementation
1. **Environment Template**: `env.example` - Contains all available environment variables with sensible defaults
2. **Local Configuration**: `.env` - Your personal environment configuration (not tracked in git)
3. **Automatic Loading**: The application automatically loads variables from `.env` on startup

### Setup Process

#### For New Developers
When you first install the application:

```bash
# Run the installation script
python scripts/install.py

# OR use the wrapper script
bash scripts/install.sh
```

The installer will automatically:
1. Copy `env.example` to `.env` (if `.env` doesn't exist)
2. Set up all necessary directories
3. Download default models
4. Create configuration files

#### Manual Setup
If you need to create `.env` manually:

```bash
# Copy the example file
cp env.example .env

# Edit .env with your preferred settings
nano .env  # or your preferred editor
```

## Configuration Priority

Environment variables are loaded in this order (later values override earlier ones):

1. **Default values** (hardcoded in the application)
2. **YAML config file** (`config/assistant_config.yaml`)
3. **Environment variables** from `.env` or system environment
4. **Command-line arguments** (if applicable)

## Key Environment Variables

### Mock Mode Settings
```bash
# Enable mock providers for testing (no hardware required)
MOCK_LLM=true
MOCK_TTS=true
MOCK_STT=true
```

### LLM Configuration
```bash
# LLM Provider (local, openai, anthropic)
LLM_PROVIDER_TYPE=local
LLM_MODEL=gemma-3n-q4-xl
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
```

### TTS Configuration
```bash
# TTS Engine (coqui, piper, xtts, mock)
TTS_ENGINE=coqui
TTS_MODEL_PATH=models/coqui
```

### Audio Configuration
```bash
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_BUFFER_SIZE=512
```

## How to Modify Configuration

### Development Mode (Recommended)
1. Edit `env.example` to add new variables with documentation
2. Copy to `.env` if you haven't already: `cp env.example .env`
3. Edit your local `.env` file with your values
4. The app will use your `.env` values at runtime

### Production Mode
For production deployments using Docker:
- Edit values directly in `docker-compose.yml` or `docker-compose.prod.yml`
- Or use environment variables from your hosting platform
- Never commit `.env` file - it's gitignored

## File Locations

```
project-root/
├── env.example          # Template (committed to git)
├── .env                  # Your config (gitignored)
├── config/
│   └── assistant_config.yaml  # YAML config alternative
└── src/
    └── core/
        └── config.py    # Configuration loader
```

## Best Practices

1. **Never commit `.env`**: Already in `.gitignore`
2. **Always update `env.example`**: When adding new variables, update the example file
3. **Use mock mode for testing**: Set `MOCK_*` variables to `true` when developing without hardware
4. **Document variables**: Add comments in `env.example` for clarity
5. **Keep secrets out**: Don't put API keys or secrets in the example file

## Docker Deployment

When using Docker, environment variables can be set in:

1. **docker-compose.yml** - For development
2. **docker-compose.prod.yml** - For production
3. **Host environment** - When starting containers
4. **Runtime** - Using `docker run -e KEY=value`

Example:
```yaml
services:
  app:
    environment:
      - MOCK_LLM=false
      - LLM_MODEL=gemma-3n-q4-xl
```

## Troubleshooting

### Variables Not Loading
- Check if `.env` exists in the project root
- Verify file permissions
- Check logs for "Loaded environment variables from..." message
- Ensure `python-dotenv` is installed: `pip install python-dotenv`

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Manual Load Test
```bash
python -c "from src.core.config import ConfigManager; cm = ConfigManager(); print('Config loaded successfully')"
```

## Summary of Changes Made

### What Was Fixed
1. ✅ Added `python-dotenv` to `requirements.txt`
2. ✅ Updated `src/core/config.py` to automatically load `.env` file
3. ✅ Modified `scripts/install.py` to copy `env.example` to `.env` during installation
4. ✅ `.env` is properly gitignored

### How to Use
1. Install dependencies: `pip install -r requirements.txt`
2. Run installer: `python scripts/install.py`
3. Edit `.env` if needed
4. Run the application: `python src/main.py`

The application will now:
- Automatically load variables from `.env` on startup
- Respect the configuration hierarchy (defaults → YAML → env → overrides)
- Work seamlessly with or without `.env` file (graceful fallback)


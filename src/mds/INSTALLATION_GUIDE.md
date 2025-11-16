# Installation Guide

This guide covers the installation process for the Voice Assistant application, including model downloads and configuration setup.

## Quick Start

### Using Docker (Recommended)

The easiest way to run the Voice Assistant is using Docker, which handles all installation steps automatically:

```bash
# Development environment
docker-compose up --build

# Production environment
docker-compose -f docker-compose.prod.yml up --build
```

### Manual Installation (Linux/macOS)

For development or custom setups, you can install manually:

```bash
# Using the installation script
./scripts/install.sh

# Or directly with Python
python scripts/install.py
```

## Installation Steps

The installation script performs the following steps:

### 1. Create Directories
Creates necessary directories for the application:
- `models/` - AI model files
- `config/` - Configuration files
- `data/` - Application data and databases
- `voices/` - Voice synthesis files
- `logs/` - Application logs

### 2. Download Default Model
Downloads the default AI model (Gemma-3n Q4_K_XL):
- **Model**: `unsloth/gemma-3n-E4B-it-GGUF`
- **Quantization**: Q4_K_XL
- **Size**: ~2.3 GB
- **Context Window**: 8192 tokens

### 3. Setup Configuration
Creates default configuration files:
- `config/assistant_config.yaml` - Main configuration
- Model manager configuration
- Component-specific settings

### 4. Verify Installation
Verifies that all components are properly installed and configured.

## Command Line Options

### Installation Script Options

```bash
python scripts/install.py [OPTIONS]

Options:
  --verbose, -v           Enable verbose logging
  --skip STEPS           Skip installation steps (space-separated)
  --models-dir DIR       Models directory (default: models)
  --config-dir DIR       Configuration directory (default: config)
  --data-dir DIR         Data directory (default: data)
  --voices-dir DIR       Voices directory (default: voices)
  --logs-dir DIR         Logs directory (default: logs)
  --help, -h             Show help message
```

### Available Steps to Skip

- `create_directories` - Skip directory creation
- `download_default_model` - Skip model download
- `setup_configuration` - Skip configuration setup
- `verify_installation` - Skip installation verification

### Examples

```bash
# Full installation with verbose output
python scripts/install.py --verbose

# Skip model download (if already downloaded)
python scripts/install.py --skip download_default_model

# Custom directories
python scripts/install.py --models-dir /custom/models --config-dir /custom/config

# Skip multiple steps
python scripts/install.py --skip download_default_model verify_installation
```

## Manual Installation

If you prefer to install manually or need to customize the process:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Directories

```bash
mkdir -p models config data voices logs
```

### 3. Download Model

```bash
python scripts/setup_gemma_model.py
```

### 4. Setup Configuration

Copy and modify configuration files from the `config/` directory.

## Docker Installation

### Development Environment

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f voice-assistant

# Stop environment
docker-compose down
```

### Production Environment

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f voice-assistant
```

### Custom Docker Build

```bash
# Build development image
docker build --target development -t voice-assistant:dev .

# Build production image
docker build --target production -t voice-assistant:prod .

# Build ARM64 image for Raspberry Pi
docker build --target production-arm64 -t voice-assistant:arm64 .
```

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Error**: `Failed to download default model`

**Solutions**:
- Check internet connection
- Ensure sufficient disk space (~3GB)
- Try running with `--verbose` for detailed logs
- Manually download model files

#### 2. Permission Errors

**Error**: `Permission denied`

**Solutions**:
- Run with appropriate permissions
- Check directory ownership
- Use `sudo` if necessary (Linux/macOS)

#### 3. Python Import Errors

**Error**: `ModuleNotFoundError`

**Solutions**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path and virtual environment
- Verify you're in the correct directory

#### 4. Docker Build Fails

**Error**: Docker build fails during model download

**Solutions**:
- Check Docker has sufficient resources
- Ensure stable internet connection
- Try building without cache: `docker build --no-cache`

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Python script
python scripts/install.py --verbose

# Docker with debug environment
docker-compose up --build
```

### Log Files

Check installation logs in:
- `logs/assistant.log` - Application logs
- `logs/install.log` - Installation logs (if created)
- Docker logs: `docker-compose logs voice-assistant`

## Post-Installation

After successful installation:

### 1. Verify Installation

```bash
# Check model availability
python scripts/model_manager_lazy.py list

# Test model
python scripts/model_manager_lazy.py test models/unsloth/gemma-3n-E4B-it-GGUF
```

### 2. Start Application

```bash
# Direct execution
python src/main.py

# Docker
docker-compose up
```

### 3. Configuration

Edit `config/assistant_config.yaml` to customize:
- Model settings
- Audio configuration
- Display settings
- Camera settings

## Advanced Configuration

### Custom Models

To use different models:

1. Download model files to `models/` directory
2. Update configuration:
   ```yaml
   llm:
     model_id: "your-model-id"
     model_path: "path/to/your/model.gguf"
   ```

### Environment Variables

Override configuration with environment variables:

```bash
export LLM_MODEL_ID="your-model-id"
export LLM_TEMPERATURE="0.8"
export LLM_CONTEXT_WINDOW="4096"
```

### Model Manager

Use the model manager for advanced model operations:

```bash
# List available models
python scripts/model_manager_lazy.py list

# Switch models
python scripts/model_manager_lazy.py switch models/your-model.gguf

# Test models
python scripts/model_manager_lazy.py test models/your-model.gguf --prompt "Hello"
```

## Support

For additional help:

1. Check the logs in `logs/` directory
2. Review configuration files in `config/`
3. Consult the documentation files:
   - `MODEL_MANAGEMENT_GUIDE.md`
   - `UNIFIED_LLM_GUIDE.md`
   - `DOCKER_DEPLOYMENT.md`
4. Run with `--verbose` flag for detailed output

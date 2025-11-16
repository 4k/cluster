# Quick Start Guide

This guide will walk you through getting the Voice Assistant up and running in just a few minutes.

## Prerequisites

- **Docker Desktop** (Windows/macOS) or **Docker + Docker Compose** (Linux)
- At least **4GB free disk space** for models
- Internet connection for downloading models

> 💡 Don't have Docker? The installation script will check and guide you to install it.

## Installation Steps

### 1. Clone or Download the Repository

```bash
git clone <your-repo-url>
cd cluster
```

### 2. Choose Installation Method

You have two options based on whether Docker is already installed:

#### Option A: Docker Not Installed / Fresh System Setup

If you're installing on a Raspberry Pi, VPS, or fresh system:

```bash
chmod +x install-docker.sh
./install-docker.sh
```

This script will:
- Install Docker if needed
- Set up the environment
- Build images
- Start the application

#### Option B: Docker Already Installed

If you already have Docker running:

```bash
# 1. Configure environment
cp env.example .env

# 2. Edit configuration
nano .env  # or use your preferred editor

# 3. Build and start
docker-compose build
docker-compose up -d
```

**The application will automatically download models on first startup.**

> **On Windows (Git Bash or WSL):** The scripts work directly.  
> **On Windows (PowerShell):** You may need to run: `bash install-docker.sh`

## What Happens During Installation

### install-docker.sh (for new systems):
1. ✅ Detects your operating system
2. ✅ Installs Docker if needed
3. ✅ Verifies Docker Compose is available
4. ✅ Creates `.env` from `env.example`
5. ✅ Builds Docker images
6. ✅ Starts the container
7. ✅ Container automatically runs `install-app.sh` on first startup
8. ✅ Models are downloaded and app is ready!

### install-app.sh (runs automatically inside container):
1. ✅ Creates necessary directories
2. ✅ Runs Python installation script
3. ✅ Downloads configured AI models
4. ✅ Sets up configuration files
5. ✅ Marks installation as complete

## Viewing Logs

Watch the application start and download models:

```bash
./install.sh --logs
```

Or use Docker Compose directly:

```bash
docker-compose logs -f
```

## Managing the Application

```bash
# View application status
docker-compose ps

# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart the application
docker-compose restart

# Rebuild and restart
docker-compose up -d --build
```

## First Run

When you first start the application:

1. **The container starts** and runs the entrypoint script
2. **Installation script runs** (downloads models, sets up directories)
3. **Application starts** with your configured settings

This process takes a few minutes the first time as models are downloaded.

## Configuration

### Quick Configuration

Edit `.env` to change settings:

```bash
# Use mock providers (no real hardware needed)
MOCK_LLM=true
MOCK_TTS=true
MOCK_STT=true

# Or use real hardware
MOCK_LLM=false
MOCK_TTS=false
MOCK_STT=false
```

### Model Configuration

The default model is **Gemma-3n** (~2.3 GB). To change it:

```bash
# In .env file
LLM_MODEL=gemma-3n-q4-xl
LLM_MODEL_ID=gemma-3n-q4-xl
```

Available models are listed in `models.json`.

## Troubleshooting

### Docker is Not Running

**Error:** `Cannot connect to Docker daemon`

**Solution:**
- **Windows/macOS:** Start Docker Desktop
- **Linux:** `sudo systemctl start docker`

### Permission Denied

**Error:** `Permission denied` on install.sh

**Solution:**
```bash
chmod +x install.sh
```

### Models Not Downloading

**Error:** Models stuck at 0%

**Solutions:**
1. Check internet connection
2. Check Docker has sufficient resources (increase memory in Docker Desktop)
3. Check `models.json` for correct model configuration
4. View logs: `docker-compose logs -f voice-assistant`

### Container Keeps Restarting

**Error:** Container exits immediately

**Solution:**
```bash
# Check what's wrong
docker-compose logs voice-assistant

# Common issues:
# - Missing .env file
# - Corrupted model files
# - Insufficient disk space
```

### .env File Not Being Used

**Error:** Configuration changes not taking effect

**Solution:**
```bash
# Restart the container to reload .env
./install.sh --restart

# Or directly:
docker-compose down
docker-compose up -d
```

## Accessing the Container

You can access the running container for debugging:

```bash
# Open a shell in the container
docker-compose exec voice-assistant sh

# Run commands inside the container
docker-compose exec voice-assistant python scripts/model_manager_lazy.py list

# Check environment variables
docker-compose exec voice-assistant env | grep LLM
```

## Files and Directories

```
cluster/
├── .env                    # Your configuration (not in git)
├── install.sh             # Main installation script
├── docker-compose.yml      # Development Docker configuration
├── data/                   # Application data (persistent)
├── models/                 # Downloaded AI models (persistent)
├── config/                 # Configuration files
├── logs/                   # Application logs
└── voices/                 # Voice files (persistent)
```

## Environment Variables

Key environment variables in `.env`:

### Mock Mode (Testing)
- `MOCK_LLM=true/false` - Use mock LLM
- `MOCK_TTS=true/false` - Use mock TTS  
- `MOCK_STT=true/false` - Use mock STT

### Model Settings
- `LLM_MODEL` - Model identifier
- `LLM_TEMPERATURE` - Creativity (0.0-1.0)
- `LLM_CONTEXT_WINDOW` - Token window size

### Audio Settings
- `AUDIO_SAMPLE_RATE` - Audio quality
- `AUDIO_WAKE_WORD_THRESHOLD` - Wake word sensitivity

See `env.example` for all available options.

## Next Steps

After successful installation:

1. **Test the application** - Check logs to ensure it started
2. **Configure your model** - Edit `.env` if needed
3. **Try mock mode first** - Verify everything works
4. **Connect hardware** - When ready, switch off mock mode
5. **Customize configuration** - Edit `config/assistant_config.yaml`

## Production Deployment

For production (Raspberry Pi, servers):

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up --build -d

# Or build ARM64 image for Raspberry Pi
docker build --target production-arm64 -t voice-assistant:arm64 .
```

## Getting Help

- **Logs:** `docker-compose logs -f`
- **Container status:** `docker-compose ps`
- **Environment:** Check `.env` file
- **Documentation:** See other `*.md` files in this directory

## Common Commands Reference

```bash
# Fresh installation (installs Docker + app)
./install-docker.sh          # Full system installation

# If Docker already installed
docker-compose up -d          # Start application
docker-compose down           # Stop application
docker-compose restart        # Restart application
docker-compose ps             # Show status
docker-compose logs -f        # Follow logs

# Debugging
docker-compose exec voice-assistant sh           # Open shell in container
docker-compose exec voice-assistant env         # Check environment variables
docker-compose logs voice-assistant             # View application logs

# Inside container (troubleshooting)
docker-compose exec voice-assistant sh
./install-app.sh             # Manually run app installation
exit                         # Exit container
```

---

Happy voice assisting! 🎙️✨


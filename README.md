# Voice Assistant - Simplified Docker Setup

A voice assistant application that works seamlessly across Windows WSL, macOS, Linux, and Raspberry Pi 5.

## 🚀 **Quick Start**

### **1. One-Time Setup**
```bash
# Clone the repository
git clone <repository-url>
cd voice-assistant

# Copy environment file and configure
cp env.example .env
# Edit .env to customize settings (all configuration is now in environment variables)
```

### **2. Build & Start** (Choose One Method)

**Option A: Automated Build Scripts (Recommended)** ⭐
```powershell
# Windows PowerShell
.\build.ps1 -Detached
```
```bash
# Linux/macOS
./build.sh --detached
```

**Option B: Makefile**
```bash
make dev
```

**Option C: Direct Docker Compose**
```bash
docker-compose up --build -d
```

> 💡 **Tip:** Build scripts automatically enable BuildKit for faster builds! See [DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md) for all commands.

### **3. Daily Usage**
```bash
# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart
docker-compose restart
```

## ⚙️ **Configuration**

All configuration is now handled through environment variables in the `.env` file. No more YAML config files!

### **Environment Variables**
- **Copy `env.example` to `.env`** and customize settings
- **Model settings** are defined in `models.json` (single source of truth)
- **All other settings** are in environment variables

### **Key Configuration Areas**
- `LLM_*` - Language model settings (provider, model_id, performance)
- `TTS_*` - Text-to-speech settings (Piper only)
- `AUDIO_*` - Audio input/output settings
- `AMBIENT_STT_*` - Ambient listening configuration (always-on speech recognition)
- `DISPLAY_*` - Display and animation settings
- `MOCK_*` - Mock mode flags for testing
- `DEVELOPMENT_*` - Development and debugging settings

### **Model Configuration**
Models are defined in `models.json` with all their settings:
- Model paths and file patterns
- Performance parameters (GPU layers, threads, etc.)
- Generation parameters (temperature, top_p, etc.)
- Context window and quantization info

### **Performance Optimization**
For Raspberry Pi 5 with GPU acceleration:
```bash
# In your .env file
LLM_GPU_LAYERS=33
LLM_THREADS=8
LLM_USE_MLOCK=true
LLM_NGL=33
LLM_MAIN_GPU=0
```

## 📋 **What the Setup Does**

The `setup.sh` script runs **once** and:
- ✅ Checks prerequisites (Docker, Docker Compose)
- ✅ Creates necessary directories (`data`, `models`, `voices`, `logs`)
- ✅ Builds the Docker image
- ✅ Shows you how to start/stop the application

## 🎯 **Mock Mode (Default)**

The application runs in **mock mode by default** - no real hardware or models needed:
- **Mock LLM**: Returns hardcoded responses
- **Mock TTS**: Plays sample audio or generates tones
- **Mock STT**: Returns hardcoded text
- **Mock Audio**: No real microphone/speaker needed

## 🎙️ **Ambient Listening Mode**

The assistant supports **always-on ambient listening** that continuously monitors for speech in the background:

### **How It Works**
- **Ambient Mode**: Continuously listens and logs detected speech without processing it
- **Wake Word Mode**: After wake word detection, speech is processed for 5 seconds (configurable)
- **Dual-Mode Tagging**: All detected speech is tagged as either "ambient" or "wakeword"

### **Configuration**
```bash
# Enable ambient listening
AMBIENT_STT_ENABLED=true

# Use lighter model for better performance
AMBIENT_STT_MODEL_PATH=models/vosk-model-small-en-us-0.15

# Lower confidence threshold for ambient mode
AMBIENT_STT_CONFIDENCE_THRESHOLD=0.3

# Wake word timeout (seconds to stay in wakeword mode)
AMBIENT_STT_WAKE_WORD_TIMEOUT=5.0

# Frame skip for performance (1=all frames, 2=every other frame)
AMBIENT_STT_FRAME_SKIP=1

# Use mock mode for testing without Vosk models
MOCK_AMBIENT_STT=true
```

### **Behavior**
- **Ambient speech**: Logged only, no animations or responses
- **Wake word speech**: Full processing with animations, LLM, and TTS

### **Performance Tips**
- Use `vosk-model-small-en-us-0.15` (40MB) instead of larger models
- Increase `AMBIENT_STT_FRAME_SKIP` to reduce CPU usage
- Set `AMBIENT_STT_ENABLED=false` to disable when not needed

## 🔧 **Configuration**

### **Simple Provider-Based Configuration**

Edit `.env` to change providers:

```bash
# Mock Mode (default: true for testing)
MOCK_MODE=true

# LLM Provider (mock = hardcoded responses, local = real model)
LLM_PROVIDER_TYPE=mock
LLM_MODEL=mock-llm-v1.0
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# TTS Engine (mock = hardcoded audio, piper = real TTS)
TTS_ENGINE=mock
TTS_MODEL_PATH=mock
TTS_RATE=150
TTS_VOLUME=1.0
```

### **How It Works**

- **`LLM_PROVIDER_TYPE=mock`** → All LLM config ignored, uses hardcoded responses
- **`TTS_ENGINE=mock`** → All TTS config ignored, uses hardcoded audio
- **`MOCK_MODE=true`** → Forces all providers to mock mode

### **Real Hardware Mode**

To use real hardware, change providers in `.env`:
```bash
MOCK_MODE=false
LLM_PROVIDER_TYPE=local
TTS_ENGINE=piper
```

## 🎮 **Common Commands**

### **Using Build Scripts (Recommended)**
```powershell
# Windows - Build and start
.\build.ps1 -Detached

# Windows - Fresh build (no cache)
.\build.ps1 -NoCache -Detached
```
```bash
# Linux/macOS - Build and start
./build.sh --detached

# Linux/macOS - Fresh build
./build.sh --no-cache --detached
```

### **Using Makefile**
```bash
make dev          # Build and start
make build        # Just build
make logs         # View logs
make down         # Stop
make restart      # Restart
make help         # See all commands
```

### **Using Docker Compose Directly**
```bash
# Start application
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Restart application
docker-compose restart

# Check status
docker-compose ps
```

📚 **See [DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md) for complete command reference**

## 🐛 **Troubleshooting**

### **Container Won't Start**
```bash
# Check logs
docker-compose logs

# Rebuild if needed
docker-compose down
docker-compose up --build -d
```

### **Mock Mode Not Working**
```bash
# Check environment variables
docker exec voice-assistant env | grep MOCK

# Restart with mock mode
docker-compose restart
```

## 📚 **Documentation**

- **[DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md)** - Quick command reference and common tasks
- **[DOCKER_BUILD_OPTIMIZATION.md](DOCKER_BUILD_OPTIMIZATION.md)** - BuildKit setup and optimization details
- **[QUICK_START.md](QUICK_START.md)** - Detailed getting started guide
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Configuration options and environment variables
- **[DEVELOPMENT_DEPLOYMENT_GUIDE.md](DEVELOPMENT_DEPLOYMENT_GUIDE.md)** - Full deployment guide

## ✅ **That's It!**

1. **Use `.\build.ps1 -Detached` (Windows) or `./build.sh --detached` (Linux)** to build and start
2. **Or use `make dev`** for quick development
3. **Use `docker-compose down`** to stop
4. **Edit `.env`** to change settings

**Simple, clean, and works everywhere!**
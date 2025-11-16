# Voice Assistant - Development & Deployment Guide

## 🏗️ **Architecture Overview**

Your Voice Assistant now uses an **Event Bus Architecture** instead of HTTP APIs:

- ✅ **Direct Execution**: Runs `python src/main.py` directly
- ✅ **Event Bus**: Components communicate via events
- ✅ **No HTTP API**: No web server or REST endpoints
- ✅ **Lazy Loading**: Models load/unload automatically
- ✅ **Single Model**: Only one model in memory at a time

## 🚀 **1. Development Setup (Windows WSL)**

### **Prerequisites**
- Docker Desktop installed and running
- WSL2 with audio support (optional)
- At least 4GB RAM available

### **Quick Start**
```bash
# Navigate to project directory
cd C:\Users\4k\Projects\4kStudio\Cluster\cluster

# Deploy development environment
./scripts/deploy.sh dev

# View logs
docker-compose logs -f voice-assistant

# Stop when done
docker-compose down
```

### **Detailed Development Process**

#### **Step 1: Initial Setup**
```bash
# Make scripts executable (if needed)
chmod +x scripts/*.sh

# Deploy development environment
./scripts/deploy.sh dev
```

#### **Step 2: Verify Deployment**
```bash
# Check container status
docker-compose ps

# View application logs
docker-compose logs -f voice-assistant

# Check if application is running
docker-compose exec voice-assistant ps aux
```

#### **Step 3: Test Model Management**
```bash
# List available models
docker-compose exec voice-assistant python scripts/model_manager_lazy.py list

# Test model switching
docker-compose exec voice-assistant python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf

# Test model inference
docker-compose exec voice-assistant python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Hello!"
```

#### **Step 4: Development Workflow**
```bash
# Make code changes (files are mounted live)
# Edit src/ai/conversation.py or other files

# Restart container to apply changes
docker-compose restart voice-assistant

# Or rebuild if needed
docker-compose up -d --build
```

### **Development Commands**

| Command | Purpose |
|---------|---------|
| `./scripts/deploy.sh dev` | Deploy development environment |
| `docker-compose logs -f` | View live logs |
| `docker-compose restart` | Restart application |
| `docker-compose down` | Stop environment |
| `docker-compose exec voice-assistant bash` | Access container shell |

### **Model Management (Development)**

```bash
# List models
docker-compose exec voice-assistant python scripts/model_manager_lazy.py list

# Switch model
docker-compose exec voice-assistant python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf

# Test model
docker-compose exec voice-assistant python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Tell me a joke"

# Check memory usage
docker-compose exec voice-assistant python scripts/model_manager_lazy.py memory

# Unload current model
docker-compose exec voice-assistant python scripts/model_manager_lazy.py unload
```

## 🍓 **2. Production Deployment (Raspberry Pi)**

### **Prerequisites**
- Raspberry Pi OS (64-bit recommended)
- Docker installed
- At least 4GB RAM, 20GB storage
- Audio, display, and camera hardware (optional)

### **Quick Start**
```bash
# Copy files to Raspberry Pi
scp -r . pi@raspberry-pi-ip:/home/pi/voice-assistant/

# SSH to Raspberry Pi
ssh pi@raspberry-pi-ip
cd voice-assistant

# Deploy production environment
./scripts/deploy.sh prod
```

### **Detailed Production Process**

#### **Step 1: System Preparation**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker (if not installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Log out and back in
exit
ssh pi@raspberry-pi-ip
```

#### **Step 2: Project Setup**
```bash
# Navigate to project directory
cd voice-assistant

# Make scripts executable
chmod +x scripts/*.sh

# Deploy production environment
./scripts/deploy.sh prod
```

#### **Step 3: Verify Deployment**
```bash
# Check container status
docker-compose -f docker-compose.prod.yml ps

# View application logs
docker-compose -f docker-compose.prod.yml logs -f voice-assistant

# Check system resources
docker stats voice-assistant-prod
```

#### **Step 4: Test Production Setup**
```bash
# Test model management
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py list

# Test model switching
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf

# Test inference
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Hello from Raspberry Pi!"
```

### **Production Commands**

| Command | Purpose |
|---------|---------|
| `./scripts/deploy.sh prod` | Deploy production environment |
| `docker-compose -f docker-compose.prod.yml logs -f` | View live logs |
| `docker-compose -f docker-compose.prod.yml restart` | Restart application |
| `docker-compose -f docker-compose.prod.yml down` | Stop environment |
| `sudo systemctl restart voice-assistant` | Restart via systemd |

### **Model Management (Production)**

```bash
# List models
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py list

# Switch model
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf

# Test model
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Hello!"

# Check memory usage
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py memory

# Unload current model
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py unload
```

## 🔧 **3. Configuration**

### **Environment Variables**

Both environments support these environment variables:

```bash
# Core settings
PYTHONPATH=/app
ASSISTANT_DEBUG=true          # Development: true, Production: false
ASSISTANT_LOG_LEVEL=DEBUG     # Development: DEBUG, Production: INFO

# TTS settings
TTS_ENGINE=piper

# Display settings
DISPLAY_STATIC_MODE=false
CAMERA_ENABLED=false          # Development: false, Production: true

# Lazy loading settings
LLM_LAZY_LOADING=true
LLM_CACHE_SIZE=1
LLM_AUTO_UNLOAD=true
LLM_UNLOAD_TIMEOUT=300
LLM_MODEL_PATH=/app/models/phi3-mini.gguf
```

### **Configuration Files**

Create `config/assistant_config.yaml`:

```yaml
name: "Voice Assistant"
version: "1.0.0"
debug: true
log_level: "DEBUG"

llm:
  provider_type: "local"
  model_path: "models/phi3-mini.gguf"
  context_window: 2048
  temperature: 0.7
  max_tokens: 512
  n_gpu_layers: 0
  n_threads: 4
  lazy_loading: true
  cache_size: 1
  auto_unload: true
  unload_timeout: 300

audio:
  sample_rate: 16000
  channels: 1
  wake_word_threshold: 0.5
  vad_threshold: 0.3

display:
  eyes_display: true
  mouth_display: true
  resolution: [800, 600]
  fps: 30
  static_mode: false

camera:
  enabled: false
  resolution: [640, 480]
  fps: 15
```

## 🧪 **4. Testing**

### **Integration Testing**

```bash
# Test event bus integration
docker-compose exec voice-assistant python scripts/test_event_bus_integration.py

# Test with specific model
docker-compose exec voice-assistant python scripts/test_event_bus_integration.py --model-path models/phi3-mini.gguf

# Test with custom prompt
docker-compose exec voice-assistant python scripts/test_event_bus_integration.py --prompt "Tell me about AI"
```

### **Model Testing**

```bash
# Test model discovery
docker-compose exec voice-assistant python scripts/model_manager_lazy.py list

# Test model switching
docker-compose exec voice-assistant python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf

# Test model inference
docker-compose exec voice-assistant python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Hello!"

# Test memory management
docker-compose exec voice-assistant python scripts/model_manager_lazy.py memory

# Test lazy loading
docker-compose exec voice-assistant python scripts/model_manager_lazy.py unload
```

## 🔍 **5. Monitoring & Debugging**

### **Logs**

```bash
# Development
docker-compose logs -f voice-assistant

# Production
docker-compose -f docker-compose.prod.yml logs -f voice-assistant

# Follow logs with timestamps
docker-compose logs -f -t voice-assistant
```

### **Resource Monitoring**

```bash
# Check container resources
docker stats voice-assistant-dev
docker stats voice-assistant-prod

# Check memory usage
docker-compose exec voice-assistant python scripts/model_manager_lazy.py memory

# Check disk usage
docker system df
```

### **Debugging**

```bash
# Access container shell
docker-compose exec voice-assistant bash

# Check running processes
docker-compose exec voice-assistant ps aux

# Check environment variables
docker-compose exec voice-assistant env | grep LLM

# Check mounted volumes
docker-compose exec voice-assistant ls -la /app/
```

## 🚨 **6. Troubleshooting**

### **Common Issues**

#### **Container Won't Start**
```bash
# Check Docker status
docker info

# Check logs
docker-compose logs voice-assistant

# Rebuild container
docker-compose up -d --build
```

#### **Model Loading Fails**
```bash
# Check model files
docker-compose exec voice-assistant ls -la /app/models/

# Check model path
docker-compose exec voice-assistant python scripts/model_manager_lazy.py list

# Test model loading
docker-compose exec voice-assistant python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf
```

#### **Out of Memory**
```bash
# Check memory usage
docker-compose exec voice-assistant python scripts/model_manager_lazy.py memory

# Reduce memory limits in docker-compose.yml
# Increase swap space on Raspberry Pi
```

#### **Audio Issues**
```bash
# Check audio devices
docker-compose exec voice-assistant ls -la /dev/snd/

# Check audio permissions
docker-compose exec voice-assistant groups
```

### **Performance Optimization**

#### **Development**
- Use smaller models for faster loading
- Enable lazy loading
- Set appropriate memory limits

#### **Production (Raspberry Pi)**
- Use quantized models (Q4_0, Q5_0)
- Enable ARM64 optimizations
- Monitor memory usage
- Use SSD storage for models

## 📋 **7. Quick Reference**

### **Development Commands**
```bash
# Start development
./scripts/deploy.sh dev

# View logs
docker-compose logs -f

# Test models
docker-compose exec voice-assistant python scripts/model_manager_lazy.py list

# Stop development
docker-compose down
```

### **Production Commands**
```bash
# Start production
./scripts/deploy.sh prod

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Test models
docker-compose -f docker-compose.prod.yml exec voice-assistant python scripts/model_manager_lazy.py list

# Stop production
docker-compose -f docker-compose.prod.yml down
```

### **Model Management Commands**
```bash
# List models
python scripts/model_manager_lazy.py list

# Switch model
python scripts/model_manager_lazy.py switch --model-path models/phi3-mini.gguf

# Test model
python scripts/model_manager_lazy.py test --model-path models/phi3-mini.gguf --prompt "Hello!"

# Check memory
python scripts/model_manager_lazy.py memory

# Unload model
python scripts/model_manager_lazy.py unload
```

This guide covers everything you need to develop and deploy your Voice Assistant with the new event bus architecture! 🎉

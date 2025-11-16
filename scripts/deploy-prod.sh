#!/bin/bash

# Production deployment script for Voice Assistant
# Optimized for Raspberry Pi deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check if running on ARM64
    if [[ $(uname -m) != "aarch64" ]]; then
        print_warning "This script is designed for ARM64 (Raspberry Pi). Current architecture: $(uname -m)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $MEMORY_GB -lt 2 ]]; then
        print_warning "Low memory detected: ${MEMORY_GB}GB. Recommended: 4GB+"
    fi
    
    # Check available disk space
    DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $DISK_GB -lt 10 ]]; then
        print_warning "Low disk space: ${DISK_GB}GB. Recommended: 20GB+"
    fi
    
    print_status "System requirements checked ✓"
}

# Function to check Docker status
check_docker() {
    print_status "Checking Docker status..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        print_status "Install Docker on Raspberry Pi:"
        echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "  sudo sh get-docker.sh"
        echo "  sudo usermod -aG docker \$USER"
        exit 1
    fi
    
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker."
        print_status "Start Docker: sudo systemctl start docker"
        exit 1
    fi
    
    # Check if user is in docker group
    if ! groups | grep -q docker; then
        print_warning "User not in docker group. You may need to use sudo for Docker commands."
    fi
    
    print_status "Docker is running ✓"
}

# Function to check hardware access
check_hardware_access() {
    print_status "Checking hardware access..."
    
    # Check audio devices
    if [[ ! -d "/dev/snd" ]]; then
        print_warning "Audio devices not found. Audio features may not work."
    else
        print_status "Audio devices found ✓"
    fi
    
    # Check display devices
    if [[ ! -e "/dev/fb0" ]]; then
        print_warning "Display device not found. Display features may not work."
    else
        print_status "Display device found ✓"
    fi
    
    # Check camera
    if ! ls /dev/video* > /dev/null 2>&1; then
        print_warning "Camera devices not found. Camera features may not work."
    else
        print_status "Camera devices found ✓"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data models voices config logs
    mkdir -p monitoring
    
    # Set proper permissions
    chmod 755 data models voices config logs monitoring
    
    print_status "Directories created ✓"
}

# Function to create production configuration
create_production_config() {
    print_status "Creating production configuration..."
    
    if [[ ! -f "config/assistant_config.yaml" ]]; then
        mkdir -p config
        cat > config/assistant_config.yaml << EOF
name: "Voice Assistant Production"
version: "1.0.0"
debug: false
log_level: "INFO"

tts:
  engine_type: "piper"
  model_path: "models/piper"
  emotion_support: true
  phoneme_output: true

audio:
  sample_rate: 16000
  channels: 1
  buffer_size: 1024
  device: "default"
  wake_word_threshold: 0.5
  vad_threshold: 0.3
  wake_words: ["hey assistant", "computer"]
  silence_timeout: 2.0
  max_audio_length: 30.0

display:
  eyes_display: true
  mouth_display: true
  resolution: [800, 600]
  fps: 30
  static_mode: false
  touch_enabled: true

camera:
  enabled: true
  interface: "usb"
  port: 0
  resolution: [640, 480]
  fps: 15
  detection_confidence: 0.7
  tracking_enabled: true

llm:
  model_path: "models/phi3-mini.gguf"
  context_window: 2048
  temperature: 0.7
  max_tokens: 512
  quantization: "q4_0"

conversation:
  buffer_size: 10
  max_age_seconds: 120
  include_visual_context: true
  response_threshold: 0.3

development:
  windows_testing: false
  mock_displays: false
  mock_audio: false
  mock_camera: false
  test_mode: false

data_dir: "data"
models_dir: "models"
voices_dir: "voices"
config_dir: "config"
logs_dir: "logs"
EOF
        print_status "Production configuration created ✓"
    else
        print_status "Configuration already exists ✓"
    fi
}

# Function to download models
download_models() {
    print_status "Downloading required models..."
    
    # Create models directory
    mkdir -p models
    
    # Download Piper TTS model (example)
    if [[ ! -d "models/piper" ]]; then
        print_status "Downloading Piper TTS model..."
        # This would be replaced with actual model download
        print_warning "Model download not implemented. Please download models manually."
    fi
    
    # Download LLM model (example)
    if [[ ! -f "models/phi3-mini.gguf" ]]; then
        print_status "Downloading LLM model..."
        # This would be replaced with actual model download
        print_warning "Model download not implemented. Please download models manually."
        print_status "For llama.cpp, you can download models from:"
        echo "  - Hugging Face: https://huggingface.co/models"
        echo "  - Example: wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf -O models/phi3-mini.gguf"
    fi
    
    print_status "Models setup completed ✓"
}

# Function to create SQLite database
create_sqlite_database() {
    print_status "Creating SQLite database..."
    
    mkdir -p data
    
    # Create a simple SQLite database for conversation history
    python3 -c "
import sqlite3
import os

db_path = 'data/conversations.db'
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            speaker TEXT NOT NULL,
            text TEXT NOT NULL,
            confidence REAL,
            session_id TEXT
        )
    ''')
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print('SQLite database created successfully')
else:
    print('SQLite database already exists')
"
    
    print_status "SQLite database setup completed ✓"
}

# Function to build production image
build_production_image() {
    print_status "Building production Docker image..."
    
    ./scripts/build.sh -t production-arm64 -p linux/arm64
    
    if [[ $? -eq 0 ]]; then
        print_status "Production image built successfully ✓"
    else
        print_error "Failed to build production image"
        exit 1
    fi
}

# Function to start production environment
start_production() {
    print_status "Starting production environment..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    
    # Start production environment
    docker-compose -f docker-compose.prod.yml up -d
    
    if [[ $? -eq 0 ]]; then
        print_status "Production environment started ✓"
    else
        print_error "Failed to start production environment"
        exit 1
    fi
}

# Function to setup systemd service
setup_systemd_service() {
    print_status "Setting up systemd service..."
    
    sudo tee /etc/systemd/system/voice-assistant.service > /dev/null << EOF
[Unit]
Description=Voice Assistant
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable voice-assistant.service
    
    print_status "Systemd service created ✓"
}

# Function to show status
show_status() {
    print_status "Production environment status:"
    echo ""
    docker-compose -f docker-compose.prod.yml ps
    echo ""
    print_status "Application Status:"
    echo "  - Voice Assistant: Running in container"
    echo "  - Architecture: Event Bus (No HTTP API)"
    echo "  - Direct execution mode"
    echo ""
    print_status "To view logs: docker-compose -f docker-compose.prod.yml logs -f voice-assistant"
    print_status "To stop: docker-compose -f docker-compose.prod.yml down"
    print_status "To restart: sudo systemctl restart voice-assistant"
    print_status "To interact: Use model_manager_lazy.py script"
}

# Main deployment process
main() {
    print_header "Starting Voice Assistant Production Deployment"
    
    check_system_requirements
    check_docker
    check_hardware_access
    create_directories
    create_production_config
    create_sqlite_database
    download_models
    build_production_image
    start_production
    setup_systemd_service
    show_status
    
    print_header "Production deployment completed successfully!"
    print_status "The Voice Assistant will start automatically on boot."
}

# Run main function
main "$@"

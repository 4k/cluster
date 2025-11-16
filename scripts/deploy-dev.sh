#!/bin/bash

# Development deployment script for Voice Assistant
# Optimized for Windows/WSL development environment

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

# Function to check Docker status
check_docker() {
    print_status "Checking Docker status..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker Desktop."
        exit 1
    fi
    
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_status "Docker is running ✓"
}

# Function to check WSL audio setup
check_audio_setup() {
    print_status "Checking audio setup..."
    
    if [[ ! -d "/dev/snd" ]]; then
        print_warning "Audio devices not found. Audio features may not work."
        print_warning "Make sure you're running in WSL2 with audio support enabled."
    else
        print_status "Audio devices found ✓"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data models voices config logs
    mkdir -p monitoring
    
    print_status "Directories created ✓"
}

# Function to create default configuration
create_default_config() {
    print_status "Creating default configuration..."
    
    if [[ ! -f "config/assistant_config.yaml" ]]; then
        mkdir -p config
        cat > config/assistant_config.yaml << EOF
name: "Voice Assistant Development"
version: "1.0.0"
debug: true
log_level: "DEBUG"

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
  touch_enabled: false

camera:
  enabled: false
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
  windows_testing: true
  mock_displays: true
  mock_audio: false
  mock_camera: true
  test_mode: false

data_dir: "data"
models_dir: "models"
voices_dir: "voices"
config_dir: "config"
logs_dir: "logs"
EOF
        print_status "Default configuration created ✓"
    else
        print_status "Configuration already exists ✓"
    fi
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

# Function to build development image
build_development_image() {
    print_status "Building development Docker image..."
    
    ./scripts/build.sh -t development
    
    if [[ $? -eq 0 ]]; then
        print_status "Development image built successfully ✓"
    else
        print_error "Failed to build development image"
        exit 1
    fi
}

# Function to start development environment
start_development() {
    print_status "Starting development environment..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Start development environment
    docker-compose up -d
    
    if [[ $? -eq 0 ]]; then
        print_status "Development environment started ✓"
    else
        print_error "Failed to start development environment"
        exit 1
    fi
}

# Function to show status
show_status() {
    print_status "Development environment status:"
    echo ""
    docker-compose ps
    echo ""
    print_status "Application Status:"
    echo "  - Voice Assistant: Running in container"
    echo "  - Architecture: Event Bus (No HTTP API)"
    echo "  - Direct execution mode"
    echo ""
    print_status "To view logs: docker-compose logs -f voice-assistant"
    print_status "To stop: docker-compose down"
    print_status "To interact: Use model_manager_lazy.py script"
}

# Main deployment process
main() {
    print_header "Starting Voice Assistant Development Deployment"
    
    check_docker
    check_audio_setup
    create_directories
    create_default_config
    create_sqlite_database
    build_development_image
    start_development
    show_status
    
    print_header "Development deployment completed successfully!"
}

# Run main function
main "$@"

#!/bin/bash
# Voice Assistant - Docker Installation Script
# Installs Docker (if needed) and sets up the complete application
# Use this script when setting up on a fresh machine or Raspberry Pi

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${CYAN}   Voice Assistant - Docker Installation${NC}"
    echo -e "${CYAN}================================================${NC}"
    echo ""
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    print_status "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Detected: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "Detected: macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_success "Detected: Windows"
    else
        OS="unknown"
        print_warning "Unknown operating system"
    fi
    echo ""
}

# Check if running as root (Linux)
check_root() {
    if [[ "$OS" == "linux" ]] && [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This script should be run as a regular user."
        print_warning "The script will use 'sudo' when necessary."
    fi
}

# Install Docker on Linux
install_docker_linux() {
    print_status "Checking Docker installation on Linux..."
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_success "Docker is already installed: $DOCKER_VERSION"
        
        # Check if user is in docker group
        if groups | grep -q docker; then
            print_success "User is in docker group"
            return 0
        else
            print_warning "User not in docker group. Adding..."
            echo ""
            echo "To complete Docker setup, you may need to:"
            echo "  1. Log out and back in, OR"
            echo "  2. Run: newgrp docker"
            echo "  3. Then run this script again"
            echo ""
            read -p "Would you like to add yourself to docker group and logout? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo usermod -aG docker $USER
                print_success "User added to docker group. Please logout and login, then run this script again."
                exit 0
            fi
        fi
    else
        print_warning "Docker is not installed. Installing..."
        
        # Detect distribution
        if command -v apt-get &> /dev/null; then
            print_status "Detected Debian/Ubuntu. Installing Docker..."
            
            # Add Docker's official GPG key
            print_status "Adding Docker's official GPG key..."
            sudo apt-get update
            sudo apt-get install -y ca-certificates curl gnupg lsb-release
            
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg
            
            # Setup Docker repository
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
              $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Install Docker
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            
            # Add user to docker group
            sudo usermod -aG docker $USER
            print_success "Docker installed successfully!"
            print_warning "You need to logout and login again, or run: newgrp docker"
            
            # Try to activate docker group
            newgrp docker <<EOF
print_success "Docker group activated in this session"
EOF
            
        elif command -v yum &> /dev/null; then
            print_status "Detected RedHat/CentOS. Installing Docker..."
            sudo yum install -y docker
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            print_success "Docker installed successfully!"
        else
            print_error "Unsupported Linux distribution"
            print_warning "Please install Docker manually: https://docs.docker.com/get-docker/"
            exit 1
        fi
    fi
}

# Install Docker on macOS
install_docker_macos() {
    print_status "Checking Docker on macOS..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker appears to be installed"
    else
        print_error "Docker Desktop is not installed on macOS"
        echo ""
        print_warning "Please install Docker Desktop for macOS:"
        echo "  1. Download from: https://docs.docker.com/desktop/install/mac-install/"
        echo "  2. Install and start Docker Desktop"
        echo "  3. Run this script again"
        exit 1
    fi
}

# Install Docker on Windows
install_docker_windows() {
    print_status "Checking Docker on Windows..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker appears to be installed"
    else
        print_error "Docker Desktop is not installed on Windows"
        echo ""
        print_warning "Please install Docker Desktop for Windows:"
        echo "  1. Download from: https://docs.docker.com/desktop/install/windows-install/"
        echo "  2. Install and start Docker Desktop"
        echo "  3. Run this script again"
        exit 1
    fi
}

# Check Docker installation based on OS
check_docker() {
    print_status "Checking Docker installation..."
    
    case $OS in
        linux)
            install_docker_linux
            ;;
        macos)
            install_docker_macos
            ;;
        windows)
            install_docker_windows
            ;;
        *)
            print_error "Cannot determine how to install Docker for this OS"
            exit 1
            ;;
    esac
    
    # Verify Docker is working
    if docker info &> /dev/null; then
        print_success "Docker is running ✓"
    else
        print_error "Docker is not running!"
        print_warning "Please start Docker Desktop or the Docker daemon"
        
        if [[ "$OS" == "linux" ]]; then
            print_status "Attempting to start Docker daemon..."
            sudo systemctl start docker
        fi
        
        if ! docker info &> /dev/null; then
            print_error "Docker is still not running. Please start it manually."
            exit 1
        fi
    fi
}

# Check Docker Compose
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    # Try new compose command first
    if docker compose version &> /dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
        print_success "Docker Compose V2 is available ✓"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
        print_success "Docker Compose is installed ✓"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi
    echo ""
}

# Check/create .env file
check_env_file() {
    print_status "Checking for .env file..."
    
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from env.example..."
        
        if [ -f "env.example" ]; then
            cp env.example .env
            print_success "Created .env file from env.example"
            echo ""
            print_warning "IMPORTANT: Please review .env file with your preferred settings!"
            echo ""
            read -p "Press Enter to continue after reviewing .env..."
        else
            print_error "env.example not found!"
            exit 1
        fi
    else
        print_success ".env file exists ✓"
    fi
    echo ""
}

# Build and start application
build_and_start() {
    print_status "Building Docker images (this may take several minutes)..."
    echo ""
    
    if $DOCKER_COMPOSE_CMD build; then
        print_success "Docker images built successfully ✓"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
    echo ""
    
    print_status "Creating necessary directories..."
    mkdir -p data models voices config logs
    print_success "Directories created ✓"
    echo ""
    
    print_status "Starting Voice Assistant application..."
    $DOCKER_COMPOSE_CMD up -d
    print_success "Application started ✓"
    echo ""
}

# Show next steps
show_next_steps() {
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}   Installation Complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "Your Voice Assistant is now running!"
    echo ""
    echo "Useful commands:"
    echo "  View logs:        docker-compose logs -f"
    echo "  Stop app:         docker-compose down"
    echo "  Restart app:      docker-compose restart"
    echo "  Check status:     docker-compose ps"
    echo ""
    echo "The application will download models on first run."
    echo "Check logs with: docker-compose logs -f"
    echo ""
}

# Main installation
main() {
    print_header
    
    detect_os
    check_root
    
    check_env_file
    check_docker
    check_docker_compose
    build_and_start
    show_next_steps
}

# Run installation
main "$@"


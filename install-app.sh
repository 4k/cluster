#!/bin/bash
# Voice Assistant - Application Installation Script  
# THIS SCRIPT RUNS INSIDE THE DOCKER CONTAINER
# Downloads models and sets up the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data models voices config logs
    print_success "Directories created ✓"
}

# Run application installation
run_installation() {
    print_status "Running application installation..."
    echo ""
    
    # The install.py script handles:
    # - Creating directories
    # - Downloading models
    # - Setting up configuration
    python scripts/install.py
    
    print_success "Application installation complete ✓"
}

# Main installation workflow
main() {
    echo ""
    print_status "================================================"
    print_status "Voice Assistant - Application Installation"
    print_status "================================================"
    echo ""
    
    create_directories
    run_installation
    
    echo ""
    print_success "================================================"
    print_success "Installation Complete!"
    print_success "================================================"
    echo ""
}

# Run installation
main "$@"

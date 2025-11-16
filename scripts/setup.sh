#!/bin/bash
# Simple Setup Script
# Just builds the Docker image and shows usage instructions

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Voice Assistant Setup${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        echo "   Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose first."
        echo "   Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_status "Prerequisites check completed."
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data models voices logs config
    print_status "Directories created successfully."
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    print_status "Docker image built successfully."
}

# Show usage instructions
show_usage() {
    echo ""
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}To start the Voice Assistant:${NC}"
    echo "  docker-compose up -d"
    echo ""
    echo -e "${BLUE}To view logs:${NC}"
    echo "  docker-compose logs -f"
    echo ""
    echo -e "${BLUE}To stop the Voice Assistant:${NC}"
    echo "  docker-compose down"
    echo ""
    echo -e "${BLUE}To restart:${NC}"
    echo "  docker-compose restart"
    echo ""
    echo -e "${BLUE}Mock Mode:${NC}"
    echo "  The application uses mock providers by default (no real hardware needed)"
    echo "  To use real hardware, edit .env and set MOCK_MODE=false"
    echo ""
}

# Main setup function
main() {
    print_header
    check_prerequisites
    create_directories
    build_image
    show_usage
}

# Show help
show_help() {
    echo "Voice Assistant Setup Script"
    echo ""
    echo "Usage: $0"
    echo ""
    echo "This script:"
    echo "  - Checks prerequisites (Docker, Docker Compose)"
    echo "  - Creates necessary directories"
    echo "  - Builds the Docker image"
    echo "  - Shows usage instructions"
    echo ""
    echo "After setup, use:"
    echo "  docker-compose up -d  # Start the application"
    echo "  docker-compose down  # Stop the application"
}

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
#!/bin/bash
# Simple deployment script for Voice Assistant

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Parse command line arguments
ENVIRONMENT=""
BUILD=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        dev|development)
            ENVIRONMENT="dev"
            shift
            ;;
        prod|production)
            ENVIRONMENT="prod"
            shift
            ;;
        --build|-b)
            BUILD=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            HELP=true
            shift
            ;;
    esac
done

# Show help
if [ "$HELP" = true ] || [ -z "$ENVIRONMENT" ]; then
    echo "Voice Assistant Deployment Script"
    echo ""
    echo "Usage: $0 [ENVIRONMENT] [OPTIONS]"
    echo ""
    echo "Environments:"
    echo "  dev, development    Deploy for development (Windows WSL)"
    echo "  prod, production    Deploy for production (Raspberry Pi)"
    echo ""
    echo "Options:"
    echo "  --build, -b         Force rebuild of Docker images"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev              Deploy development environment"
    echo "  $0 prod --build     Deploy production with rebuild"
    echo "  $0 dev -b           Deploy development with rebuild"
    exit 0
fi

# Set compose file based on environment
if [ "$ENVIRONMENT" = "dev" ]; then
    COMPOSE_FILE="docker-compose.yml"
    print_status "Deploying development environment (Windows WSL)"
elif [ "$ENVIRONMENT" = "prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    print_status "Deploying production environment (Raspberry Pi)"
fi

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "Docker Compose file not found: $COMPOSE_FILE"
    exit 1
fi

# Build images if requested
if [ "$BUILD" = true ]; then
    print_status "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose -f "$COMPOSE_FILE" down

# Start containers
print_status "Starting containers..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 5

# Check if services are running
if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
    print_status "Deployment successful!"
    echo ""
    echo "Services:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    echo "Application Status:"
    echo "  Voice Assistant: Running in container"
    echo "  Architecture: Event Bus (No HTTP API)"
    echo "  Logs: Available via docker-compose logs"
    echo ""
    echo "Useful commands:"
    echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  Stop: docker-compose -f $COMPOSE_FILE down"
    echo "  Restart: docker-compose -f $COMPOSE_FILE restart"
else
    print_error "Deployment failed. Check logs:"
    docker-compose -f "$COMPOSE_FILE" logs
    exit 1
fi

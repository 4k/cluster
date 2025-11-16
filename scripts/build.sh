#!/bin/bash

# Build script for Voice Assistant Docker images
# Supports both development and production builds

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TARGET="development"
PLATFORM="linux/amd64"
TAG="voice-assistant"
PUSH=false
REGISTRY=""

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET     Build target (development|production|production-arm64) [default: development]"
    echo "  -p, --platform PLATFORM Platform to build for [default: linux/amd64]"
    echo "  --tag TAG              Image tag [default: voice-assistant]"
    echo "  --push                 Push image to registry after building"
    echo "  --registry REGISTRY    Registry to push to (e.g., your-registry.com)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build development image for current platform"
    echo "  $0 -t production                      # Build production image"
    echo "  $0 -t production-arm64 -p linux/arm64 # Build ARM64 image for Raspberry Pi"
    echo "  $0 --push --registry my-registry.com # Build and push to registry"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$TARGET" != "development" && "$TARGET" != "production" && "$TARGET" != "production-arm64" ]]; then
    print_error "Invalid target: $TARGET. Must be one of: development, production, production-arm64"
    exit 1
fi

# Set platform based on target if not specified
if [[ "$TARGET" == "production-arm64" && "$PLATFORM" == "linux/amd64" ]]; then
    PLATFORM="linux/arm64"
    print_warning "Automatically setting platform to linux/arm64 for production-arm64 target"
fi

# Build image name
if [[ -n "$REGISTRY" ]]; then
    IMAGE_NAME="${REGISTRY}/${TAG}:${TARGET}-$(date +%Y%m%d-%H%M%S)"
    IMAGE_NAME_LATEST="${REGISTRY}/${TAG}:${TARGET}-latest"
else
    IMAGE_NAME="${TAG}:${TARGET}-$(date +%Y%m%d-%H%M%S)"
    IMAGE_NAME_LATEST="${TAG}:${TARGET}-latest"
fi

print_status "Building Voice Assistant Docker image"
print_status "Target: $TARGET"
print_status "Platform: $PLATFORM"
print_status "Image: $IMAGE_NAME"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    print_warning "Docker buildx not available. Using regular docker build."
    BUILD_CMD="docker build"
else
    print_status "Using Docker buildx for multi-platform builds"
    BUILD_CMD="docker buildx build"
fi

# Build the image
print_status "Starting build process..."

if [[ "$BUILD_CMD" == "docker buildx build" ]]; then
    # Use buildx for multi-platform builds
    $BUILD_CMD \
        --platform "$PLATFORM" \
        --target "$TARGET" \
        --tag "$IMAGE_NAME" \
        --tag "$IMAGE_NAME_LATEST" \
        --load \
        .
else
    # Use regular docker build
    $BUILD_CMD \
        --target "$TARGET" \
        --tag "$IMAGE_NAME" \
        --tag "$IMAGE_NAME_LATEST" \
        .
fi

if [[ $? -eq 0 ]]; then
    print_status "Build completed successfully!"
    print_status "Image: $IMAGE_NAME"
    print_status "Latest: $IMAGE_NAME_LATEST"
    
    # Push to registry if requested
    if [[ "$PUSH" == true ]]; then
        if [[ -z "$REGISTRY" ]]; then
            print_error "Registry not specified. Use --registry to specify a registry."
            exit 1
        fi
        
        print_status "Pushing image to registry..."
        docker push "$IMAGE_NAME"
        docker push "$IMAGE_NAME_LATEST"
        
        if [[ $? -eq 0 ]]; then
            print_status "Image pushed successfully!"
        else
            print_error "Failed to push image to registry"
            exit 1
        fi
    fi
    
    # Show image size
    print_status "Image size:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
else
    print_error "Build failed!"
    exit 1
fi

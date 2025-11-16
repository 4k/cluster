#!/bin/bash
# Docker Build Script with BuildKit
# Automatically enables BuildKit for builds
# Usage: ./build.sh [options]

set -e

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Parse arguments
NO_CACHE=""
PULL=""
DETACHED=false
TARGET="development"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --pull)
            PULL="--pull"
            shift
            ;;
        -d|--detached)
            DETACHED=true
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}🔧 Building Voice Assistant with BuildKit...${NC}"

# Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build command
BUILD_CMD="docker-compose build"

if [ -n "$NO_CACHE" ]; then
    BUILD_CMD="$BUILD_CMD $NO_CACHE"
    echo -e "   ${YELLOW}Using --no-cache (fresh build)${NC}"
fi

if [ -n "$PULL" ]; then
    BUILD_CMD="$BUILD_CMD $PULL"
    echo -e "   ${YELLOW}Using --pull (update base images)${NC}"
fi

echo -e "   ${GREEN}Target: $TARGET${NC}"
echo -e "   ${GREEN}BuildKit: Enabled ✓${NC}"
echo ""

# Execute build
eval $BUILD_CMD

echo -e "${GREEN}✓ Build completed successfully!${NC}"

# Optionally start the container
if [ "$DETACHED" = true ]; then
    echo ""
    echo -e "${CYAN}🚀 Starting container in detached mode...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✓ Container started!${NC}"
    echo ""
    echo -e "${GRAY}View logs with: docker-compose logs -f${NC}"
else
    echo ""
    echo -e "${GRAY}To start the container, run: docker-compose up -d${NC}"
fi


# Voice Assistant - Makefile
# Automatically enables BuildKit for all Docker operations
# Usage: make [target]

# Export BuildKit environment variables for all commands
export DOCKER_BUILDKIT := 1
export COMPOSE_DOCKER_CLI_BUILD := 1

.PHONY: help build build-no-cache up down restart logs clean dev prod

# Default target
help:
	@echo "Voice Assistant - Docker Commands"
	@echo "=================================="
	@echo ""
	@echo "Development:"
	@echo "  make build         - Build container with cache"
	@echo "  make build-fresh   - Build container without cache"
	@echo "  make up            - Start container (detached)"
	@echo "  make dev           - Build and start in development mode"
	@echo ""
	@echo "Operations:"
	@echo "  make down          - Stop and remove container"
	@echo "  make restart       - Restart container"
	@echo "  make logs          - Show container logs (follow)"
	@echo "  make shell         - Open shell in container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Remove container and volumes"
	@echo "  make prune         - Clean Docker system (images, cache)"
	@echo ""
	@echo "Production:"
	@echo "  make prod          - Build and run production image"
	@echo ""
	@echo "Note: BuildKit is automatically enabled ✓"

# Build with cache
build:
	@echo "🔧 Building with BuildKit (cache enabled)..."
	docker-compose build

# Build without cache (fresh build)
build-fresh:
	@echo "🔧 Building with BuildKit (no cache)..."
	docker-compose build --no-cache

# Build with latest base images
build-pull:
	@echo "🔧 Building with BuildKit (pulling latest images)..."
	docker-compose build --pull

# Start container
up:
	@echo "🚀 Starting container..."
	docker-compose up -d
	@echo "✓ Container started! View logs with: make logs"

# Build and start (common workflow)
dev: build up
	@echo "✓ Development environment ready!"

# Stop container
down:
	@echo "⏹️  Stopping container..."
	docker-compose down

# Restart container
restart:
	@echo "🔄 Restarting container..."
	docker-compose restart

# Show logs (follow mode)
logs:
	docker-compose logs -f

# Open shell in running container
shell:
	docker-compose exec voice-assistant /bin/bash

# Clean everything (containers, volumes)
clean:
	@echo "🧹 Cleaning containers and volumes..."
	docker-compose down -v
	@echo "✓ Cleaned!"

# Prune Docker system
prune:
	@echo "🧹 Pruning Docker system..."
	docker system prune -af
	docker builder prune -af
	@echo "✓ Docker system cleaned!"

# Production build and run
prod:
	@echo "🚀 Building production image..."
	docker-compose -f docker-compose.yml build --target production
	docker-compose up -d
	@echo "✓ Production container started!"

# Check Docker and BuildKit status
status:
	@echo "Docker Status:"
	@echo "=============="
	@docker version --format 'Client: {{.Client.Version}}'
	@docker version --format 'Server: {{.Server.Version}}'
	@echo ""
	@echo "BuildKit Status:"
	@echo "================"
	@echo "DOCKER_BUILDKIT: $(DOCKER_BUILDKIT)"
	@echo "COMPOSE_DOCKER_CLI_BUILD: $(COMPOSE_DOCKER_CLI_BUILD)"
	@echo ""
	@echo "Containers:"
	@echo "==========="
	@docker-compose ps

# Quick rebuild (down, build, up)
rebuild: down build up
	@echo "✓ Rebuild complete!"


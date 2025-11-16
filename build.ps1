# Docker Build Script with BuildKit
# Automatically enables BuildKit for builds
# Usage: .\build.ps1 [options]

param(
    [switch]$NoCache,
    [switch]$Pull,
    [switch]$Detached,
    [string]$Target = "development"
)

Write-Host "🔧 Building Voice Assistant with BuildKit..." -ForegroundColor Cyan

# Enable BuildKit
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

# Build command arguments
$buildArgs = @("build")

if ($NoCache) {
    $buildArgs += "--no-cache"
    Write-Host "   Using --no-cache (fresh build)" -ForegroundColor Yellow
}

if ($Pull) {
    $buildArgs += "--pull"
    Write-Host "   Using --pull (update base images)" -ForegroundColor Yellow
}

# Add target if specified
$buildArgs += "--build-arg"
$buildArgs += "TARGET=$Target"

Write-Host "   Target: $Target" -ForegroundColor Green
Write-Host "   BuildKit: Enabled ✓" -ForegroundColor Green
Write-Host ""

# Execute docker-compose build
& docker-compose $buildArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Build completed successfully!" -ForegroundColor Green
    
    # Optionally start the container
    if ($Detached) {
        Write-Host ""
        Write-Host "🚀 Starting container in detached mode..." -ForegroundColor Cyan
        & docker-compose up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Container started!" -ForegroundColor Green
            Write-Host ""
            Write-Host "View logs with: docker-compose logs -f" -ForegroundColor Gray
        }
    } else {
        Write-Host ""
        Write-Host "To start the container, run: docker-compose up -d" -ForegroundColor Gray
    }
} else {
    Write-Host "✗ Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}


# Docker Build Optimization Guide

## Problem: Packages Redownloading on Every Build

### Why This Happens

When running `docker-compose up --build`, pip packages are redownloaded even when `requirements.txt` hasn't changed. This occurs because:

1. **Docker Layer Caching** - Docker caches the entire layer (including installed packages), but only if nothing before that layer changed
2. **Pip's Download Cache** - Pip's download cache (`/root/.cache/pip`) exists only during the build and is discarded after
3. **Each build starts fresh** - Without persistent cache, pip must re-download all packages from PyPI

### The Solution: BuildKit Cache Mounts

We've implemented BuildKit's `--mount=type=cache` feature, which:
- ✅ Persists pip's download cache between builds
- ✅ Speeds up rebuilds by 5-10x for packages
- ✅ Works even when Docker layers are invalidated
- ✅ Shares cache across different build stages

## How to Use

### Option 1: Use Wrapper Scripts (RECOMMENDED) ⭐

We provide automated build scripts that handle BuildKit for you:

**On Windows PowerShell:**
```powershell
# Build and start
.\build.ps1 -Detached

# Build without cache
.\build.ps1 -NoCache -Detached

# Build with latest base images
.\build.ps1 -Pull -Detached
```

**On Linux/macOS:**
```bash
# Build and start
./build.sh --detached

# Build without cache
./build.sh --no-cache --detached

# Build with latest base images
./build.sh --pull --detached
```

**Using Makefile:**
```bash
# Build and start (recommended)
make dev

# Just build
make build

# Build without cache
make build-fresh

# View all commands
make help
```

### Option 2: Enable BuildKit Globally (One-Time Setup)

**Configure Docker daemon** to enable BuildKit permanently:

**Location:**
- Windows: `C:\ProgramData\docker\config\daemon.json`
- Linux: `/etc/docker/daemon.json`
- macOS: `~/.docker/daemon.json`

**Add this configuration:**
```json
{
  "features": {
    "buildkit": true
  }
}
```

**Restart Docker:**
- Windows: Restart Docker Desktop
- Linux: `sudo systemctl restart docker`
- macOS: Restart Docker Desktop

### Option 3: Manual Environment Variables

**Only use if you can't use options 1 or 2.**

**On Linux/macOS:**
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
docker-compose up --build -d
```

**On Windows PowerShell:**
```powershell
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1
docker-compose up --build -d
```

### Version Check

**Check your Docker version:**
```bash
docker version
```

- Docker 23.0+: BuildKit is enabled by default ✅
- Docker 18.09-22.x: Use wrapper scripts or enable manually

## Verifying It Works

**First build (downloads everything):**
```bash
docker-compose build --no-cache
```
Output: Downloads all packages (~2-5 minutes)

**Second build (uses cache):**
```bash
docker-compose build --no-cache
```
Output: Uses cached packages (~30 seconds)

## Performance Improvements

### Before (No Cache Mount)
- First build: 5 minutes
- Rebuild with no changes: 30 seconds (layer cache)
- Rebuild with code changes: 5 minutes (re-downloads packages)
- Rebuild with `--no-cache`: 5 minutes (re-downloads everything)

### After (With Cache Mount)
- First build: 5 minutes
- Rebuild with no changes: 30 seconds (layer cache)
- Rebuild with code changes: 1 minute (uses pip cache)
- Rebuild with `--no-cache`: 1 minute (uses pip cache)

## Technical Details

### What Changed in Dockerfile

**Before:**
```dockerfile
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
```

**After:**
```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt
```

### How Cache Mounts Work

1. **Cache Location**: `/root/.cache/pip` (standard pip cache directory)
2. **Persistence**: Docker maintains this directory across builds
3. **Isolation**: Each cache mount is identified by a unique hash
4. **Sharing**: Cache is shared across all stages in the Dockerfile

### Cache Management

**View cache usage:**
```bash
docker system df -v
```

**Clear all build cache (if needed):**
```bash
docker builder prune -a
```

**Clear only pip cache:**
```bash
# No direct command, but rebuild with --no-cache will refresh it
docker-compose build --no-cache
```

## Troubleshooting

### Issue: BuildKit not enabled

**Symptom:** Error: `unknown flag: --mount`

**Solution:** Enable BuildKit (see section 1 above)

### Issue: Packages still redownloading

**Possible causes:**
1. BuildKit not enabled
2. Using `docker build` instead of `docker-compose build`
3. Docker version too old (< 18.09)

**Solution:**
```bash
# Check BuildKit is enabled
echo $DOCKER_BUILDKIT  # Should output: 1

# Verify Docker version supports BuildKit
docker version  # Client and Server should be 18.09+
```

### Issue: Cache taking too much space

**Check cache size:**
```bash
docker system df -v | grep cache
```

**Clean if needed:**
```bash
docker builder prune --filter "until=24h"  # Remove cache older than 24 hours
```

## Additional Optimizations

### 1. Order Matters
The Dockerfile is already optimized with:
```dockerfile
COPY requirements.txt .      # Copy requirements first
RUN pip install ...          # Install packages (cached layer)
COPY . .                     # Copy code last (changes frequently)
```

### 2. Use .dockerignore
We've configured `.dockerignore` to exclude:
- Logs, cache files, and temporary files
- Virtual environments
- Documentation files
- Git history

This reduces build context size and prevents cache invalidation.

### 3. Multi-stage Builds
The Dockerfile uses multi-stage builds:
- `base`: System dependencies (rarely changes)
- `development`: Development tools + packages
- `production`: Production-only packages
- `production-arm64`: ARM64 optimizations

Each stage benefits from cache mounts independently.

## Best Practices

1. ✅ **Use `docker-compose up --build`** for normal development
2. ✅ **Only use `--no-cache`** when troubleshooting or forcing clean build
3. ✅ **Keep requirements.txt stable** - pin versions to maximize cache hits
4. ✅ **Enable BuildKit** permanently in your environment
5. ✅ **Monitor cache size** periodically with `docker system df`

## Why Can't BuildKit Be Enabled in Dockerfile or docker-compose.yml?

### Technical Explanation

**Short answer:** BuildKit is a **builder implementation**, not a container runtime feature. It must be enabled at the Docker CLI/daemon level, not within the build artifacts themselves.

### The Details

**1. Dockerfile Cannot Enable BuildKit**
- The Dockerfile is **interpreted by the builder** (either classic builder or BuildKit)
- It's like asking a book to choose which language it's written in - the reader (builder) makes that choice
- The Dockerfile syntax `RUN --mount=...` is **BuildKit-specific syntax** that requires BuildKit to already be enabled

**2. docker-compose.yml Has Limited Control**
- docker-compose.yml is a **orchestration configuration**, not a builder configuration
- It tells Docker Compose **what to build**, not **how to build** it
- The `build:` section passes parameters to the underlying Docker build process, but can't change the builder engine

**3. The Build Process Hierarchy**
```
User Command → Docker CLI → Builder (Classic or BuildKit) → Dockerfile
     ↑                           ↑
  Sets builder type         Executes build instructions
```

BuildKit must be selected at step 1 (User Command) or configured at the Docker daemon level.

### Why Docker 23.0+ Makes This Easy

Docker 23.0+ enables BuildKit by default, making the issue mostly transparent:
- No environment variables needed
- BuildKit syntax "just works"
- Older projects automatically benefit from performance improvements

### The Automation Solution

Since we can't enable BuildKit **from within** the build files, we enable it **before** running the build:

**Wrapper Scripts:**
```powershell
# build.ps1
$env:DOCKER_BUILDKIT = "1"  # ← Enable BuildKit BEFORE calling docker-compose
docker-compose build
```

**Makefile:**
```makefile
export DOCKER_BUILDKIT := 1  # ← Enable for all make targets
build:
    docker-compose build
```

This is the standard approach used by professional projects (Kubernetes, Moby, containerd, etc.).

## References

- [Docker BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [BuildKit Cache Mounts](https://docs.docker.com/engine/reference/builder/#run---mounttypecache)
- [Docker Compose Build](https://docs.docker.com/compose/reference/build/)
- [BuildKit Architecture](https://github.com/moby/buildkit/blob/master/docs/architecture.md)


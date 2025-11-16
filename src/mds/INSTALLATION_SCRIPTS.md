# Installation Scripts Overview

This project has **two distinct installation scripts** for different purposes:

## 1. `install-docker.sh` - Full System Installation

**Purpose:** Installs Docker and sets up the complete application on target hardware (Raspberry Pi, VPS, etc.)

**When to use:**
- ✅ Fresh system installation (Raspberry Pi, Linux server)
- ✅ First-time setup on new hardware
- ✅ Setting up production deployment

**What it does:**
1. Detects operating system (Linux, macOS, Windows)
2. Installs Docker if not present
3. Sets up Docker Compose
4. Configures `.env` file from template
5. Builds Docker images
6. Starts the application
7. Downloads models automatically on first run

**Usage:**
```bash
chmod +x install-docker.sh
./install-docker.sh
```

**Example scenarios:**
```bash
# Fresh Raspberry Pi installation
./install-docker.sh

# Will automatically:
# - Detect it's a Raspberry Pi
# - Install Docker if needed
# - Build ARM64 optimized images
# - Start the application
```

---

## 2. `install-app.sh` - Application Setup (Runs INSIDE Container)

**Purpose:** Runs inside the Docker container to setup the application and download models

**When to use:**
- ✅ Automatically called by the container entrypoint
- ✅ Called manually inside the container for troubleshooting
- ✅ NOT meant to be run on the host!

**What it does:**
1. Creates necessary directories
2. Runs `scripts/install.py` to download models
3. Sets up configuration files
4. Marks installation as complete

**Usage:**

**From outside container (Docker Compose):**
```bash
# Start container - will automatically run install-app.sh
docker-compose up -d

# View installation logs
docker-compose logs -f
```

**From inside container (for troubleshooting):**
```bash
# Enter the running container
docker-compose exec voice-assistant sh

# Manually run installation
./install-app.sh
```

**Architecture:**
```
Container Startup Flow:
┌─────────────────────────────┐
│  docker-entrypoint.sh      │
│  - Checks if installed     │
│  - Calls install-app.sh    │
│  - Marks as installed      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  install-app.sh             │
│  - Creates directories      │
│  - Calls scripts/install.py │
│  - Downloads models         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  scripts/install.py         │
│  - Python installation      │
│  - Model download           │
│  - Config setup             │
└─────────────────────────────┘
```

---

## Summary

| Script | Runs On | Purpose | User Action |
|--------|---------|---------|-------------|
| `install-docker.sh` | **Host system** | Install Docker + build images | Manual run |
| `install-app.sh` | **Inside container** | Setup app + download models | Auto on startup |

---

## Typical Workflows

### Fresh System Setup (Raspberry Pi)

```bash
# 1. User clones repository
git clone <repo>
cd cluster

# 2. Run full installation
chmod +x install-docker.sh
./install-docker.sh

# What happens:
# - Docker is installed
# - .env is created from env.example
# - Images are built
# - Container starts
# - install-app.sh runs INSIDE container (auto)
# - Models are downloaded
# - Application starts
```

### Development Setup (Already has Docker)

```bash
# 1. User has Docker running
docker --version ✓

# 2. Configure .env
cp env.example .env
nano .env  # Edit configuration

# 3. Build and start
docker-compose build
docker-compose up -d

# What happens:
# - Images are built (if not cached)
# - Container starts
# - install-app.sh runs INSIDE container (auto)
# - Models are downloaded
# - Application starts
```

---

## Troubleshooting

### Problem: "install-app.sh not found"
**Solution:** You're trying to run it on the host. It only runs inside the container.

### Problem: Container exits immediately
**Solution:** Check logs
```bash
docker-compose logs voice-assistant
```

### Problem: Models not downloading
**Solution:** Check .env file is configured correctly
```bash
# Verify .env exists
ls -la .env

# Check environment inside container
docker-compose exec voice-assistant env | grep LLM
```

### Problem: Manual installation needed
**Solution:** Run install-app.sh manually inside container
```bash
# Enter container
docker-compose exec voice-assistant sh

# Run installation
./install-app.sh

# Exit container
exit
```

---

## File Locations

### On Host System
```
cluster/
├── install-docker.sh          # Run this on host
├── install-app.sh             # Copied into container
├── docker-compose.yml         # Docker Compose config
├── .env                       # Configuration (loaded into container)
└── Dockerfile                 # Container build instructions
```

### Inside Container
```
/app/
├── install-app.sh             # Runs during container startup
├── scripts/
│   └── install.py             # Python installation script
├── models/                    # Downloaded models (mounted from host)
├── config/                    # Configuration files (mounted from host)
└── data/                      # Application data (mounted from host)
```

---

## Key Points to Remember

1. ✅ **`install-docker.sh`** = Run on HOST to install Docker + build images
2. ✅ **`install-app.sh`** = Runs INSIDE container automatically (don't run on host!)
3. ✅ Models are downloaded when container first starts
4. ✅ Configuration comes from `.env` file (loaded as environment variables)
5. ✅ Installation only runs once (tracked by `/app/models/.installed`)


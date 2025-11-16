# Docker Quick Reference

## 🚀 Quick Start

### First Time Setup (Choose One)

**Option A: Use Build Scripts (Easiest)**
```powershell
# Windows
.\build.ps1 -Detached
```
```bash
# Linux/macOS
./build.sh --detached
```

**Option B: Use Makefile**
```bash
make dev
```

**Option C: Manual with BuildKit**
```powershell
# Windows PowerShell
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1
docker-compose up --build -d
```
```bash
# Linux/macOS
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
docker-compose up --build -d
```

## 📋 Common Commands

### Using Build Scripts

| Task | Windows | Linux/macOS |
|------|---------|-------------|
| Build & Start | `.\build.ps1 -Detached` | `./build.sh --detached` |
| Fresh Build | `.\build.ps1 -NoCache -Detached` | `./build.sh --no-cache --detached` |
| Update Base Images | `.\build.ps1 -Pull -Detached` | `./build.sh --pull --detached` |

### Using Makefile

| Task | Command |
|------|---------|
| Build & Start | `make dev` |
| Just Build | `make build` |
| Fresh Build | `make build-fresh` |
| Start Container | `make up` |
| Stop Container | `make down` |
| Restart | `make restart` |
| View Logs | `make logs` |
| Shell Access | `make shell` |
| Clean Everything | `make clean` |
| Help | `make help` |

### Direct Docker Compose

| Task | Command |
|------|---------|
| Build | `docker-compose build` |
| Build & Start | `docker-compose up --build -d` |
| Start | `docker-compose up -d` |
| Stop | `docker-compose down` |
| View Logs | `docker-compose logs -f` |
| Shell Access | `docker-compose exec voice-assistant bash` |
| Rebuild Fresh | `docker-compose build --no-cache` |

## 🔍 Debugging

### Check Container Status
```bash
docker-compose ps
```

### View Logs
```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service logs
docker-compose logs -f voice-assistant
```

### Access Container Shell
```bash
docker-compose exec voice-assistant bash
```

### Inspect Build Process
```bash
# See build output
docker-compose build --progress=plain

# Check BuildKit status
make status  # or docker version
```

## 🛠️ Maintenance

### Clean Up

```bash
# Stop and remove containers
make down  # or: docker-compose down

# Remove containers and volumes
make clean  # or: docker-compose down -v

# Clean Docker system (free up space)
make prune  # or: docker system prune -af
```

### Check Disk Usage
```bash
docker system df -v
```

### Remove Old Images
```bash
docker image prune -a
```

## ⚙️ Configuration

### Environment Variables
Edit `.env` file in project root:
```bash
# Example
MODEL_NAME=gemma-3n-E4B
DISPLAY_MODE=animated
LOG_LEVEL=INFO
```

### Docker Resources
Edit `docker-compose.yml` to adjust limits:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '2.0'
```

## 🐛 Troubleshooting

### BuildKit Not Enabled
**Symptom:** `unknown flag: --mount`

**Solution:**
- Check Docker version: `docker version`
- Use build scripts (automatically enable BuildKit)
- Or enable globally in `daemon.json`

### Packages Still Redownloading
**Check:**
1. BuildKit is enabled (see above)
2. Using docker-compose, not plain docker build
3. Docker version 18.09+ (BuildKit support)

### Container Won't Start
**Check logs:**
```bash
docker-compose logs
```

**Common issues:**
- Port conflicts: Change ports in `docker-compose.yml`
- Device access: Check `/dev/snd` permissions
- Memory limits: Increase in `docker-compose.yml`

### Performance Issues
**Check resources:**
```bash
docker stats
```

**Solutions:**
- Increase memory limit in `docker-compose.yml`
- Reduce CPU throttling
- Check host system resources

## 📚 Documentation

- [DOCKER_BUILD_OPTIMIZATION.md](DOCKER_BUILD_OPTIMIZATION.md) - Detailed BuildKit explanation
- [DEVELOPMENT_DEPLOYMENT_GUIDE.md](DEVELOPMENT_DEPLOYMENT_GUIDE.md) - Full deployment guide
- [QUICK_START.md](QUICK_START.md) - Getting started guide

## 🔑 Key Points

✅ **Always use BuildKit** for faster builds (use scripts or enable globally)  
✅ **Use build scripts** for automated, consistent builds across team  
✅ **Pin package versions** in requirements.txt for cache hits  
✅ **Mount volumes** for hot-reload during development  
✅ **Monitor disk usage** periodically with `docker system df`  
✅ **Clean up regularly** with `make clean` or `docker system prune`  

## 🎯 Recommended Workflow

**Daily Development:**
```bash
# Start working
make dev       # or .\build.ps1 -Detached

# View logs
make logs      # or docker-compose logs -f

# Make code changes (hot-reload via volumes)

# Restart if needed
make restart   # or docker-compose restart

# Done working
make down      # or docker-compose down
```

**After Changing Dependencies:**
```bash
# Rebuild container
make rebuild   # or .\build.ps1 -Detached
```

**Weekly Maintenance:**
```bash
# Clean up unused resources
make prune     # or docker system prune -af
```


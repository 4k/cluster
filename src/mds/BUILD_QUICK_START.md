# 🚀 Build Quick Start

## First Time? Start Here!

### Windows (PowerShell)
```powershell
.\build.ps1 -Detached
```

### Linux/macOS
```bash
./build.sh --detached
```

### Alternative: Makefile
```bash
make dev
```

## That's It!

The scripts automatically:
- ✅ Enable BuildKit for faster builds
- ✅ Build the Docker image
- ✅ Start the container
- ✅ Show you the logs command

## Common Tasks

| Task | Windows | Linux/macOS | Make |
|------|---------|-------------|------|
| **Build & Start** | `.\build.ps1 -Detached` | `./build.sh -d` | `make dev` |
| **Fresh Build** | `.\build.ps1 -NoCache -Detached` | `./build.sh --no-cache -d` | `make build-fresh && make up` |
| **View Logs** | `docker-compose logs -f` | `docker-compose logs -f` | `make logs` |
| **Stop** | `docker-compose down` | `docker-compose down` | `make down` |
| **Restart** | `docker-compose restart` | `docker-compose restart` | `make restart` |

## Why Use These Scripts?

❌ **Without scripts:** Must remember to set BuildKit env vars every time  
✅ **With scripts:** Automatic! Just run the script

## More Info

- **Quick Reference:** [DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md)
- **Why BuildKit?** [DOCKER_BUILD_OPTIMIZATION.md](DOCKER_BUILD_OPTIMIZATION.md)
- **Technical Details:** [BUILDKIT_AUTOMATION_SUMMARY.md](BUILDKIT_AUTOMATION_SUMMARY.md)

---

**TL;DR:** Run `.\build.ps1 -Detached` (Windows) or `./build.sh --detached` (Linux) and you're done! 🎉


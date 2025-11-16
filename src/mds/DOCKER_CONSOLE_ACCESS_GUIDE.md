# Docker Container Console Access Guide

This guide explains how to access the Docker container console to run Python test scripts and perform debugging tasks.

## Overview

The Voice Assistant application runs in Docker containers and doesn't use virtual environments. All Python dependencies are installed directly in the container's Python environment, making it easy to run test scripts and debug the application.

## Container Access Methods

### 1. Interactive Shell Access

#### For Running Container
```bash
# Access the running container's shell
docker exec -it voice-assistant-dev bash

# Or if using docker-compose
docker-compose exec voice-assistant bash
```

#### For New Container Instance
```bash
# Start a new container instance with shell access
docker run -it --rm voice-assistant-dev bash

# Or with docker-compose (overrides the default command)
docker-compose run --rm voice-assistant bash
```

### 2. Direct Python Execution

#### Run Python Scripts Directly
```bash
# Execute a Python script in the running container
docker exec voice-assistant-dev python /app/test_essential.py

# Or with docker-compose
docker-compose exec voice-assistant python /app/test_essential.py
```

#### Run Python with Interactive Mode
```bash
# Start Python interactive shell in the container
docker exec -it voice-assistant-dev python

# Or with docker-compose
docker-compose exec voice-assistant python
```

## Running Test Scripts

### Available Test Scripts

The following test scripts are available in the project root:

- `test_essential.py` - Essential functionality tests
- `test_mock_providers.py` - Mock provider tests
- `test_wake_word_fix.py` - Wake word detection tests

### Running Tests

#### Method 1: Interactive Shell
```bash
# 1. Access the container shell
docker exec -it voice-assistant-dev bash

# 2. Navigate to the app directory
cd /app

# 3. Run test scripts
python test_essential.py
python test_mock_providers.py
python test_wake_word_fix.py
```

#### Method 2: Direct Execution
```bash
# Run tests directly without shell access
docker exec voice-assistant-dev python /app/test_essential.py
docker exec voice-assistant-dev python /app/test_mock_providers.py
docker exec voice-assistant-dev python /app/test_wake_word_fix.py
```

#### Method 3: Using Docker Compose
```bash
# Run tests using docker-compose
docker-compose exec voice-assistant python /app/test_essential.py
docker-compose exec voice-assistant python /app/test_mock_providers.py
docker-compose exec voice-assistant python /app/test_wake_word_fix.py
```

## Development and Debugging

### Running Custom Python Scripts

#### Create and Run Custom Scripts
```bash
# 1. Access container shell
docker exec -it voice-assistant-dev bash

# 2. Create a test script
cat > /app/my_test.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/src')

from ai.model_manager import get_model_manager
from core.config import get_config_manager

print("Testing model manager...")
manager = get_model_manager()
stats = manager.get_model_stats()
print(f"Model stats: {stats}")

print("Testing config manager...")
config = get_config_manager()
print(f"Config loaded: {config is not None}")
EOF

# 3. Run the script
python /app/my_test.py
```

#### Copy Scripts from Host
```bash
# Copy a script from host to container
docker cp my_test.py voice-assistant-dev:/app/my_test.py

# Run the copied script
docker exec voice-assistant-dev python /app/my_test.py
```

### Debugging with Python Debugger

#### Interactive Debugging
```bash
# Access container shell
docker exec -it voice-assistant-dev bash

# Start Python with debugging
python -c "
import sys
sys.path.insert(0, '/app/src')
import pdb

# Your debugging code here
from ai.model_manager import get_model_manager
manager = get_model_manager()
pdb.set_trace()  # Set breakpoint
stats = manager.get_model_stats()
print(stats)
"
```

#### Using IPython for Enhanced Debugging
```bash
# Install IPython in the container (if not already installed)
docker exec voice-assistant-dev pip install ipython

# Start IPython shell
docker exec -it voice-assistant-dev ipython
```

## Container Management Commands

### Starting and Stopping Containers

```bash
# Start the application
docker-compose up -d

# Stop the application
docker-compose down

# Restart the application
docker-compose restart

# View logs
docker-compose logs -f
```

### Container Information

```bash
# List running containers
docker ps

# Get container details
docker inspect voice-assistant-dev

# View container logs
docker logs voice-assistant-dev
```

## Environment Variables and Configuration

### Viewing Environment Variables
```bash
# Access container shell
docker exec -it voice-assistant-dev bash

# View all environment variables
env

# View specific environment variables
echo $PYTHONPATH
echo $ASSISTANT_DEBUG
echo $LLM_PROVIDER_TYPE
```

### Modifying Configuration
```bash
# Access container shell
docker exec -it voice-assistant-dev bash

# Edit configuration files
nano /app/config/assistant_config.yaml

# Or use vi
vi /app/config/assistant_config.yaml
```

## Volume Mounts and File Access

### Accessing Host Files
The container has the following volume mounts:
- `./src:/app/src` - Source code
- `./config:/app/config` - Configuration files
- `./data:/app/data` - Data files
- `./models:/app/models` - AI models
- `./voices:/app/voices` - Voice files
- `./logs:/app/logs` - Log files

### File Operations
```bash
# Access container shell
docker exec -it voice-assistant-dev bash

# List files in mounted directories
ls -la /app/src
ls -la /app/config
ls -la /app/models

# Create files that persist on host
echo "test data" > /app/data/test.txt

# View logs
tail -f /app/logs/assistant.log
```

## Troubleshooting

### Common Issues

#### Container Not Running
```bash
# Check container status
docker ps -a

# Start container if stopped
docker-compose up -d

# View error logs
docker-compose logs voice-assistant
```

#### Permission Issues
```bash
# Fix file permissions
docker exec voice-assistant-dev chown -R assistant:assistant /app

# Or run as root for debugging
docker exec -it --user root voice-assistant-dev bash
```

#### Python Import Errors
```bash
# Check Python path
docker exec voice-assistant-dev python -c "import sys; print(sys.path)"

# Verify source code is mounted
docker exec voice-assistant-dev ls -la /app/src
```

### Getting Help

#### Container Health Check
```bash
# Check if all services are running
docker exec voice-assistant-dev ps aux

# Check Python environment
docker exec voice-assistant-dev python --version
docker exec voice-assistant-dev pip list
```

#### Resource Usage
```bash
# Check container resource usage
docker stats voice-assistant-dev

# Check disk usage
docker exec voice-assistant-dev df -h
```

## Best Practices

1. **Use Interactive Shell for Development**: Access the container shell for interactive development and testing.

2. **Run Tests Before Deployment**: Always run test scripts before deploying changes.

3. **Monitor Logs**: Keep an eye on the application logs for any issues.

4. **Backup Important Data**: Ensure important data in mounted volumes is backed up.

5. **Use Environment Variables**: Modify behavior using environment variables rather than hardcoded values.

## Quick Reference

| Task | Command |
|------|---------|
| Access shell | `docker exec -it voice-assistant-dev bash` |
| Run test script | `docker exec voice-assistant-dev python /app/test_essential.py` |
| View logs | `docker logs voice-assistant-dev` |
| Restart container | `docker-compose restart` |
| Copy file to container | `docker cp file.py voice-assistant-dev:/app/` |
| Check Python packages | `docker exec voice-assistant-dev pip list` |

This guide provides comprehensive instructions for accessing the Docker container console and running Python test scripts without needing virtual environments.

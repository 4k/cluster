#!/bin/bash
# Docker entrypoint script
# Runs application installation on first startup, then starts the main application

set -e

echo "[ENTRYPOINT] Starting Voice Assistant container..."

# Run application installation if needed (models not downloaded yet)
if [ ! -f "/app/models/.installed" ]; then
    echo "[ENTRYPOINT] Running application installation script..."
    chmod +x install-app.sh
    ./install-app.sh
    touch /app/models/.installed
    echo "[ENTRYPOINT] Application installation completed."
else
    echo "[ENTRYPOINT] Application already installed, skipping installation."
fi

# Start the main application
echo "[ENTRYPOINT] Starting main application..."
exec "$@"


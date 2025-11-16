# Multi-stage Dockerfile for Voice Assistant
# Supports both x86_64 (development) and ARM64 (Raspberry Pi) architectures

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # llama.cpp dependencies
    git \
    wget \
    curl \
    # SQLite (usually included, but ensure it's available)
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development environment
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    # Additional tools for llama.cpp development
    gdb \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Install development dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    jupyter

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data models voices config logs

# Remove any Python cache files that might have been copied
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true

# No installation script needed - just run the main application

# No ports needed for direct execution

# Default command for development
CMD ["python", "src/main.py"]

# Stage 3: Production environment
FROM base as production

# Create non-root user
RUN groupadd -r assistant && useradd -r -g assistant assistant

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with llama.cpp optimizations and cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    CMAKE_ARGS="-DLLAMA_OPENBLAS=on" pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories and set ownership
RUN mkdir -p data models voices config logs && \
    chown -R assistant:assistant /app

# Copy and set up entrypoint script (runs installation at container startup)
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chown assistant:assistant /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER assistant

# Use entrypoint that runs installation before starting app
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# No ports needed for direct execution

# Default command for production
CMD ["python", "src/main.py"]

# Stage 4: ARM64 optimized for Raspberry Pi
FROM base as production-arm64

# Install ARM64 specific dependencies
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libblas-dev \
    liblapack-dev \
    gfortran \
    # llama.cpp ARM64 optimizations
    libopenblas-dev \
    libomp-dev \
    # SQLite for ARM64
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r assistant && useradd -r -g assistant assistant

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with ARM64 optimizations and cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    CMAKE_ARGS="-DLLAMA_OPENBLAS=on -DLLAMA_NATIVE=on" pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories and set ownership
RUN mkdir -p data models voices config logs && \
    chown -R assistant:assistant /app

# Copy and set up entrypoint script (runs installation at container startup)
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chown assistant:assistant /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER assistant

# Use entrypoint that runs installation before starting app
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# No ports needed for direct execution

# Default command for production
CMD ["python", "src/main.py"]

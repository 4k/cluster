#!/bin/bash
# Virtual Environment Setup Script for Linux/WSL

echo "=== AI Assistant Virtual Environment Setup ==="
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists!"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf venv
    else
        echo "❌ Setup cancelled."
        exit 0
    fi
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pyaudio portaudio19-dev python3-dev espeak

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✨ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To download voice models, run:"
echo "  python download_voice.py"

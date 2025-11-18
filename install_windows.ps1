# Windows Installation Script for AI Assistant
# Run this in PowerShell

Write-Host "=== AI Assistant - Windows Installation ===" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists, skipping..." -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Try to install PyAudio using pipwin (easier on Windows)
Write-Host ""
Write-Host "Installing PyAudio (may take a moment)..." -ForegroundColor Yellow
pip install pipwin
pipwin install pyaudio

# Install all other requirements
Write-Host ""
Write-Host "Installing remaining dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "=== Installation Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To test STT service:" -ForegroundColor Cyan
Write-Host "  python stt_service.py" -ForegroundColor White
Write-Host ""
Write-Host "To test TTS service:" -ForegroundColor Cyan
Write-Host '  python tts_service.py "Hello World"' -ForegroundColor White
Write-Host ""

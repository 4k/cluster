# Virtual Environment Setup Script for Windows PowerShell

Write-Host "=== AI Assistant Virtual Environment Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to delete and recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "🗑️  Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "❌ Setup cancelled." -ForegroundColor Red
        exit 0
    }
}

# Create virtual environment
Write-Host "🐍 Creating virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate virtual environment
Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements
Write-Host "📚 Installing Python dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Check if PyAudio installation was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠️  PyAudio installation may have failed." -ForegroundColor Yellow
    Write-Host "Try installing it separately:" -ForegroundColor Yellow
    Write-Host "  pip install pipwin" -ForegroundColor White
    Write-Host "  pipwin install pyaudio" -ForegroundColor White
    Write-Host "  pip install -r requirements.txt" -ForegroundColor White
}

Write-Host ""
Write-Host "✨ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "If you get an execution policy error, run:" -ForegroundColor Cyan
Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
Write-Host ""
Write-Host "To download voice models, run:" -ForegroundColor Cyan
Write-Host "  python download_voice.py" -ForegroundColor White

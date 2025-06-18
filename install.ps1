# install.ps1 - Installation script for Windows PowerShell
# Quranic Transcription Pipeline Setup

Write-Host "Setting up Quranic Transcription Pipeline..." -ForegroundColor Green

# Check if Python is installed and get version
try {
    $pythonVersion = python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
    Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan
    
    # Check if Python version is 3.8 or higher
    $versionCheck = python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Python 3.8 or higher is required" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv quran_pipeline_env

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\quran_pipeline_env\Scripts\Activate.ps1"

# Check if activation script exists
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch (CPU version - adjust for GPU if needed)
Write-Host "Installing PyTorch..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

# Install other requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
$packages = @(
    "openai-whisper>=20231117",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "numpy>=1.24.0",
    "PyYAML>=6.0",
    "requests>=2.31.0",
    "scipy>=1.11.0"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to install $package" -ForegroundColor Yellow
    }
}

# Create directory structure
Write-Host "Creating directory structure..." -ForegroundColor Yellow
$directories = @(
    "output\transcriptions",
    "output\preprocessed_audio", 
    "output\reports",
    "input_audio",
    "temp"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Cyan
    }
}

# Download initial Whisper model
Write-Host "Downloading Whisper model (this may take a few minutes)..." -ForegroundColor Yellow
python -c "import whisper; whisper.load_model('large-v3')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To use the pipeline:" -ForegroundColor Cyan
    Write-Host "1. Activate the virtual environment: .\quran_pipeline_env\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "2. Place your audio files in the 'input_audio' directory" -ForegroundColor White
    Write-Host "3. Run: python quran_pipeline.py input_audio\your_file.mp3" -ForegroundColor White
    Write-Host "   Or for batch processing: python quran_pipeline.py --batch input_audio\" -ForegroundColor White
    Write-Host ""
    Write-Host "Configuration can be modified in pipeline_config.yaml" -ForegroundColor Yellow
} else {
    Write-Host "Setup completed with warnings. Please check the output above." -ForegroundColor Yellow
}
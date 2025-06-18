# install.sh - Installation script
#!/bin/bash

echo "Setting up Quranic Transcription Pipeline..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv quran_pipeline_env
source quran_pipeline_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version - adjust for GPU if needed)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "Installing requirements..."
pip install openai-whisper>=20231117
pip install librosa>=0.10.0
pip install soundfile>=0.12.0
pip install numpy>=1.24.0
pip install PyYAML>=6.0
pip install requests>=2.31.0
pip install scipy>=1.11.0

# Create directory structure
echo "Creating directory structure..."
mkdir -p output/transcriptions
mkdir -p output/preprocessed_audio
mkdir -p output/reports
mkdir -p input_audio
mkdir -p temp

# Download initial Whisper model
echo "Downloading Whisper model (this may take a few minutes)..."
python3 -c "import whisper; whisper.load_model('large-v3')"

echo "Setup complete!"
echo ""
echo "To use the pipeline:"
echo "1. Activate the virtual environment: source quran_pipeline_env/bin/activate"
echo "2. Place your audio files in the 'input_audio' directory"
echo "3. Run: python quran_pipeline.py input_audio/your_file.mp3"
echo "   Or for batch processing: python quran_pipeline.py --batch input_audio/"
echo ""
echo "Configuration can be modified in pipeline_config.yaml"

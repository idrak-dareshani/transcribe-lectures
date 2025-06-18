# README.md
# Quranic Lecture Transcription Pipeline

A comprehensive, production-ready pipeline for transcribing and correcting Urdu/Arabic Quranic lectures using OpenAI Whisper with specialized corrections for Islamic terminology.

## Features

- **Audio Preprocessing**: Noise reduction, normalization, and silence removal
- **Optimized Transcription**: Uses Whisper Large-v3 with optimized settings for Urdu/Arabic
- **Intelligent Correction**: Extensive dictionary of Quranic terms and Islamic vocabulary
- **Pattern-based Corrections**: Fixes common Arabic transcription patterns
- **Batch Processing**: Process multiple files automatically
- **Detailed Reports**: Processing statistics, confidence scores, and correction logs
- **Configurable**: Flexible YAML-based configuration
- **CLI Interface**: Easy command-line usage
- **Docker Support**: Containerized deployment option

## Installation

### Quick Setup
```bash
# Clone or download the pipeline files
chmod +x install.sh
./install.sh
```

### Manual Installation
```bash
# Create virtual environment
python3 -m venv quran_pipeline_env
source quran_pipeline_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Whisper model
python -c "import whisper; whisper.load_model('large-v3')"
```

## Usage

### Single File Processing
```bash
python quran_pipeline.py input_audio/lecture.mp3
```

### Batch Processing
```bash
python quran_pipeline.py --batch input_audio/
```

### Custom Configuration
```bash
python quran_pipeline.py --config custom_config.yaml input_audio/lecture.mp3
```

## Configuration

Edit `pipeline_config.yaml` to customize:

```yaml
model_size: "large-v3"
language: "ur"
confidence_threshold: 0.8
output_dir: "output"
save_preprocessed_audio: true
generate_report: true
```

## Output Structure

```
output/
├── transcriptions/          # JSON and text files
├── preprocessed_audio/      # Cleaned audio files
└── reports/                # Processing reports
```

## Correction Dictionary

The pipeline includes 100+ corrections for common Quranic terms:
- سلس → ثلث
- سمد → صمد  
- قرآان → قرآن
- And many more...

Add custom corrections in `quran_corrections.yaml` or programmatically.

## Docker Usage

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t quran-pipeline .
docker run -v $(pwd)/input_audio:/app/input_audio quran-pipeline
```

## API Usage

```python
from quran_pipeline import QuranicTranscriptionPipeline

# Initialize
pipeline = QuranicTranscriptionPipeline()

# Process file
result = pipeline.process_file("lecture.mp3")

# Access results
print(result.corrected_text)
print(f"Corrections made: {len(result.corrections_made)}")
```

## Requirements

- Python 3.8+
- 4GB+ RAM (for Whisper Large-v3)
- ffmpeg (for audio processing)

## Troubleshooting

1. **Out of memory**: Use smaller model (`base` or `small`)
2. **Slow processing**: Consider `medium` model for speed/accuracy balance
3. **Poor accuracy**: Ensure clean, clear audio input
4. **Missing corrections**: Add custom terms to correction dictionary

## Contributing

1. Add new corrections to the dictionary
2. Improve audio preprocessing
3. Add support for other languages
4. Enhance the web interface

## License

Open source - feel free to modify and distribute.

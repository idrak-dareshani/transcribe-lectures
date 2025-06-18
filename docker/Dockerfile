# docker/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper model
RUN python -c "import whisper; whisper.load_model('large-v3')"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p output/transcriptions output/preprocessed output/reports input_audio

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for web interface later)
EXPOSE 8000

# Default command
CMD ["python", "quran_pipeline.py", "--help"]

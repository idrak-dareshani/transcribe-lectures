# setup.py
from setuptools import setup, find_packages

setup(
    name="quran-transcription-pipeline",
    version="1.0.0",
    description="A comprehensive pipeline for transcribing and correcting Quranic lectures",
    author="Idrak Dareshani",
    packages=find_packages(),
    install_requires=[
        #"openai-whisper>=20231117",
        "librosa>=0.10.0", 
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "requests>=2.31.0",
        "scipy>=1.11.0",
        "torch>=2.1.0",
        "torchaudio>=2.1.0"
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'quran-transcribe=quran_pipeline:main',
        ],
    },
)

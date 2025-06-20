#!/usr/bin/env python3
"""
Quranic Lecture Transcription Pipeline
A comprehensive solution for transcribing and correcting Urdu/Arabic Quranic lectures
"""

import os
import json
import re
import whisper
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import requests
from dataclasses import dataclass
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quran_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Data class for transcription results"""
    original_text: str
    corrected_text: str
    confidence_scores: List[float]
    corrections_made: List[Dict]
    processing_time: float
    audio_duration: float

class AudioPreprocessor:
    """Handles audio preprocessing and enhancement"""
    
    def __init__(self):
        self.target_sr = 16000
        self.min_duration = 1.0  # seconds
    
    def load_and_preprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence
            audio = self._remove_silence(audio, sr)
            
            # Enhance quality
            audio = self._enhance_audio(audio)
            
            logger.info(f"Preprocessed audio: {len(audio)/sr:.2f}s duration")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def _remove_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove silence from audio"""
        # Use librosa's trim function
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=20,  # Threshold in dB
            frame_length=2048,
            hop_length=512
        )
        return audio_trimmed
    
    def _enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Basic audio enhancement"""
        # Apply preemphasis filter
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Normalize again after filtering
        return librosa.util.normalize(audio)
    
    def save_preprocessed(self, audio: np.ndarray, sr: int, output_path: str):
        """Save preprocessed audio"""
        sf.write(output_path, audio, sr)
        logger.info(f"Saved preprocessed audio to {output_path}")

class QuranicTermCorrector:
    """Handles correction of Quranic terms and Arabic text"""
    
    def __init__(self, corrections_file: str = "quran_corrections.yaml"):
        self.corrections_file = corrections_file
        self.corrections = self._load_corrections()
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]+')
        
    def _load_corrections(self) -> Dict[str, str]:
        """Load correction dictionary from file"""
        corrections = {}
        
        # Try to load from file if exists
        try:
            if os.path.exists(self.corrections_file):
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    file_data = yaml.safe_load(f)
                    if isinstance(file_data, dict):
                        if 'corrections' in file_data:
                            corrections.update(file_data['corrections'])
                        else:
                            corrections.update(file_data)
        except Exception as e:
            logger.warning(f"Could not load corrections file: {e}")
        
        return corrections
    
    def correct_text(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply corrections to text"""
        corrected_text = text
        corrections_made = []
        
        # Apply word-level corrections
        for incorrect, correct in self.corrections.items():
            if incorrect in corrected_text:
                corrected_text = corrected_text.replace(incorrect, correct)
                corrections_made.append({
                    'type': 'word_correction',
                    'original': incorrect,
                    'corrected': correct,
                    'position': text.find(incorrect)
                })
        
        # Apply pattern-based corrections
        corrected_text, pattern_corrections = self._apply_pattern_corrections(corrected_text)
        corrections_made.extend(pattern_corrections)
        
        return corrected_text, corrections_made
    
    def _apply_pattern_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply pattern-based corrections"""
        corrections = []
        
        # Example: Fix common Arabic article issues
        # الا -> إلا
        pattern1 = re.compile(r'\bالا\b')
        if pattern1.search(text):
            text = pattern1.sub('إلا', text)
            corrections.append({
                'type': 'pattern_correction',
                'pattern': 'الا -> إلا',
                'description': 'Fixed Arabic exception particle'
            })
        
        # Add more pattern corrections as needed
        
        return text, corrections
    
    def add_correction(self, incorrect: str, correct: str):
        """Add a new correction to the dictionary"""
        self.corrections[incorrect] = correct
        self._save_corrections()
    
    def _save_corrections(self):
        """Save corrections to file"""
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.corrections, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Could not save corrections: {e}")

class WhisperTranscriber:
    """Handles Whisper transcription with optimization"""
    
    def __init__(self, config: Dict):
        self.model_size = config.get("model_size", "medium")
        self.language = config.get("language", "ur")
        self.word_timestamps=config.get("word_timestamps", True)
        self.temperature=config.get("temperature", 0.0)
        self.beam_size=config.get("beam_size", 1)
        self.best_of=config.get("best_of", 1)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio with optimized settings"""
        try:
            logger.info(f"Transcribing: {audio_path}")
            
            result = self.model.transcribe(
                audio_path,
                task="transcribe",
                language=self.language,
                word_timestamps=self.word_timestamps,
                temperature=self.temperature,               # More deterministic
                beam_size=self.beam_size,                   # Better beam search
                best_of=self.best_of,                       # Multiple attempts
                fp16=False,                                 # Better precision
                verbose=False
            )
            
            logger.info(f"Transcription completed: {len(result['text'])} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

class QuranicTranscriptionPipeline:
    """Main pipeline class that orchestrates the entire process"""
    
    def __init__(self, config_file: str = "pipeline_config.yaml"):
        self.config = self._load_config(config_file)
        self.preprocessor = AudioPreprocessor()
        self.corrector = QuranicTermCorrector()
        self.transcriber = WhisperTranscriber(self.config)
        
        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'preprocessed').mkdir(exist_ok=True)
        (self.output_dir / 'transcriptions').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            'model_size': 'medium',
            'language': 'ur',
            'output_dir': 'output',
            'save_preprocessed_audio': True,
            'generate_report': True,
            'confidence_threshold': 0.8
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def process_file(self, audio_path: str) -> TranscriptionResult:
        """Process a single audio file through the complete pipeline"""
        start_time = datetime.now()
        audio_path = Path(audio_path)
        
        logger.info(f"Starting pipeline for: {audio_path.name}")
        
        try:
            # Step 1: Preprocess audio
            audio, sr = self.preprocessor.load_and_preprocess(str(audio_path))
            audio_duration = len(audio) / sr
            
            # Save preprocessed audio if configured
            if self.config.get('save_preprocessed_audio', True):
                preprocessed_path = self.output_dir / 'preprocessed' / f"{audio_path.stem}.wav"
                self.preprocessor.save_preprocessed(audio, sr, str(preprocessed_path))
            
            # Step 2: Transcribe
            temp_audio_path = f"temp/{audio_path.stem}.wav"
            sf.write(temp_audio_path, audio, sr)
            
            whisper_result = self.transcriber.transcribe(temp_audio_path)
            
            # Cleanup temp file
            os.remove(temp_audio_path)
            
            # Step 3: Correct text
            original_text = whisper_result['text']
            corrected_text, corrections_made = self.corrector.correct_text(original_text)
            
            # Step 4: Extract confidence scores
            confidence_scores = []
            if 'segments' in whisper_result:
                for segment in whisper_result['segments']:
                    if 'words' in segment:
                        confidence_scores.extend([word.get('probability', 0.0) for word in segment['words']])
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            result = TranscriptionResult(
                original_text=original_text,
                corrected_text=corrected_text,
                confidence_scores=confidence_scores,
                corrections_made=corrections_made,
                processing_time=processing_time,
                audio_duration=audio_duration
            )
            
            # Save transcription
            self._save_transcription(audio_path.stem, result, whisper_result)
            
            # Generate report if configured
            if self.config.get('generate_report', True):
                self._generate_report(audio_path.stem, result)
            
            logger.info(f"Pipeline completed for {audio_path.name} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed for {audio_path.name}: {e}")
            raise
    
    def process_batch(self, audio_dir: str) -> List[TranscriptionResult]:
        """Process multiple audio files"""
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.m4a"))
        
        logger.info(f"Processing {len(audio_files)} files from {audio_dir}")
        
        results = []
        for audio_file in audio_files:
            try:
                result = self.process_file(str(audio_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")
        
        # Generate batch report
        self._generate_batch_report(results)
        
        return results
    
    def _save_transcription(self, filename: str, result: TranscriptionResult, whisper_result: Dict):
        """Save transcription results"""
        output_file = self.output_dir / 'transcriptions' / f"{filename}.json"
        
        output_data = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'original_text': result.original_text,
            'corrected_text': result.corrected_text,
            'corrections_made': result.corrections_made,
            'processing_time': result.processing_time,
            'audio_duration': result.audio_duration,
            'average_confidence': np.mean(result.confidence_scores) if result.confidence_scores else 0.0,
            'whisper_segments': whisper_result.get('segments', [])
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Also save plain text version
        text_file = self.output_dir / 'transcriptions' / f"{filename}_corrected.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(result.corrected_text)
    
    def _generate_report(self, filename: str, result: TranscriptionResult):
        """Generate processing report"""
        report_file = self.output_dir / 'reports' / f"{filename}.txt"
        
        avg_confidence = np.mean(result.confidence_scores) if result.confidence_scores else 0.0
        
        report = f"""
Quranic Lecture Transcription Report
===================================

File: {filename}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Audio Duration: {result.audio_duration:.2f} seconds
Processing Time: {result.processing_time:.2f} seconds
Processing Speed: {result.audio_duration/result.processing_time:.2f}x real-time

Transcription Statistics:
- Original text length: {len(result.original_text)} characters
- Corrected text length: {len(result.corrected_text)} characters
- Number of corrections: {len(result.corrections_made)}
- Average confidence: {avg_confidence:.3f}
- Low confidence segments: {sum(1 for score in result.confidence_scores if score < self.config.get('confidence_threshold', 0.8))}

Corrections Made:
"""
        
        for correction in result.corrections_made:
            if correction['type'] == 'word_correction':
                report += f"- {correction['original']} → {correction['corrected']}\n"
            else:
                report += f"- {correction['description']}\n"
        
        report += f"\nOriginal Text:\n{result.original_text}\n"
        report += f"\nCorrected Text:\n{result.corrected_text}\n"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _generate_batch_report(self, results: List[TranscriptionResult]):
        """Generate batch processing report"""
        if not results:
            return
        
        report_file = self.output_dir / 'reports' / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        total_duration = sum(r.audio_duration for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        total_corrections = sum(len(r.corrections_made) for r in results)
        
        report = f"""
Batch Processing Report
======================

Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Files Processed: {len(results)}
Total Audio Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)
Total Processing Time: {total_processing_time:.2f} seconds ({total_processing_time/60:.2f} minutes)
Average Processing Speed: {total_duration/total_processing_time:.2f}x real-time
Total Corrections Made: {total_corrections}

Per-File Summary:
"""
        
        for i, result in enumerate(results, 1):
            avg_conf = np.mean(result.confidence_scores) if result.confidence_scores else 0.0
            report += f"{i:2d}. Duration: {result.audio_duration:6.2f}s, "
            report += f"Processing: {result.processing_time:6.2f}s, "
            report += f"Confidence: {avg_conf:.3f}, "
            report += f"Corrections: {len(result.corrections_made):2d}\n"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

# CLI Interface
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quranic Lecture Transcription Pipeline")
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('--config', default='pipeline_config.yaml', help='Configuration file')
    parser.add_argument('--batch', action='store_true', help='Process directory of files')
    parser.add_argument('--model', default='medium', help='Whisper model size')
    parser.add_argument('--language', default='ur', help='Audio language')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = QuranicTranscriptionPipeline(args.config)
    
    try:
        if args.batch:
            results = pipeline.process_batch(args.input)
            print(f"Processed {len(results)} files successfully")
        else:
            result = pipeline.process_file(args.input)
            print(f"Transcription completed:")
            print(f"Original length: {len(result.original_text)} chars")
            print(f"Corrected length: {len(result.corrected_text)} chars")
            print(f"Corrections made: {len(result.corrections_made)}")
            print(f"Processing time: {result.processing_time:.2f}s")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

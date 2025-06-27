#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Audio Analysis and Whisper Processing Script
Analyzes and processes multiple audio files in a folder for Whisper compatibility
Optimized for Urdu Quran lecture transcription
"""

import os
import sys
import subprocess
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import locale
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set UTF-8 encoding for the script
if sys.platform.startswith('win'):
    # Windows encoding fix
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("⚠ faster-whisper not installed. Install with: pip install faster-whisper")

# Thread-safe printing
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

class BatchAudioProcessor:
    def __init__(self):
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
        self.processed_files = []
        self.failed_files = []
        self.skipped_files = []
        
    def find_audio_files(self, folder_path, recursive=True):
        """Find all audio files in the specified folder"""
        folder_path = Path(folder_path)
        audio_files = []
        
        if not folder_path.exists():
            safe_print(f"❌ Folder not found: {folder_path}")
            return []
        
        safe_print(f"🔍 Searching for audio files in: {folder_path}")
        
        if recursive:
            # Search recursively
            for ext in self.supported_formats:
                pattern = f"**/*{ext}"
                files = list(folder_path.glob(pattern))
                audio_files.extend(files)
        else:
            # Search only in the current folder
            for ext in self.supported_formats:
                pattern = f"*{ext}"
                files = list(folder_path.glob(pattern))
                audio_files.extend(files)
        
        # Remove duplicates and sort
        audio_files = sorted(list(set(audio_files)))
        
        safe_print(f"📁 Found {len(audio_files)} audio files:")
        for i, file in enumerate(audio_files, 1):
            safe_print(f"   {i:2d}. {file.name} ({file.suffix})")
        
        return audio_files
    
    def check_existing_transcriptions(self, audio_files, output_dir='output', skip_existing=True):
        """Check which files already have transcriptions"""
        output_dir = Path(output_dir)
        
        if not skip_existing:
            return audio_files
        
        files_to_process = []
        already_processed = []
        
        for audio_file in audio_files:
            base_name = audio_file.stem
            txt_file = output_dir / f"{base_name}.txt"
            
            if txt_file.exists():
                already_processed.append(audio_file)
            else:
                files_to_process.append(audio_file)
        
        if already_processed:
            safe_print(f"\n📋 Found existing transcriptions for {len(already_processed)} files:")
            for file in already_processed:
                safe_print(f"   ✓ {file.name}")
            safe_print(f"   (Use --overwrite to reprocess these files)")
        
        safe_print(f"\n🎯 Files to process: {len(files_to_process)}")
        return files_to_process
    
    def filter_by_duration(self, audio_files, min_duration=5, max_duration=None):
        """Filter files by duration"""
        if min_duration is None and max_duration is None:
            return audio_files
        
        safe_print(f"\n⏱️  Filtering by duration (min: {min_duration}s, max: {max_duration}s)")
        
        valid_files = []
        for audio_file in audio_files:
            try:
                # Quick duration check
                duration = librosa.get_duration(path=str(audio_file))
                
                if min_duration and duration < min_duration:
                    safe_print(f"   ⏭️  Skipping {audio_file.name} (too short: {duration:.1f}s)")
                    continue
                
                if max_duration and duration > max_duration:
                    safe_print(f"   ⏭️  Skipping {audio_file.name} (too long: {duration:.1f}s)")
                    continue
                
                valid_files.append(audio_file)
                
            except Exception as e:
                safe_print(f"   ❌ Error checking {audio_file.name}: {e}")
        
        safe_print(f"✅ {len(valid_files)} files passed duration filter")
        return valid_files

class AudioAnalyzer:
    def __init__(self, audio_path):
        self.audio_path = Path(audio_path)
        self.audio_data = None
        self.sample_rate = None
        self.duration = None
        self.analysis_results = {}
        
    def load_audio(self):
        """Load audio file and extract basic information"""
        try:
            self.audio_data, self.sample_rate = librosa.load(
                self.audio_path, 
                sr=None,  # Keep original sample rate
                mono=False  # Keep stereo if available
            )
            self.duration = len(self.audio_data) / self.sample_rate
            return True
        except Exception as e:
            safe_print(f"✗ Error loading audio {self.audio_path.name}: {e}")
            return False
    
    def analyze_audio_quality(self):
        """Analyze audio quality for Whisper compatibility"""
        
        # Convert to mono if stereo
        if self.audio_data.ndim > 1:
            audio_mono = librosa.to_mono(self.audio_data)
        else:
            audio_mono = self.audio_data
        
        # 1. Check sample rate (Whisper prefers 16kHz)
        optimal_sr = self.sample_rate == 16000
        self.analysis_results['sample_rate'] = {
            'current': self.sample_rate,
            'optimal': optimal_sr,
            'recommendation': 'Good' if optimal_sr else 'Resample to 16kHz recommended'
        }
        
        # 2. Check audio length (Whisper works best with <30min chunks)
        duration_ok = self.duration <= 1800  # 30 minutes
        self.analysis_results['duration'] = {
            'current': self.duration,
            'optimal': duration_ok,
            'recommendation': 'Good' if duration_ok else 'Split into chunks <30min'
        }
        
        # 3. Check for silence and audio activity
        frame_length = 2048
        hop_length = 512
        
        # Calculate RMS energy
        rms = librosa.feature.rms(
            y=audio_mono, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Dynamic threshold based on audio statistics
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        silence_threshold = max(0.01, rms_mean - 2 * rms_std)
        
        # Calculate speech activity
        active_frames = np.sum(rms > silence_threshold)
        total_frames = len(rms)
        speech_ratio = active_frames / total_frames
        
        self.analysis_results['speech_activity'] = {
            'speech_ratio': speech_ratio,
            'optimal': speech_ratio > 0.3,
            'recommendation': 'Good speech activity' if speech_ratio > 0.3 else 'Low speech activity detected'
        }
        
        # 4. Check for clipping
        max_amplitude = np.max(np.abs(audio_mono))
        clipping_detected = max_amplitude >= 0.99
        self.analysis_results['clipping'] = {
            'max_amplitude': max_amplitude,
            'clipping_detected': clipping_detected,
            'recommendation': 'No clipping detected' if not clipping_detected else 'Audio clipping detected - may affect quality'
        }
        
        # 5. Signal-to-Noise Ratio estimation
        speech_frames = rms > silence_threshold
        if np.sum(speech_frames) > 0 and np.sum(~speech_frames) > 0:
            speech_energy = np.mean(rms[speech_frames])
            noise_energy = np.mean(rms[~speech_frames])
            snr_estimate = 20 * np.log10(speech_energy / (noise_energy + 1e-10))
        else:
            snr_estimate = 0
        
        self.analysis_results['snr'] = {
            'estimated_snr': snr_estimate,
            'optimal': snr_estimate > 10,
            'recommendation': 'Good SNR' if snr_estimate > 10 else 'Low SNR - consider noise reduction'
        }
        
        #return self.analysis_results
    
    def needs_preprocessing(self):
        """Determine if audio needs preprocessing"""
        needs_processing = False
        processing_steps = []
        
        # Check each analysis result safely
        if 'sample_rate' in self.analysis_results and not self.analysis_results['sample_rate'].get('optimal', True):
            needs_processing = True
            processing_steps.append('resample')
        
        if 'duration' in self.analysis_results and not self.analysis_results['duration'].get('optimal', True):
            needs_processing = True
            processing_steps.append('split')
        
        if 'clipping' in self.analysis_results and self.analysis_results['clipping'].get('clipping_detected', False):
            needs_processing = True
            processing_steps.append('normalize')
        
        if 'snr' in self.analysis_results and not self.analysis_results['snr'].get('optimal', True):
            needs_processing = True
            processing_steps.append('denoise')
        
        return needs_processing, processing_steps

class AudioPreprocessor:
    def __init__(self, audio_path, output_dir="temp"):
        self.audio_path = Path(audio_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def resample_audio(self, target_sr=16000):
        """Resample audio to target sample rate"""
        audio, sr = librosa.load(self.audio_path, sr=None)
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        output_path = self.output_dir / f"{self.audio_path.stem}_resampled.wav"
        sf.write(output_path, audio_resampled, target_sr)
        
        return output_path
    
    def split_audio(self, chunk_duration=1800, overlap=30):
        """Split audio into chunks (default 30min with 30s overlap)"""
        audio, sr = librosa.load(self.audio_path, sr=None)
        total_duration = len(audio) / sr
        
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        
        chunks = []
        start = 0
        chunk_num = 1
        
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            if len(chunk) > sr:  # Only save chunks longer than 1 second
                output_path = self.output_dir / f"{self.audio_path.stem}_chunk_{chunk_num:02d}.wav"
                sf.write(output_path, chunk, sr)
                chunks.append(output_path)
                chunk_num += 1
            
            start = end - overlap_samples
            if start >= len(audio) - overlap_samples:
                break
        
        return chunks
    
    def normalize_audio(self):
        """Normalize audio to prevent clipping"""
        audio, sr = librosa.load(self.audio_path, sr=None)
        
        # Normalize to 90% of maximum to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio * (0.9 / max_val)
        else:
            audio_normalized = audio
        
        output_path = self.output_dir / f"{self.audio_path.stem}_normalized.wav"
        sf.write(output_path, audio_normalized, sr)
        
        return output_path
    
    def basic_denoise(self):
        """Apply basic noise reduction using spectral gating"""
        try:
            import noisereduce as nr
            audio, sr = librosa.load(self.audio_path, sr=None)
            
            # Apply noise reduction
            audio_denoised = nr.reduce_noise(y=audio, sr=sr)
            
            output_path = self.output_dir / f"{self.audio_path.stem}_denoised.wav"
            sf.write(output_path, audio_denoised, sr)
            
            return output_path
            
        except ImportError:
            safe_print("⚠ noisereduce not installed. Skipping noise reduction.")
            return self.audio_path

class WhisperProcessor:
    def __init__(self):
        self.model_sizes = {
            'tiny': 'Fastest, lowest quality',
            'base': 'Good balance of speed and quality', 
            'small': 'Better quality, slower',
            'medium': 'High quality, slower',
            'large-v2': 'Highest quality, slowest',
            'large-v3': 'Latest model, best quality'
        }
        self.model = None
        self.model_loaded = False
    
    def check_faster_whisper(self):
        """Check if faster-whisper is available"""
        return FASTER_WHISPER_AVAILABLE
    
    def check_whisper_installation(self):
        """Check if Whisper (original) is installed"""
        try:
            result = subprocess.run(['whisper', '--help'], 
                                 capture_output=True, text=True, encoding='utf-8', errors='ignore')
            return True
        except FileNotFoundError:
            return False
    
    def load_model(self, model_size='base', device='auto', compute_type='auto'):
        """Load faster-whisper model"""
        if not FASTER_WHISPER_AVAILABLE:
            safe_print("❌ faster-whisper not available")
            return False
        
        # Don't reload if same model is already loaded
        if self.model_loaded and hasattr(self, 'current_model_size') and self.current_model_size == model_size:
            return True
        
        try:
            safe_print(f"🔄 Loading faster-whisper model: {model_size}")
            
            # Determine best device and compute type
            if device == 'auto':
                try:
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except ImportError:
                    device = 'cpu'
            
            if compute_type == 'auto':
                if device == 'cuda':
                    compute_type = 'float16'  # Faster on GPU
                else:
                    compute_type = 'int8'     # Faster on CPU
            
            self.model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type,
                download_root=None,  # Use default cache
                local_files_only=False
            )
            
            self.model_loaded = True
            self.current_model_size = model_size
            safe_print(f"✅ Model loaded successfully on {device}")
            return True
            
        except Exception as e:
            safe_print(f"❌ Error loading model: {e}")
            return False
    
    def suggest_model_size(self, duration, quality_preference='balanced'):
        """Suggest appropriate Whisper model based on duration and quality preference"""
        if quality_preference == 'speed':
            return 'tiny' if duration > 3600 else 'base'
        elif quality_preference == 'quality':
            return 'large-v3'
        else:  # balanced
            if duration < 600:  # < 10 minutes
                return 'small'
            elif duration < 1800:  # < 30 minutes
                return 'base'
            else:
                return 'base'  # For longer files, balance speed vs quality
    
    def transcribe_with_faster_whisper(self, audio_path, language='ur', task='transcribe', 
                                     initial_prompt=None, temperature=0.0, best_of=5):
        """Transcribe using faster-whisper"""
        if not self.model:
            safe_print("❌ Model not loaded")
            return None
        
        try:
            safe_print(f"🎯 Transcribing: {Path(audio_path).name}")
            
            # Set up parameters
            transcribe_params = {
                'language': language,
                'task': task,
                'temperature': temperature,
                'best_of': best_of,
                'beam_size': 5,
                'word_timestamps': True,
                'vad_filter': True,  # Voice activity detection
                'vad_parameters': {
                    'min_silence_duration_ms': 500,
                    'speech_pad_ms': 400
                }
            }
            
            if initial_prompt:
                transcribe_params['initial_prompt'] = initial_prompt
            
            # Start transcription
            start_time = time.time()
            segments, info = self.model.transcribe(str(audio_path), **transcribe_params)
            
            # Collect results
            results = {
                'text': '',
                'segments': [],
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'all_language_probs': info.all_language_probs
            }
            
            # Process segments
            full_text = []
            for segment in segments:
                segment_info = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'words': []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_info = {
                            'start': word.start,
                            'end': word.end,
                            'word': word.word,
                            'probability': word.probability
                        }
                        segment_info['words'].append(word_info)
                
                results['segments'].append(segment_info)
                full_text.append(segment.text.strip())
            
            results['text'] = ' '.join(full_text)
            
            end_time = time.time()
            processing_time = end_time - start_time
            speed_ratio = info.duration / processing_time
            
            safe_print(f"⏱️  Processing time: {processing_time:.2f}s (speed ratio: {speed_ratio:.1f}x)")
            
            return results
            
        except Exception as e:
            safe_print(f"❌ Error during transcription: {e}")
            return None
    
    def save_results(self, results, audio_path, output_dir='output'):
        """Save transcription results in multiple formats"""
        if not results:
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        base_name = Path(audio_path).stem
        saved_files = []
        
        try:
            # Save as TXT
            txt_path = output_dir / f"{base_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(results['text'])
            saved_files.append(txt_path)
            
            # Save as JSON (detailed)
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            saved_files.append(json_path)
            
            # # Save as SRT (subtitles)
            # srt_path = output_dir / f"{base_name}.srt"
            # with open(srt_path, 'w', encoding='utf-8') as f:
            #     for i, segment in enumerate(results['segments'], 1):
            #         start_time = self.seconds_to_srt_time(segment['start'])
            #         end_time = self.seconds_to_srt_time(segment['end'])
            #         f.write(f"{i}\n")
            #         f.write(f"{start_time} --> {end_time}\n")
            #         f.write(f"{segment['text']}\n\n")
            # saved_files.append(srt_path)
            
            # # Save as VTT (WebVTT)
            # vtt_path = output_dir / f"{base_name}.vtt"
            # with open(vtt_path, 'w', encoding='utf-8') as f:
            #     f.write("WEBVTT\n\n")
            #     for segment in results['segments']:
            #         start_time = self.seconds_to_vtt_time(segment['start'])
            #         end_time = self.seconds_to_vtt_time(segment['end'])
            #         f.write(f"{start_time} --> {end_time}\n")
            #         f.write(f"{segment['text']}\n\n")
            # saved_files.append(vtt_path)
            
            # # Save as TSV (tab-separated, with word timestamps)
            # tsv_path = output_dir / f"{base_name}.tsv"
            # with open(tsv_path, 'w', encoding='utf-8') as f:
            #     f.write("start\tend\ttext\n")
            #     for segment in results['segments']:
            #         f.write(f"{segment['start']:.3f}\t{segment['end']:.3f}\t{segment['text']}\n")
            # saved_files.append(tsv_path)
            
            return saved_files
            
        except Exception as e:
            safe_print(f"❌ Error saving results: {e}")
            return saved_files
    
    def seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def seconds_to_vtt_time(self, seconds):
        """Convert seconds to VTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def process_single_file(self, audio_file, model='base', language='ur', task='transcribe', 
                          output_dir='output', analyze_first=True, force_process=False):
        """Process a single audio file"""
        audio_path = Path(audio_file)
        thread_id = threading.get_ident()
        
        safe_print(f"[Thread {thread_id}] 🎵 Processing: {audio_path.name}")
        
        try:
            # Step 1: Analyze audio if requested
            files_to_process = [audio_path]
            
            if analyze_first:
                analyzer = AudioAnalyzer(audio_path)
                
                if not analyzer.load_audio():
                    return {'success': False, 'file': audio_path, 'error': 'Failed to load audio'}
                
                analyzer.analyze_audio_quality()
                needs_processing, processing_steps = analyzer.needs_preprocessing()
                
                # Step 2: Preprocess if needed
                if needs_processing or force_process:
                    safe_print(f"[Thread {thread_id}] 🔧 Preprocessing: {', '.join(processing_steps)}")
                    preprocessor = AudioPreprocessor(audio_path)
                    
                    current_file = audio_path
                    
                    # Apply preprocessing steps in order
                    if 'resample' in processing_steps:
                        current_file = preprocessor.resample_audio()
                        preprocessor.audio_path = current_file
                    
                    if 'normalize' in processing_steps:
                        current_file = preprocessor.normalize_audio()
                        preprocessor.audio_path = current_file
                    
                    if 'denoise' in processing_steps:
                        current_file = preprocessor.basic_denoise()
                        preprocessor.audio_path = current_file
                    
                    if 'split' in processing_steps:
                        files_to_process = preprocessor.split_audio()
                    else:
                        files_to_process = [current_file]
            
            # Step 3: Transcribe
            if FASTER_WHISPER_AVAILABLE:
                initial_prompt = "This is a Quran recitation and Islamic lecture in Urdu language with Quranic verses and Islamic terminology." if task == 'transcribe' else None
                
                for file_to_process in files_to_process:
                    transcription = self.transcribe_with_faster_whisper(
                        file_to_process,
                        language=language,
                        task=task,
                        initial_prompt=initial_prompt
                    )
                    
                    if transcription:
                        saved_files = self.save_results(transcription, file_to_process, output_dir)
                        if saved_files:
                            safe_print(f"[Thread {thread_id}] ✅ Successfully processed: {audio_path.name}")
                            return {'success': True, 'file': audio_path, 'saved_files': saved_files}
                        else:
                            return {'success': False, 'file': audio_path, 'error': 'Failed to save results'}
                    else:
                        return {'success': False, 'file': audio_path, 'error': 'Failed to transcribe'}
            
            else:
                return {'success': False, 'file': audio_path, 'error': 'No Whisper installation available'}
        
        except Exception as e:
            safe_print(f"[Thread {thread_id}] ❌ Error processing {audio_path.name}: {e}")
            return {'success': False, 'file': audio_path, 'error': str(e)}
    
    def process_batch(self, audio_files, model='base', language='ur', task='transcribe',
                     output_dir='output', max_workers=2, analyze_first=True, force_process=False):
        """Process multiple audio files with threading"""
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load model once for all files
        if FASTER_WHISPER_AVAILABLE:
            if not self.load_model(model):
                safe_print("❌ Failed to load model")
                return []
        
        results = []
        total_files = len(audio_files)
        
        safe_print(f"\n🚀 Starting batch processing:")
        safe_print(f"   Files: {total_files}")
        safe_print(f"   Model: {model}")
        safe_print(f"   Language: {language}")
        safe_print(f"   Max workers: {max_workers}")
        safe_print(f"   Output: {output_dir}")
        
        start_time = time.time()
        
        # Process files with threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self.process_single_file,
                    audio_file,
                    model=model,
                    language=language,
                    task=task,
                    output_dir=output_dir,
                    analyze_first=analyze_first,
                    force_process=force_process
                ): audio_file for audio_file in audio_files
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1
                
                if result['success']:
                    safe_print(f"✅ [{completed}/{total_files}] Completed: {result['file'].name}")
                else:
                    safe_print(f"❌ [{completed}/{total_files}] Failed: {result['file'].name} - {result['error']}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = total_files - successful
        
        safe_print(f"\n📊 Batch Processing Complete!")
        safe_print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
        safe_print(f"   Successful: {successful}/{total_files}")
        safe_print(f"   Failed: {failed}")
        safe_print(f"   Average time per file: {total_time/total_files:.2f}s")
        
        if failed > 0:
            safe_print(f"\n❌ Failed files:")
            for result in results:
                if not result['success']:
                    safe_print(f"   - {result['file'].name}: {result['error']}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Batch Audio Analysis and Whisper Processing')
    parser.add_argument('input_path', help='Path to audio file or folder containing audio files')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
                       help='Whisper model size')
    parser.add_argument('--language', default='ur', help='Audio language (default: ur for Urdu)')
    parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                       help='Task to perform (default: transcribe)')
    parser.add_argument('--output-dir', default='output', help='Output directory for results')
    parser.add_argument('--recursive', action='store_true', help='Search audio files recursively in subdirectories')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing transcriptions')
    parser.add_argument('--no-analysis', action='store_true', help='Skip audio analysis and preprocessing')
    parser.add_argument('--force-process', action='store_true', help='Force preprocessing even if not needed')
    parser.add_argument('--min-duration', type=float, help='Minimum audio duration in seconds')
    parser.add_argument('--max-duration', type=float, help='Maximum audio duration in seconds')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers (default: 2)')
    parser.add_argument('--quality', choices=['speed', 'balanced', 'quality'], default='balanced',
                       help='Processing quality preference')
    
    args = parser.parse_args()
    
    # Print banner
    print("🎵 Batch Audio Analysis and Whisper Processing")
    print("=" * 50)
    print(f"📁 Input: {args.input_path}")
    print(f"🤖 Model: {args.model}")
    print(f"🌍 Language: {args.language}")
    print(f"📝 Task: {args.task}")
    print(f"📂 Output: {args.output_dir}")
    print("=" * 50)
    
    # Initialize components
    batch_processor = BatchAudioProcessor()
    whisper_processor = WhisperProcessor()
    
    # Check if input is file or directory
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"❌ Error: Input path '{input_path}' does not exist!")
        return 1
    
    # Get audio files
    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in batch_processor.supported_formats:
            audio_files = [input_path]
        else:
            print(f"❌ Error: File '{input_path}' is not a supported audio format!")
            print(f"Supported formats: {', '.join(batch_processor.supported_formats)}")
            return 1
    else:
        # Directory
        audio_files = batch_processor.find_audio_files(input_path, recursive=args.recursive)
        
        if not audio_files:
            print(f"❌ No audio files found in '{input_path}'")
            return 1
    
    # Filter by duration if specified
    if args.min_duration or args.max_duration:
        audio_files = batch_processor.filter_by_duration(
            audio_files, 
            min_duration=args.min_duration,
            max_duration=args.max_duration
        )
        
        if not audio_files:
            print("❌ No files passed duration filter")
            return 1
    
    # Check for existing transcriptions
    audio_files = batch_processor.check_existing_transcriptions(
        audio_files, 
        output_dir=args.output_dir,
        skip_existing=not args.overwrite
    )
    
    if not audio_files:
        print("✅ All files already processed!")
        return 0
    
    # Auto-suggest model based on total duration and quality preference
    if args.model == 'base' and len(audio_files) > 1:
        total_duration = 0
        try:
            for file in audio_files[:5]:  # Sample first 5 files
                duration = librosa.get_duration(path=str(file))
                total_duration += duration
            
            avg_duration = total_duration / min(len(audio_files), 5)
            suggested_model = whisper_processor.suggest_model_size(avg_duration, args.quality)
            
            if suggested_model != args.model:
                print(f"💡 Suggestion: Model '{suggested_model}' might be better for your files")
                print(f"   Current: {args.model}, Suggested: {suggested_model}")
        except:
            pass
    
    # Check Whisper installation
    if not whisper_processor.check_faster_whisper():
        print("❌ faster-whisper not available!")
        print("Install with: pip install faster-whisper")
        return 1
    
    # Process files
    try:
        results = whisper_processor.process_batch(
            audio_files,
            model=args.model,
            language=args.language,
            task=args.task,
            output_dir=args.output_dir,
            max_workers=args.workers,
            analyze_first=not args.no_analysis,
            force_process=args.force_process
        )
        
        # Final summary
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"\n🎉 Processing Complete!")
        print(f"✅ Successfully processed: {successful}/{total} files")
        
        if successful > 0:
            print(f"📁 Results saved in: {args.output_dir}")
            print(f"📝 Formats: TXT, JSON")
        
        return 0 if successful > 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
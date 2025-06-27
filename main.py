#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Analysis and Whisper Processing Script
Analyzes audio files for Whisper compatibility and processes them accordingly
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
            print(f"✓ Audio loaded successfully")
            print(f"  - Duration: {self.duration:.2f} seconds ({self.duration/60:.2f} minutes)")
            print(f"  - Sample Rate: {self.sample_rate} Hz")
            print(f"  - Channels: {'Mono' if self.audio_data.ndim == 1 else 'Stereo'}")
            return True
        except Exception as e:
            print(f"✗ Error loading audio: {e}")
            return False
    
    def analyze_audio_quality(self):
        """Analyze audio quality for Whisper compatibility"""
        print("\n🔍 Analyzing audio quality...")
        
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
        # Use a more sensitive threshold for speech detection
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
        # Simple SNR estimation using energy in speech vs silence
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
    
    def print_analysis_results(self):
        """Print detailed analysis results"""
        print("\n📊 Audio Analysis Results:")
        print("=" * 50)
        
        for category, results in self.analysis_results.items():
            # Safely get the optimal status
            optimal = results.get('optimal', True)
            status = "✓" if optimal else "⚠"
            print(f"{status} {category.upper().replace('_', ' ')}")
            
            if category == 'sample_rate':
                print(f"   Current: {results.get('current', 'Unknown')} Hz")
            elif category == 'duration':
                current = results.get('current', 0)
                print(f"   Current: {current:.2f}s ({current/60:.2f}min)")
            elif category == 'speech_activity':
                ratio = results.get('speech_ratio', 0)
                print(f"   Speech ratio: {ratio:.2%}")
            elif category == 'clipping':
                amplitude = results.get('max_amplitude', 0)
                print(f"   Max amplitude: {amplitude:.3f}")
            elif category == 'snr':
                snr = results.get('estimated_snr', 0)
                print(f"   Estimated SNR: {snr:.1f} dB")
            
            recommendation = results.get('recommendation', 'No recommendation available')
            print(f"   → {recommendation}")
            print()
    
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
        print(f"🔄 Resampling to {target_sr} Hz...")
        
        audio, sr = librosa.load(self.audio_path, sr=None)
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        output_path = self.output_dir / f"{self.audio_path.stem}_resampled.wav"
        sf.write(output_path, audio_resampled, target_sr)
        
        print(f"✓ Resampled audio saved: {output_path}")
        return output_path
    
    def split_audio(self, chunk_duration=1800, overlap=30):
        """Split audio into chunks (default 30min with 30s overlap)"""
        print(f"✂️ Splitting audio into {chunk_duration/60:.0f}-minute chunks...")
        
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
                print(f"  ✓ Chunk {chunk_num}: {len(chunk)/sr:.1f}s -> {output_path}")
                chunk_num += 1
            
            start = end - overlap_samples
            if start >= len(audio) - overlap_samples:
                break
        
        return chunks
    
    def normalize_audio(self):
        """Normalize audio to prevent clipping"""
        print("🔧 Normalizing audio...")
        
        audio, sr = librosa.load(self.audio_path, sr=None)
        
        # Normalize to 90% of maximum to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio * (0.9 / max_val)
        else:
            audio_normalized = audio
        
        output_path = self.output_dir / f"{self.audio_path.stem}_normalized.wav"
        sf.write(output_path, audio_normalized, sr)
        
        print(f"✓ Normalized audio saved: {output_path}")
        return output_path
    
    def basic_denoise(self):
        """Apply basic noise reduction using spectral gating"""
        print("🔇 Applying basic noise reduction...")
        
        try:
            import noisereduce as nr
            audio, sr = librosa.load(self.audio_path, sr=None)
            
            # Apply noise reduction
            audio_denoised = nr.reduce_noise(y=audio, sr=sr)
            
            output_path = self.output_dir / f"{self.audio_path.stem}_denoised.wav"
            sf.write(output_path, audio_denoised, sr)
            
            print(f"✓ Denoised audio saved: {output_path}")
            return output_path
            
        except ImportError:
            print("⚠ noisereduce not installed. Skipping noise reduction.")
            print("  Install with: pip install noisereduce")
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
            print("❌ faster-whisper not available")
            return False
        
        try:
            print(f"🔄 Loading faster-whisper model: {model_size}")
            print(f"   Device: {device}, Compute type: {compute_type}")
            
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
            
            print(f"✅ Model loaded successfully on {device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
            print("❌ Model not loaded")
            return None
        
        try:
            print(f"🎯 Transcribing: {Path(audio_path).name}")
            print(f"   Language: {language}, Task: {task}")
            
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
            
            print(f"📊 Detected language: {info.language} (probability: {info.language_probability:.2f})")
            print(f"📏 Audio duration: {info.duration:.2f} seconds")
            
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
                
                # Print progress
                print(f"  [{segment.start:6.1f}s -> {segment.end:6.1f}s] {segment.text.strip()}")
            
            results['text'] = ' '.join(full_text)
            
            end_time = time.time()
            processing_time = end_time - start_time
            speed_ratio = info.duration / processing_time
            
            print(f"⏱️  Processing time: {processing_time:.2f}s (speed ratio: {speed_ratio:.1f}x)")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
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
            
            print(f"💾 Saved {len(saved_files)} output files:")
            for file_path in saved_files:
                print(f"   ✓ {file_path}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
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
    
    def build_whisper_command(self, audio_path, model='base', language='ur', 
                            task='transcribe', output_dir='output'):
        """Build optimized Whisper command for fallback to original whisper"""
        
        cmd = [
            'whisper',
            str(audio_path),
            '--model', model,
            '--language', language,
            '--task', task,
            '--output_dir', output_dir,
            '--output_format', 'json',  # 'all' - Generate multiple formats
            '--verbose', 'True',
            '--temperature', '0.0',  # More deterministic output
            '--best_of', '3',  # Try multiple decodings for better quality
            '--patience', '2.0',  # Wait longer for better results
        ]
        
        # Add specific optimizations for religious content
        if task == 'transcribe':
            # Use ASCII-safe prompt to avoid encoding issues
            cmd.extend([
                '--initial_prompt', 
                'This is a Quran recitation and Islamic lecture in Urdu language with Quranic verses and Islamic terminology.'
            ])
        
        return cmd
    
    def run_whisper(self, audio_files, model='base', language='ur', task='transcribe'):
        """Run Whisper processing (prioritizes faster-whisper, falls back to original)"""
        
        if isinstance(audio_files, (str, Path)):
            audio_files = [audio_files]
        
        # Ensure output directory exists
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Try faster-whisper first
        if FASTER_WHISPER_AVAILABLE:
            print(f"🚀 Using faster-whisper with model '{model}'")
            
            if not self.load_model(model):
                print("❌ Failed to load faster-whisper model")
                return []
            
            results = []
            initial_prompt = "This is a Quran recitation and Islamic lecture in Urdu language with Quranic verses and Islamic terminology." if task == 'transcribe' else None
            
            for i, audio_file in enumerate(audio_files, 1):
                print(f"\n📝 Processing file {i}/{len(audio_files)}: {Path(audio_file).name}")
                
                # Transcribe with faster-whisper
                transcription = self.transcribe_with_faster_whisper(
                    audio_file,
                    language=language,
                    task=task,
                    initial_prompt=initial_prompt
                )
                
                if transcription:
                    # Save results
                    saved_files = self.save_results(transcription, audio_file, output_dir)
                    if saved_files:
                        results.append(str(audio_file))
                        print(f"✅ Successfully processed: {audio_file}")
                    else:
                        print(f"⚠️  Transcribed but failed to save: {audio_file}")
                else:
                    print(f"❌ Failed to transcribe: {audio_file}")
            
            return results
        
        # Fallback to original whisper
        elif self.check_whisper_installation():
            print(f"🔄 Using original whisper with model '{model}'")
            print(f"Language: {language}")
            print(f"Files to process: {len(audio_files)}")
            
            results = []
            for i, audio_file in enumerate(audio_files, 1):
                print(f"\n📝 Processing file {i}/{len(audio_files)}: {Path(audio_file).name}")
                
                cmd = self.build_whisper_command(
                    audio_file, model=model, language=language, 
                    task=task, output_dir=str(output_dir)
                )
                
                print(f"Command: {' '.join(cmd)}")
                
                try:
                    # Run Whisper with proper encoding handling
                    import locale
                    system_encoding = locale.getpreferredencoding()
                    
                    process = subprocess.Popen(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        errors='replace',  # Replace problematic characters
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Print output in real-time with encoding safety
                    for line in process.stdout:
                        try:
                            print(f"  {line.strip()}")
                        except UnicodeEncodeError:
                            # Fallback for problematic characters
                            print(f"  {line.encode('ascii', errors='replace').decode('ascii').strip()}")
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        print(f"✅ Successfully processed: {audio_file}")
                        results.append(str(audio_file))
                    else:
                        print(f"❌ Error processing: {audio_file}")
                        
                except Exception as e:
                    print(f"❌ Error running Whisper: {e}")
            
            return results
        
        else:
            print("❌ No Whisper installation found!")
            print("Install one of:")
            print("   pip install faster-whisper  (recommended)")
            print("   pip install -U openai-whisper")
            return []

def main():
    parser = argparse.ArgumentParser(description='Audio Analysis and Whisper Processing')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
                       help='Whisper model size')
    parser.add_argument('--language', default='ur', help='Audio language (default: ur for Urdu)')
    parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                       help='Transcribe or translate to English')
    parser.add_argument('--quality', choices=['speed', 'balanced', 'quality'], default='balanced',
                       help='Quality preference for model selection')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use for faster-whisper (auto, cpu, cuda)')
    parser.add_argument('--compute-type', choices=['auto', 'int8', 'float16', 'float32'], default='auto',
                       help='Compute type for faster-whisper')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip audio analysis')
    parser.add_argument('--force-process', action='store_true', help='Force preprocessing even if not needed')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_path}")
        return 1
    
    print(f"🎵 Processing: {audio_path.name}")
    print(f"📍 Full path: {audio_path.absolute()}")
    
    # Step 1: Analyze audio
    if not args.skip_analysis:
        analyzer = AudioAnalyzer(audio_path)
        
        if not analyzer.load_audio():
            return 1
        
        analyzer.analyze_audio_quality()
        analyzer.print_analysis_results()
        
        # Check if preprocessing is needed
        needs_processing, processing_steps = analyzer.needs_preprocessing()
        
        if needs_processing and not args.force_process:
            print(f"\n⚠️  Audio needs preprocessing: {', '.join(processing_steps)}")
            response = input("Proceed with preprocessing? (y/n): ").lower().strip()
            if response != 'y':
                print("Skipping preprocessing. You may want to process the audio manually.")
                needs_processing = False
        elif args.force_process:
            needs_processing = True
            processing_steps = ['resample', 'normalize', 'denoise']
    else:
        needs_processing = args.force_process
        processing_steps = ['resample', 'normalize', 'denoise'] if needs_processing else []
    
    # Step 2: Preprocess if needed
    files_to_process = [audio_path]
    
    if needs_processing:
        print(f"\n🔧 Starting preprocessing: {', '.join(processing_steps)}")
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
    
    # Step 3: Suggest optimal model
    whisper_processor = WhisperProcessor()
    
    if not args.skip_analysis:
        total_duration = analyzer.duration
    else:
        # Quick duration check
        audio_data, sr = librosa.load(audio_path, sr=None)
        total_duration = len(audio_data) / sr
    
    suggested_model = whisper_processor.suggest_model_size(total_duration, args.quality)
    
    # Update model choices for newer models
    if suggested_model == 'large':
        suggested_model = 'large-v3'  # Use latest large model
    
    if args.model == 'base' and suggested_model != 'base':
        print(f"\n💡 Suggested model: {suggested_model} (you selected: {args.model})")
        print(f"   Reason: {whisper_processor.model_sizes[suggested_model]}")
        response = input(f"Use suggested model '{suggested_model}'? (y/n): ").lower().strip()
        if response == 'y':
            args.model = suggested_model
    
    # Step 4: Run Whisper
    print(f"\n🚀 Starting Whisper processing...")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Task: {args.task}")
    print(f"Files: {len(files_to_process)}")
    
    if FASTER_WHISPER_AVAILABLE:
        print(f"Engine: faster-whisper")
        print(f"Device: {args.device}")
        print(f"Compute type: {args.compute_type}")
    else:
        print(f"Engine: original whisper (fallback)")
    
    results = whisper_processor.run_whisper(
        files_to_process, 
        model=args.model,
        language=args.language,
        task=args.task
    )
    
    # Step 5: Summary
    print(f"\n✅ Processing Complete!")
    print(f"Successfully processed: {len(results)}/{len(files_to_process)} files")
    print(f"Output directory: output/")
    print(f"Formats generated: txt, json")
    
    if len(results) < len(files_to_process):
        print("⚠️  Some files failed to process. Check the output above for details.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
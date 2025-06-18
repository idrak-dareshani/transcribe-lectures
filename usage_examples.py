# usage_examples.py
"""
Usage examples for the Quranic Transcription Pipeline
"""

from quran_pipeline import QuranicTranscriptionPipeline
import os

def example_single_file():
    """Example: Process a single audio file"""
    # Initialize pipeline
    pipeline = QuranicTranscriptionPipeline("pipeline_config.yaml")
    
    # Process single file
    audio_file = "input_audio/lecture_001.mp3"
    if os.path.exists(audio_file):
        result = pipeline.process_file(audio_file)
        
        print(f"Transcription Results:")
        print(f"- Original text length: {len(result.original_text)} characters")
        print(f"- Corrected text length: {len(result.corrected_text)} characters") 
        print(f"- Corrections made: {len(result.corrections_made)}")
        print(f"- Processing time: {result.processing_time:.2f} seconds")
        print(f"- Audio duration: {result.audio_duration:.2f} seconds")
        
        # Print first 200 characters of corrected text
        print(f"\nFirst 200 characters of corrected text:")
        print(result.corrected_text[:200] + "..." if len(result.corrected_text) > 200 else result.corrected_text)
        
        # Show corrections made
        if result.corrections_made:
            print(f"\nCorrections made:")
            for correction in result.corrections_made[:5]:  # Show first 5
                if correction['type'] == 'word_correction':
                    print(f"- {correction['original']} → {correction['corrected']}")
    else:
        print(f"File not found: {audio_file}")

def example_batch_processing():
    """Example: Process multiple files"""
    pipeline = QuranicTranscriptionPipeline()
    
    # Process all files in directory
    input_dir = "input_audio"
    if os.path.exists(input_dir):
        results = pipeline.process_batch(input_dir)
        
        print(f"Batch processing completed:")
        print(f"- Files processed: {len(results)}")
        
        total_duration = sum(r.audio_duration for r in results)
        total_processing = sum(r.processing_time for r in results)
        total_corrections = sum(len(r.corrections_made) for r in results)
        
        print(f"- Total audio duration: {total_duration:.2f} seconds")
        print(f"- Total processing time: {total_processing:.2f} seconds")
        print(f"- Average speed: {total_duration/total_processing:.2f}x real-time")
        print(f"- Total corrections: {total_corrections}")
    else:
        print(f"Directory not found: {input_dir}")

def example_custom_corrections():
    """Example: Add custom corrections"""
    pipeline = QuranicTranscriptionPipeline()
    
    # Add custom corrections
    pipeline.corrector.add_correction('غلط_spelling', 'صحیح_spelling')
    pipeline.corrector.add_correction('another_wrong', 'another_correct')
    
    print("Custom corrections added!")

def example_configuration():
    """Example: Custom configuration"""
    import yaml
    
    # Create custom config
    custom_config = {
        'model_size': 'large-v3',
        'language': 'ur',
        'confidence_threshold': 0.85,
        'output_dir': 'custom_output',
        'save_preprocessed_audio': False,
        'generate_report': True
    }
    
    # Save custom config
    with open('custom_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    # Use custom config
    pipeline = QuranicTranscriptionPipeline('custom_config.yaml')
    print("Pipeline initialized with custom configuration!")

if __name__ == "__main__":
    print("Quranic Transcription Pipeline Examples")
    print("=" * 40)
    
    print("\n1. Single file processing:")
    example_single_file()
    
    print("\n2. Batch processing:")
    example_batch_processing()
    
    print("\n3. Custom corrections:")
    example_custom_corrections()
    
    print("\n4. Custom configuration:")
    example_configuration()

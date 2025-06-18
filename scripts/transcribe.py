#import whisper
from huggingface_hub import snapshot_download
import os
import json
from utils import clean_transcription, highlight_quranic_references
from corrector import IslamicUrduCorrector

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Download the model if not already present
# if not os.path.exists("./models/whisper-largev3-junev9"):
#     snapshot_download(repo_id="zahoor54321/whisper-largev3-junev9", local_dir="./models")

import whisper
# Load the Whisper model   
#model = whisper.load_model("large-v3", language="ur", download_root="./models/whisper-largev3-junev9")
model = whisper.load_model("large-v3", temprature=0.0, beam_size=5, patience=1.0, language="ur")

def transcribe_file(filepath):
    #wav_path = convert_to_wav(filepath)
    result = model.transcribe(filepath, language="ur", verbose=True)
    result_text = clean_transcription(result["text"])
    result_text = highlight_quranic_references(result_text)
    corrector = IslamicUrduCorrector()
    result_text = corrector.apply_corrections(result_text)  # Assuming apply_corrections is defined in utils.py

    return result_text, result["segments"]

if __name__ == "__main__":
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".mp3"):
            filename = file.replace(".mp3", "")
            print(f"Processing {file}...")
            
            result_text, segments = transcribe_file(os.path.join(INPUT_DIR, file))
            #out_path = os.path.join(OUTPUT_DIR, file.replace(".mp3", ".txt"))
            
            with open(os.path.join(OUTPUT_DIR, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write(result_text)

            with open(os.path.join(OUTPUT_DIR, f"{filename}.json"), "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

            print(f"Saved outputs for: {filename}")

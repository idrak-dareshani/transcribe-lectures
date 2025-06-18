#import ffmpeg
#import os

# def convert_to_wav(mp3_path):
#     wav_path = mp3_path.replace(".mp3", ".wav")
#     if not os.path.exists(wav_path):
#         ffmpeg.input(mp3_path).output(wav_path).run(overwrite_output=True)
#     return wav_path

def clean_transcription(text):
    # Basic cleanup: remove filler noise, normalize Urdu/Arabic punctuation
    text = text.replace("،", ",").replace("۔", ".")
    return text.strip()

import re

def highlight_quranic_references(text):
    # Example: Detect common patterns like "Surah Al-Baqarah ayah 2"
    pattern = r"(Surah\s+[A-Za-z\u0600-\u06FF\-]+(\s+ayah\s+\d+)?|سورة\s+[^\s]+)"
    matches = re.findall(pattern, text)
    for match in matches:
        reference = match[0]
        text = text.replace(reference, f"[QURAN_REF: {reference}]")
    return text

import re
import json
from typing import Dict, List, Tuple

class IslamicUrduCorrector:
    def __init__(self):
        # Common S-sound corrections (س → ص)
        self.s_corrections = {
            # Core Islamic concepts
            'سبر': 'صبر',           # patience
            'سدقہ': 'صدقہ',         # charity
            'سوم': 'صوم',           # fasting
            'سالح': 'صالح',         # righteous
            'سلاة': 'صلاة',         # prayer
            'سلوة': 'صلوٰة',        # prayer (with diacritics)
            'سراط': 'صراط',        # path
            'سحابہ': 'صحابہ',       # companions
            'سحیح': 'صحیح',         # authentic/correct
            'سف': 'صف',            # row/line
            'سفا': 'صفا',          # purity
            'سغیر': 'صغیر',        # small/minor
            'سنعت': 'صنعت',        # craft/art
            
            # Names and attributes
            'سفی': 'صفی',          # chosen one
            'سمد': 'صمد',          # eternal
            'سابر': 'صابر',        # patient
            'سادق': 'صادق',        # truthful
            'سالح': 'صالح',        # righteous
            
            # Quranic terms
            'سحف': 'صحف',          # scriptures
            'سخرہ': 'صخرہ',        # rock
            'سعب': 'صعب',          # difficult
            'سغار': 'صغار',        # humiliation
        }
        
        # Z-sound corrections (ز → ذ، ض، ظ)
        self.z_corrections = {
            'زکر': 'ذکر',           # remembrance
            'زات': 'ذات',          # essence/self
            'زریعہ': 'ذریعہ',       # means/source
            'زخیرہ': 'ذخیرہ',       # treasure
            'زہن': 'ذہن',          # mind
            'زمہ': 'ذمہ',          # responsibility
            'زلت': 'ذلت',          # humiliation
            'زوق': 'ذوق',          # taste/preference
            
            # ض corrections
            'زرور': 'ضرور',        # necessary
            'زروری': 'ضروری',       # necessary
            'زعیف': 'ضعیف',        # weak
            'زرر': 'ضرر',          # harm
            'زبط': 'ضبط',          # control
            
            # ظ corrections
            'زالم': 'ظالم',        # oppressor
            'زلم': 'ظلم',          # oppression
            'زاہر': 'ظاہر',        # apparent
            'زرف': 'ظرف',          # container/capacity
        }
        
        # Common extra letter removals
        self.extra_letter_patterns = [
            (r'اللہہ', 'اللہ'),      # Allah with extra ہ
            (r'محمدد', 'محمد'),       # Muhammad with extra د
            (r'قرآنن', 'قرآن'),       # Quran with extra ن
            (r'اسلامم', 'اسلام'),     # Islam with extra م
            (r'نمازز', 'نماز'),       # prayer with extra ز
            (r'روزہہ', 'روزہ'),       # fast with extra ہ
            (r'حجج', 'حج'),          # Hajj with extra ج
            (r'عمرہہ', 'عمرہ'),       # Umrah with extra ہ
        ]
        
        # Context-based corrections for common religious phrases
        self.phrase_corrections = {
            'سبحان اللہ': 'سبحان اللہ',
            'الحمد للہ': 'الحمدللہ',
            'لا الہ الا اللہ': 'لا الٰہ الا اللہ',
            'اللہ اکبر': 'اللہ اکبر',
            'بسم اللہ': 'بسم اللہ',
            'ان شاء اللہ': 'انشاءاللہ',
            'ماشاء اللہ': 'ماشاءاللہ',
            'استغفر اللہ': 'استغفراللہ',
        }
        
        # Diacritics and proper spelling
        self.diacritic_corrections = {
            'قران': 'قرآن',          # Quran with proper آ
            'رحمن': 'رحمٰن',         # Rahman with proper diacritic
            'رحیم': 'رحیم',          # Raheem
            'علیہ السلام': 'علیہ السلام',
            'صلی اللہ علیہ وسلم': 'صلی اللہ علیہ وسلم',
            'رضی اللہ عنہ': 'رضی اللہ عنہ',
            'رضی اللہ عنہا': 'رضی اللہ عنہا',
        }
    
    def apply_corrections(self, text: str) -> str:
        """Apply all correction rules to the input text"""
        corrected_text = text
        
        # Apply S-sound corrections
        for incorrect, correct in self.s_corrections.items():
            corrected_text = re.sub(r'\b' + re.escape(incorrect) + r'\b', 
                                  correct, corrected_text)
        
        # Apply Z-sound corrections
        for incorrect, correct in self.z_corrections.items():
            corrected_text = re.sub(r'\b' + re.escape(incorrect) + r'\b', 
                                  correct, corrected_text)
        
        # Remove extra letters
        for pattern, replacement in self.extra_letter_patterns:
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        # Apply phrase corrections
        for incorrect, correct in self.phrase_corrections.items():
            corrected_text = re.sub(re.escape(incorrect), correct, corrected_text)
        
        # Apply diacritic corrections
        for incorrect, correct in self.diacritic_corrections.items():
            corrected_text = re.sub(r'\b' + re.escape(incorrect) + r'\b', 
                                  correct, corrected_text)
        
        return corrected_text
    
    def batch_correct_file(self, input_file: str, output_file: str):
        """Process a file with Whisper transcriptions"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            corrected_content = self.apply_corrections(content)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corrected_content)
            
            print(f"Corrected transcription saved to: {output_file}")
            
        except FileNotFoundError:
            print(f"Error: File {input_file} not found")
        except Exception as e:
            print(f"Error processing file: {str(e)}")
    
    def add_custom_correction(self, incorrect: str, correct: str, category: str = 's_corrections'):
        """Add custom corrections to the corrector"""
        if category == 's_corrections':
            self.s_corrections[incorrect] = correct
        elif category == 'z_corrections':
            self.z_corrections[incorrect] = correct
        elif category == 'phrase_corrections':
            self.phrase_corrections[incorrect] = correct
        elif category == 'diacritic_corrections':
            self.diacritic_corrections[incorrect] = correct
        else:
            print(f"Unknown category: {category}")
    
    def get_correction_stats(self, original_text: str, corrected_text: str) -> Dict:
        """Get statistics about corrections made"""
        stats = {
            'total_s_corrections': 0,
            'total_z_corrections': 0,
            'total_phrase_corrections': 0,
            'total_diacritic_corrections': 0,
            'corrections_made': []
        }
        
        # Count S-corrections
        for incorrect, correct in self.s_corrections.items():
            count = len(re.findall(r'\b' + re.escape(incorrect) + r'\b', original_text))
            if count > 0:
                stats['total_s_corrections'] += count
                stats['corrections_made'].append(f"{incorrect} → {correct} ({count} times)")
        
        # Count Z-corrections
        for incorrect, correct in self.z_corrections.items():
            count = len(re.findall(r'\b' + re.escape(incorrect) + r'\b', original_text))
            if count > 0:
                stats['total_z_corrections'] += count
                stats['corrections_made'].append(f"{incorrect} → {correct} ({count} times)")
        
        return stats

# Example usage
def main():
    # Initialize the corrector
    corrector = IslamicUrduCorrector()
    
    # Example text with common errors
    sample_text = """
    اللہ تعالیٰ نے ہمیں سبر کرنے کا حکم دیا ہے۔ سدقہ دینا بہت اہم ہے۔
    ہمیں زکر اللہ کرنا چاہیے اور سوم رکھنا چاہیے۔
    قران میں بہت سی باتیں ہیں جو ہماری رہنمائی کرتی ہیں۔
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Apply corrections
    corrected_text = corrector.apply_corrections(sample_text)
    
    print("Corrected text:")
    print(corrected_text)
    print("\n" + "="*50 + "\n")
    
    # Get correction statistics
    stats = corrector.get_correction_stats(sample_text, corrected_text)
    print("Correction Statistics:")
    print(f"S-sound corrections: {stats['total_s_corrections']}")
    print(f"Z-sound corrections: {stats['total_z_corrections']}")
    print(f"Phrase corrections: {stats['total_phrase_corrections']}")
    print(f"Diacritic corrections: {stats['total_diacritic_corrections']}")
    
    if stats['corrections_made']:
        print("\nCorrections made:")
        for correction in stats['corrections_made']:
            print(f"  • {correction}")

if __name__ == "__main__":
    main()
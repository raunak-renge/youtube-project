#!/usr/bin/env python3
"""
Interactive Reel Shorts Designer
================================
Creates viral short-form video content using:
- Ollama for AI script generation
- Bing Image Crawler for visuals
- IndexTTS2 for voice synthesis
- MoviePy for video compilation with Ken Burns effects
"""

import os
import sys
import json
import random
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import math

# Check and install required packages
def install_packages():
    """Install required packages"""
    packages = [
        "icrawler",
        "moviepy",
        "numpy",
        "Pillow",
        "requests",
        "scipy",
        "soundfile",
        "openai-whisper",
        "google-auth",
        "google-auth-oauthlib",
        "google-auth-httplib2",
        "google-api-python-client",
    ]
    for pkg in packages:
        try:
            pkg_import = pkg.replace("-", "_").lower()
            if pkg == "openai-whisper":
                pkg_import = "whisper"
            elif pkg == "google-api-python-client":
                pkg_import = "googleapiclient"
            elif pkg == "google-auth-oauthlib":
                pkg_import = "google_auth_oauthlib"
            elif pkg == "google-auth-httplib2":
                pkg_import = "google_auth_httplib2"
            elif pkg == "google-auth":
                pkg_import = "google.auth"
            __import__(pkg_import)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"])

install_packages()

from datetime import datetime
import glob
import time
import pickle
import http.client
import httplib2

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import whisper
# MoviePy 2.x API
from moviepy.video.VideoClip import ImageClip, VideoClip, TextClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
from icrawler.builtin import BingImageCrawler

# Google API imports for YouTube upload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the reel designer"""
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = Path("./reel_output")
    IMAGES_DIR = OUTPUT_DIR / "images"
    AUDIO_DIR = OUTPUT_DIR / "audio"
    VIDEO_DIR = OUTPUT_DIR / "video"
    SCRIPTS_DIR = OUTPUT_DIR / "scripts"  # Separate scripts folder
    
    # Audio
    MUSIC_DIR = BASE_DIR / "background-sounds" / "music"
    CLICKS_DIR = BASE_DIR / "background-sounds" / "clicks"
    
    # Kokoro TTS
    KOKORO_MODEL = BASE_DIR / "kokoro-v1.0.onnx"
    KOKORO_VOICES = BASE_DIR / "voices-v1.0.bin"
    TTS_VOICE = "af_bella"
    TTS_SPEED = 1.0  # Normal speed for longer content
    
    # Video (YouTube Shorts 9:16)
    VIDEO_WIDTH = 1080
    VIDEO_HEIGHT = 1920
    FPS = 30
    
    # Ken Burns effect settings
    ZOOM_RANGE = (1.0, 1.3)  # Zoom from 100% to 130%
    PAN_RANGE = 0.15  # Max 15% pan in any direction
    
    # Ollama settings
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen3:8b"
    
    # Audio levels
    MUSIC_VOLUME = 0.12  # Background music volume
    CLICK_VOLUME = 0.30  # Transition click volume
    
    # YouTube Upload settings
    GOOGLE_CONSOLE_DIR = BASE_DIR / "google-console"
    YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    MAX_TITLE_LENGTH = 100  # YouTube title limit
    MAX_TAGS_LENGTH = 500  # YouTube tags character limit
    
    @classmethod
    def setup_dirs(cls):
        """Create output directories"""
        for d in [cls.OUTPUT_DIR, cls.IMAGES_DIR, cls.AUDIO_DIR, cls.VIDEO_DIR, cls.SCRIPTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_music(cls) -> Optional[str]:
        """Get a random background music file"""
        if not cls.MUSIC_DIR.exists():
            return None
        files = list(cls.MUSIC_DIR.glob("*.wav")) + list(cls.MUSIC_DIR.glob("*.mp3"))
        return str(random.choice(files)) if files else None
    
    @classmethod
    def get_click(cls) -> Optional[str]:
        """Get a random click sound file"""
        if not cls.CLICKS_DIR.exists():
            return None
        files = list(cls.CLICKS_DIR.glob("*.wav")) + list(cls.CLICKS_DIR.glob("*.mp3"))
        return str(random.choice(files)) if files else None


# ============================================================================
# WORD-BY-WORD TRANSCRIPT (Using Whisper)
# ============================================================================

class WordTranscriber:
    """Transcribe audio to get word-level timestamps using Whisper"""
    
    def __init__(self, model_size: str = "turbo"):
        """Initialize Whisper model (base is fast, large is accurate)"""
        self.model_size = model_size
        self._model = None
    
    @property
    def model(self):
        """Lazy load Whisper model"""
        if self._model is None:
            print(f"  📝 Loading Whisper ({self.model_size}) model...")
            self._model = whisper.load_model(self.model_size)
        return self._model
    
    def transcribe(self, audio_path: Path, original_text: str = None) -> List[Dict]:
        """
        Transcribe audio and return word-level timestamps.
        If original_text is provided, aligns Whisper output with it.
        Returns: List of {"word": str, "start": float, "end": float}
        """
        try:
            result = self.model.transcribe(
                str(audio_path),
                word_timestamps=True
            )
            
            words = []
            for segment in result.get('segments', []):
                for word_info in segment.get('words', []):
                    words.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end']
                    })
            
            # Align with original text if provided
            if original_text and words:
                words = self.align_with_original(words, original_text)
            
            return words
        except Exception as e:
            print(f"    ⚠️ Whisper transcription error: {e}")
            return []
    
    def align_with_original(self, whisper_words: List[Dict], original_text: str) -> List[Dict]:
        """
        Align Whisper transcription with original narration text.
        
        ROBUST ALGORITHM:
        - Keep Whisper's word COUNT and TIMING exactly as-is
        - Only CORRECT SPELLINGS by finding best matching original word
        - Never alter the sequence of words
        - Handle cases where Whisper has more or fewer words than original
        
        Returns: List with corrected spellings but original Whisper timestamps
        """
        if not whisper_words:
            return []
        
        if not original_text:
            return whisper_words
        
        # Tokenize original text
        original_words = self._tokenize_text(original_text)
        
        if not original_words:
            return whisper_words
        
        # Create aligned result - same length as whisper_words
        aligned = []
        
        # Track which original words we've used
        used_original = [False] * len(original_words)
        
        for w_idx, whisper_word in enumerate(whisper_words):
            whisper_text = whisper_word['word']
            whisper_norm = self._normalize_word(whisper_text)
            
            # Find the best matching original word
            best_match_idx = -1
            best_similarity = 0.0
            
            # Search in a window around the expected position
            # Expected position based on proportional mapping
            expected_pos = int((w_idx / len(whisper_words)) * len(original_words))
            
            # Search window: prefer nearby words, but check all if needed
            search_order = self._get_search_order(expected_pos, len(original_words))
            
            for orig_idx in search_order:
                if used_original[orig_idx]:
                    continue
                
                orig_word = original_words[orig_idx]
                similarity = self._word_similarity(whisper_text, orig_word)
                
                # Prefer exact matches or very high similarity
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = orig_idx
                
                # If we found an exact match, stop searching
                if similarity >= 0.95:
                    break
            
            # Determine which word to use
            if best_match_idx >= 0 and best_similarity >= 0.4:
                # Good match found - use original spelling with Whisper timing
                corrected_word = original_words[best_match_idx]
                used_original[best_match_idx] = True
            else:
                # No good match - keep Whisper's word as-is
                corrected_word = whisper_text
            
            aligned.append({
                'word': corrected_word,
                'start': whisper_word['start'],
                'end': whisper_word['end']
            })
        
        return aligned
    
    def _get_search_order(self, expected_pos: int, total_len: int) -> List[int]:
        """
        Generate search order starting from expected position, expanding outward.
        This prioritizes finding matches near the expected position.
        """
        order = []
        # Start at expected position, then alternate left and right
        left = expected_pos
        right = expected_pos + 1
        
        while left >= 0 or right < total_len:
            if left >= 0:
                order.append(left)
                left -= 1
            if right < total_len:
                order.append(right)
                right += 1
        
        return order
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words, preserving punctuation attached to words"""
        # Remove extra whitespace and split
        text = ' '.join(text.split())
        # Split on whitespace but keep contractions together
        words = []
        for word in text.split():
            # Clean but preserve the word
            cleaned = word.strip()
            if cleaned:
                words.append(cleaned)
        return words
    
    def _normalize_word(self, word: str) -> str:
        """Normalize word for comparison (lowercase, remove punctuation)"""
        return re.sub(r'[^\w\s]', '', word.lower()).strip()
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words (0 to 1).
        Uses multiple methods for robust matching.
        """
        w1 = self._normalize_word(word1)
        w2 = self._normalize_word(word2)
        
        # Exact match
        if w1 == w2:
            return 1.0
        
        # Empty check
        if not w1 or not w2:
            return 0.0
        
        # One contains the other (handles contractions, suffixes)
        if w1 in w2 or w2 in w1:
            return 0.85
        
        # Same first letter and similar length (likely same word)
        if w1[0] == w2[0] and abs(len(w1) - len(w2)) <= 2:
            # Calculate Levenshtein-like ratio
            matches = sum(1 for c1, c2 in zip(w1, w2) if c1 == c2)
            max_len = max(len(w1), len(w2))
            ratio = matches / max_len
            if ratio >= 0.6:
                return 0.7 + (ratio * 0.2)
        
        # Check for common transcription errors
        # Numbers: "70,000" vs "seventy thousand"
        # We'll handle these by checking if both represent numbers
        
        # Character-level similarity (Levenshtein ratio approximation)
        # Count matching characters in order
        matches = 0
        j = 0
        for c in w1:
            while j < len(w2):
                if w2[j] == c:
                    matches += 1
                    j += 1
                    break
                j += 1
        
        max_len = max(len(w1), len(w2))
        return matches / max_len if max_len > 0 else 0.0


# ============================================================================
# SMART IMAGE SELECTOR (Prefer Portrait for 9:16)
# ============================================================================

class SmartImageSelector:
    """
    Smart image selection that prefers images matching the target orientation.
    For 9:16 shorts, portrait images look better and require less cropping.
    """
    
    TARGET_ASPECT = Config.VIDEO_WIDTH / Config.VIDEO_HEIGHT  # 0.5625 for 9:16
    
    @staticmethod
    def get_image_score(image_path: Path) -> Tuple[float, Dict]:
        """
        Score an image based on how well it fits 9:16 format.
        Higher score = better fit.
        Returns: (score, metadata)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect = width / height
                
                # Calculate aspect ratio similarity (1.0 = perfect match)
                aspect_score = 1.0 - abs(aspect - SmartImageSelector.TARGET_ASPECT)
                
                # Prefer portrait over landscape for shorts
                orientation_bonus = 0.3 if height > width else 0.0
                
                # Prefer higher resolution
                resolution = width * height
                resolution_score = min(resolution / (1920 * 1080), 1.0) * 0.2
                
                # Check if image has enough height for vertical video
                height_score = min(height / 1080, 1.0) * 0.3
                
                total_score = aspect_score + orientation_bonus + resolution_score + height_score
                
                metadata = {
                    'width': width,
                    'height': height,
                    'aspect': aspect,
                    'orientation': 'portrait' if height > width else 'landscape',
                    'resolution': resolution
                }
                
                return total_score, metadata
        except Exception as e:
            return 0.0, {'error': str(e)}
    
    @staticmethod
    def select_best_image(images: List[Path]) -> Optional[Path]:
        """Select the best image from a list based on 9:16 compatibility"""
        if not images:
            return None
        
        scored_images = []
        for img_path in images:
            score, metadata = SmartImageSelector.get_image_score(img_path)
            scored_images.append((score, img_path, metadata))
        
        # Sort by score (highest first)
        scored_images.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_path, best_meta = scored_images[0]
        print(f"      → Selected: {best_meta.get('orientation', 'unknown')} "
              f"({best_meta.get('width', 0)}x{best_meta.get('height', 0)}) "
              f"score={best_score:.2f}")
        
        return best_path
    
    @staticmethod
    def smart_crop_for_vertical(image_path: Path, target_width: int, target_height: int) -> np.ndarray:
        """
        Intelligently crop image to fit vertical format.
        Uses center-weighted cropping with smart aspect handling.
        """
        img = Image.open(image_path).convert('RGB')
        img_w, img_h = img.size
        target_aspect = target_width / target_height
        img_aspect = img_w / img_h
        
        if img_aspect > target_aspect:
            # Image is wider than target - crop sides
            new_width = int(img_h * target_aspect)
            left = (img_w - new_width) // 2
            img = img.crop((left, 0, left + new_width, img_h))
        else:
            # Image is taller than target - crop top/bottom (favor upper portion for faces)
            new_height = int(img_w / target_aspect)
            # Favor upper 40% for faces/subjects
            top = int((img_h - new_height) * 0.3)
            img = img.crop((0, top, img_w, top + new_height))
        
        # Resize to exact dimensions
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return np.array(img)


# ============================================================================
# ANIMATED TEXT OVERLAY (Smart Phrase Captions - Like TikTok/Reels)
# ============================================================================

class SmartPhraseGrouper:
    """
    Intelligently groups words for captions.
    - Keeps numbers with their units together ("7 million dollars")
    - Doesn't show words after punctuation in same frame
    - Creates natural reading chunks
    """
    
    # Patterns that should be grouped together
    NUMBER_WORDS = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
                    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
                    'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
                    'hundred', 'thousand', 'million', 'billion', 'trillion'}
    
    UNIT_WORDS = {'dollars', 'euros', 'pounds', 'percent', 'percentage', 'years', 
                  'months', 'days', 'hours', 'minutes', 'seconds', 'times', 'people',
                  'users', 'customers', 'followers', 'views', 'likes', 'subscribers'}
    
    MAGNITUDE_WORDS = {'thousand', 'million', 'billion', 'trillion', 'hundred'}
    
    @staticmethod
    def group_words_for_display(words: List[Dict]) -> List[Dict]:
        """
        Group words intelligently for caption display.
        Returns list of phrase groups with combined timing.
        """
        if not words:
            return []
        
        grouped = []
        i = 0
        
        while i < len(words):
            word = words[i]
            word_lower = word['word'].lower().strip()
            word_clean = re.sub(r'[^\w]', '', word_lower)
            
            # Check if this word ends with sentence-ending punctuation
            ends_sentence = word['word'].rstrip().endswith(('.', '!', '?', '…', '...'))
            
            # Check if this is a number or starts a number phrase
            is_number = word_clean.isdigit() or word_clean in SmartPhraseGrouper.NUMBER_WORDS
            
            if is_number:
                # Collect the full number phrase (e.g., "7 million dollars")
                phrase_words = [word]
                j = i + 1
                
                while j < len(words):
                    next_word = words[j]
                    next_lower = next_word['word'].lower().strip()
                    next_clean = re.sub(r'[^\w]', '', next_lower)
                    
                    # Check if next word is part of number phrase
                    is_magnitude = next_clean in SmartPhraseGrouper.MAGNITUDE_WORDS
                    is_unit = next_clean in SmartPhraseGrouper.UNIT_WORDS
                    is_num = next_clean.isdigit() or next_clean in SmartPhraseGrouper.NUMBER_WORDS
                    
                    if is_magnitude or is_unit or is_num:
                        phrase_words.append(next_word)
                        j += 1
                        # Stop after unit word
                        if is_unit:
                            break
                    else:
                        break
                
                # Create grouped phrase
                combined_text = ' '.join(w['word'] for w in phrase_words)
                grouped.append({
                    'word': combined_text,
                    'start': phrase_words[0]['start'],
                    'end': phrase_words[-1]['end'],
                    'is_phrase': True
                })
                i = j
            else:
                # Single word - just add it
                grouped.append({
                    'word': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'is_phrase': False,
                    'ends_sentence': ends_sentence
                })
                i += 1
        
        return grouped
    
    @staticmethod
    def should_show_word(current_phrase: Dict, prev_phrase: Optional[Dict], current_time: float) -> bool:
        """
        Determine if we should show this phrase at current time.
        Don't show word immediately after sentence-ending punctuation in same timeframe.
        """
        if not current_phrase:
            return False
        
        # Basic timing check
        if not (current_phrase['start'] <= current_time <= current_phrase['end']):
            return False
        
        # If previous phrase ended a sentence, ensure small gap
        if prev_phrase and prev_phrase.get('ends_sentence', False):
            # Add a small visual pause after sentences
            if current_time < current_phrase['start'] + 0.1:
                return False
        
        return True


class AnimatedCaptions:
    """
    Create beautiful phrase-aware animated captions.
    Shows ONE word/phrase at a time, centered and highlighted in YELLOW.
    Keeps numbers with units together for natural reading.
    """
    
    # Text styling - Large, bold, yellow text
    FONT_SIZE = 90  # Slightly smaller for phrases
    FONT_COLOR = (255, 255, 0)  # Bright Yellow
    STROKE_COLOR = (0, 0, 0)  # Black outline for readability
    STROKE_WIDTH = 5  # Thicker stroke for visibility
    BG_COLOR = (0, 0, 0, 180)  # Semi-transparent black background
    
    # Position (center of screen, lower third area)
    POSITION_Y_RATIO = 0.72  # 72% from top (lower third)
    
    # Animation settings
    PADDING = 30
    MAX_CHARS = 25  # Max characters before reducing font size
    
    def __init__(self, width: int = Config.VIDEO_WIDTH, height: int = Config.VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.font = self._load_font()
        self.small_font = self._load_font(size=72)
        self.phrase_grouper = SmartPhraseGrouper()
        self._grouped_cache = {}
    
    def _load_font(self, size: int = None) -> ImageFont.FreeTypeFont:
        """Load a bold font for captions"""
        size = size or self.FONT_SIZE
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for path in font_paths:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except:
                    continue
        return ImageFont.load_default()
    
    def _get_grouped_words(self, words: List[Dict]) -> List[Dict]:
        """Get or create grouped words (cached)"""
        # Create a simple hash for caching
        cache_key = len(words)
        if cache_key not in self._grouped_cache:
            self._grouped_cache[cache_key] = SmartPhraseGrouper.group_words_for_display(words)
        return self._grouped_cache[cache_key]
    
    def create_caption_frame(self, words: List[Dict], current_time: float) -> np.ndarray:
        """
        Create a transparent frame with a SINGLE word or phrase.
        Intelligently groups numbers with units.
        """
        # Create transparent image
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Get grouped words/phrases
        grouped = self._get_grouped_words(words)
        
        # Find the current phrase being spoken
        current_phrase, prev_phrase = self._get_current_phrase(grouped, current_time)
        
        if not current_phrase:
            return np.array(img)
        
        # Check if we should show this phrase
        if not SmartPhraseGrouper.should_show_word(current_phrase, prev_phrase, current_time):
            return np.array(img)
        
        # Draw the phrase centered
        self._draw_phrase(draw, current_phrase['word'])
        
        return np.array(img)
    
    def _get_current_phrase(self, grouped: List[Dict], current_time: float) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get the current phrase and previous phrase at current time"""
        prev_phrase = None
        for i, phrase in enumerate(grouped):
            if phrase['start'] <= current_time <= phrase['end']:
                prev_phrase = grouped[i-1] if i > 0 else None
                return phrase, prev_phrase
            if phrase['start'] > current_time:
                break
            prev_phrase = phrase
        return None, prev_phrase
    
    def _draw_phrase(self, draw: ImageDraw.Draw, text: str):
        """Draw a word or phrase centered on screen with yellow highlight"""
        # Clean and format the text
        display_text = text.upper().strip()
        
        # Remove trailing punctuation for cleaner display (but keep internal)
        display_text = display_text.rstrip('.,!?;:')
        
        # Choose font size based on text length
        font = self.font if len(display_text) <= self.MAX_CHARS else self.small_font
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center position
        x = (self.width - text_width) // 2
        y = int(self.height * self.POSITION_Y_RATIO) - text_height // 2
        
        # Draw background pill/rectangle
        bg_padding_x = 50
        bg_padding_y = 25
        bg_rect = [
            x - bg_padding_x,
            y - bg_padding_y,
            x + text_width + bg_padding_x,
            y + text_height + bg_padding_y
        ]
        # Rounded rectangle background
        draw.rounded_rectangle(bg_rect, radius=20, fill=self.BG_COLOR)
        
        # Draw text with stroke (outline) for readability
        for dx in range(-self.STROKE_WIDTH, self.STROKE_WIDTH + 1):
            for dy in range(-self.STROKE_WIDTH, self.STROKE_WIDTH + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), display_text, 
                             font=font, fill=self.STROKE_COLOR)
        
        # Draw main text in YELLOW
        draw.text((x, y), display_text, font=font, fill=self.FONT_COLOR)


# ============================================================================
# TEXT PREPROCESSING FOR TTS (Expand abbreviations)
# ============================================================================

def expand_abbreviations_for_tts(text: str) -> str:
    """
    Expand abbreviations and symbols for proper TTS pronunciation.
    Converts $100B → 100 billion dollars, 50% → 50 percent, etc.
    """
    import re
    
    # Currency with magnitude (must come before simple currency)
    # $100B, $5M, $2T, etc.
    def expand_currency_magnitude(match):
        symbol = match.group(1)
        number = match.group(2)
        magnitude = match.group(3).upper()
        
        currency = "dollars" if symbol == "$" else "euros" if symbol == "€" else "pounds" if symbol == "£" else "units"
        magnitudes = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}
        mag_word = magnitudes.get(magnitude, "")
        
        return f"{number} {mag_word} {currency}"
    
    text = re.sub(r'([$€£])(\d+(?:\.\d+)?)\s*([KkMmBbTt])\b', expand_currency_magnitude, text)
    
    # Simple currency ($100, $5.50)
    text = re.sub(r'\$(\d+(?:\.\d+)?)', r'\1 dollars', text)
    text = re.sub(r'€(\d+(?:\.\d+)?)', r'\1 euros', text)
    text = re.sub(r'£(\d+(?:\.\d+)?)', r'\1 pounds', text)
    
    # Percentages
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
    
    # Common abbreviations
    abbreviations = {
        r'\bYoY\b': 'year over year',
        r'\bMoM\b': 'month over month',
        r'\bQoQ\b': 'quarter over quarter',
        r'\bROI\b': 'return on investment',
        r'\bGDP\b': 'GDP',
        r'\bCEO\b': 'CEO',
        r'\bAI\b': 'AI',
        r'\bUS\b': 'US',
        r'\bUK\b': 'UK',
        r'\bvs\.?\b': 'versus',
        r'\betc\.?\b': 'etcetera',
        r'\be\.g\.?\b': 'for example',
        r'\bi\.e\.?\b': 'that is',
        r'\bw/\b': 'with',
        r'\bw/o\b': 'without',
        r'\b&\b': 'and',
        r'\bhr\b': 'hour',
        r'\bhrs\b': 'hours',
        r'\bmin\b': 'minute',
        r'\bmins\b': 'minutes',
        r'\bsec\b': 'second',
        r'\bsecs\b': 'seconds',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Numbers with K/M/B suffix (not currency)
    def expand_number_magnitude(match):
        number = match.group(1)
        magnitude = match.group(2).upper()
        magnitudes = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}
        return f"{number} {magnitudes.get(magnitude, '')}"
    
    text = re.sub(r'(\d+(?:\.\d+)?)\s*([KkMmBbTt])\b(?![a-zA-Z])', expand_number_magnitude, text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================================
# VIRAL SCRIPT FRAMEWORK
# ============================================================================

VIRAL_SCRIPT_PROMPT = '''Create a viral {duration}-second short video script for: "{topic}"

CRITICAL RULES:
1. First segment MUST be a hook (pattern interrupt, curiosity gap, or bold claim)
2. Last segment MUST loop back naturally - when video replays, it flows into the hook again
3. Each segment: 2-3 sentences, conversational tone, 8-10 seconds of speech each
4. Include specific numbers/facts for credibility
5. Create curiosity gaps between segments
6. Start the video with the statement "Do you know... "

DURATION GUIDE (IMPORTANT - generate enough content!):
- 30 seconds = 5-6 segments, ~75 words total
- 45 seconds = 7-8 segments, ~110 words total  
- 60 seconds = 9-10 segments, ~150 words total
- Each segment should have 12-18 words (2-3 sentences)
- Average speaking rate: 2.5 words per second

TEXT RULES (for TTS narration):
- NO EMOJIS in "text" field - this is spoken by TTS!
- Write numbers fully: "100 billion dollars" NOT "$100B"
- Write "50 percent" NOT "50%"
- Avoid abbreviations: "year over year" NOT "YoY"
- Use natural spoken language
- Make each segment substantial (2-3 sentences, not just one short sentence)

IMAGE KEYWORD RULES:
- Keywords are used for Bing/Google IMAGE SEARCH
- If topic is about a PERSON/CELEBRITY, ALWAYS include their NAME in keywords
- Use COMMON, SEARCHABLE terms that return good stock photos
- Be GENERIC enough to find quality images, but relevant to the topic
- AVOID: Technical jargon, specific charts, abstract concepts
- GOOD for person topics: "cristiano ronaldo football", "elon musk tesla", "taylor swift concert"
- GOOD for general: "person counting money", "gold bars stack", "businessman thinking"
- BAD: "inflation hedge visualization", "silver price spike chart", "abstract success"
- Think: "What photo would a stock photo site have?"
- **IMPORTANT**: For person-based topics, include the person's name in EVERY image keyword

YOUTUBE SEO METADATA RULES:
- video_title: SEO-optimized with hashtags (e.g., "Ronaldo's Secret Salary 💰 #shorts #football #cr7 #viral")
- Must include #shorts at the end for YouTube Shorts algorithm
- Include relevant trending hashtags: #viral #trending #fyp #tiktok etc.
- description: 2-3 sentences (100-150 words) SEO-friendly, include main keywords
- hashtags_text: 450-500 characters total, comma-separated hashtags for copy-paste without '#' symbol
- Include mix of: niche tags, broad tags, trending tags
- uploaded: false (status tracking)
- visibility: "public" (or "private"/"unlisted")

SEGMENT STRUCTURE:
- Segment 1: HOOK (attention grabber)
- Segments 2-4: Value/Story (setup → conflict → payoff)
- Last Segment: LOOP CLOSER (teases back to start, creates rewatch urge)

EXAMPLE OUTPUT for "silver investing" (45-second script, 7 segments):
{{
  "title": "Silver Investment Strategy",
  "video_title": "Why Silver Is About To EXPLODE 🚀 #shorts #silver #investing #money #viral",
  "description": "Discover the shocking reason why silver prices are skyrocketing right now! China's massive silver purchases are driving prices up 50%% year over year. Smart investors are using this precious metal to hedge against inflation. Don't miss out on this incredible opportunity!",
  "segments": [
    {{"text": "Do you know there is a precious metal that has outperformed gold for two straight years? And most investors have no idea.", "image_keyword": "silver coins pile", "emotion": "curious"}},
    {{"text": "China just bought 30 percent more silver than last year. They are stockpiling it like there is no tomorrow.", "image_keyword": "china flag money", "emotion": "serious"}},
    {{"text": "Here is what makes this interesting. Silver is not just a precious metal, it is essential for electronics and solar panels.", "image_keyword": "silver industrial use", "emotion": "serious"}},
    {{"text": "This dual demand is driving prices up 50 percent year over year. And supply simply cannot keep up.", "image_keyword": "gold silver bars", "emotion": "excited"}},
    {{"text": "Smart investors have already figured this out. They are using silver to hedge against inflation and currency collapse.", "image_keyword": "investor analyzing stocks", "emotion": "inspiring"}},
    {{"text": "The question is, will you get in before the masses? Or will you watch from the sidelines?", "image_keyword": "person thinking decision", "emotion": "curious"}},
    {{"text": "But here is the part nobody wants you to know. Do you know what silver could be worth next year?", "image_keyword": "secret document mystery", "emotion": "curious"}}
  ],
  "hashtags_text": "#silver #silverinvesting #preciousmetals #investing #finance #money #wealth #silverstack #silverbullion #investingtips #financialfreedom #stockmarket #crypto #gold #goldinvesting #inflation #economy #trading #investor #passiveincome #makemoney #financetips #wealthbuilding #invest #investments #stocks #forex #bitcoin #entrepreneur #business #shorts #viral #trending #fyp #foryou #tiktok #reels #youtube",
  "uploaded": false,
  "visibility": "public"
}}

EXAMPLE OUTPUT for "Cristiano Ronaldo" (45-second script, 7 segments with person's name):
{{
  "title": "Ronaldo's Insane Net Worth",
  "video_title": "Ronaldo's Salary Can Buy A COUNTRY?! 🤯 #shorts #ronaldo #football #cr7 #viral",
  "description": "Ever wondered how much Cristiano Ronaldo really makes? His salary from football alone is 100 million euros annually, but his brand deals add another 500 million euros yearly! Discover the shocking truth about CR7's massive wealth and income sources. You won't believe these numbers!",
  "segments": [
    {{"text": "Do you know Cristiano Ronaldo makes more money than some entire countries? His income is absolutely insane.", "image_keyword": "cristiano ronaldo football", "emotion": "curious"}},
    {{"text": "Ronaldo earns 100 million euros annually from football alone. But that is just his base salary.", "image_keyword": "ronaldo playing soccer", "emotion": "serious"}},
    {{"text": "What most people do not realize is his sponsorship deals. Nike pays him 20 million euros per year just to wear their shoes.", "image_keyword": "ronaldo nike sponsorship", "emotion": "excited"}},
    {{"text": "His brand deals add another 500 million euros yearly. That includes Nike, Clear, and his own CR7 brand.", "image_keyword": "ronaldo sponsor deal", "emotion": "excited"}},
    {{"text": "He even makes 3 million dollars per Instagram post. That is more than most people earn in a lifetime.", "image_keyword": "ronaldo instagram social media", "emotion": "serious"}},
    {{"text": "With all his investments, Ronaldo's net worth is over 1 billion dollars. He is one of the richest athletes ever.", "image_keyword": "cristiano ronaldo luxury wealthy", "emotion": "inspiring"}},
    {{"text": "So the real question is, do you know how much Ronaldo will be worth by 2030?", "image_keyword": "cristiano ronaldo portrait", "emotion": "curious"}}
  ],
  "hashtags_text": "#cristianoronaldo #ronaldo #cr7 #football #soccer #footballplayer #soccerplayer #messi #neymar #sports #fifa #championsleague #premierleague #realmadrid #manchester #juventus #portugal #goat #footballskills #goals #salary #networth #wealth #rich #money #millionaire #billionaire #celebrity #famous #athlete #sports #inspiration #motivation #success #shorts #viral #trending #fyp #foryou #tiktok #reels #youtube #explore",
  "uploaded": false,
  "visibility": "public"
}}

OUTPUT FORMAT (strict JSON only, no other text):
{{
  "title": "Internal title for organization",
  "video_title": "SEO YouTube title with emojis and hashtags #shorts",
  "description": "2-3 sentence SEO description with keywords (100-150 words)",
  "segments": [
    {{"text": "Spoken narration (natural language, no abbreviations)", "image_keyword": "simple searchable image terms", "emotion": "serious/excited/neutral/inspiring/curious"}}
  ],
  "hashtags_text": "450-500 chars of space-separated hashtags for copy-paste",
  "uploaded": false,
  "visibility": "public"
}}

Generate script now (follow DURATION GUIDE above for segment count, {duration} seconds total):'''


IMAGE_KEYWORDS_PROMPT = '''Generate a simple Bing IMAGE SEARCH keyword (2-5 words).

Narration: "{text}"
Topic: "{topic}"

RULES:
- If the topic is about a SPECIFIC PERSON/CELEBRITY, INCLUDE THEIR NAME in the keyword
- Use common words that return good stock photos
- Avoid technical/abstract terms
- Think "what photo exists on stock sites?"
- GOOD for person: "cristiano ronaldo playing", "elon musk speaking", "taylor swift stage"
- GOOD for general: "person working laptop", "money cash pile", "city skyline night"
- BAD: "productivity optimization", "financial growth chart", "abstract success"

Return ONLY the search keyword:'''


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = Config.OLLAMA_URL, model: str = Config.OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
    
    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 2000
                    }
                },
                timeout=300  # 5 minutes timeout for complex prompts
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            print(f"Ollama error: {e}")
        return ""


# ============================================================================
# SCRIPT GENERATOR
# ============================================================================

class ScriptGenerator:
    """Generate viral video scripts using Ollama"""
    
    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama
    
    def generate_script(self, topic: str, duration: int = 45, max_retries: int = 3) -> Dict:
        """Generate a viral script for the given topic with retry logic"""
        prompt = VIRAL_SCRIPT_PROMPT.format(topic=topic, duration=duration)
        
        print("\n🎬 Generating viral script...")
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt + 1}/{max_retries}...")
            
            response = self.ollama.generate(prompt, temperature=0.7 + (attempt * 0.1))
            
            if not response:
                continue
            
            # Extract JSON from response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group()
                    # Fix common JSON issues
                    json_str = json_str.replace('\n', ' ')
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    script_data = json.loads(json_str)
                    
                    # Validate that it has required fields
                    if "segments" in script_data and len(script_data["segments"]) > 0:
                        # Ensure all required fields exist
                        if 'video_title' not in script_data:
                            script_data['video_title'] = f"{script_data.get('title', topic)} #shorts #viral"
                        if 'description' not in script_data:
                            script_data['description'] = f"Discover the truth about {topic}. Watch now!"
                        if 'hashtags_text' not in script_data:
                            script_data['hashtags_text'] = self._generate_default_hashtags(topic)
                        
                        print(f"✅ Generated {len(script_data['segments'])} segments")
                        return script_data
                        
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    print(f"⚠️  JSON parse error after {max_retries} attempts: {e}")
                    print(f"   Raw response preview: {response[:200]}...")
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️  Unexpected error: {e}")
        
        # All retries failed - use fallback
        print("⚠️  Using fallback script after failed retries...")
        return self._create_fallback_script(topic, response if response else "")
    
    def _generate_default_hashtags(self, topic: str) -> str:
        """Generate default hashtags text (450-500 chars)"""
        base_tags = topic.lower().replace(" ", "").split()[:3]
        niche_tags = " ".join([f"#{tag}" for tag in base_tags])
        
        trending = "#shorts #viral #trending #fyp #foryou #tiktok #reels #youtube #explore #viralvideos #viralshorts"
        return f"{niche_tags} {trending} #education #learning #facts #knowledge #motivation #inspiration #success #tips #advice #life"
    
    def _create_fallback_script(self, topic: str, raw_response: str) -> Dict:
        """Create fallback script structure with loop-back ending and full metadata"""
        print("   📝 Creating fallback script with generic template...")
        
        # DON'T use raw response as it contains JSON syntax
        # Create clean, professional script
        
        # Detect if topic is a person's name (has multiple words with capital letters)
        words = topic.split()
        is_person = len(words) >= 2 and any(w[0].isupper() for w in words if w)
        
        if is_person:
            # Person-focused script (7 segments for ~45 seconds)
            name = words[0]  # First name
            segments = [
                {"text": f"Do you know the incredible story behind {topic}? Most people have never heard this.", "image_keyword": f"{topic} portrait", "emotion": "curious"},
                {"text": f"{name} is making waves in their field right now. And the world is finally paying attention.", "image_keyword": f"{topic} working", "emotion": "serious"},
                {"text": f"What most people do not realize is how much sacrifice it took. Years of hard work behind the scenes.", "image_keyword": f"{topic} training", "emotion": "serious"},
                {"text": f"Their achievements are truly remarkable. Breaking records that no one thought possible.", "image_keyword": f"{topic} success", "emotion": "excited"},
                {"text": f"This is what sets {name} apart from everyone else. Pure dedication and relentless focus.", "image_keyword": f"{topic} achievement", "emotion": "inspiring"},
                {"text": f"The question is, can anyone else replicate what they have done? Probably not.", "image_keyword": f"{topic} famous", "emotion": "curious"},
                {"text": f"But do you really know what makes {topic} so special? The answer might surprise you.", "image_keyword": f"{topic} inspiring", "emotion": "curious"}
            ]
            title = f"{topic} - Untold Story"
            video_title = f"{topic}'s Incredible Journey 🌟 #shorts #inspiration #success #viral"
            description = f"Discover the amazing story of {topic}. Learn about their journey, achievements, and what makes them truly exceptional. You won't believe what they've accomplished!"
        else:
            # Topic-focused script (7 segments for ~45 seconds)
            segments = [
                {"text": f"Do you know the real secret about {topic}? Most people get this completely wrong.", "image_keyword": f"{topic} explained", "emotion": "curious"},
                {"text": f"The truth is not what you expect. What you have been told is only half the story.", "image_keyword": f"{topic} facts", "emotion": "serious"},
                {"text": f"Here is what actually makes a difference. This one insight changes everything.", "image_keyword": f"{topic} tips", "emotion": "excited"},
                {"text": f"Smart people already figured this out. They use this knowledge to get ahead of everyone else.", "image_keyword": f"{topic} success", "emotion": "inspiring"},
                {"text": f"The problem is most people never take action. They hear this and then do nothing.", "image_keyword": f"{topic} problem", "emotion": "serious"},
                {"text": f"Will you be different? Or will you stay stuck like everyone else?", "image_keyword": f"{topic} decision", "emotion": "curious"},
                {"text": f"But do you really understand {topic}? The answer determines your future.", "image_keyword": f"{topic} knowledge", "emotion": "curious"}
            ]
            title = f"{topic.title()} - The Truth"
            video_title = f"{topic.title()} Explained 🔥 #shorts #viral #trending"
            description = f"Discover the truth about {topic}! This video reveals what most people don't know. Watch till the end to learn the secrets that can change everything. Don't miss out!"
        
        # Generate comprehensive hashtags
        hashtags_text = self._generate_default_hashtags(topic)
        
        return {
            "title": title,
            "video_title": video_title,
            "description": description,
            "segments": segments,
            "hashtags_text": hashtags_text,
            "uploaded": False,
            "visibility": "public"
        }
    
    def enhance_image_keywords(self, script: Dict, topic: str) -> Dict:
        """Enhance image search keywords for each segment"""
        print("\n🖼️  Optimizing image search keywords...")
        
        for segment in script.get("segments", []):
            prompt = IMAGE_KEYWORDS_PROMPT.format(
                text=segment["text"],
                topic=topic
            )
            keyword = self.ollama.generate(prompt, temperature=0.5).strip()
            if keyword and len(keyword) < 50:
                segment["image_keyword"] = keyword
        
        return script


# ============================================================================
# IMAGE CRAWLER
# ============================================================================

class ImageDownloader:
    """Download images from Bing with smart selection for vertical videos"""
    
    def __init__(self, output_dir: Path = Config.IMAGES_DIR):
        self.output_dir = output_dir
        self.smart_selector = SmartImageSelector()
    
    def download_images(self, keywords: List[str], images_per_keyword: int = 5) -> Dict[str, List[Path]]:
        """
        Download images for each keyword.
        Downloads more images than needed to allow smart selection.
        """
        print("\n📥 Downloading images from Bing (with smart selection)...")
        
        result = {}
        
        for i, keyword in enumerate(keywords):
            keyword_dir = self.output_dir / f"segment_{i:02d}"
            keyword_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear previous images
            for f in keyword_dir.glob("*"):
                f.unlink()
            
            # Add "vertical" or "portrait" to search for better results
            enhanced_keyword = f"{keyword} vertical portrait"
            print(f"  Searching: '{keyword}'...")
            
            try:
                crawler = BingImageCrawler(
                    storage={'root_dir': str(keyword_dir)},
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=2
                )
                crawler.crawl(
                    keyword=enhanced_keyword,
                    max_num=images_per_keyword,
                    min_size=(640, 800)  # Prefer taller images
                )
                
                # Also try original keyword if we got few results
                downloaded = list(keyword_dir.glob("*.*"))
                if len(downloaded) < 2:
                    crawler.crawl(
                        keyword=keyword,
                        max_num=images_per_keyword,
                        min_size=(640, 480)
                    )
                
                # Collect downloaded images
                images = list(keyword_dir.glob("*.*"))
                result[keyword] = images
                print(f"    ✓ Downloaded {len(result[keyword])} images")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                result[keyword] = []
        
        return result
    
    def select_best_images(self, images_dict: Dict[str, List[Path]], segments: List[Dict]) -> List[Path]:
        """Select the best image for each segment using smart scoring"""
        print("\n🎯 Selecting best images for 9:16 format...")
        selected = []
        
        for i, segment in enumerate(segments):
            keyword = segment.get("image_keyword", "")
            images = images_dict.get(keyword, [])
            
            print(f"    Segment {i+1}: '{keyword[:30]}...'")
            
            if images:
                # Use smart selection based on orientation scoring
                best_image = SmartImageSelector.select_best_image(images)
                selected.append(best_image)
            else:
                # Use a fallback or placeholder
                print(f"      → No images found!")
                selected.append(None)
        
        return selected


# ============================================================================
# TEXT-TO-SPEECH (Kokoro TTS)
# ============================================================================

class VoiceGenerator:
    """Generate voice narration using Kokoro TTS"""
    
    def __init__(self):
        self._bin = self._find_bin()
    
    def _find_bin(self) -> str:
        """Find kokoro-tts binary"""
        paths = [
            "/opt/anaconda3/envs/youtube-env/bin/kokoro-tts",
            str(Config.BASE_DIR / "venv" / "bin" / "kokoro-tts"),
            "/opt/homebrew/bin/kokoro-tts",
            "/usr/local/bin/kokoro-tts",
        ]
        for p in paths:
            if Path(p).exists():
                return p
        result = subprocess.run(["which", "kokoro-tts"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else paths[0]
    
    def initialize(self) -> bool:
        """Check if Kokoro TTS is available"""
        if not Config.KOKORO_MODEL.exists():
            print(f"  ✗ Kokoro model not found: {Config.KOKORO_MODEL}")
            return False
        if not Config.KOKORO_VOICES.exists():
            print(f"  ✗ Kokoro voices not found: {Config.KOKORO_VOICES}")
            return False
        print("  ✓ Kokoro TTS ready!")
        return True
    
    def generate_audio(self, text: str, output_path: Path, 
                       voice: str = None, emotion: str = "neutral") -> Optional[Path]:
        """Generate audio for the given text using Kokoro TTS"""
        if not Config.KOKORO_MODEL.exists():
            return self._fallback_audio(text, output_path)
        
        voice = voice or Config.TTS_VOICE
        tmp = output_path.parent / f"tmp_{random.randint(1000,9999)}.txt"
        tmp.write_text(text)
        
        try:
            cmd = [
                self._bin, str(tmp), str(output_path),
                "--speed", str(Config.TTS_SPEED),
                "--voice", voice,
                "--model", str(Config.KOKORO_MODEL),
                "--voices", str(Config.KOKORO_VOICES)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                print(f"    ✗ TTS failed: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                return self._fallback_audio(text, output_path)
                
        except Exception as e:
            print(f"    ✗ TTS error: {e}")
            return self._fallback_audio(text, output_path)
        finally:
            tmp.unlink(missing_ok=True)
    
    def _fallback_audio(self, text: str, output_path: Path) -> Optional[Path]:
        """Create placeholder audio if TTS fails"""
        # Create silent audio as fallback
        return self._create_silent_audio(output_path, duration=len(text) * 0.08)
    
    def _create_silent_audio(self, output_path: Path, duration: float = 3.0) -> Path:
        """Create silent audio file using ffmpeg"""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(duration), str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        return output_path
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 3.0  # Default fallback
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for natural TTS pauses.
        Handles multiple sentence-ending punctuation marks.
        IMPORTANT: Don't split on ellipsis (...) as it's a pause, not sentence end.
        """
        # First, protect ellipsis from being split
        text = text.replace('...', '<<<ELLIPSIS>>>')
        text = text.replace('…', '<<<ELLIPSIS>>>')
        
        # Pattern to split on sentence-ending punctuation followed by space
        # Only split on . ! ? when followed by a space and capital letter or end
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Restore ellipsis
        sentences = [s.replace('<<<ELLIPSIS>>>', '...').strip() for s in sentences]
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        # If no split occurred (single sentence), return original
        if not sentences:
            return [text.replace('<<<ELLIPSIS>>>', '...')]
        
        return sentences
    
    def _concatenate_audio_files(self, audio_files: List[Path], output_path: Path) -> Optional[Path]:
        """
        Concatenate multiple audio files into one using ffmpeg.
        """
        if not audio_files:
            return None
        
        if len(audio_files) == 1:
            # Just copy the single file
            import shutil
            shutil.copy(audio_files[0], output_path)
            return output_path
        
        # Create a temporary file list for ffmpeg
        list_file = output_path.parent / f"concat_list_{random.randint(1000, 9999)}.txt"
        try:
            with open(list_file, 'w') as f:
                for audio in audio_files:
                    # Escape single quotes in path
                    safe_path = str(audio).replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")
            
            # Run ffmpeg concat
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                print(f"    ⚠ Concat failed, using first sentence only")
                import shutil
                shutil.copy(audio_files[0], output_path)
                return output_path
                
        except Exception as e:
            print(f"    ⚠ Concat error: {e}")
            return None
        finally:
            list_file.unlink(missing_ok=True)
            # Clean up temp sentence files
            for audio in audio_files:
                if "_sent_" in str(audio):
                    audio.unlink(missing_ok=True)
    
    def generate_audio_sentence_by_sentence(self, text: str, output_path: Path,
                                            voice: str = None, emotion: str = "neutral") -> Optional[Path]:
        """
        Generate audio sentence by sentence for natural TTS pauses.
        This prevents the TTS from running sentences together unnaturally.
        """
        sentences = self._split_into_sentences(text)
        
        # If only one sentence, use regular generation
        if len(sentences) <= 1:
            return self.generate_audio(text, output_path, voice, emotion)
        
        print(f"      📜 Processing {len(sentences)} sentences...")
        
        # Generate audio for each sentence
        sentence_audios = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            sent_path = output_path.parent / f"temp_sent_{output_path.stem}_{i:02d}.wav"
            result = self.generate_audio(sentence, sent_path, voice, emotion)
            
            if result and result.exists():
                sentence_audios.append(result)
        
        if not sentence_audios:
            return self.generate_audio(text, output_path, voice, emotion)
        
        # Concatenate all sentence audios
        return self._concatenate_audio_files(sentence_audios, output_path)


# ============================================================================
# KEN BURNS EFFECT (Zoom & Pan Animation)
# ============================================================================

class KenBurnsEffect:
    """Create Ken Burns zoom and pan effects on images with smart cropping"""
    
    def __init__(self, width: int = Config.VIDEO_WIDTH, height: int = Config.VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
    
    def prepare_image(self, image_path: Path) -> np.ndarray:
        """
        Prepare image for video using smart cropping.
        Intelligently crops to fit 9:16 with minimal content loss.
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        img_w, img_h = img.size
        img_aspect = img_w / img_h
        target_aspect = self.aspect_ratio
        
        # Smart crop based on orientation
        if img_aspect > target_aspect:
            # Image is wider than target - crop sides (center crop)
            new_width = int(img_h * target_aspect * 1.4)  # Extra for Ken Burns
            new_height = int(img_h * 1.4)
            
            # Ensure we don't exceed original dimensions
            if new_width > img_w:
                scale = img_w / new_width
                new_width = img_w
                new_height = int(new_height * scale)
            
            # Center crop horizontally
            left = max(0, (img_w - new_width) // 2)
            top = 0
            crop_w = min(new_width, img_w - left)
            crop_h = min(new_height, img_h)
            
        else:
            # Image is taller or square - use more of vertical space
            new_width = int(self.width * 1.4)
            new_height = int(new_width / img_aspect)
            
            # Favor upper portion for faces (30% from top instead of center)
            left = 0
            top = max(0, int((img_h - new_height) * 0.3))
            crop_w = min(new_width, img_w)
            crop_h = min(new_height, img_h - top)
        
        # Crop if needed
        if left > 0 or top > 0 or crop_w < img_w or crop_h < img_h:
            img = img.crop((left, top, left + crop_w, top + crop_h))
        
        # Resize with extra padding for Ken Burns movement
        ken_burns_size = (int(self.width * 1.4), int(self.height * 1.4))
        img = img.resize(ken_burns_size, Image.Resampling.LANCZOS)
        
        return np.array(img)
    
    def create_clip(self, image_path: Path, duration: float, 
                    effect_type: str = "random") -> VideoClip:
        """Create an animated clip with Ken Burns effect"""
        img_array = self.prepare_image(image_path)
        img_h, img_w = img_array.shape[:2]
        
        # Choose effect parameters
        if effect_type == "random":
            effect_type = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"])
        
        # Define animation parameters
        effects = {
            "zoom_in": {"start_zoom": 1.0, "end_zoom": 1.25, "pan": (0, 0)},
            "zoom_out": {"start_zoom": 1.25, "end_zoom": 1.0, "pan": (0, 0)},
            "pan_left": {"start_zoom": 1.15, "end_zoom": 1.15, "pan": (0.15, 0)},
            "pan_right": {"start_zoom": 1.15, "end_zoom": 1.15, "pan": (-0.15, 0)},
            "pan_up": {"start_zoom": 1.15, "end_zoom": 1.15, "pan": (0, 0.15)},
            "pan_down": {"start_zoom": 1.15, "end_zoom": 1.15, "pan": (0, -0.15)},
        }
        
        params = effects[effect_type]
        
        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            progress = self._ease_in_out(progress)
            
            # Interpolate zoom
            zoom = params["start_zoom"] + (params["end_zoom"] - params["start_zoom"]) * progress
            
            # Calculate crop dimensions
            crop_w = int(self.width / zoom)
            crop_h = int(self.height / zoom)
            
            # Calculate pan offset
            pan_x = params["pan"][0] * progress * (img_w - crop_w)
            pan_y = params["pan"][1] * progress * (img_h - crop_h)
            
            # Center position with pan offset
            center_x = (img_w - crop_w) // 2 + int(pan_x)
            center_y = (img_h - crop_h) // 2 + int(pan_y)
            
            # Clamp to valid range
            center_x = max(0, min(center_x, img_w - crop_w))
            center_y = max(0, min(center_y, img_h - crop_h))
            
            # Crop and resize
            cropped = img_array[center_y:center_y + crop_h, center_x:center_x + crop_w]
            
            # Resize to output dimensions
            pil_img = Image.fromarray(cropped)
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            
            return np.array(pil_img)
        
        # Use VideoClip for custom frame generation in MoviePy 2.x
        clip = VideoClip(make_frame, duration=duration)
        return clip.with_fps(Config.FPS)
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function"""
        return t * t * (3 - 2 * t)


# ============================================================================
# VIDEO COMPOSER
# ============================================================================

class VideoComposer:
    """Compose final video from clips and audio with captions, music, and click sounds"""
    
    def __init__(self):
        self.ken_burns = KenBurnsEffect()
        self.transcriber = WordTranscriber(model_size="base")  # Use base for speed
        self.captions = AnimatedCaptions()
    
    def create_video(self, segments: List[Dict], images: List[Path], 
                     audio_files: List[Path], output_path: Path,
                     add_captions: bool = True) -> Path:
        """Create the final video with word-by-word captions, music, and transition sounds"""
        print("\n🎥 Composing video with animated captions...")
        
        clips = []
        durations = []
        all_word_data = []  # Store transcription data for each segment
        
        for i, (segment, image, audio) in enumerate(zip(segments, images, audio_files)):
            print(f"  Processing segment {i+1}/{len(segments)}...")
            
            if image is None or audio is None:
                continue
            
            # Get audio duration
            audio_clip = AudioFileClip(str(audio))
            duration = audio_clip.duration
            durations.append(duration)
            
            # Transcribe audio for word timestamps
            if add_captions:
                print(f"    📝 Transcribing for captions...")
                # Get original narration text for alignment
                original_text = segment.get('text', '')
                words = self.transcriber.transcribe(audio, original_text=original_text)
                all_word_data.append(words)
                print(f"    ✓ Aligned {len(words)} words with narration")
            else:
                all_word_data.append([])
            
            # Create Ken Burns effect on image
            effect_types = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
            effect = effect_types[i % len(effect_types)]
            
            video_clip = self.ken_burns.create_clip(image, duration, effect)
            
            # Add word-by-word captions overlay
            if add_captions and all_word_data[-1]:
                caption_clip = self._create_caption_clip(all_word_data[-1], duration)
                video_clip = CompositeVideoClip([video_clip, caption_clip])
            
            video_clip = video_clip.with_audio(audio_clip)
            clips.append(video_clip)
        
        if not clips:
            raise ValueError("No clips to compose!")
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Calculate transition times for click sounds
        transition_times = []
        current_time = 0
        for dur in durations[:-1]:  # No click after last segment
            current_time += dur
            transition_times.append(current_time)
        
        # Export initial video
        temp_path = output_path.parent / "temp_video.mp4"
        print(f"\n📤 Exporting video with captions...")
        try:
            final_video.write_videofile(
                str(temp_path),
                fps=Config.FPS,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                threads=4,
                logger="bar"  # Show progress bar
            )
        except Exception as e:
            print(f"  ⚠️ MoviePy export failed: {e}")
            print("  → Trying ffmpeg fallback...")
            # Fallback: export without audio first, then add audio
            temp_video_only = output_path.parent / "temp_video_only.mp4"
            final_video.write_videofile(
                str(temp_video_only),
                fps=Config.FPS,
                codec='libx264',
                audio=False,
                preset='fast',
                logger="bar"
            )
            # Skip audio processing if video-only works
            if temp_video_only.exists():
                shutil.move(str(temp_video_only), str(temp_path))
        
        # Cleanup moviepy clips
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        try:
            final_video.close()
        except:
            pass
        
        # Check if temp file was created
        if not temp_path.exists():
            raise RuntimeError("Video export failed - temp file not created")
        
        current_video = str(temp_path)
        
        # Add background music
        music_path = Config.get_music()
        if music_path:
            with_music = str(output_path.parent / "with_music.mp4")
            print(f"  🎵 Adding background music: {Path(music_path).name}")
            if self._add_music(current_video, music_path, with_music):
                current_video = with_music
        
        # Add click sounds at transitions
        click_path = Config.get_click()
        if click_path and transition_times:
            with_clicks = str(output_path.parent / "with_clicks.mp4")
            print(f"  🔊 Adding {len(transition_times)} transition click sounds")
            if self._add_clicks(current_video, click_path, transition_times, with_clicks):
                current_video = with_clicks
        
        # Add fade in/out for smooth looping
        with_fade = str(output_path.parent / "with_fade.mp4")
        print("  🔄 Adding fade in/out for loopability...")
        if self._add_fade(current_video, with_fade):
            current_video = with_fade
        
        # Move to final output
        if Path(current_video).exists():
            shutil.copy(current_video, output_path)
            print(f"  ✅ Video saved to: {output_path}")
        else:
            raise RuntimeError(f"Final video not found: {current_video}")
        
        # Cleanup temp files
        temp_files = [
            temp_path, 
            output_path.parent / "with_music.mp4", 
            output_path.parent / "with_clicks.mp4", 
            output_path.parent / "with_fade.mp4",
            output_path.parent / "temp_video_only.mp4"
        ]
        for temp_file in temp_files:
            try:
                if temp_file.exists() and temp_file != output_path:
                    temp_file.unlink()
            except:
                pass
        
        return output_path
    
    def _create_caption_clip(self, words: List[Dict], duration: float) -> VideoClip:
        """Create a transparent video clip with animated word-by-word captions"""
        
        def make_caption_frame(t):
            # Create caption frame at time t
            frame = self.captions.create_caption_frame(words, t)
            # Convert RGBA to RGB with alpha compositing on transparent background
            return frame[:, :, :3]  # Return RGB only, alpha handled by compositing
        
        # Create VideoClip with caption frames
        clip = VideoClip(make_caption_frame, duration=duration)
        clip = clip.with_fps(Config.FPS)
        
        # Make it a mask clip (transparent overlay)
        # The captions class returns RGBA, we need to handle transparency
        def make_mask_frame(t):
            frame = self.captions.create_caption_frame(words, t)
            # Return alpha channel normalized to 0-1
            return frame[:, :, 3] / 255.0
        
        mask_clip = VideoClip(make_mask_frame, duration=duration, is_mask=True)
        mask_clip = mask_clip.with_fps(Config.FPS)
        
        clip = clip.with_mask(mask_clip)
        
        return clip
    
    def _add_music(self, video: str, music: str, output: str) -> bool:
        """Add looping background music"""
        if not Path(music).exists():
            shutil.copy(video, output)
            return True
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video, "-i", music,
            "-filter_complex",
            f"[1:a]volume={Config.MUSIC_VOLUME},aloop=loop=-1:size=2e9[m];"
            f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and Path(output).exists()
    
    def _add_clicks(self, video: str, click: str, times: List[float], output: str) -> bool:
        """Add click sounds at slide transitions"""
        if not times or not Path(click).exists():
            shutil.copy(video, output)
            return True
        
        inputs = ["-i", video]
        filters = []
        
        for i, t in enumerate(times):
            inputs.extend(["-i", click])
            filters.append(f"[{i+1}:a]volume={Config.CLICK_VOLUME},adelay={int(t*1000)}|{int(t*1000)}[c{i}]")
        
        mix = "[0:a]" + "".join(f"[c{i}]" for i in range(len(times)))
        filters.append(f"{mix}amix=inputs={len(times)+1}:duration=first[out]")
        
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", ";".join(filters),
            "-map", "0:v", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and Path(output).exists()
    
    def _add_fade(self, video: str, output: str) -> bool:
        """Add fade in/out for loopability"""
        cmd = [
            "ffmpeg", "-y", "-i", video,
            "-vf", "fade=t=in:st=0:d=0.3,fade=t=out:st=-0.3:d=0.3",
            "-af", "afade=t=in:st=0:d=0.2,afade=t=out:st=-0.2:d=0.2",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and Path(output).exists()


# ============================================================================
# YOUTUBE UPLOADER (Multi-credential with failover)
# ============================================================================

class YouTubeUploader:
    """
    YouTube video uploader with multi-credential failover support.
    Reads OAuth credentials from google-console folder and retries with
    different credentials if one fails.
    """
    
    # Trending hashtags for Shorts (SEO optimized)
    TRENDING_HASHTAGS = [
        "#shorts", "#viral", "#trending", "#fyp", "#foryou", "#tiktok", 
        "#reels", "#youtube", "#explore", "#viralshorts", "#youtubeshorts"
    ]
    
    # Niche-based SEO tags database
    SEO_TAGS_DATABASE = {
        "finance": ["money", "investing", "stocks", "crypto", "wealth", "finance", "millionaire", "passive income", "trading", "entrepreneur"],
        "motivation": ["motivation", "success", "mindset", "inspiration", "goals", "hustle", "grind", "self improvement", "growth"],
        "education": ["education", "learning", "facts", "knowledge", "science", "history", "tips", "howto", "tutorial"],
        "entertainment": ["funny", "comedy", "memes", "entertainment", "lol", "hilarious", "jokes"],
        "sports": ["sports", "football", "soccer", "basketball", "fitness", "gym", "workout", "athlete"],
        "tech": ["tech", "technology", "gadgets", "ai", "coding", "software", "innovation", "future"],
        "lifestyle": ["lifestyle", "life", "daily", "routine", "vlog", "day in my life"],
        "celebrity": ["celebrity", "famous", "star", "hollywood", "gossip", "news", "breaking"],
    }
    
    def __init__(self):
        self.credentials_files = self._get_credential_files()
        self.current_cred_index = 0
        self.youtube_service = None
        
    def _get_credential_files(self) -> List[Path]:
        """Get all OAuth credential JSON files from google-console folder"""
        cred_dir = Config.GOOGLE_CONSOLE_DIR
        if not cred_dir.exists():
            print(f"⚠️  Google console directory not found: {cred_dir}")
            return []
        
        # Find all client secret JSON files
        cred_files = list(cred_dir.glob("client_secret*.json"))
        if not cred_files:
            # Also try other naming patterns
            cred_files = list(cred_dir.glob("*.json"))
            # Filter out token files
            cred_files = [f for f in cred_files if "token" not in f.name.lower()]
        
        print(f"📁 Found {len(cred_files)} credential file(s) in google-console/")
        return cred_files
    
    def _get_authenticated_service(self, cred_file: Path):
        """Authenticate and return YouTube service using specific credential file"""
        token_file = Config.GOOGLE_CONSOLE_DIR / f"token_{cred_file.stem}.pickle"
        creds = None
        
        # Load existing token if available
        if token_file.exists():
            try:
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                print(f"   ⚠️ Error loading token: {e}")
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"   ⚠️ Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(cred_file), 
                        Config.YOUTUBE_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    print(f"   ❌ Authentication failed: {e}")
                    return None
            
            # Save the credentials for next time
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build YouTube service
        try:
            return build(
                Config.YOUTUBE_API_SERVICE_NAME,
                Config.YOUTUBE_API_VERSION,
                credentials=creds
            )
        except Exception as e:
            print(f"   ❌ Failed to build YouTube service: {e}")
            return None
    
    def _optimize_title(self, title: str, topic: str) -> str:
        """
        Optimize title for YouTube Shorts SEO.
        - Max 100 characters
        - Include #shorts and viral hashtags
        - Make it catchy and clickable
        """
        # Essential hashtags for Shorts algorithm
        required_tags = " #shorts #viral"
        
        # Clean the title
        title = title.strip()
        
        # If title doesn't have hashtags, add them
        if "#shorts" not in title.lower():
            # Calculate available space
            available = Config.MAX_TITLE_LENGTH - len(required_tags)
            
            if len(title) > available:
                # Truncate title smartly (at word boundary)
                title = title[:available-3].rsplit(' ', 1)[0] + "..."
            
            title = title + required_tags
        else:
            # Title already has hashtags, just ensure length
            if len(title) > Config.MAX_TITLE_LENGTH:
                # Keep hashtags, truncate description part
                parts = title.rsplit('#', 1)
                if len(parts) == 2:
                    desc_part = parts[0][:Config.MAX_TITLE_LENGTH - len('#' + parts[1]) - 3] + "..."
                    title = desc_part + '#' + parts[1]
                else:
                    title = title[:Config.MAX_TITLE_LENGTH-3] + "..."
        
        return title[:Config.MAX_TITLE_LENGTH]
    
    def _generate_seo_tags(self, topic: str, existing_tags: str = "") -> List[str]:
        """
        Generate SEO-optimized tags for YouTube (max 500 characters total).
        Mix of: topic-specific, niche, trending, and broad reach tags.
        """
        tags = []
        
        # Parse existing tags if provided
        if existing_tags:
            # Remove # symbols and split
            existing = [t.strip().replace('#', '') for t in existing_tags.replace(',', ' ').split()]
            tags.extend([t for t in existing if t and len(t) > 1])
        
        # Add topic-based tags
        topic_words = topic.lower().replace('-', ' ').replace('_', ' ').split()
        tags.extend([w for w in topic_words if len(w) > 2])
        
        # Add combined topic tag
        topic_tag = ''.join(w.capitalize() for w in topic_words[:3])
        if topic_tag and len(topic_tag) > 2:
            tags.append(topic_tag)
        
        # Detect niche and add relevant tags
        topic_lower = topic.lower()
        for niche, niche_tags in self.SEO_TAGS_DATABASE.items():
            # Check if topic relates to this niche
            if any(keyword in topic_lower for keyword in niche_tags[:3]):
                tags.extend(niche_tags)
        
        # Add trending Shorts tags (without #)
        trending = [t.replace('#', '') for t in self.TRENDING_HASHTAGS]
        tags.extend(trending)
        
        # Add broad reach tags
        broad_tags = [
            "viral", "trending", "fyp", "foryou", "explore",
            "viralvideo", "trendingshorts", "youtubeshorts",
            "shortsvideo", "shortsyoutube", "viralshort"
        ]
        tags.extend(broad_tags)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            tag_lower = tag.lower().strip()
            if tag_lower and tag_lower not in seen and len(tag_lower) > 1:
                seen.add(tag_lower)
                unique_tags.append(tag)
        
        # Trim to fit 500 character limit
        final_tags = []
        total_length = 0
        for tag in unique_tags:
            # Account for comma separator
            tag_length = len(tag) + 1
            if total_length + tag_length <= Config.MAX_TAGS_LENGTH:
                final_tags.append(tag)
                total_length += tag_length
            else:
                break
        
        return final_tags
    
    def _generate_description(self, script: Dict, topic: str) -> str:
        """Generate SEO-optimized description for YouTube"""
        description = script.get('description', '')
        
        if not description:
            # Generate default description
            description = f"Discover the truth about {topic}! Watch till the end for a surprising twist. "
            description += "Don't forget to like, subscribe, and turn on notifications! "
        
        # Add hashtags at the end (YouTube shows first 3)
        hashtags = script.get('hashtags_text', '')
        if hashtags:
            # Take first 10 hashtags for description
            tags_list = hashtags.split()[:10]
            hashtag_line = ' '.join([f"#{t.replace('#', '')}" for t in tags_list])
            description += f"\n\n{hashtag_line}"
        
        # Add call to action
        description += "\n\n📌 Follow for more amazing content!"
        description += "\n💬 Comment what you want to see next!"
        description += "\n❤️ Like if you found this valuable!"
        
        return description[:5000]  # YouTube description limit
    
    def upload_video(self, video_path: str, script: Dict, topic: str) -> Tuple[bool, str]:
        """
        Upload video to YouTube with multi-credential failover.
        Returns (success, video_id or error_message)
        """
        if not self.credentials_files:
            return False, "No credential files found in google-console/"
        
        # Get optimized metadata
        title = self._optimize_title(
            script.get('video_title', script.get('title', f'{topic} #shorts')),
            topic
        )
        
        description = self._generate_description(script, topic)
        tags = self._generate_seo_tags(topic, script.get('hashtags_text', ''))
        visibility = script.get('visibility', 'public')
        
        print(f"\n📤 Uploading to YouTube...")
        print(f"   📺 Title: {title}")
        print(f"   🏷️  Tags: {len(tags)} tags ({sum(len(t) for t in tags)} chars)")
        print(f"   🔒 Visibility: {visibility}")
        
        # Try each credential file until success
        last_error = ""
        for i, cred_file in enumerate(self.credentials_files):
            print(f"\n   🔑 Trying credential {i+1}/{len(self.credentials_files)}: {cred_file.name[:30]}...")
            
            try:
                service = self._get_authenticated_service(cred_file)
                if not service:
                    last_error = "Authentication failed"
                    continue
                
                # Prepare upload body
                body = {
                    'snippet': {
                        'title': title,
                        'description': description,
                        'tags': tags,
                        'categoryId': '22'  # People & Blogs (good for Shorts)
                    },
                    'status': {
                        'privacyStatus': visibility,
                        'selfDeclaredMadeForKids': False,
                        'madeForKids': False
                    }
                }
                
                # Create media upload
                media = MediaFileUpload(
                    video_path,
                    mimetype='video/mp4',
                    resumable=True,
                    chunksize=1024*1024  # 1MB chunks
                )
                
                # Execute upload with retry logic
                request = service.videos().insert(
                    part=','.join(body.keys()),
                    body=body,
                    media_body=media
                )
                
                response = None
                retry_count = 0
                max_retries = 3
                
                while response is None:
                    try:
                        print(f"   ⬆️  Uploading...", end='', flush=True)
                        status, response = request.next_chunk()
                        if status:
                            progress = int(status.progress() * 100)
                            print(f"\r   ⬆️  Uploading... {progress}%", end='', flush=True)
                    except HttpError as e:
                        if e.resp.status in [500, 502, 503, 504] and retry_count < max_retries:
                            retry_count += 1
                            print(f"\n   ⚠️ Server error, retry {retry_count}/{max_retries}...")
                            time.sleep(5 * retry_count)
                        else:
                            raise
                
                print(f"\r   ✅ Upload complete!                    ")
                video_id = response.get('id', '')
                video_url = f"https://youtube.com/shorts/{video_id}"
                print(f"   🔗 URL: {video_url}")
                
                return True, video_id
                
            except HttpError as e:
                last_error = f"HTTP Error: {e.resp.status} - {e.content.decode()[:100]}"
                print(f"   ❌ {last_error}")
                
                # Check if quota exceeded - try next credential
                if e.resp.status == 403 and 'quotaExceeded' in str(e.content):
                    print(f"   ⚠️ Quota exceeded, trying next credential...")
                    continue
                    
            except Exception as e:
                last_error = str(e)
                print(f"   ❌ Error: {last_error}")
        
        return False, f"All credentials failed. Last error: {last_error}"
    
    def upload_pending_videos(self) -> Dict[str, any]:
        """
        Upload all pending videos from scripts folder.
        Returns summary of upload results.
        """
        scripts_dir = Config.SCRIPTS_DIR
        if not scripts_dir.exists():
            return {"error": "Scripts directory not found"}
        
        # Find all script files with uploaded: false
        pending = []
        for script_file in scripts_dir.glob("script_*.json"):
            try:
                with open(script_file, 'r') as f:
                    script = json.load(f)
                if not script.get('uploaded', False) and script.get('video_file_path'):
                    if Path(script['video_file_path']).exists():
                        pending.append((script_file, script))
            except Exception as e:
                print(f"⚠️ Error reading {script_file}: {e}")
        
        if not pending:
            print("✅ No pending videos to upload!")
            return {"uploaded": 0, "failed": 0, "pending": 0}
        
        print(f"\n📋 Found {len(pending)} pending video(s) to upload")
        
        results = {"uploaded": 0, "failed": 0, "errors": []}
        
        for script_file, script in pending:
            topic = script.get('title', 'Unknown')
            video_path = script['video_file_path']
            
            print(f"\n{'='*50}")
            print(f"📹 Uploading: {topic}")
            
            success, result = self.upload_video(video_path, script, topic)
            
            if success:
                # Update script file with upload status
                script['uploaded'] = True
                script['youtube_video_id'] = result
                script['youtube_url'] = f"https://youtube.com/shorts/{result}"
                script['upload_date'] = datetime.now().isoformat()
                
                with open(script_file, 'w') as f:
                    json.dump(script, f, indent=2, ensure_ascii=False)
                
                results["uploaded"] += 1
                print(f"✅ Successfully uploaded: {script['youtube_url']}")
            else:
                results["failed"] += 1
                results["errors"].append({"topic": topic, "error": result})
                print(f"❌ Failed to upload: {result}")
            
            # Wait between uploads to avoid rate limiting
            if len(pending) > 1:
                print("⏳ Waiting 5 seconds before next upload...")
                time.sleep(5)
        
        print(f"\n{'='*50}")
        print(f"📊 Upload Summary:")
        print(f"   ✅ Uploaded: {results['uploaded']}")
        print(f"   ❌ Failed: {results['failed']}")
        
        return results


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class ReelDesigner:
    """Main application for designing viral reels"""
    
    def __init__(self):
        Config.setup_dirs()
        self.ollama = OllamaClient()
        self.script_gen = ScriptGenerator(self.ollama)
        self.image_dl = ImageDownloader()
        self.voice_gen = VoiceGenerator()
        self.composer = VideoComposer()
        self.uploader = YouTubeUploader()
    
    def check_requirements(self) -> Dict[str, bool]:
        """Check if all requirements are met"""
        requirements = {}
        
        # Check Ollama
        print("\n🔍 Checking requirements...")
        requirements["ollama"] = self.ollama.check_connection()
        print(f"  Ollama: {'✓' if requirements['ollama'] else '✗'}")
        
        if requirements["ollama"]:
            models = self.ollama.list_models()
            print(f"    Available models: {', '.join(models[:5])}")
        
        # Check Kokoro TTS
        requirements["kokoro_tts"] = self.voice_gen.initialize()
        
        # Check background audio
        music = Config.get_music()
        click = Config.get_click()
        print(f"  Background Music: {'✓' if music else '✗'}")
        print(f"  Transition Clicks: {'✓' if click else '✗'}")
        
        return requirements
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*60)
        print("🎬 INTERACTIVE REEL SHORTS DESIGNER")
        print("="*60)
        print("  Features: Kokoro TTS, Background Music, Transition Clicks")
        print("="*60)
        
        # Check requirements
        reqs = self.check_requirements()
        
        if not reqs["ollama"]:
            print("\n⚠️  Ollama is not running!")
            print("   Please start Ollama: ollama serve")
            print(f"   And pull a model: ollama pull {Config.OLLAMA_MODEL}")
            return
        
        # Get topic from user
        print("\n" + "-"*40)
        topic = input("📝 Enter your reel topic: ").strip()
        
        if not topic:
            topic = "productivity tips for busy professionals"
            print(f"   Using default topic: {topic}")
        
        # Get duration preference
        duration_input = input("⏱️  Target duration (30/45/60 seconds) [45]: ").strip()
        duration = int(duration_input) if duration_input.isdigit() else 45
        
        # Get voice preference
        print(f"🎤 Using Kokoro TTS voice: {Config.TTS_VOICE}")
        voice = input("   Enter different voice name or press Enter to use default: ").strip()
        if not voice:
            voice = Config.TTS_VOICE
        
        # Generate the reel
        self.create_reel(topic, duration, voice)
    
    def create_reel(self, topic: str, duration: int = 45, 
                    voice: str = None, add_captions: bool = True,
                    whisper_model: str = "base", auto_upload: bool = False) -> Path:
        """Create a complete reel from topic to video with word-by-word captions"""
        voice = voice or Config.TTS_VOICE
        
        # Generate unique timestamp for this video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_')[:30]
        
        print("\n" + "="*60)
        print(f"🎬 Creating Reel: {topic}")
        print(f"   📝 Word-by-word captions: {'Enabled' if add_captions else 'Disabled'}")
        print(f"   🆔 Video ID: {timestamp}_{safe_topic}")
        print("="*60)
        
        # Update whisper model if captions are enabled
        if add_captions:
            self.composer.transcriber = WordTranscriber(model_size=whisper_model)
        
        # Step 1: Generate script
        script = self.script_gen.generate_script(topic, duration)
        script = self.script_gen.enhance_image_keywords(script, topic)
        
        # Add metadata fields
        script['uploaded'] = False
        script['visibility'] = 'public'
        script['video_file_path'] = None
        script['created_at'] = datetime.now().isoformat()
        script['topic'] = topic
        script['duration_target'] = duration
        
        # Save script to separate timestamped file in scripts folder
        script_filename = f"script_{timestamp}_{safe_topic}.json"
        script_path = Config.SCRIPTS_DIR / script_filename
        
        # Also save to main script.json for backward compatibility
        main_script_path = Config.OUTPUT_DIR / "script.json"
        
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2, ensure_ascii=False)
        with open(main_script_path, 'w') as f:
            json.dump(script, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Script saved to: {script_path}")
        
        # Display script with YouTube metadata
        print("\n" + "-"*40)
        print("📜 GENERATED SCRIPT (Loop-optimized):")
        print("-"*40)
        print(f"Title: {script.get('title', 'Untitled')}")
        
        # Show optimized YouTube title
        video_title = script.get('video_title', script.get('title', 'Untitled'))
        if len(video_title) > Config.MAX_TITLE_LENGTH:
            video_title = video_title[:Config.MAX_TITLE_LENGTH-3] + "..."
        print(f"YouTube Title ({len(video_title)} chars): {video_title}")
        
        if script.get('description'):
            print(f"\nDescription:\n{script.get('description', '')[:150]}...")
        print("\nSegments (Hook → Content → Loop Closer):")
        segments = script.get("segments", [])
        for i, seg in enumerate(segments, 1):
            marker = "🎣 HOOK" if i == 1 else ("🔄 LOOP" if i == len(segments) else f"   {i}")
            print(f"  {marker}: {seg.get('text', '')[:70]}...")
            print(f"        📷 {seg.get('image_keyword', '')} | 😊 {seg.get('emotion', 'neutral')}")
        
        # Display hashtags with character count
        hashtags_text = script.get('hashtags_text', '')
        if hashtags_text:
            print(f"\n#️⃣  Hashtags ({len(hashtags_text)}/{Config.MAX_TAGS_LENGTH} chars):")
            print(f"   {hashtags_text[:200]}...")
        else:
            print(f"\n#️⃣  Hashtags: {', '.join(script.get('hashtags', []))}")
        
        # Step 2: Download images (with smart selection for 9:16)
        keywords = [s.get("image_keyword", topic) for s in script.get("segments", [])]
        images_dict = self.image_dl.download_images(keywords, images_per_keyword=5)
        selected_images = self.image_dl.select_best_images(images_dict, script.get("segments", []))
        
        # Step 3: Generate audio for each segment using Kokoro TTS
        # Uses sentence-by-sentence generation for natural pauses
        print(f"\n🎤 Generating voice narration with Kokoro TTS (voice: {voice})...")
        print("   Using sentence-by-sentence processing for natural speech flow...")
        audio_files = []
        
        for i, segment in enumerate(script.get("segments", [])):
            text = segment.get("text", "")
            emotion = segment.get("emotion", "neutral")
            audio_path = Config.AUDIO_DIR / f"segment_{i:02d}.wav"
            
            # Expand abbreviations for proper TTS pronunciation
            tts_text = expand_abbreviations_for_tts(text)
            
            print(f"  Segment {i+1}: {text[:50]}...")
            if tts_text != text:
                print(f"    📝 TTS: {tts_text[:50]}...")
            
            # Use sentence-by-sentence TTS for natural pauses
            result = self.voice_gen.generate_audio_sentence_by_sentence(tts_text, audio_path, voice, emotion)
            
            if result:
                audio_files.append(result)
                audio_duration = self.voice_gen.get_audio_duration(result)
                print(f"    ✓ Duration: {audio_duration:.2f}s")
            else:
                audio_files.append(None)
        
        # Step 4: Compose video with captions, music and click sounds
        output_video = Config.VIDEO_DIR / f"reel_{timestamp}_{safe_topic}.mp4"
        
        # Filter out None values
        valid_data = [
            (seg, img, aud) 
            for seg, img, aud in zip(script.get("segments", []), selected_images, audio_files)
            if img is not None and aud is not None
        ]
        
        if valid_data:
            segments, images, audios = zip(*valid_data)
            final_video = self.composer.create_video(
                list(segments), list(images), list(audios), output_video,
                add_captions=add_captions
            )
            
            # Update script with video file path
            script['video_file_path'] = str(final_video.absolute())
            script['video_filename'] = final_video.name
            
            # Save updated script to both locations
            with open(script_path, 'w') as f:
                json.dump(script, f, indent=2, ensure_ascii=False)
            with open(main_script_path, 'w') as f:
                json.dump(script, f, indent=2, ensure_ascii=False)
            
            print("\n" + "="*60)
            print("✅ REEL CREATED SUCCESSFULLY!")
            print("="*60)
            print(f"📁 Video: {final_video}")
            print(f"📝 Script: {script_path}")
            print(f"📺 YouTube Title: {script.get('video_title', '')[:Config.MAX_TITLE_LENGTH]}")
            print(f"📊 Status: {'Uploaded' if script.get('uploaded') else 'Pending Upload'}")
            print(f"🔒 Visibility: {script.get('visibility', 'public')}")
            if add_captions:
                print(f"✨ Features: Word-by-word captions, Ken Burns effects, Background music")
            
            # Auto-upload if requested
            if auto_upload:
                print("\n" + "-"*40)
                print("🚀 Auto-uploading to YouTube...")
                success, result = self.uploader.upload_video(
                    str(final_video), script, topic
                )
                if success:
                    script['uploaded'] = True
                    script['youtube_video_id'] = result
                    script['youtube_url'] = f"https://youtube.com/shorts/{result}"
                    script['upload_date'] = datetime.now().isoformat()
                    
                    with open(script_path, 'w') as f:
                        json.dump(script, f, indent=2, ensure_ascii=False)
                    with open(main_script_path, 'w') as f:
                        json.dump(script, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ Uploaded: {script['youtube_url']}")
                else:
                    print(f"❌ Upload failed: {result}")
            
            return final_video
        else:
            print("\n❌ Failed to create reel: No valid segments")
            return None
    
    def upload_pending(self) -> Dict:
        """Upload all pending videos"""
        return self.uploader.upload_pending_videos()
    
    def list_pending(self) -> List[Dict]:
        """List all pending (not uploaded) videos"""
        scripts_dir = Config.SCRIPTS_DIR
        if not scripts_dir.exists():
            return []
        
        pending = []
        for script_file in sorted(scripts_dir.glob("script_*.json")):
            try:
                with open(script_file, 'r') as f:
                    script = json.load(f)
                if not script.get('uploaded', False):
                    pending.append({
                        'file': script_file.name,
                        'title': script.get('title', 'Unknown'),
                        'video_title': script.get('video_title', ''),
                        'video_exists': Path(script.get('video_file_path', '')).exists() if script.get('video_file_path') else False,
                        'created': script.get('created_at', 'Unknown')
                    })
            except Exception as e:
                print(f"⚠️ Error reading {script_file}: {e}")
        
        return pending


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive Reel Shorts Designer with YouTube Upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode
  python main.py -t "Bitcoin investing"    # Create reel for topic
  python main.py -t "Elon Musk" --upload   # Create and auto-upload
  python main.py --upload-pending          # Upload all pending videos
  python main.py --list-pending            # List pending uploads
        """
    )
    parser.add_argument("--topic", "-t", type=str, help="Topic for the reel")
    parser.add_argument("--duration", "-d", type=int, default=45, help="Target duration in seconds")
    parser.add_argument("--voice", "-v", type=str, default=Config.TTS_VOICE, help="Kokoro TTS voice name (e.g., af_bella, af_nicole)")
    parser.add_argument("--model", "-m", type=str, default=Config.OLLAMA_MODEL, help="Ollama model to use")
    parser.add_argument("--speed", "-s", type=float, default=Config.TTS_SPEED, help="TTS speech speed (default: 1.0)")
    parser.add_argument("--no-captions", action="store_true", help="Disable word-by-word captions")
    parser.add_argument("--whisper-model", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size for transcription (default: base)")
    
    # YouTube upload options
    parser.add_argument("--upload", "-u", action="store_true", default=True, help="Auto-upload to YouTube after creation (default: True)")
    parser.add_argument("--no-upload", action="store_true", help="Disable auto-upload to YouTube")
    parser.add_argument("--upload-pending", action="store_true", help="Upload all pending videos")
    parser.add_argument("--list-pending", action="store_true", help="List all pending (not uploaded) videos")
    
    args = parser.parse_args()
    
    # Handle --no-upload flag
    if args.no_upload:
        args.upload = False
    
    # Update config if specified
    Config.OLLAMA_MODEL = args.model
    Config.TTS_SPEED = args.speed
    
    designer = ReelDesigner()
    
    # Handle upload-only commands
    if args.upload_pending:
        print("\n" + "="*60)
        print("📤 UPLOADING PENDING VIDEOS")
        print("="*60)
        designer.upload_pending()
        return
    
    if args.list_pending:
        print("\n" + "="*60)
        print("📋 PENDING VIDEOS")
        print("="*60)
        pending = designer.list_pending()
        if not pending:
            print("✅ No pending videos found!")
        else:
            print(f"Found {len(pending)} pending video(s):\n")
            for i, video in enumerate(pending, 1):
                status = "✅ Ready" if video['video_exists'] else "❌ Missing"
                print(f"  {i}. {video['title']}")
                print(f"     📝 {video['file']}")
                print(f"     📺 {video['video_title'][:50]}...")
                print(f"     📁 {status}")
                print(f"     📅 {video['created']}")
                print()
        return
    
    if args.topic:
        # Non-interactive mode
        designer.create_reel(
            args.topic, 
            args.duration, 
            args.voice, 
            add_captions=not args.no_captions,
            whisper_model=args.whisper_model,
            auto_upload=args.upload
        )
    else:
        # Interactive mode
        designer.interactive_mode()


if __name__ == "__main__":
    main()
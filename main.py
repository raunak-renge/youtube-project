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
    ]
    for pkg in packages:
        try:
            pkg_import = pkg.replace("-", "_").lower()
            if pkg == "openai-whisper":
                pkg_import = "whisper"
            __import__(pkg_import)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"])

install_packages()

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import whisper
# MoviePy 2.x API
from moviepy.video.VideoClip import ImageClip, VideoClip, TextClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
from icrawler.builtin import BingImageCrawler


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
    
    @classmethod
    def setup_dirs(cls):
        """Create output directories"""
        for d in [cls.OUTPUT_DIR, cls.IMAGES_DIR, cls.AUDIO_DIR, cls.VIDEO_DIR]:
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
        Uses the timestamps from Whisper but corrects the words from the original text.
        This ensures captions match exactly what was intended to be spoken.
        """
        # Clean and tokenize original text
        original_words = self._tokenize_text(original_text)
        
        if not original_words or not whisper_words:
            return whisper_words
        
        # Use dynamic time warping-like alignment
        aligned = self._dtw_align(whisper_words, original_words)
        
        return aligned
    
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
        """Calculate similarity between two words (0 to 1)"""
        w1 = self._normalize_word(word1)
        w2 = self._normalize_word(word2)
        
        if w1 == w2:
            return 1.0
        
        # Check if one contains the other
        if w1 in w2 or w2 in w1:
            return 0.8
        
        # Simple character-level similarity (Levenshtein-like)
        if not w1 or not w2:
            return 0.0
        
        # Count matching characters
        matches = sum(1 for c1, c2 in zip(w1, w2) if c1 == c2)
        max_len = max(len(w1), len(w2))
        return matches / max_len if max_len > 0 else 0.0
    
    def _dtw_align(self, whisper_words: List[Dict], original_words: List[str]) -> List[Dict]:
        """
        Align Whisper words with original words using a greedy alignment approach.
        Returns aligned words with corrected text but Whisper timestamps.
        """
        aligned = []
        whisper_idx = 0
        original_idx = 0
        
        while original_idx < len(original_words) and whisper_idx < len(whisper_words):
            orig_word = original_words[original_idx]
            whisper_word = whisper_words[whisper_idx]
            
            # Calculate similarity
            similarity = self._word_similarity(orig_word, whisper_word['word'])
            
            if similarity >= 0.5:
                # Good match - use original word with Whisper timing
                aligned.append({
                    'word': orig_word,
                    'start': whisper_word['start'],
                    'end': whisper_word['end']
                })
                whisper_idx += 1
                original_idx += 1
            else:
                # Check if Whisper skipped a word or added extra word
                # Look ahead in both sequences
                next_whisper_sim = 0
                next_orig_sim = 0
                
                if whisper_idx + 1 < len(whisper_words):
                    next_whisper_sim = self._word_similarity(
                        orig_word, whisper_words[whisper_idx + 1]['word']
                    )
                
                if original_idx + 1 < len(original_words):
                    next_orig_sim = self._word_similarity(
                        original_words[original_idx + 1], whisper_word['word']
                    )
                
                if next_whisper_sim > similarity and next_whisper_sim >= 0.5:
                    # Whisper has an extra word, skip it
                    whisper_idx += 1
                elif next_orig_sim > similarity and next_orig_sim >= 0.5:
                    # Original has a word Whisper missed, interpolate timing
                    if aligned:
                        prev_end = aligned[-1]['end']
                        next_start = whisper_word['start']
                        duration = (next_start - prev_end) / 2
                        aligned.append({
                            'word': orig_word,
                            'start': prev_end,
                            'end': prev_end + duration
                        })
                    original_idx += 1
                else:
                    # Force alignment with current words
                    aligned.append({
                        'word': orig_word,
                        'start': whisper_word['start'],
                        'end': whisper_word['end']
                    })
                    whisper_idx += 1
                    original_idx += 1
        
        # Handle remaining original words (interpolate timing)
        if original_idx < len(original_words) and aligned:
            last_end = aligned[-1]['end']
            remaining = len(original_words) - original_idx
            # Estimate duration per word
            avg_duration = sum(w['end'] - w['start'] for w in aligned) / len(aligned) if aligned else 0.3
            
            for i, word in enumerate(original_words[original_idx:]):
                start = last_end + i * avg_duration
                aligned.append({
                    'word': word,
                    'start': start,
                    'end': start + avg_duration
                })
        
        return aligned


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
# ANIMATED TEXT OVERLAY (Single Word Captions - Like TikTok/Reels)
# ============================================================================

class AnimatedCaptions:
    """
    Create beautiful single-word animated captions.
    Shows ONE word at a time, centered and highlighted in YELLOW.
    Styled like viral TikTok/Instagram Reels captions.
    """
    
    # Text styling - Large, bold, yellow text
    FONT_SIZE = 96  # Bigger for single word
    FONT_COLOR = (255, 255, 0)  # Bright Yellow (like the reference image)
    STROKE_COLOR = (0, 0, 0)  # Black outline for readability
    STROKE_WIDTH = 6  # Thicker stroke for visibility
    BG_COLOR = (0, 0, 0, 160)  # Semi-transparent black background
    
    # Position (center of screen, lower third area)
    POSITION_Y_RATIO = 0.70  # 70% from top (lower third)
    
    # Animation settings
    PADDING = 30
    
    def __init__(self, width: int = Config.VIDEO_WIDTH, height: int = Config.VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.font = self._load_font()
        self.small_font = self._load_font(size=48)  # For secondary text if needed
    
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
    
    def create_caption_frame(self, words: List[Dict], current_time: float) -> np.ndarray:
        """
        Create a transparent frame with a SINGLE highlighted word.
        Only shows the current word being spoken, centered in yellow.
        """
        # Create transparent image
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Find the current word being spoken
        current_word = self._get_current_word(words, current_time)
        
        if not current_word:
            return np.array(img)
        
        # Draw the single word centered
        self._draw_single_word(draw, current_word['word'])
        
        return np.array(img)
    
    def _get_current_word(self, words: List[Dict], current_time: float) -> Optional[Dict]:
        """Get the single word being spoken at current time"""
        for word in words:
            # Check if current time falls within this word's timing
            if word['start'] <= current_time <= word['end']:
                return word
        
        # If between words, show nothing (or could show the last word)
        return None
    
    def _draw_single_word(self, draw: ImageDraw.Draw, word: str):
        """Draw a single word centered on screen with yellow highlight"""
        # Clean the word (remove extra punctuation for display, keep it readable)
        display_word = word.upper()  # Uppercase for impact (like the reference)
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), display_word, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center position
        x = (self.width - text_width) // 2
        y = int(self.height * self.POSITION_Y_RATIO) - text_height // 2
        
        # Draw background pill/rectangle
        bg_padding_x = 40
        bg_padding_y = 20
        bg_rect = [
            x - bg_padding_x,
            y - bg_padding_y,
            x + text_width + bg_padding_x,
            y + text_height + bg_padding_y
        ]
        # Rounded rectangle background
        draw.rounded_rectangle(bg_rect, radius=15, fill=self.BG_COLOR)
        
        # Draw text with stroke (outline) for readability
        # Draw stroke first (black outline)
        for dx in range(-self.STROKE_WIDTH, self.STROKE_WIDTH + 1):
            for dy in range(-self.STROKE_WIDTH, self.STROKE_WIDTH + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), display_word, 
                             font=self.font, fill=self.STROKE_COLOR)
        
        # Draw main text in YELLOW
        draw.text((x, y), display_word, font=self.font, fill=self.FONT_COLOR)


# ============================================================================
# VIRAL SCRIPT FRAMEWORK (Based on uploaded document)
# ============================================================================

VIRAL_SCRIPT_PROMPT = '''You are a viral content creator specializing in short-form video scripts for Instagram Reels and YouTube Shorts.

Create a highly engaging script for the topic: "{topic}"

FOLLOW THIS EXACT FRAMEWORK:

1. **HOOK (0-2 seconds)** - Create a pattern interrupt using ONE of these:
   - Contradiction: "I was doing this wrong for 3 years…"
   - Big claim: "This is the fastest way to ___ (without ___)."
   - Curiosity gap: "Nobody talks about the *real* reason ___."
   - Specific number: "3 mistakes killing your ____."
   - Relatable pain: "If you feel stuck... watch this."
   - Cold open action: Start mid-moment

2. **STAKES (1 line)** - Why they should care:
   - "If you don't fix this, you'll keep ___."
   - "This saved me ___ (time/money/embarrassment)."

3. **3-BEAT ARC**:
   - Setup: what's happening (fast)
   - Conflict: the problem/tension
   - Payoff: the reveal + result + lesson

4. **OPEN LOOPS** - Keep attention:
   - "In a second, I'll show you..."
   - "The weirdest part was..."
   - Reveal every 3-7 seconds

5. **PROOF** - Build credibility:
   - Include a specific number, result, or before-after

6. **REPLAY LINE** - Shareable quote:
   - Something memorable they'll want to rewatch

7. **CTA** - One action:
   - "Save this for later"
   - "Comment if you want part 2"
   - "Follow for more"

OUTPUT FORMAT (JSON):
{{
    "title": "Video title for the reel",
    "hook": "The opening hook line (2-3 seconds spoken)",
    "segments": [
        {{
            "text": "Narration text for this segment",
            "image_keyword": "Specific image search term for this part",
            "duration_hint": "short/medium/long",
            "emotion": "neutral/excited/serious/inspiring"
        }}
    ],
    "cta": "Call to action at the end",
    "hashtags": ["relevant", "hashtags", "list"],
    "total_estimated_duration": "30-60 seconds"
}}

RULES:
- Keep each segment to 1-2 short sentences max
- Make it conversational, not formal
- Use contrast and specificity
- Create curiosity gaps
- Make every line earn the next second
- Target {duration} second video

Generate the script now:'''


IMAGE_KEYWORDS_PROMPT = '''Based on this video script segment, generate the BEST image search keyword for Bing Images.

Segment text: "{text}"
Topic: "{topic}"

Requirements:
- Keyword should be 2-4 words
- Should find visually striking, high-quality images
- Should relate directly to the content
- Avoid generic terms, be specific

Return ONLY the search keyword, nothing else.'''


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
                timeout=120
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
    
    def generate_script(self, topic: str, duration: int = 45) -> Dict:
        """Generate a viral script for the given topic"""
        prompt = VIRAL_SCRIPT_PROMPT.format(topic=topic, duration=duration)
        
        print("\n🎬 Generating viral script...")
        response = self.ollama.generate(prompt, temperature=0.8)
        
        if not response:
            print("⚠️  Ollama returned empty response, using fallback...")
            return self._create_fallback_script(topic, "")
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                script_data = json.loads(json_match.group())
                # Validate that it has required fields
                if "segments" in script_data and len(script_data["segments"]) > 0:
                    print(f"✅ Generated {len(script_data['segments'])} segments")
                    return script_data
                else:
                    print("⚠️  JSON missing segments, using fallback...")
                    return self._create_fallback_script(topic, response)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}, using fallback...")
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}, using fallback...")
        
        # Fallback: create a simple structure
        return self._create_fallback_script(topic, response)
    
    def _create_fallback_script(self, topic: str, raw_response: str) -> Dict:
        """Create fallback script structure from raw response"""
        # Try to extract meaningful lines from response
        lines = [l.strip() for l in raw_response.split('\n') if l.strip() and len(l.strip()) > 10]
        
        # If we have very few lines, create a default viral script structure
        if len(lines) < 3:
            lines = [
                f"Watch how I mastered {topic}!",
                f"Most people don't know this about {topic}.",
                f"Here's the secret to {topic}.",
                f"This changed everything for me.",
                f"You can do this too!",
                f"Save this for later!"
            ]
        
        segments = []
        for i, line in enumerate(lines[:6]):  # Max 6 segments
            segments.append({
                "text": line[:200],  # Limit length
                "image_keyword": f"{topic} {['action', 'tutorial', 'result', 'before', 'after', 'success'][i % 6]}",
                "duration_hint": "medium",
                "emotion": "neutral"
            })
        
        return {
            "title": f"🔥 Master {topic} - Viral Tips",
            "hook": segments[0]["text"] if segments else f"You won't believe this about {topic}...",
            "segments": segments,
            "cta": "Follow for more amazing content!",
            "hashtags": [topic.replace(" ", ""), "viral", "shorts", "reels"],
            "total_estimated_duration": "30-45 seconds"
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
            "/Users/sutirthachakraborty/Library/Python/3.11/bin/kokoro-tts",
            "/opt/homebrew/bin/kokoro-tts",
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
            print("   And pull a model: ollama pull qwen3:8b")
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
                    whisper_model: str = "base") -> Path:
        """Create a complete reel from topic to video with word-by-word captions"""
        voice = voice or Config.TTS_VOICE
        
        print("\n" + "="*60)
        print(f"🎬 Creating Reel: {topic}")
        print(f"   📝 Word-by-word captions: {'Enabled' if add_captions else 'Disabled'}")
        print("="*60)
        
        # Update whisper model if captions are enabled
        if add_captions:
            self.composer.transcriber = WordTranscriber(model_size=whisper_model)
        
        # Step 1: Generate script
        script = self.script_gen.generate_script(topic, duration)
        script = self.script_gen.enhance_image_keywords(script, topic)
        
        # Save script
        script_path = Config.OUTPUT_DIR / "script.json"
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2)
        print(f"\n📝 Script saved to: {script_path}")
        
        # Display script
        print("\n" + "-"*40)
        print("📜 GENERATED SCRIPT:")
        print("-"*40)
        print(f"Title: {script.get('title', 'Untitled')}")
        print(f"Hook: {script.get('hook', '')}")
        print("\nSegments:")
        for i, seg in enumerate(script.get("segments", []), 1):
            print(f"  {i}. {seg.get('text', '')[:80]}...")
            print(f"     Image: {seg.get('image_keyword', '')}")
        print(f"\nCTA: {script.get('cta', '')}")
        print(f"Hashtags: {', '.join(script.get('hashtags', []))}")
        
        # Step 2: Download images (with smart selection for 9:16)
        keywords = [s.get("image_keyword", topic) for s in script.get("segments", [])]
        images_dict = self.image_dl.download_images(keywords, images_per_keyword=5)  # More images for better selection
        selected_images = self.image_dl.select_best_images(images_dict, script.get("segments", []))
        
        # Step 3: Generate audio for each segment using Kokoro TTS
        print(f"\n🎤 Generating voice narration with Kokoro TTS (voice: {voice})...")
        audio_files = []
        
        for i, segment in enumerate(script.get("segments", [])):
            text = segment.get("text", "")
            emotion = segment.get("emotion", "neutral")
            audio_path = Config.AUDIO_DIR / f"segment_{i:02d}.wav"
            
            print(f"  Segment {i+1}: {text[:50]}...")
            result = self.voice_gen.generate_audio(text, audio_path, voice, emotion)
            
            if result:
                audio_files.append(result)
                audio_duration = self.voice_gen.get_audio_duration(result)
                print(f"    ✓ Duration: {audio_duration:.2f}s")
            else:
                audio_files.append(None)
        
        # Step 4: Compose video with captions, music and click sounds
        output_video = Config.VIDEO_DIR / f"reel_{topic.replace(' ', '_')[:30]}.mp4"
        
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
            
            print("\n" + "="*60)
            print("✅ REEL CREATED SUCCESSFULLY!")
            print("="*60)
            print(f"📁 Output: {final_video}")
            print(f"📝 Script: {script_path}")
            if add_captions:
                print(f"📝 Features: Word-by-word captions, Ken Burns effects, Background music")
            
            return final_video
        else:
            print("\n❌ Failed to create reel: No valid segments")
            return None


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Reel Shorts Designer with Kokoro TTS and Word-by-Word Captions")
    parser.add_argument("--topic", "-t", type=str, help="Topic for the reel")
    parser.add_argument("--duration", "-d", type=int, default=45, help="Target duration in seconds")
    parser.add_argument("--voice", "-v", type=str, default=Config.TTS_VOICE, help="Kokoro TTS voice name (e.g., af_bella, af_nicole)")
    parser.add_argument("--model", "-m", type=str, default=Config.OLLAMA_MODEL, help="Ollama model to use")
    parser.add_argument("--speed", "-s", type=float, default=Config.TTS_SPEED, help="TTS speech speed (default: 1.0)")
    parser.add_argument("--no-captions", action="store_true", help="Disable word-by-word captions")
    parser.add_argument("--whisper-model", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size for transcription (default: base)")
    
    args = parser.parse_args()
    
    # Update config if specified
    Config.OLLAMA_MODEL = args.model
    Config.TTS_SPEED = args.speed
    
    designer = ReelDesigner()
    
    if args.topic:
        # Non-interactive mode
        designer.create_reel(args.topic, args.duration, args.voice, 
                            add_captions=not args.no_captions,
                            whisper_model=args.whisper_model)
    else:
        # Interactive mode
        designer.interactive_mode()


if __name__ == "__main__":
    main()
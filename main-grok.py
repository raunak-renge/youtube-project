#!/usr/bin/env python3
"""
Interactive Reel Shorts Designer
================================
Creates viral short-form video content using:
- Gemini AI for script generation (single API call)
- Bing Image Crawler for visuals
- Kokoro TTS for voice synthesis
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
        "google-generativeai",
        "mediapipe",
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
            elif pkg == "google-generativeai":
                pkg_import = "google.genai"
            elif pkg == "mediapipe":
                pkg_import = "mediapipe"
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
import mediapipe as mp
# MoviePy 2.x API
from moviepy.video.VideoClip import ImageClip, VideoClip, TextClip, ColorClip
from moviepy.video.io.VideoFileClip import VideoFileClip
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

# Gemini AI for script generation
from google import genai


# ============================================================================
# FACE DETECTION (MediaPipe) - Ensures faces aren't cut off in crops
# ============================================================================

class FaceDetector:
    """
    Detect faces in images using MediaPipe to ensure faces aren't cut off
    when cropping images for 9:16 video format.
    """
    
    _instance = None
    _detector = None
    
    def __new__(cls):
        """Singleton pattern - reuse detector instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if FaceDetector._detector is None:
            try:
                FaceDetector._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 1 = full range model (better for various distances)
                    min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"    ⚠️ MediaPipe init failed: {e}")
                FaceDetector._detector = None
    
    def detect_faces(self, image_path: Path) -> List[Dict]:
        """
        Detect faces in an image and return their bounding boxes.
        
        Returns list of dicts with:
        - 'bbox': (x, y, width, height) as fractions of image size (0-1)
        - 'center': (cx, cy) center point as fractions
        - 'confidence': detection confidence score
        """
        if FaceDetector._detector is None:
            return []
        
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Detect faces
            results = FaceDetector._detector.process(img_array)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Get bounding box (normalized 0-1)
                    x = max(0, bbox.xmin)
                    y = max(0, bbox.ymin)
                    w = min(bbox.width, 1 - x)
                    h = min(bbox.height, 1 - y)
                    
                    # Calculate center
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    faces.append({
                        'bbox': (x, y, w, h),
                        'center': (cx, cy),
                        'confidence': detection.score[0] if detection.score else 0.5
                    })
            
            return faces
            
        except Exception as e:
            # Silently fail - just return no faces
            return []
    
    def get_face_safe_crop_region(self, image_path: Path, target_aspect: float) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate a crop region that keeps all detected faces visible.
        
        Args:
            image_path: Path to the image
            target_aspect: Target aspect ratio (width/height, e.g., 0.5625 for 9:16)
        
        Returns:
            Tuple (left, top, right, bottom) in pixels, or None if no special handling needed
        """
        faces = self.detect_faces(image_path)
        
        if not faces:
            return None  # No faces, use default cropping
        
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size
            img_aspect = img_w / img_h
            
            # Find bounding box containing all faces with padding
            min_x = min(f['bbox'][0] for f in faces)
            min_y = min(f['bbox'][1] for f in faces)
            max_x = max(f['bbox'][0] + f['bbox'][2] for f in faces)
            max_y = max(f['bbox'][1] + f['bbox'][3] for f in faces)
            
            # Add padding around faces (20% of face size)
            padding_x = (max_x - min_x) * 0.3
            padding_y = (max_y - min_y) * 0.3
            
            face_left = max(0, min_x - padding_x)
            face_top = max(0, min_y - padding_y)
            face_right = min(1, max_x + padding_x)
            face_bottom = min(1, max_y + padding_y)
            
            # Convert to pixels
            face_left_px = int(face_left * img_w)
            face_top_px = int(face_top * img_h)
            face_right_px = int(face_right * img_w)
            face_bottom_px = int(face_bottom * img_h)
            
            # Calculate crop region that includes faces and matches target aspect
            if img_aspect > target_aspect:
                # Image is wider - crop sides, but keep faces centered
                new_width = int(img_h * target_aspect)
                
                # Find center that keeps faces visible
                face_center_x = (face_left_px + face_right_px) // 2
                
                # Calculate crop bounds centered on faces
                left = face_center_x - new_width // 2
                
                # Clamp to image bounds
                if left < 0:
                    left = 0
                elif left + new_width > img_w:
                    left = img_w - new_width
                
                return (left, 0, left + new_width, img_h)
                
            else:
                # Image is taller - crop top/bottom, but keep faces visible
                new_height = int(img_w / target_aspect)
                
                # Find vertical center that keeps faces visible
                face_center_y = (face_top_px + face_bottom_px) // 2
                
                # Prefer showing faces with some headroom (faces in upper 60%)
                # But ensure face isn't cut off
                ideal_top = face_center_y - int(new_height * 0.4)  # Face at 40% from top
                
                # Ensure entire face region is visible
                if ideal_top > face_top_px - int(new_height * 0.1):
                    ideal_top = face_top_px - int(new_height * 0.1)
                if ideal_top + new_height < face_bottom_px + int(new_height * 0.1):
                    ideal_top = face_bottom_px + int(new_height * 0.1) - new_height
                
                # Clamp to image bounds
                top = max(0, min(ideal_top, img_h - new_height))
                
                return (0, top, img_w, top + new_height)
                
        except Exception as e:
            return None
    
    def close(self):
        """Clean up detector resources"""
        if FaceDetector._detector:
            FaceDetector._detector.close()
            FaceDetector._detector = None


# Global face detector instance
_face_detector = None

def get_face_detector() -> FaceDetector:
    """Get or create the global face detector instance"""
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector()
    return _face_detector


# ============================================================================
# API KEY LOADERS
# ============================================================================

def load_gemini_api_key() -> str:
    """Load Gemini API key from key.txt file"""
    key_file = Path(__file__).parent / "key.txt"
    if not key_file.exists():
        raise FileNotFoundError(f"API key file not found: {key_file}")
    
    with open(key_file, 'r') as f:
        content = f.read()
    
    # Parse geminikey="..." format
    match = re.search(r'geminikey\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    
    # Try direct key format
    key = content.strip()
    if key.startswith('AIza'):
        return key
    
    raise ValueError("Could not find valid Gemini API key in key.txt")


def load_pexels_api_key() -> str:
    """Load Pexels API key from key.txt file"""
    key_file = Path(__file__).parent / "key.txt"
    if not key_file.exists():
        raise FileNotFoundError(f"API key file not found: {key_file}")
    
    with open(key_file, 'r') as f:
        content = f.read()
    
    # Parse pexelkey="..." format
    match = re.search(r'pexel(?:s)?key\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    
    raise ValueError("Could not find valid Pexels API key in key.txt")


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
    
    # Gemini AI settings
    GEMINI_MODEL = "gemini-2.0-flash"
    
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
        Uses FACE DETECTION to ensure faces aren't cut off, then falls back to center-weighted cropping.
        """
        img = Image.open(image_path).convert('RGB')
        img_w, img_h = img.size
        target_aspect = target_width / target_height
        img_aspect = img_w / img_h
        
        # Try face-aware cropping first
        face_detector = get_face_detector()
        face_crop = face_detector.get_face_safe_crop_region(image_path, target_aspect)
        
        if face_crop:
            # Use face-aware crop region
            left, top, right, bottom = face_crop
            img = img.crop((left, top, right, bottom))
        elif img_aspect > target_aspect:
            # No face detected - Image is wider than target - crop sides (center)
            new_width = int(img_h * target_aspect)
            left = (img_w - new_width) // 2
            img = img.crop((left, 0, left + new_width, img_h))
        else:
            # No face detected - Image is taller than target - crop top/bottom (favor upper portion)
            new_height = int(img_w / target_aspect)
            top = int((img_h - new_height) * 0.3)
            img = img.crop((0, top, img_w, top + new_height))
        
        # Resize to exact dimensions
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return np.array(img)


# ============================================================================
# PEXELS API CLIENT (Videos & Photos)
# ============================================================================

class PexelsClient:
    """Client for Pexels API - Videos first, then Photos fallback"""
    
    VIDEO_API_URL = "https://api.pexels.com/videos/search"
    PHOTO_API_URL = "https://api.pexels.com/v1/search"
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Config.IMAGES_DIR
        self._api_key = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize Pexels client with API key"""
        if self._initialized:
            return True
        
        try:
            self._api_key = load_pexels_api_key()
            self._initialized = True
            print("  ✓ Pexels API initialized")
            return True
        except Exception as e:
            print(f"  ✗ Pexels initialization failed: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers"""
        return {"Authorization": self._api_key}
    
    def search_videos(self, query: str, per_page: int = 3, orientation: str = "portrait") -> List[Dict]:
        """
        Search for videos on Pexels.
        Returns list of video info with download URLs.
        """
        if not self.initialize():
            return []
        
        try:
            params = {
                "query": query,
                "orientation": orientation,
                "per_page": per_page,
                "size": "medium"  # medium quality for faster downloads
            }
            
            response = requests.get(
                self.VIDEO_API_URL,
                headers=self._get_headers(),
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                videos = []
                
                for video in data.get("videos", []):
                    # Get the best quality HD file with portrait dimensions
                    video_files = video.get("video_files", [])
                    best_file = None
                    
                    # Prefer HD quality, portrait orientation
                    for vf in video_files:
                        # Check for portrait orientation (height > width)
                        if vf.get("height", 0) > vf.get("width", 0):
                            if vf.get("quality") == "hd" or best_file is None:
                                best_file = vf
                    
                    # Fallback to any HD file if no portrait found
                    if not best_file:
                        for vf in video_files:
                            if vf.get("quality") == "hd":
                                best_file = vf
                                break
                    
                    # Fallback to first file
                    if not best_file and video_files:
                        best_file = video_files[0]
                    
                    if best_file:
                        videos.append({
                            "id": video.get("id"),
                            "url": best_file.get("link"),
                            "width": best_file.get("width"),
                            "height": best_file.get("height"),
                            "duration": video.get("duration"),
                            "quality": best_file.get("quality")
                        })
                
                return videos
            else:
                print(f"    Pexels video search failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"    Pexels video search error: {e}")
            return []
    
    def search_photos(self, query: str, per_page: int = 5, orientation: str = "portrait") -> List[Dict]:
        """
        Search for photos on Pexels.
        Returns list of photo info with download URLs.
        """
        if not self.initialize():
            return []
        
        try:
            params = {
                "query": query,
                "orientation": orientation,
                "per_page": per_page,
                "size": "large"
            }
            
            response = requests.get(
                self.PHOTO_API_URL,
                headers=self._get_headers(),
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                photos = []
                
                for photo in data.get("photos", []):
                    src = photo.get("src", {})
                    photos.append({
                        "id": photo.get("id"),
                        "url": src.get("large2x") or src.get("large") or src.get("original"),
                        "width": photo.get("width"),
                        "height": photo.get("height")
                    })
                
                return photos
            else:
                print(f"    Pexels photo search failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"    Pexels photo search error: {e}")
            return []
    
    def download_video(self, url: str, output_path: Path) -> Optional[Path]:
        """Download a video from URL"""
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return output_path
        except Exception as e:
            print(f"    Download error: {e}")
        return None
    
    def download_photo(self, url: str, output_path: Path) -> Optional[Path]:
        """Download a photo from URL"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return output_path
        except Exception as e:
            print(f"    Download error: {e}")
        return None
    
    def trim_video_to_duration(self, video_path: Path, duration: float, output_path: Path = None) -> Optional[Path]:
        """
        Trim video to specified duration using ffmpeg.
        Returns path to trimmed video.
        """
        if not video_path.exists():
            return None
        
        output_path = output_path or video_path.parent / f"trimmed_{video_path.name}"
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-an",  # Remove audio from stock video
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                print(f"    FFmpeg trim error: {result.stderr.decode()[:200]}")
                return None
                
        except Exception as e:
            print(f"    Video trim error: {e}")
            return None


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
# VIRAL SCRIPT FRAMEWORK (Single Gemini API Call)
# ============================================================================

VIRAL_SCRIPT_PROMPT = '''You are an expert viral video script writer. Create a {duration}-second engaging YouTube Short about: "{topic}"

🎯 YOUR MISSION: Create a captivating, informative mini-story that hooks viewers instantly and keeps them watching.

📋 CONTENT STRATEGY:
1. HOOK (First 3 seconds): Start with a BANG - shocking stat, bold claim, or intriguing question
   ❌ NEVER start with: "This is interesting", "Let's explore", "Did you know that", "Have you heard"
   ✅ START directly: "73 percent of people fail at...", "The truth about X will shock you", "X just hit 10 billion dollars"

2. STORY FLOW: Tell a complete mini-story with a clear beginning → middle → end
   - Build tension and curiosity
   - Each segment adds NEW information (no repetition)
   - Use transitions: "But here's the twist", "Even more shocking", "The real reason"

3. DATA-DRIVEN: Pack REAL facts, numbers, and statistics
   - Every segment needs specific data: percentages, dollar amounts, dates, measurements
   - Use 2024-2025 information when possible
   - Be specific: "increased 340 percent" not "grew rapidly"

4. ENDING: Close with impact + CTA
   - Summarize key insight or shocking conclusion
   - Add: "Like, share, and subscribe to Buzz Today for more!"

⏱️ TIMING:
- 30 sec = 5-6 segments (~75 words) | 45 sec = 7-8 segments (~110 words) | 60 sec = 9-10 segments (~150 words)
- Each segment: 12-18 words, 2-3 sentences
- Speaking rate: 2.5 words/second

📝 WRITING RULES:
- NO emojis (text will be spoken)
- Write numbers fully: "50 percent" not "50%", "100 billion dollars" not "$100B"
- No abbreviations: spell out everything
- Conversational but informative tone
- Short, punchy sentences
- Each segment must feel different from the last

🎬 FORBIDDEN PHRASES (Never use these):
❌ "This is an interesting topic about..."
❌ "Let's explore what makes..."
❌ "There's so much to discover..."
❌ "Understanding X can open perspectives..."
❌ "The key insights are..."
❌ "Stay tuned for..."
❌ "Keep exploring and learning..."

🎨 VISUAL STRATEGY (CRITICAL - You control what visuals are shown):

For each segment, specify "visual_priority" - an ordered list of functions to try for the BEST visual:

AVAILABLE FUNCTIONS:
1. "pexels_video" - Stock VIDEO from Pexels (motion, dynamic, great for action/concepts)
2. "pexels_photo" - Stock PHOTO from Pexels (high-quality, generic subjects)
3. "bing_image" - Web image search (celebrities, specific people, brands, logos, news)

WHEN TO USE EACH:

🎬 pexels_video FIRST for:
- Action scenes (running, working out, cooking, driving)
- Nature/landscapes (ocean, mountains, city timelapse)
- Abstract concepts (money falling, stock charts, technology)
- Generic activities (office work, meetings, celebrations)

📷 pexels_photo FIRST for:
- Static subjects (food, products, buildings)
- Portrait-style generic people
- Objects, textures, backgrounds

🔍 bing_image FIRST for:
- CELEBRITIES (Ronaldo,  Messi, Elon Musk, Taylor Swift)
- Specific BRANDS/LOGOS (Nike, Apple, Tesla logo)
- FAMOUS PEOPLE (politicians, actors, athletes)
- Historical photos, news events
- Specific characters (anime, movies, games)

📊 OUTPUT FORMAT (strict JSON):
{{
  "title": "Short internal title",
  "video_title": "Catchy YouTube title with emojis #shorts #viral #BuzzToday (max 100 chars)",
  "description": "SEO-optimized 100-150 word description ending with: Subscribe to Buzz Today for daily viral content!",
  "segments": [
    {{
      "text": "Segment narration - direct, engaging, fact-packed",
      "image_keyword": "search term for media",
      "visual_priority": ["function1", "function2", "function3"],
      "emotion": "curious/shocked/excited/serious/dramatic/hopeful/confident"
    }}
  ],
  "hashtags_text": "450-500 chars of trending + niche hashtags including #BuzzToday",
  "uploaded": false,
  "visibility": "public"
}}

💡 EXAMPLES:

Topic: "Why Ronaldo is GOAT"
{{
  "text": "900 career goals. 5 Champions League titles. Numbers that define greatness.",
  "image_keyword": "Cristiano Ronaldo celebration goal",
  "visual_priority": ["bing_image", "pexels_photo", "pexels_video"],
  "emotion": "excited"
}}

Topic: "Bitcoin Future"  
{{
  "text": "Bitcoin crossed 95,000 dollars in December 2024.",
  "image_keyword": "bitcoin cryptocurrency trading",
  "visual_priority": ["pexels_video", "pexels_photo", "bing_image"],
  "emotion": "excited"
}}

Topic: "Morning Routine for Success"
{{
  "text": "Successful CEOs wake up at 5 AM. Here's why early mornings change everything.",
  "image_keyword": "person waking up morning",
  "visual_priority": ["pexels_video", "pexels_photo", "bing_image"],
  "emotion": "inspiring"
}}

Topic: "Elon Musk Net Worth"
{{
  "text": "From 0 to 250 billion dollars. The richest person on Earth.",
  "image_keyword": "Elon Musk Tesla SpaceX",
  "visual_priority": ["bing_image", "pexels_photo", "pexels_video"],
  "emotion": "shocked"
}}

NOW CREATE: Write the complete engaging script for "{topic}" ({duration} seconds). Make it viral-worthy!
'''


# ============================================================================
# GEMINI AI CLIENT
# ============================================================================

class GeminiClient:
    """Client for interacting with Google Gemini AI API (new google.genai package)"""
    
    def __init__(self, model: str = None):
        self.model_name = model or Config.GEMINI_MODEL
        self._client = None
        self._initialized = False
        self._validated = False
    
    def initialize(self) -> bool:
        """Initialize Gemini with API key"""
        if self._initialized:
            return True
        
        try:
            api_key = load_gemini_api_key()
            self._client = genai.Client(api_key=api_key)
            self._initialized = True
            print(f"  ✓ Gemini AI initialized ({self.model_name})")
            return True
        except Exception as e:
            print(f"  ✗ Gemini initialization failed: {e}")
            return False
    
    def validate_api_key(self) -> Tuple[bool, str]:
        """
        Validate Gemini API key by making a simple test request.
        Returns: (is_valid, error_message)
        """
        if not self.initialize():
            return False, "Failed to initialize Gemini client"
        
        try:
            # Make a minimal test request
            response = self._client.models.generate_content(
                model=self.model_name,
                contents="Say 'OK' in one word.",
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 10,
                }
            )
            
            if response and response.text:
                self._validated = True
                return True, "API key is valid"
            else:
                return False, "API returned empty response"
                
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                return False, "API key is INVALID. Get a new key from: https://aistudio.google.com/app/apikey"
            elif "QUOTA" in error_msg.upper():
                return False, "API quota exceeded. Try again later or get a new key."
            else:
                return False, f"API error: {error_msg[:100]}"
    
    def check_connection(self) -> bool:
        """Check if Gemini is properly configured"""
        return self.initialize()
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini"""
        if not self.initialize():
            return ""
        
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": 4096,
                }
            )
            
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
        return ""


# ============================================================================
# SCRIPT GENERATOR (Gemini - Single API Call)
# ============================================================================

class ScriptGenerator:
    """Generate viral video scripts using Gemini AI in a single API call"""
    
    def __init__(self, gemini: GeminiClient):
        self.gemini = gemini
    
    def generate_script(self, topic: str, duration: int = 45, max_retries: int = 3) -> Dict:
        """Generate a complete viral script in a single Gemini API call"""
        prompt = VIRAL_SCRIPT_PROMPT.format(topic=topic, duration=duration)
        
        print("\n🎬 Generating viral script with Gemini AI (single API call)...")
        print(f"   📝 Topic: {topic}")
        print(f"   ⏱️  Duration: {duration} seconds")
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt + 1}/{max_retries}...")
            
            response = self.gemini.generate(prompt, temperature=0.7 + (attempt * 0.1))
            
            if not response:
                continue
            
            # Extract JSON from response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group()
                    # Fix common JSON issues
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
                        
                        print(f"✅ Generated {len(script_data['segments'])} segments with SEO metadata")
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
        """Create engaging fallback script when AI fails - with smart visual_priority"""
        print("   ⚠️ Creating engaging fallback script...")
        
        # Clean the topic for better formatting
        clean_topic = topic.strip().rstrip('?').rstrip('.').lower()
        
        # Detect topic type and create specific content
        topic_lower = clean_topic.lower()
        
        # Check if topic involves specific people/celebrities (use Bing first)
        celebrity_keywords = ['ronaldo', 'messi', 'neymar', 'mbappe', 'lebron', 'jordan', 
                             'elon', 'musk', 'bezos', 'gates', 'zuckerberg', 'trump', 'obama',
                             'taylor swift', 'drake', 'kanye', 'kardashian']
        is_celebrity_topic = any(celeb in topic_lower for celeb in celebrity_keywords)
        
        # Priority presets
        BING_FIRST = ["bing_image", "pexels_photo", "pexels_video"]
        PEXELS_VIDEO_FIRST = ["pexels_video", "pexels_photo", "bing_image"]
        PEXELS_PHOTO_FIRST = ["pexels_photo", "pexels_video", "bing_image"]
        
        # Sports/Athletes fallback
        if any(word in topic_lower for word in ['ronaldo', 'messi', 'neymar', 'mbappe', 'football', 'soccer', 'player', 'athlete', 'goat', 'best']):
            # Determine if specific athlete or generic sports
            if any(name in topic_lower for name in ['ronaldo', 'messi', 'neymar', 'mbappe', 'lebron']):
                # Specific athlete - use Bing for their photos
                athlete_name = next((name for name in ['ronaldo', 'messi', 'neymar', 'mbappe'] if name in topic_lower), 'athlete')
                segments = [
                    {"text": "900 career goals. 5 Champions League titles. 5 Ballon d'Or awards. These numbers tell an incredible story.", "image_keyword": f"Cristiano {athlete_name} celebration goal", "visual_priority": BING_FIRST, "emotion": "excited"},
                    {"text": "At 39 years old, still scoring hat-tricks while others struggle to walk. That's not talent, that's obsession.", "image_keyword": f"Cristiano {athlete_name} training", "visual_priority": BING_FIRST, "emotion": "serious"},
                    {"text": "From the streets of Madeira to the biggest stages in world football. A journey that changed sports forever.", "image_keyword": f"Cristiano {athlete_name} Real Madrid", "visual_priority": BING_FIRST, "emotion": "inspiring"},
                    {"text": "The critics said he was finished at 30. He responded with 450 more goals. Legends don't retire, they evolve.", "image_keyword": f"Cristiano {athlete_name} champion trophy", "visual_priority": BING_FIRST, "emotion": "confident"},
                    {"text": "Numbers don't lie. Impact doesn't fade. The debate ends here. Like and subscribe to Buzz Today for more!", "image_keyword": f"Cristiano {athlete_name} celebration", "visual_priority": BING_FIRST, "emotion": "excited"}
                ]
            else:
                # Generic sports - use Pexels for stock footage
                segments = [
                    {"text": "900 career goals. 5 Champions League titles. 5 Ballon d'Or awards. These numbers tell an incredible story.", "image_keyword": "soccer goal celebration", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "excited"},
                    {"text": "At 39 years old, still scoring hat-tricks while others struggle to walk. That's not talent, that's obsession.", "image_keyword": "athlete training gym", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "serious"},
                    {"text": "From humble beginnings to the biggest stages in world football. A journey that changed sports forever.", "image_keyword": "stadium crowd cheering", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "inspiring"},
                    {"text": "The critics said he was finished at 30. He responded with 450 more goals. Legends don't retire, they evolve.", "image_keyword": "victory celebration sports", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "confident"},
                    {"text": "Numbers don't lie. Impact doesn't fade. The debate ends here. Like and subscribe to Buzz Today for more!", "image_keyword": "trophy award gold", "visual_priority": PEXELS_PHOTO_FIRST, "emotion": "excited"}
                ]
            title = "The Greatest of All Time"
            
        # Money/Finance fallback (use Pexels video - stock footage)
        elif any(word in topic_lower for word in ['money', 'rich', 'wealth', 'millionaire', 'billionaire', 'invest', 'stock', 'crypto', 'bitcoin']):
            segments = [
                {"text": "The top 1 percent own more wealth than the bottom 50 percent combined. Here's how they did it.", "image_keyword": "money cash dollars", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "serious"},
                {"text": "Warren Buffett made 99 percent of his wealth after age 50. Compound interest is the eighth wonder of the world.", "image_keyword": "stock market trading", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "curious"},
                {"text": "The average millionaire has 7 income streams. Your job is just one. Time to multiply.", "image_keyword": "business meeting office", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "confident"},
                {"text": "Rich people buy assets. Poor people buy liabilities. The difference? Understanding what puts money in your pocket.", "image_keyword": "real estate house luxury", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "serious"},
                {"text": "Start investing just 100 dollars monthly. In 30 years, you could have over 300,000 dollars. Like and subscribe to Buzz Today!", "image_keyword": "finance calculator money", "visual_priority": PEXELS_PHOTO_FIRST, "emotion": "excited"}
            ]
            title = "Wealth Building Secrets"
            
        # Technology/AI fallback (use Pexels video - generic stock)
        elif any(word in topic_lower for word in ['ai', 'technology', 'tech', 'future', 'robot', 'computer', 'software', 'app', 'chatgpt', 'artificial']):
            segments = [
                {"text": "AI can now write code, create art, and pass medical exams. In 2020, none of this existed.", "image_keyword": "technology computer coding", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "shocked"},
                {"text": "By 2030, 85 million jobs will be replaced by automation. But 97 million new ones will be created.", "image_keyword": "robot technology futuristic", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "serious"},
                {"text": "ChatGPT reached 100 million users in 2 months. Instagram took 2 years. TikTok took 9 months.", "image_keyword": "smartphone app typing", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "excited"},
                {"text": "The companies investing in AI today will dominate tomorrow. Those ignoring it will disappear.", "image_keyword": "tech office startup", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "confident"},
                {"text": "The future belongs to those who learn AI skills now. Are you ready? Like and subscribe to Buzz Today!", "image_keyword": "person laptop coding", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "inspiring"}
            ]
            title = "AI Revolution Facts"
            
        # Health/Fitness fallback (use Pexels video - action footage)
        elif any(word in topic_lower for word in ['health', 'fitness', 'workout', 'exercise', 'diet', 'weight', 'muscle', 'gym', 'body']):
            segments = [
                {"text": "Just 30 minutes of walking daily reduces heart disease risk by 35 percent. Small steps, massive impact.", "image_keyword": "person walking nature", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "inspiring"},
                {"text": "Your body replaces itself every 7 years. The food you eat today becomes who you are tomorrow.", "image_keyword": "healthy food vegetables", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "serious"},
                {"text": "People who exercise regularly earn 9 percent more than those who don't. Health equals wealth.", "image_keyword": "gym workout fitness", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "confident"},
                {"text": "Sleep deprivation costs the US economy 411 billion dollars yearly. Rest is not laziness, it's strategy.", "image_keyword": "person sleeping bed", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "thoughtful"},
                {"text": "Start with 10 pushups today. In one year, you won't recognize yourself. Like and subscribe to Buzz Today!", "image_keyword": "fitness training exercise", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "excited"}
            ]
            title = "Health Facts That Matter"
            
        # Default engaging fallback for any other topic
        else:
            # Check if it mentions a specific person - use Bing first
            default_priority = BING_FIRST if is_celebrity_topic else PEXELS_VIDEO_FIRST
            segments = [
                {"text": f"Most people get {clean_topic} completely wrong. Here's what the experts actually know.", "image_keyword": "person thinking idea", "visual_priority": default_priority, "emotion": "curious"},
                {"text": f"After analyzing thousands of cases, one pattern emerged. The truth about {clean_topic} is surprising.", "image_keyword": "research data analysis", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "serious"},
                {"text": f"The top performers in this field all share one secret. It's not what you think it is.", "image_keyword": "success business meeting", "visual_priority": PEXELS_VIDEO_FIRST, "emotion": "confident"},
                {"text": f"Studies show 90 percent of people miss this crucial detail about {clean_topic}. Don't be one of them.", "image_keyword": "statistics chart graph", "visual_priority": PEXELS_PHOTO_FIRST, "emotion": "shocked"},
                {"text": f"Now you know what others don't. Use this knowledge wisely. Like and subscribe to Buzz Today for more!", "image_keyword": "thumbs up success", "visual_priority": PEXELS_PHOTO_FIRST, "emotion": "excited"}
            ]
            title = f"The Truth About {topic.title()}"
        
        return {
            "title": title,
            "video_title": f"{title} 🔥 #shorts #viral #BuzzToday",
            "description": f"Discover the truth about {clean_topic}. Mind-blowing facts and insights you need to know. Subscribe to Buzz Today for daily viral content!",
            "segments": segments,
            "hashtags_text": self._generate_default_hashtags(topic),
            "uploaded": False,
            "visibility": "public"
        }


# ============================================================================
# MEDIA DOWNLOADER (Smart Priority: Gemini decides Bing vs Pexels)
# ============================================================================

class MediaDownloader:
    """
    Download media for segments with SMART priority based on Gemini's recommendation.
    - For celebrities/specific people: Bing first (has real photos)
    - For generic concepts: Pexels first (has stock videos/photos)
    """
    
    def __init__(self, output_dir: Path = Config.IMAGES_DIR):
        self.output_dir = output_dir
        self.pexels = PexelsClient(output_dir)
        
        # Map function names to actual methods
        self._function_map = {
            "pexels_video": self._try_pexels_video,
            "pexels_photo": self._try_pexels_photo,
            "bing_image": self._try_bing_image,
        }
    
    def download_media_for_segments(
        self, 
        segments: List[Dict], 
        audio_durations: List[float] = None
    ) -> Dict[str, Dict]:
        """
        Download media for each segment using Gemini's visual_priority.
        Gemini specifies the exact order of functions to try for best visuals.
        
        Returns dict with keyword -> {type, path, duration} info
        """
        print("\n📥 Downloading media (Gemini controls visual priority)...")
        
        result = {}
        
        for i, segment in enumerate(segments):
            keyword = segment.get("image_keyword", "")
            duration = audio_durations[i] if audio_durations and i < len(audio_durations) else 8.0
            
            # Get visual priority from Gemini (or use default)
            visual_priority = segment.get("visual_priority", ["pexels_video", "pexels_photo", "bing_image"])
            
            # Handle legacy media_source field (backward compatibility)
            if "media_source" in segment and "visual_priority" not in segment:
                media_source = segment.get("media_source", "pexels").lower()
                if media_source == "bing":
                    visual_priority = ["bing_image", "pexels_photo", "pexels_video"]
                else:
                    visual_priority = ["pexels_video", "pexels_photo", "bing_image"]
            
            segment_dir = self.output_dir / f"segment_{i:02d}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear previous files
            for f in segment_dir.glob("*"):
                f.unlink()
            
            priority_str = " → ".join(visual_priority)
            print(f"  Segment {i+1}: '{keyword}' ({duration:.1f}s)")
            print(f"    📋 Priority: {priority_str}")
            
            # Try each function in the priority order specified by Gemini
            media_info = self._download_with_priority(keyword, segment_dir, duration, visual_priority)
            
            if media_info:
                result[keyword] = media_info
                print(f"    ✓ Got {media_info['type']} from {media_info.get('source', 'unknown')}: {media_info['path'].name}")
            else:
                result[keyword] = {"type": "none", "path": None}
                print(f"    ✗ No media found!")
        
        return result
    
    def _download_with_priority(self, keyword: str, output_dir: Path, duration: float, 
                                 visual_priority: List[str]) -> Optional[Dict]:
        """
        Download media using the exact priority order specified by Gemini.
        Tries each function in order until one succeeds.
        """
        for func_name in visual_priority:
            func = self._function_map.get(func_name)
            if not func:
                print(f"    ⚠️ Unknown function: {func_name}")
                continue
            
            print(f"    → Trying {func_name}...")
            
            # Call the function with appropriate parameters
            if func_name == "pexels_video":
                media_info = func(keyword, output_dir, duration)
            else:
                media_info = func(keyword, output_dir)
            
            if media_info:
                return media_info
        
        return None
    
    def _try_pexels_video(self, keyword: str, output_dir: Path, duration: float) -> Optional[Dict]:
        """Try to download and trim a Pexels video"""
        videos = self.pexels.search_videos(keyword, per_page=3, orientation="portrait")
        
        if not videos:
            return None
        
        for idx, video in enumerate(videos):
            video_url = video.get("url")
            if not video_url:
                continue
            
            # Download video
            video_path = output_dir / f"video_{idx}.mp4"
            downloaded = self.pexels.download_video(video_url, video_path)
            
            if downloaded and downloaded.exists():
                # Trim to segment duration
                trimmed_path = output_dir / f"trimmed_{idx}.mp4"
                trimmed = self.pexels.trim_video_to_duration(downloaded, duration, trimmed_path)
                
                if trimmed and trimmed.exists():
                    # Clean up original
                    if downloaded.exists() and downloaded != trimmed:
                        downloaded.unlink()
                    
                    return {
                        "type": "video",
                        "path": trimmed,
                        "duration": duration,
                        "source": "pexels"
                    }
        
        return None
    
    def _try_pexels_photo(self, keyword: str, output_dir: Path) -> Optional[Dict]:
        """Try to download a Pexels photo"""
        photos = self.pexels.search_photos(keyword, per_page=5, orientation="portrait")
        
        if not photos:
            return None
        
        downloaded_photos = []
        for idx, photo in enumerate(photos):
            photo_url = photo.get("url")
            if not photo_url:
                continue
            
            photo_path = output_dir / f"photo_{idx}.jpg"
            downloaded = self.pexels.download_photo(photo_url, photo_path)
            
            if downloaded and downloaded.exists():
                downloaded_photos.append(downloaded)
        
        if downloaded_photos:
            # Select best photo for 9:16 format
            best_photo = SmartImageSelector.select_best_image(downloaded_photos)
            return {
                "type": "image",
                "path": best_photo,
                "source": "pexels"
            }
        
        return None
    
    def _try_bing_image(self, keyword: str, output_dir: Path) -> Optional[Dict]:
        """Search Bing for images (best for celebrities, brands, specific people)"""
        try:
            # Add portrait/vertical hint
            enhanced_keyword = f"{keyword} vertical portrait"
            
            crawler = BingImageCrawler(
                storage={'root_dir': str(output_dir)},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=2
            )
            crawler.crawl(
                keyword=enhanced_keyword,
                max_num=5,
                min_size=(640, 800)
            )
            
            # Also try original keyword if few results
            downloaded = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
            if len(downloaded) < 2:
                crawler.crawl(
                    keyword=keyword,
                    max_num=5,
                    min_size=(640, 480)
                )
            
            downloaded = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
            
            if downloaded:
                best_image = SmartImageSelector.select_best_image(downloaded)
                return {
                    "type": "image",
                    "path": best_image,
                    "source": "bing"
                }
                
        except Exception as e:
            print(f"      Bing error: {e}")
        
        return None
    
    def get_selected_media(self, media_dict: Dict[str, Dict], segments: List[Dict]) -> List[Dict]:
        """
        Get ordered list of media for each segment.
        Returns list of {type, path, source} dicts matching segment order.
        """
        selected = []
        
        for segment in segments:
            keyword = segment.get("image_keyword", "")
            media_info = media_dict.get(keyword, {"type": "none", "path": None})
            selected.append(media_info)
        
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
    """Create Ken Burns zoom and pan effects on images with smart cropping and face detection"""
    
    def __init__(self, width: int = Config.VIDEO_WIDTH, height: int = Config.VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
    
    def prepare_image(self, image_path: Path) -> np.ndarray:
        """
        Prepare image for video using FACE-AWARE smart cropping.
        Uses MediaPipe to detect faces and ensures they aren't cut off.
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        img_w, img_h = img.size
        img_aspect = img_w / img_h
        target_aspect = self.aspect_ratio
        
        # Try face-aware cropping first
        face_detector = get_face_detector()
        face_crop = face_detector.get_face_safe_crop_region(image_path, target_aspect)
        
        if face_crop:
            # Use face-aware crop with extra padding for Ken Burns
            left, top, right, bottom = face_crop
            crop_w = right - left
            crop_h = bottom - top
            
            # Add 40% extra for Ken Burns movement while keeping faces centered
            extra_w = int(crop_w * 0.2)
            extra_h = int(crop_h * 0.2)
            
            left = max(0, left - extra_w)
            top = max(0, top - extra_h)
            right = min(img_w, right + extra_w)
            bottom = min(img_h, bottom + extra_h)
            
            img = img.crop((left, top, right, bottom))
            
        elif img_aspect > target_aspect:
            # No face - Image is wider than target - crop sides (center crop)
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
            
            img = img.crop((left, top, left + crop_w, top + crop_h))
            
        else:
            # No face - Image is taller or square - use more of vertical space
            new_width = int(self.width * 1.4)
            new_height = int(new_width / img_aspect)
            
            # Favor upper portion (30% from top instead of center)
            left = 0
            top = max(0, int((img_h - new_height) * 0.3))
            crop_w = min(new_width, img_w)
            crop_h = min(new_height, img_h - top)
            
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
        self.transcriber = WordTranscriber(model_size="turbo")  # Use turbo for speed
        self.captions = AnimatedCaptions()
    
    def create_video(self, segments: List[Dict], images: List[Path], 
                     audio_files: List[Path], output_path: Path,
                     add_captions: bool = True, media_types: List[str] = None) -> Path:
        """
        Create the final video with word-by-word captions, music, and transition sounds.
        Supports both video clips (from Pexels) and images.
        
        Args:
            segments: Script segments with text and keywords
            images: List of media paths (images or videos)
            audio_files: Audio files for each segment
            output_path: Output video path
            add_captions: Whether to add word-by-word captions
            media_types: List of "video" or "image" for each segment
        """
        print("\n🎥 Composing video with animated captions...")
        
        # Default media types to "image" if not provided
        if media_types is None:
            media_types = ["image"] * len(images)
        
        clips = []
        durations = []
        all_word_data = []  # Store transcription data for each segment
        
        for i, (segment, media_path, audio) in enumerate(zip(segments, images, audio_files)):
            print(f"  Processing segment {i+1}/{len(segments)}...")
            
            if media_path is None or audio is None:
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
            
            # Create video clip based on media type
            media_type = media_types[i] if i < len(media_types) else "image"
            
            if media_type == "video" and str(media_path).endswith(('.mp4', '.mov', '.avi', '.webm')):
                # Use stock video clip (already trimmed to duration)
                print(f"    🎬 Using stock video: {media_path.name}")
                video_clip = self._create_video_clip(media_path, duration)
            else:
                # Use Ken Burns effect on image
                effect_types = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
                effect = effect_types[i % len(effect_types)]
                video_clip = self.ken_burns.create_clip(media_path, duration, effect)
            
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
    
    def _create_video_clip(self, video_path: Path, duration: float) -> VideoClip:
        """
        Create a video clip from a stock video file.
        Resizes and crops to fit 9:16 format.
        """
        try:
            # Load video clip
            clip = VideoFileClip(str(video_path))
            
            # Get original dimensions
            orig_w, orig_h = clip.size
            
            # Calculate target dimensions (9:16)
            target_w = Config.VIDEO_WIDTH
            target_h = Config.VIDEO_HEIGHT
            target_aspect = target_w / target_h
            clip_aspect = orig_w / orig_h
            
            # Resize and crop to fit 9:16
            if clip_aspect > target_aspect:
                # Clip is wider - scale by height and crop sides
                new_h = target_h
                new_w = int(orig_w * (target_h / orig_h))
                clip = clip.resized(height=target_h)
                # Center crop
                x_center = new_w // 2
                clip = clip.cropped(x1=x_center - target_w // 2, 
                                   x2=x_center + target_w // 2,
                                   y1=0, y2=target_h)
            else:
                # Clip is taller or square - scale by width and crop top/bottom
                new_w = target_w
                new_h = int(orig_h * (target_w / orig_w))
                clip = clip.resized(width=target_w)
                # Favor top portion for faces
                y_start = int((new_h - target_h) * 0.3)
                clip = clip.cropped(x1=0, x2=target_w,
                                   y1=y_start, y2=y_start + target_h)
            
            # Trim to exact duration if needed
            if clip.duration > duration:
                clip = clip.subclipped(0, duration)
            elif clip.duration < duration:
                # Loop video if too short
                loops_needed = int(duration / clip.duration) + 1
                clips = [clip] * loops_needed
                clip = concatenate_videoclips(clips)
                clip = clip.subclipped(0, duration)
            
            # Remove audio from stock video (we use TTS audio)
            clip = clip.without_audio()
            
            return clip
            
        except Exception as e:
            print(f"    ⚠️ Video clip error: {e}")
            # Fallback: create black frame
            return ColorClip(size=(Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT), 
                           color=(0, 0, 0), duration=duration)
    
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
        self.gemini = GeminiClient()
        self.script_gen = ScriptGenerator(self.gemini)
        self.media_dl = MediaDownloader()  # Pexels videos/photos + Bing fallback
        self.voice_gen = VoiceGenerator()
        self.composer = VideoComposer()
        self.uploader = YouTubeUploader()
    
    def check_requirements(self) -> Dict[str, bool]:
        """Check if all requirements are met"""
        requirements = {}
        
        # Check Gemini AI
        print("\n🔍 Checking requirements...")
        requirements["gemini"] = self.gemini.check_connection()
        print(f"  Gemini AI: {'✓' if requirements['gemini'] else '✗'}")
        
        # Check Pexels API
        requirements["pexels"] = self.media_dl.pexels.initialize()
        print(f"  Pexels API: {'✓' if requirements['pexels'] else '✗'}")
        
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
        print("  Features: Gemini AI, Pexels Videos/Photos, Kokoro TTS")
        print("="*60)
        
        # Check requirements
        reqs = self.check_requirements()
        
        if not reqs["gemini"]:
            print("\n⚠️  Gemini AI is not configured!")
            print("   Please ensure key.txt contains valid Gemini API key")
            return
        
        # Validate Gemini API key with a test request
        print("\n🔑 Validating Gemini API key...")
        is_valid, error_msg = self.gemini.validate_api_key()
        
        use_fallback = False
        if not is_valid:
            print(f"\n❌ Gemini API Key Error: {error_msg}")
            print("\n" + "-"*40)
            print("Options:")
            print("  1. Fix your API key in key.txt and restart")
            print("  2. Continue with fallback scripts (topic-based templates)")
            print("-"*40)
            
            choice = input("\n⚠️  Continue with fallback scripts? (y/n): ").strip().lower()
            if choice != 'y':
                print("\n🔑 To fix: Get a new API key from https://aistudio.google.com/app/apikey")
                print("   Then update key.txt with: geminikey=\"YOUR_NEW_KEY\"")
                return
            use_fallback = True
            print("\n✓ Using fallback scripts (pre-built topic templates)")
        else:
            print("✅ Gemini API key validated successfully!")
        
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
        
        # Generate the reel (with auto-upload enabled by default)
        self.create_reel(topic, duration, voice, auto_upload=True, force_fallback=use_fallback)
    
    def create_reel(self, topic: str, duration: int = 45, 
                    voice: str = None, add_captions: bool = True,
                    whisper_model: str = "base", auto_upload: bool = True,
                    force_fallback: bool = False) -> Path:
        """Create a complete reel from topic to video with word-by-word captions"""
        voice = voice or Config.TTS_VOICE
        
        # Generate unique timestamp for this video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_')[:30]
        
        # Video filename (used for both video and JSON)
        video_base_name = f"reel_{timestamp}_{safe_topic}"
        
        print("\n" + "="*60)
        print(f"🎬 Creating Reel: {topic}")
        print(f"   📝 Word-by-word captions: {'Enabled' if add_captions else 'Disabled'}")
        print(f"   🆔 Video ID: {video_base_name}")
        if force_fallback:
            print(f"   ⚠️  Using fallback script (Gemini unavailable)")
        print("="*60)
        
        # Update whisper model if captions are enabled
        if add_captions:
            self.composer.transcriber = WordTranscriber(model_size=whisper_model)
        
        # Step 1: Generate script (single Gemini API call - includes all content)
        if force_fallback:
            # Directly use fallback script without trying Gemini
            print("\n🎬 Using fallback script generator...")
            script = self.script_gen._create_fallback_script(topic, "")
        else:
            script = self.script_gen.generate_script(topic, duration)
        
        # Add metadata fields
        script['uploaded'] = False
        script['visibility'] = 'public'
        script['video_file_path'] = None
        script['created_at'] = datetime.now().isoformat()
        script['topic'] = topic
        script['duration_target'] = duration
        
        # Save script with same name as video (JSON alongside MP4)
        script_filename = f"{video_base_name}.json"
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
        
        # Step 2: Generate audio for each segment using Kokoro TTS
        # (Do this BEFORE downloading media so we know durations for video trimming)
        print(f"\n🎤 Generating voice narration with Kokoro TTS (voice: {voice})...")
        print("   Using sentence-by-sentence processing for natural speech flow...")
        audio_files = []
        audio_durations = []
        
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
                audio_durations.append(audio_duration)
                print(f"    ✓ Duration: {audio_duration:.2f}s")
            else:
                audio_files.append(None)
                audio_durations.append(8.0)  # Default duration
        
        # Step 3: Download media (Pexels videos → photos → Bing images)
        # Videos are trimmed to match each segment's audio duration
        segments_list = script.get("segments", [])
        media_dict = self.media_dl.download_media_for_segments(segments_list, audio_durations)
        selected_media = self.media_dl.get_selected_media(media_dict, segments_list)
        
        # Extract paths for video composition (supports both videos and images)
        selected_images = [m.get("path") for m in selected_media]
        media_types = [m.get("type", "image") for m in selected_media]
        
        # Step 4: Compose video with captions, music and click sounds
        output_video = Config.VIDEO_DIR / f"{video_base_name}.mp4"
        
        # Filter out None values (keep media_types aligned)
        valid_data = [
            (seg, img, aud, mtype) 
            for seg, img, aud, mtype in zip(script.get("segments", []), selected_images, audio_files, media_types)
            if img is not None and aud is not None
        ]
        
        if valid_data:
            segments, images, audios, valid_media_types = zip(*valid_data)
            final_video = self.composer.create_video(
                list(segments), list(images), list(audios), output_video,
                add_captions=add_captions,
                media_types=list(valid_media_types)
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
        # Match both old (script_*) and new (reel_*) naming conventions
        for script_file in sorted(list(scripts_dir.glob("script_*.json")) + list(scripts_dir.glob("reel_*.json"))):
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
        description="Interactive Reel Shorts Designer with Gemini AI & YouTube Upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode (auto-uploads)
  python main.py -t "Bitcoin investing"    # Create reel (auto-uploads)
  python main.py -t "Elon Musk" --no-upload # Create without uploading
  python main.py --upload-pending          # Upload all pending videos
  python main.py --list-pending            # List pending uploads

Note: Requires Gemini API key in key.txt file (format: geminikey="YOUR_API_KEY")
        """
    )
    parser.add_argument("--topic", "-t", type=str, help="Topic for the reel")
    parser.add_argument("--duration", "-d", type=int, default=45, help="Target duration in seconds (30/45/60)")
    parser.add_argument("--voice", "-v", type=str, default=Config.TTS_VOICE, help="Kokoro TTS voice name (e.g., af_bella, af_nicole)")
    parser.add_argument("--speed", "-s", type=float, default=Config.TTS_SPEED, help="TTS speech speed (default: 1.0)")
    parser.add_argument("--no-captions", action="store_true", help="Disable word-by-word captions")
    parser.add_argument("--whisper-model", type=str, default="turbo", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size for transcription (default: turbo)")
    
    # YouTube upload options
    parser.add_argument("--no-upload", action="store_true", help="Disable auto-upload to YouTube (default: auto-upload enabled)")
    parser.add_argument("--upload-pending", action="store_true", help="Upload all pending videos")
    parser.add_argument("--list-pending", action="store_true", help="List all pending (not uploaded) videos")
    
    args = parser.parse_args()
    
    # Auto-upload by default, unless --no-upload is specified
    args.upload = not args.no_upload
    
    # Update config if specified
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
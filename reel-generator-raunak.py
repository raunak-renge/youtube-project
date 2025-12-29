#!/usr/bin/env python3
"""
🎬 ULTIMATE VIRAL REEL GENERATOR
================================
A professional-grade automated reel/shorts generator that creates viral content.

Features:
- 🤖 Gemini AI for script generation with SEO optimization
- 📥 yt-dlp for downloading source videos from YouTube
- 🎬 PySceneDetect for intelligent scene selection
- 🎤 TTS for voice narration (Edge TTS / Kokoro)
- 📝 Whisper for word-level timestamp captions
- 🖼️ g4f for AI thumbnail generation
- 📤 YouTube upload with multi-credential support
- 📊 Comprehensive JSON metadata output

Author: AI Reel Generator
Version: 2.0.0
"""

import os
import sys
import json
import random
import subprocess
import shutil
import tempfile
import re
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pickle
import logging

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

def install_package(package: str, import_name: str = None) -> bool:
    """Install a package if not present"""
    import_name = import_name or package.replace("-", "_")
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"📦 Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, 
             "--break-system-packages", "-q"],
            capture_output=True
        )
        return result.returncode == 0

# Required packages
REQUIRED_PACKAGES = [
    ("yt-dlp", "yt_dlp"),
    ("scenedetect[opencv]", "scenedetect"),
    ("openai-whisper", "whisper"),
    ("moviepy", "moviepy"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("google-genai", "google.genai"),
    ("g4f", "g4f"),
    ("google-api-python-client", "googleapiclient"),
    ("google-auth-oauthlib", "google_auth_oauthlib"),
    ("soundfile", "soundfile"),
    ("paddleocr", "paddleocr"),
    ("opencv-python", "cv2"),
    ("tqdm", "tqdm"),
]

def check_and_install_dependencies():
    """Check and install all required dependencies"""
    print("🔍 Checking dependencies...")
    for package, import_name in REQUIRED_PACKAGES:
        if not install_package(package, import_name):
            print(f"⚠️  Failed to install {package}")
    print("✅ Dependencies ready!")

# Run dependency check
check_and_install_dependencies()

# Now import everything
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import whisper
import yt_dlp
from scenedetect import detect, ContentDetector, AdaptiveDetector
from scenedetect import open_video, SceneManager
from moviepy.video.VideoClip import VideoClip, ColorClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from g4f.client import Client as G4FClient
    G4F_AVAILABLE = True
except ImportError:
    G4F_AVAILABLE = False

# PaddleOCR for text removal from videos (excellent for stylized/colored text)
try:
    from paddleocr import PaddleOCR
    import cv2
    PADDLEOCR_AVAILABLE = True
    print("✅ PaddleOCR available for text removal")
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠️  PaddleOCR not available - text removal disabled")

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the reel generator"""
    # Directories
    BASE_DIR: Path = Path("./reel_output_youtube")
    VIDEOS_DIR: Path = field(default_factory=lambda: Path("./reel_output_youtube/source_videos"))
    SCENES_DIR: Path = field(default_factory=lambda: Path("./reel_output_youtube/scenes"))
    AUDIO_DIR: Path = field(default_factory=lambda: Path("./reel_output_youtube/audio"))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("./reel_output_youtube/final"))
    THUMBNAILS_DIR: Path = field(default_factory=lambda: Path("./reel_output_youtube/thumbnails"))
    SCRIPTS_DIR: Path = field(default_factory=lambda: Path("./reel_output_youtube/scripts"))
    
    # Background audio directories
    MUSIC_DIR: Path = field(default_factory=lambda: Path("./background-sounds/music"))
    CLICKS_DIR: Path = field(default_factory=lambda: Path("./background-sounds/clicks"))
    
    # Video settings (YouTube Shorts 9:16)
    VIDEO_WIDTH: int = 1080
    VIDEO_HEIGHT: int = 1920
    FPS: int = 30
    
    # Kokoro TTS Settings
    KOKORO_MODEL: Path = field(default_factory=lambda: Path("./kokoro-v1.0.onnx"))
    KOKORO_VOICES: Path = field(default_factory=lambda: Path("./voices-v1.0.bin"))
    TTS_VOICE: str = "af_bella"  # Kokoro TTS voice
    TTS_SPEED: float = 1.0  # Speech speed
    
    # Audio levels
    MUSIC_VOLUME: float = 0.12
    CLICK_VOLUME: float = 0.25
    
    # Gemini settings
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # YouTube settings
    GOOGLE_CONSOLE_DIR: Path = field(default_factory=lambda: Path("./google-console"))
    YOUTUBE_SCOPES: List[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/youtube.upload"])
    
    # Video download settings
    MAX_VIDEO_DURATION: int = 300  # 5 minutes max per source video
    MIN_VIDEO_DURATION: int = 30   # Minimum 30 seconds
    VIDEOS_PER_SEARCH: int = 3     # Number of videos to download per search term
    
    # Scene detection
    SCENE_THRESHOLD: float = 27.0  # Content detector threshold
    MIN_SCENE_DURATION: float = 1.0  # Minimum scene duration in seconds
    
    # Text removal settings
    REMOVE_TEXT: bool = True  # Enable/disable text removal from source videos
    TEXT_REMOVAL_SAMPLE_RATE: int = 5  # Process every Nth frame for speed (1 = all frames)
    
    # Parallelization settings (for faster processing)
    PARALLEL_DOWNLOADS: bool = True    # Enable parallel video downloads
    PARALLEL_SCENE_DETECTION: bool = True  # Enable parallel scene detection
    PARALLEL_TRANSCRIPTION: bool = True    # Enable parallel Whisper transcription
    MAX_DOWNLOAD_WORKERS: int = 3      # Max parallel download workers
    MAX_SCENE_WORKERS: int = 4         # Max parallel scene detection workers
    MAX_TRANSCRIBE_WORKERS: int = 2    # Max parallel transcription workers (GPU limited)
    
    # Channel settings for default keywords
    CHANNEL_NAME: str = "Buzz Today"  # Your channel name
    
    # Default keywords to fill 500 character limit (add your channel-specific keywords)
    DEFAULT_KEYWORDS: List[str] = field(default_factory=lambda: [
        # Channel branding
        "Buzz Today", "buzz today", "daily buzz",
        # Generic viral shorts keywords  
        "shorts", "viral", "trending", "fyp", "foryou", "foryoupage",
        "youtube shorts", "short video", "viral video", "trending now",
        # Engagement keywords
        "must watch", "amazing", "incredible", "mindblowing", "shocking",
        "unbelievable", "insane", "crazy", "awesome", "epic",
        # Content type keywords
        "facts", "did you know", "fun facts", "interesting facts",
        "learn something new", "education", "knowledge", "informative",
        # Call to action keywords
        "follow for more", "subscribe", "like and share",
        # Discovery keywords
        "explore", "discover", "new", "latest", "2024", "2025",
        # Platform keywords
        "reels", "tiktok", "instagram", "social media",
        # Emotion keywords
        "motivation", "inspiration", "entertainment", "funny",
    ])
    
    def setup_dirs(self):
        """Create all required directories"""
        for attr in ['BASE_DIR', 'VIDEOS_DIR', 'SCENES_DIR', 'AUDIO_DIR', 
                     'OUTPUT_DIR', 'THUMBNAILS_DIR', 'SCRIPTS_DIR']:
            path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)
    
    def get_music(self) -> Optional[Path]:
        """Get random background music file"""
        if not self.MUSIC_DIR.exists():
            return None
        files = list(self.MUSIC_DIR.glob("*.wav")) + list(self.MUSIC_DIR.glob("*.mp3"))
        return random.choice(files) if files else None
    
    def get_click(self) -> Optional[Path]:
        """Get random click sound file"""
        if not self.CLICKS_DIR.exists():
            return None
        files = list(self.CLICKS_DIR.glob("*.wav")) + list(self.CLICKS_DIR.glob("*.mp3"))
        return random.choice(files) if files else None

# Global config
config = Config()


def sanitize_tag(tag: str) -> str:
    """
    Sanitize a single tag for YouTube compliance.
    YouTube tag rules:
    - Max 30 characters per tag
    - No < or > characters
    - No leading/trailing whitespace
    - Should be alphanumeric with spaces allowed
    """
    if not tag:
        return ""
    
    # Strip whitespace
    tag = tag.strip()
    
    # Remove invalid characters (keep alphanumeric, spaces, and common punctuation)
    # YouTube rejects: < > 
    tag = re.sub(r'[<>"\']', '', tag)
    
    # Replace multiple spaces with single space
    tag = re.sub(r'\s+', ' ', tag)
    
    # Truncate to 30 characters (YouTube limit per tag)
    if len(tag) > 30:
        tag = tag[:30].strip()
    
    return tag


def fill_tags_to_limit(tags: List[str], topic: str = "", limit: int = 500) -> List[str]:
    """
    Fill tags list with default keywords until reaching the character limit.
    YouTube allows up to 500 characters for tags.
    
    Args:
        tags: Initial tags from Gemini
        topic: The video topic for additional keyword variations
        limit: Character limit (default 500 for YouTube)
    
    Returns:
        Extended list of tags that fills the limit (all sanitized for YouTube)
    """
    # Sanitize and start with the provided tags
    final_tags = []
    for tag in (tags or []):
        sanitized = sanitize_tag(tag)
        if sanitized:
            final_tags.append(sanitized)
    
    # Calculate current character count (tags joined by commas)
    def get_char_count(tag_list):
        return len(", ".join(tag_list))
    
    # Generate topic-specific variations
    topic_variations = []
    if topic:
        topic_lower = topic.lower()
        topic_words = topic_lower.split()
        topic_variations = [
            topic_lower,
            f"{topic_lower} facts",
            f"{topic_lower} explained",
            f"{topic_lower} shorts",
            f"{topic_lower} viral",
            f"about {topic_lower}",
            f"learn {topic_lower}",
            f"{topic_lower} 2025",
        ]
        # Add individual words if multi-word topic
        if len(topic_words) > 1:
            topic_variations.extend(topic_words)
    
    # Combine all potential keywords: topic variations + channel defaults
    all_keywords = topic_variations + list(config.DEFAULT_KEYWORDS)
    
    # Add keywords until we reach the limit
    existing_tags_lower = {t.lower() for t in final_tags}
    
    for keyword in all_keywords:
        # Sanitize each keyword
        keyword = sanitize_tag(keyword)
        if not keyword or keyword.lower() in existing_tags_lower:
            continue
        
        # Check if adding this keyword would exceed the limit
        test_tags = final_tags + [keyword]
        if get_char_count(test_tags) <= limit:
            final_tags.append(keyword)
            existing_tags_lower.add(keyword.lower())
        elif get_char_count(final_tags) >= limit - 20:
            # Close enough to limit, stop adding
            break
    
    logger.debug(f"Tags filled: {get_char_count(final_tags)}/{limit} characters, {len(final_tags)} tags")
    return final_tags


# ============================================================================
# TEXT REMOVER (PaddleOCR + Advanced Inpainting)
# ============================================================================

class TextRemover:
    """
    Remove text/watermarks from video frames using PaddleOCR and OpenCV inpainting.
    
    PaddleOCR advantages:
    - Excellent for large, colorful, stylized text (TikTok/YouTube captions)
    - Better detection of watermarks and channel handles
    - Multi-angle and curved text detection
    - Fast and accurate with GPU support
    - Supports 80+ languages
    
    This helps create cleaner videos by removing existing text/watermarks
    before adding our own captions.
    """
    
    _ocr = None  # Singleton OCR instance to avoid reloading model
    
    def __init__(self):
        self._initialized = False
    
    @classmethod
    def get_ocr(cls):
        """Get or create the PaddleOCR instance (singleton - loads model once)"""
        if cls._ocr is None and PADDLEOCR_AVAILABLE:
            print("🔤 Loading PaddleOCR model (first time, may take a moment)...")
            try:
                # Suppress PaddleOCR verbose output
                import os
                os.environ['PADDLEOCR_SHOW_LOG'] = 'False'
                
                # Initialize PaddleOCR - use simple initialization for compatibility
                # PaddleOCR API varies between versions, so keep it minimal
                cls._ocr = PaddleOCR(lang='en')
                print("✅ PaddleOCR model loaded!")
            except Exception as e:
                logger.warning(f"Failed to load PaddleOCR: {e}")
                cls._ocr = None
        return cls._ocr
    
    @staticmethod
    def midpoint(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
        """Calculate midpoint between two points"""
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    @staticmethod
    def expand_bbox(pts: np.ndarray, padding: int, img_shape: Tuple[int, int]) -> np.ndarray:
        """Expand bounding box by padding pixels while staying within image bounds"""
        h, w = img_shape[:2]
        # Calculate center
        center = pts.mean(axis=0)
        # Expand each point away from center
        expanded = []
        for pt in pts:
            direction = pt - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            new_pt = pt + direction * padding
            # Clamp to image bounds
            new_pt[0] = max(0, min(w - 1, new_pt[0]))
            new_pt[1] = max(0, min(h - 1, new_pt[1]))
            expanded.append(new_pt)
        return np.array(expanded, dtype=np.int32)
    
    def remove_text_from_image(self, img: np.ndarray) -> np.ndarray:
        """
        Remove text from a single image using PaddleOCR detection and OpenCV inpainting.
        
        Args:
            img: Input image as numpy array (RGB format)
        
        Returns:
            Image with text removed (RGB format)
        """
        if not PADDLEOCR_AVAILABLE:
            return img
        
        ocr = self.get_ocr()
        if ocr is None:
            return img
        
        try:
            # PaddleOCR expects BGR format for numpy arrays
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Run OCR prediction
            result = ocr.predict(input=img_bgr)
            
            if not result:
                return img  # No results
            
            # Create mask for inpainting
            mask = np.zeros(img.shape[:2], dtype="uint8")
            text_found = False
            
            for res in result:
                # Access the detection results - PaddleOCR returns dict-like objects
                # Check for rec_texts (recognized text)
                rec_texts = res.get('rec_texts', []) if isinstance(res, dict) else getattr(res, 'rec_texts', [])
                if not rec_texts:
                    continue
                
                # Get bounding boxes from dt_polys
                dt_polys = res.get('dt_polys', []) if isinstance(res, dict) else getattr(res, 'dt_polys', [])
                if dt_polys is not None:
                    for i, poly in enumerate(dt_polys):
                        text_found = True
                        # poly is typically [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
                        pts = np.array(poly, dtype=np.int32)
                        
                        # Expand the bounding box for better coverage
                        padding = 15  # pixels of padding around detected text
                        pts_expanded = self.expand_bbox(pts, padding, img.shape)
                        
                        # Fill the polygon in the mask
                        cv2.fillPoly(mask, [pts_expanded], 255)
                        
                        # Also draw thick lines connecting the corners for better coverage
                        for j in range(4):
                            p1 = tuple(pts_expanded[j])
                            p2 = tuple(pts_expanded[(j + 1) % 4])
                            cv2.line(mask, p1, p2, 255, 8)
            
            if not text_found:
                return img  # No text detected
            
            # Apply morphological operations for better mask
            # Dilate to expand coverage
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(mask, kernel, iterations=3)
            
            # Optional: Close small gaps in the mask
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Inpaint using multiple techniques for best results
            # First pass with INPAINT_NS (Navier-Stokes based)
            inpainted = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_NS)
            
            # Second pass with INPAINT_TELEA for smoother results
            inpainted = cv2.inpaint(inpainted, mask, 3, cv2.INPAINT_TELEA)
            
            # Convert back to RGB
            result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
            
            return result
            
        except Exception as e:
            logger.warning(f"Text removal failed: {e}")
            return img
    
    def remove_text_from_video(self, input_path: Path, output_path: Path,
                               sample_rate: int = None) -> Optional[Path]:
        """
        Remove text from all frames in a video.
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            sample_rate: Process every Nth frame (1 = all frames, higher = faster but may miss text)
        
        Returns:
            Path to processed video, or None if failed
        """
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available, skipping text removal")
            return input_path
        
        sample_rate = sample_rate or config.TEXT_REMOVAL_SAMPLE_RATE
        
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {input_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            logger.info(f"Removing text from video: {total_frames} frames...")
            
            # Import tqdm for progress bar
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=total_frames, 
                    desc="🔤 OCR Text Removal", 
                    unit="frame",
                    bar_format='{l_bar}{bar:30}{r_bar}',
                    colour='green',
                    ncols=100
                )
            except ImportError:
                pbar = None
                print(f"   Processing {total_frames} frames...")
            
            frame_count = 0
            last_processed_frame = None
            last_mask = None
            text_detected_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame for speed
                if frame_count % sample_rate == 0:
                    # Convert BGR to RGB for PaddleOCR processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Remove text
                    processed_rgb = self.remove_text_from_image(frame_rgb)
                    # Check if text was actually removed (frames differ)
                    if not np.array_equal(frame_rgb, processed_rgb):
                        text_detected_count += 1
                    # Convert back to BGR for OpenCV
                    processed_frame = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                    last_processed_frame = processed_frame
                else:
                    # For non-processed frames, use temporal consistency
                    # This reduces flickering between processed frames
                    if last_processed_frame is not None:
                        # Blend slightly with last processed to reduce artifacts
                        processed_frame = frame
                    else:
                        processed_frame = frame
                
                out.write(processed_frame)
                frame_count += 1
                
                # Update progress bar
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'text_found': text_detected_count,
                        'rate': f'{sample_rate}x'
                    })
                elif frame_count % 50 == 0:
                    # Fallback progress without tqdm
                    pct = int(frame_count / total_frames * 100)
                    print(f"\r   Progress: {pct}% ({frame_count}/{total_frames} frames, {text_detected_count} text found)", end="")
            
            # Close progress bar
            if pbar:
                pbar.close()
            else:
                print()  # New line after progress
            
            print(f"   ✅ OCR Complete: {frame_count} frames processed, text removed from {text_detected_count} frames")
            
            cap.release()
            out.release()
            
            # Re-add audio from original video using ffmpeg
            temp_output = output_path.parent / f"temp_{output_path.name}"
            shutil.move(str(output_path), str(temp_output))
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_output),
                "-i", str(input_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            temp_output.unlink(missing_ok=True)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Text removal complete: {output_path}")
                return output_path
            else:
                # If ffmpeg fails, return the video without audio
                if temp_output.exists():
                    shutil.move(str(temp_output), str(output_path))
                return output_path
            
        except Exception as e:
            logger.error(f"Video text removal failed: {e}")
            return None
    
    def process_scene(self, scene_path: Path) -> Path:
        """
        Process a scene video to remove text.
        Returns the processed path (or original if processing fails).
        """
        if not config.REMOVE_TEXT or not PADDLEOCR_AVAILABLE:
            return scene_path
        
        output_path = scene_path.parent / f"clean_{scene_path.name}"
        
        result = self.remove_text_from_video(scene_path, output_path)
        
        if result and result.exists():
            # Delete original, rename cleaned version
            scene_path.unlink(missing_ok=True)
            result.rename(scene_path)
            return scene_path
        
        return scene_path


# Global text remover instance
text_remover = TextRemover()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ScriptSegment:
    """A single segment of the video script"""
    text: str
    search_term: str
    emotion: str = "neutral"
    duration_hint: float = 5.0
    visual_style: str = "dynamic"
    
@dataclass
class VideoScript:
    """Complete video script with metadata"""
    title: str
    youtube_title: str
    description: str
    segments: List[ScriptSegment]
    hashtags: List[str]
    tags: List[str]
    topic: str
    duration_target: int
    hook_text: str = ""
    cta_text: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "youtube_title": self.youtube_title,
            "description": self.description,
            "segments": [
                {"text": s.text, "search_term": s.search_term, 
                 "emotion": s.emotion, "duration_hint": s.duration_hint}
                for s in self.segments
            ],
            "hashtags": self.hashtags,
            "tags": self.tags,
            "topic": self.topic,
            "duration_target": self.duration_target,
            "hook_text": self.hook_text,
            "cta_text": self.cta_text,
            "created_at": self.created_at
        }

@dataclass
class SceneInfo:
    """Information about a detected scene"""
    video_path: Path
    start_time: float
    end_time: float
    duration: float
    score: float = 0.0  # Quality/relevance score
    
@dataclass
class WordTimestamp:
    """Word with timing information"""
    word: str
    start: float
    end: float

@dataclass
class ReelMetadata:
    """Complete metadata for a generated reel"""
    video_path: str
    script: Dict
    audio_duration: float
    word_timestamps: List[Dict]
    scenes_used: List[Dict]
    thumbnail_path: str = ""
    youtube_video_id: str = ""
    youtube_url: str = ""
    uploaded: bool = False
    upload_date: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# GEMINI AI SCRIPT GENERATOR
# ============================================================================

VIRAL_SCRIPT_PROMPT = '''You are an expert viral short-form video scriptwriter. Create an engaging {duration}-second YouTube Short about: "{topic}"

🎯 MISSION: Create a captivating mini-story that hooks viewers instantly and keeps them watching till the end.

📋 CONTENT STRATEGY:
1. HOOK (First 3 seconds): Start with IMPACT - shocking stat, bold claim, or intriguing question
   ❌ NEVER use: "Did you know", "Let's explore", "In this video"
   ✅ DO use: Direct statements, surprising facts, emotional triggers

2. FLOW: Clear beginning → middle → end with building tension
   - Each segment adds NEW information
   - Use transitions: "But here's the twist", "What's even crazier"
   - Build to a satisfying conclusion

3. DATA-DRIVEN: Include real facts, numbers, statistics
   - Every claim should have specific data when possible
   - Be specific: "340 percent increase" not "big growth"

4. ENDING: Strong CTA + impact
   - "Follow for more" or "Like and subscribe"
   - Channel name : "Buzz Today"

⏱️ TIMING GUIDE:
- 30 sec = 5-6 segments (~75 words)
- 45 sec = 7-8 segments (~110 words)  
- 60 sec = 9-10 segments (~150 words)
- Each segment: 2-3 sentences, 12-18 words

📝 WRITING RULES:
- NO emojis (text will be spoken)
- Write numbers as words: "fifty percent" not "50%"
- Conversational, punchy tone
- Short sentences for impact
- loopable content
- tags and keywords for SEO (400-500 characters put as much as possible) , description (100-150 words) with SEO keywords, title (max 100 characters) with hook and viral tags
- Hook in first line for maximum retention
- dont use abbreviations and sentences like 'in this video', 'lets explore', 'lets dive in'

🎬 SEARCH TERMS: For each segment, provide a YouTube search term to find relevant B-roll footage
- Be specific: "person typing laptop office" not just "work"
- Include action words for dynamic footage
- Think about what visuals would enhance the message

📊 OUTPUT FORMAT (strict JSON):
{{
  "title": "Short internal title",
  "youtube_title": "Catchy title with hook + emojis #shorts #viral (max 100 chars)",
  "description": "SEO description (100-150 words) with keywords and CTA",
  "hook_text": "The attention-grabbing first line",
  "cta_text": "The call-to-action ending",
  "segments": [
    {{
      "text": "Segment narration text",
      "search_term": "YouTube search query for B-roll video",
      "emotion": "curious/shocked/excited/serious/inspiring",
      "duration_hint": 5.0
    }}
  ],
  "hashtags": ["shorts", "viral", "trending", "fyp", "topic-specific-tags"],
  "tags": ["seo", "keyword", "tags", "for", "youtube"]
}}

NOW CREATE: An engaging viral script for "{topic}" ({duration} seconds)'''


class GeminiScriptGenerator:
    """Generate viral video scripts using Gemini AI"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self._client = None
        self._initialized = False
    
    def initialize(self, api_key: str = None) -> bool:
        """Initialize Gemini client"""
        if self._initialized:
            return True
        
        if api_key:
            self.api_key = api_key
        
        if not self.api_key:
            # Try to load from key.txt
            key_file = Path("key.txt")
            if key_file.exists():
                content = key_file.read_text()
                match = re.search(r'geminikey\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    self.api_key = match.group(1)
        
        if not self.api_key or not GEMINI_AVAILABLE:
            logger.warning("Gemini API not available")
            return False
        
        try:
            self._client = genai.Client(api_key=self.api_key)
            self._initialized = True
            logger.info("Gemini AI initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            return False
    
    def get_trending_topics(self, count: int = 5, category: str = None) -> List[str]:
        """
        Get trending topics from Google Trends using Gemini AI.
        
        Args:
            count: Number of topics to generate
            category: Optional category filter (tech, entertainment, sports, etc.)
        
        Returns:
            List of trending topic strings
        """
        if not self.initialize():
            logger.warning("Gemini not available, using fallback trending topics")
            return self._fallback_trending_topics(count)
        
        category_filter = f" in the {category} category" if category else ""
        
        prompt = f'''You are a Google Trends expert. Generate {count} trending topics that are currently viral{category_filter}.

Rules:
- Topics must be CURRENTLY trending on Google, YouTube, Twitter/X, TikTok
- Focus on topics that would make great YouTube Shorts content
- Include a mix of: breaking news, viral moments, tech updates, pop culture, interesting facts
- Each topic should be specific enough to create engaging content
- Avoid controversial political topics or sensitive content
- Topics should be searchable and have lots of related video content available

Today's date: {datetime.now().strftime("%B %d, %Y")}

Return ONLY a JSON array of {count} topic strings, nothing else:
["topic 1", "topic 2", "topic 3", ...]

Examples of good topics:
- "iPhone 16 Pro Max camera features"
- "Taylor Swift Eras Tour highlights"
- "ChatGPT new features 2025"
- "Cristiano Ronaldo Al-Nassr goals"
- "SpaceX Starship launch update"
- "Netflix most watched shows this week"

Generate {count} CURRENT trending topics:'''
        
        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.9, "max_output_tokens": 1024}
            )
            
            if not response or not response.text:
                return self._fallback_trending_topics(count)
            
            # Parse JSON array
            json_match = re.search(r'\[([^\]]+)\]', response.text)
            if json_match:
                topics = json.loads(f'[{json_match.group(1)}]')
                logger.info(f"Generated {len(topics)} trending topics")
                return topics[:count]
            
            return self._fallback_trending_topics(count)
            
        except Exception as e:
            logger.error(f"Trending topics generation failed: {e}")
            return self._fallback_trending_topics(count)
    
    def _fallback_trending_topics(self, count: int) -> List[str]:
        """Fallback trending topics when Gemini is unavailable"""
        fallback_topics = [
            "AI technology breakthroughs 2025",
            "Cryptocurrency market update",
            "Viral TikTok trends today",
            "Latest smartphone features",
            "Celebrity news this week",
            "Gaming industry updates",
            "Health and fitness tips",
            "Money saving hacks",
            "Travel destinations 2025",
            "Science discoveries explained",
        ]
        return fallback_topics[:count]
    
    def generate_script(self, topic: str, duration: int = 45) -> Optional[VideoScript]:
        """Generate a video script for the given topic"""
        if not self.initialize():
            return self._fallback_script(topic, duration)
        
        prompt = VIRAL_SCRIPT_PROMPT.format(topic=topic, duration=duration)
        
        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.8, "max_output_tokens": 4096}
            )
            
            if not response or not response.text:
                return self._fallback_script(topic, duration)
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if not json_match:
                return self._fallback_script(topic, duration)
            
            json_str = json_match.group()
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            data = json.loads(json_str)
            
            # Convert to VideoScript
            segments = [
                ScriptSegment(
                    text=s.get("text", ""),
                    search_term=s.get("search_term", topic),
                    emotion=s.get("emotion", "neutral"),
                    duration_hint=s.get("duration_hint", 5.0)
                )
                for s in data.get("segments", [])
            ]
            
            # Fill tags to 500 characters with default keywords
            gemini_tags = data.get("tags", [])
            filled_tags = fill_tags_to_limit(gemini_tags, topic, limit=500)
            
            return VideoScript(
                title=data.get("title", topic),
                youtube_title=data.get("youtube_title", f"{topic} #shorts"),
                description=data.get("description", f"Learn about {topic}"),
                segments=segments,
                hashtags=data.get("hashtags", ["shorts", "viral"]),
                tags=filled_tags,
                topic=topic,
                duration_target=duration,
                hook_text=data.get("hook_text", ""),
                cta_text=data.get("cta_text", "Follow for more!")
            )
            
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return self._fallback_script(topic, duration)
    
    def _fallback_script(self, topic: str, duration: int) -> VideoScript:
        """Create a fallback script when Gemini is unavailable"""
        logger.info("Using fallback script generator")
        
        segments = [
            ScriptSegment(
                text=f"Here's something incredible about {topic} that most people don't know.",
                search_term=f"{topic} explained",
                emotion="curious",
                duration_hint=5.0
            ),
            ScriptSegment(
                text="The facts behind this will completely change how you think.",
                search_term=f"{topic} facts amazing",
                emotion="excited",
                duration_hint=5.0
            ),
            ScriptSegment(
                text="Studies show that understanding this can make a huge difference in your life.",
                search_term=f"{topic} benefits results",
                emotion="serious",
                duration_hint=5.0
            ),
            ScriptSegment(
                text="But here's what makes this truly remarkable.",
                search_term=f"{topic} incredible surprising",
                emotion="shocked",
                duration_hint=5.0
            ),
            ScriptSegment(
                text="Now you know the truth. Follow for more amazing content like this!",
                search_term=f"{topic} conclusion motivation",
                emotion="inspiring",
                duration_hint=5.0
            )
        ]
        
        # Fill tags to 500 characters with default keywords
        base_tags = [topic.lower(), "facts", "trending", "viral", "shorts"]
        filled_tags = fill_tags_to_limit(base_tags, topic, limit=500)
        
        return VideoScript(
            title=f"The Truth About {topic.title()}",
            youtube_title=f"🔥 {topic.title()} - What They Don't Tell You #shorts #viral",
            description=f"Discover the truth about {topic}. Amazing facts that will blow your mind!",
            segments=segments,
            hashtags=["shorts", "viral", "trending", "fyp", "facts"],
            tags=filled_tags,
            topic=topic,
            duration_target=duration,
            hook_text=f"Here's something incredible about {topic}",
            cta_text="Follow for more!"
        )


# ============================================================================
# YT-DLP VIDEO DOWNLOADER
# ============================================================================

class VideoDownloader:
    """Download videos from YouTube using yt-dlp"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or config.VIDEOS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def search_and_download(self, search_term: str, max_results: int = 3,
                           max_duration: int = 180) -> List[Path]:
        """
        Search YouTube and download videos matching the search term.
        Returns list of downloaded video paths.
        """
        downloaded = []
        
        # Clean search term for filename
        safe_term = re.sub(r'[^\w\s-]', '', search_term).replace(' ', '_')[:30]
        search_dir = self.output_dir / safe_term
        search_dir.mkdir(exist_ok=True)
        
        # yt-dlp options
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
            'outtmpl': str(search_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'ignoreerrors': True,
            'noplaylist': True,
            'max_downloads': max_results,
            'match_filter': lambda info: None if info.get('duration', 0) <= max_duration else "Video too long",
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        
        search_url = f"ytsearch{max_results}:{search_term} vertical short"
        
        try:
            logger.info(f"Searching: '{search_term}'")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(search_url, download=False)
                
                if not info or 'entries' not in info:
                    logger.warning(f"No results for: {search_term}")
                    return []
                
                # Download each video
                for entry in info['entries'][:max_results]:
                    if not entry:
                        continue
                    
                    video_id = entry.get('id', '')
                    video_url = entry.get('webpage_url', f"https://youtube.com/watch?v={video_id}")
                    
                    try:
                        ydl.download([video_url])
                        
                        # Find downloaded file
                        video_files = list(search_dir.glob(f"{video_id}.*"))
                        if video_files:
                            downloaded.append(video_files[0])
                            logger.info(f"Downloaded: {video_files[0].name}")
                    except Exception as e:
                        logger.warning(f"Download failed for {video_id}: {e}")
                        continue
            
            return downloaded
            
        except Exception as e:
            logger.error(f"Search/download error: {e}")
            return []
    
    def download_for_script(self, script: VideoScript, parallel: bool = True) -> Dict[str, List[Path]]:
        """
        Download videos for all segments in a script.
        Returns dict mapping search terms to downloaded video paths.
        Uses parallel downloads for faster processing.
        """
        results = {}
        
        # Get unique search terms
        search_terms = list(set(seg.search_term for seg in script.segments))
        
        logger.info(f"Downloading videos for {len(search_terms)} search terms...")
        
        use_parallel = parallel and config.PARALLEL_DOWNLOADS and len(search_terms) > 1
        
        if use_parallel:
            # Parallel download using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            
            def download_term(term):
                videos = self.search_and_download(
                    term, 
                    max_results=config.VIDEOS_PER_SEARCH,
                    max_duration=config.MAX_VIDEO_DURATION
                )
                if not videos:
                    # Try a simpler search
                    simple_term = ' '.join(term.split()[:3])
                    videos = self.search_and_download(simple_term, max_results=2)
                return term, videos
            
            # Use configured parallel workers
            max_workers = min(config.MAX_DOWNLOAD_WORKERS, len(search_terms))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(download_term, term): term for term in search_terms}
                
                pbar = tqdm(
                    total=len(futures),
                    desc="📥 Downloading Videos",
                    unit="term",
                    bar_format='{l_bar}{bar:30}{r_bar}',
                    colour='blue',
                    ncols=100
                )
                
                for future in as_completed(futures):
                    try:
                        term, videos = future.result()
                        results[term] = videos
                        pbar.set_postfix({'term': term[:20] + '...' if len(term) > 20 else term})
                    except Exception as e:
                        term = futures[future]
                        logger.warning(f"Download failed for '{term}': {e}")
                        results[term] = []
                    pbar.update(1)
                
                pbar.close()
        else:
            # Sequential download
            for term in search_terms:
                videos = self.search_and_download(
                    term, 
                    max_results=config.VIDEOS_PER_SEARCH,
                    max_duration=config.MAX_VIDEO_DURATION
                )
                results[term] = videos
                
                if not videos:
                    # Try a simpler search
                    simple_term = ' '.join(term.split()[:3])
                    videos = self.search_and_download(simple_term, max_results=2)
                    results[term] = videos
        
        return results


# ============================================================================
# PYSCENEDETECT SCENE ANALYZER
# ============================================================================

class SceneAnalyzer:
    """Analyze videos and extract the best scenes using PySceneDetect"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or config.SCENES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._scene_cache = {}  # Cache for scene detection results
    
    def detect_scenes(self, video_path: Path, 
                     threshold: float = None) -> List[SceneInfo]:
        """
        Detect scenes in a video using content-aware detection.
        Returns list of SceneInfo objects. Uses caching to avoid re-analysis.
        """
        threshold = threshold or config.SCENE_THRESHOLD
        
        # Check cache first
        cache_key = (str(video_path), threshold)
        if cache_key in self._scene_cache:
            logger.debug(f"Using cached scenes for {video_path.name}")
            return self._scene_cache[cache_key]
        
        try:
            scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
            
            scenes = []
            for start, end in scene_list:
                duration = end.get_seconds() - start.get_seconds()
                
                if duration >= config.MIN_SCENE_DURATION:
                    scenes.append(SceneInfo(
                        video_path=video_path,
                        start_time=start.get_seconds(),
                        end_time=end.get_seconds(),
                        duration=duration
                    ))
            
            logger.info(f"Detected {len(scenes)} scenes in {video_path.name}")
            
            # Cache the result
            self._scene_cache[cache_key] = scenes
            return scenes
            return scenes
            
        except Exception as e:
            logger.error(f"Scene detection failed for {video_path}: {e}")
            # Return whole video as single scene
            try:
                clip = VideoFileClip(str(video_path))
                duration = clip.duration
                clip.close()
                return [SceneInfo(
                    video_path=video_path,
                    start_time=0,
                    end_time=duration,
                    duration=duration
                )]
            except:
                return []
    
    def analyze_videos(self, videos: Dict[str, List[Path]], parallel: bool = True) -> Dict[str, List[SceneInfo]]:
        """
        Analyze all downloaded videos and detect scenes.
        Returns dict mapping search terms to scene lists.
        Uses parallel processing for faster analysis.
        """
        all_scenes = {}
        
        # Flatten all video paths with their terms
        video_tasks = []
        for term, video_paths in videos.items():
            for video_path in video_paths:
                video_tasks.append((term, video_path))
        
        use_parallel = parallel and config.PARALLEL_SCENE_DETECTION and len(video_tasks) > 1
        
        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            
            # Initialize results dict
            for term in videos.keys():
                all_scenes[term] = []
            
            # Use configured parallel workers
            max_workers = min(config.MAX_SCENE_WORKERS, len(video_tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.detect_scenes, vp): (term, vp) 
                          for term, vp in video_tasks}
                
                pbar = tqdm(
                    total=len(futures),
                    desc="🎬 Detecting Scenes",
                    unit="video",
                    bar_format='{l_bar}{bar:30}{r_bar}',
                    colour='magenta',
                    ncols=100
                )
                
                for future in as_completed(futures):
                    term, video_path = futures[future]
                    try:
                        scenes = future.result()
                        all_scenes[term].extend(scenes)
                        pbar.set_postfix({'scenes': len(scenes)})
                    except Exception as e:
                        logger.warning(f"Scene detection failed for {video_path.name}: {e}")
                    pbar.update(1)
                
                pbar.close()
            
            # Sort scenes by duration preference
            for term in all_scenes:
                all_scenes[term].sort(key=lambda s: abs(s.duration - 5.0))
        else:
            # Sequential processing
            for term, video_paths in videos.items():
                term_scenes = []
                
                for video_path in video_paths:
                    scenes = self.detect_scenes(video_path)
                    term_scenes.extend(scenes)
                
                # Sort by duration (prefer medium-length scenes)
                term_scenes.sort(key=lambda s: abs(s.duration - 5.0))
                all_scenes[term] = term_scenes
        
        return all_scenes
    
    def extract_scene(self, scene: SceneInfo, output_path: Path) -> Optional[Path]:
        """Extract a scene from a video and save to file"""
        try:
            clip = VideoFileClip(str(scene.video_path))
            
            # Extract subclip
            subclip = clip.subclipped(scene.start_time, scene.end_time)
            
            # Write to file
            subclip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=False,
                preset='fast',
                logger=None
            )
            
            subclip.close()
            clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Scene extraction failed: {e}")
            return None
    
    def select_best_scenes(self, scenes: Dict[str, List[SceneInfo]], 
                          segment_durations: List[float]) -> List[SceneInfo]:
        """
        Select the best scenes to match the required segment durations.
        Uses intelligent matching to find scenes that fit well.
        """
        selected = []
        used_scenes = set()
        
        # Get all available scenes in a flat list with their search terms
        scene_pool = []
        for term, scene_list in scenes.items():
            for scene in scene_list:
                scene_pool.append((term, scene))
        
        for duration in segment_durations:
            best_scene = None
            best_score = float('inf')
            
            for term, scene in scene_pool:
                # Skip if already used
                scene_key = (scene.video_path, scene.start_time)
                if scene_key in used_scenes:
                    continue
                
                # Score based on how close duration is to target
                duration_diff = abs(scene.duration - duration)
                score = duration_diff
                
                # Prefer scenes that are slightly longer (can trim)
                if scene.duration >= duration:
                    score *= 0.8
                
                if score < best_score:
                    best_score = score
                    best_scene = scene
            
            if best_scene:
                selected.append(best_scene)
                used_scenes.add((best_scene.video_path, best_scene.start_time))
            else:
                # Use any available scene
                for term, scene in scene_pool:
                    scene_key = (scene.video_path, scene.start_time)
                    if scene_key not in used_scenes:
                        selected.append(scene)
                        used_scenes.add(scene_key)
                        break
        
        return selected


# ============================================================================
# TEXT-TO-SPEECH ENGINE (Kokoro TTS)
# ============================================================================

class KokoroTTSEngine:
    """Text-to-speech engine using Kokoro TTS (same as main.py)"""
    
    def __init__(self, voice: str = None, speed: float = None):
        self.voice = voice or config.TTS_VOICE
        self.speed = speed or config.TTS_SPEED
        self._bin = self._find_bin()
    
    def _find_bin(self) -> str:
        """Find kokoro-tts binary"""
        paths = [
            "/opt/anaconda3/envs/youtube-env/bin/kokoro-tts",
            str(Path("./venv/bin/kokoro-tts").absolute()),
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
        if not config.KOKORO_MODEL.exists():
            logger.warning(f"Kokoro model not found: {config.KOKORO_MODEL}")
            return False
        if not config.KOKORO_VOICES.exists():
            logger.warning(f"Kokoro voices not found: {config.KOKORO_VOICES}")
            return False
        logger.info("Kokoro TTS ready!")
        return True
    
    def generate(self, text: str, output_path: Path) -> Optional[Path]:
        """Generate audio for the given text using Kokoro TTS"""
        # Expand abbreviations for proper pronunciation
        text = self._expand_abbreviations(text)
        
        if not config.KOKORO_MODEL.exists():
            logger.warning("Kokoro model not found, using fallback")
            return self._fallback_audio(text, output_path)
        
        tmp = output_path.parent / f"tmp_{random.randint(1000,9999)}.txt"
        tmp.write_text(text)
        
        try:
            cmd = [
                self._bin, str(tmp), str(output_path),
                "--speed", str(self.speed),
                "--voice", self.voice,
                "--model", str(config.KOKORO_MODEL),
                "--voices", str(config.KOKORO_VOICES)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                logger.warning(f"TTS failed: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                return self._fallback_audio(text, output_path)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return self._fallback_audio(text, output_path)
        finally:
            tmp.unlink(missing_ok=True)
    
    def _fallback_audio(self, text: str, output_path: Path) -> Optional[Path]:
        """Create placeholder audio if TTS fails"""
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
    
    def generate_for_script(self, script: VideoScript, output_dir: Path) -> List[Tuple[Path, float]]:
        """
        Generate audio for all segments in a script.
        Returns list of (audio_path, duration) tuples.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        
        # Progress bar for voice generation
        try:
            from tqdm import tqdm
            pbar = tqdm(
                total=len(script.segments),
                desc="🎤 Generating Voice",
                unit="segment",
                bar_format='{l_bar}{bar:30}{r_bar}',
                colour='cyan',
                ncols=100
            )
        except ImportError:
            pbar = None
        
        for i, segment in enumerate(script.segments):
            audio_path = output_dir / f"segment_{i:02d}.wav"  # Kokoro outputs WAV
            
            if self.generate(segment.text, audio_path):
                duration = self._get_audio_duration(audio_path)
                results.append((audio_path, duration))
                if pbar:
                    pbar.set_postfix({'duration': f'{duration:.1f}s'})
            else:
                results.append((None, segment.duration_hint))
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        return results
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations for proper TTS pronunciation"""
        # Currency with magnitude (must come before simple currency)
        text = re.sub(r'([\$€£])(\d+(?:\.\d+)?)\s*([KkMmBbTt])\b', 
                     lambda m: f"{m.group(2)} {self._magnitude(m.group(3))} {self._currency(m.group(1))}", text)
        text = re.sub(r'\$(\d+(?:\.\d+)?)', r'\1 dollars', text)
        text = re.sub(r'€(\d+(?:\.\d+)?)', r'\1 euros', text)
        text = re.sub(r'£(\d+(?:\.\d+)?)', r'\1 pounds', text)
        
        # Percentages
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
        
        # Common abbreviations
        replacements = {
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
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Numbers with K/M/B suffix (not currency)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*([KkMmBbTt])\b(?![a-zA-Z])', 
                     lambda m: f"{m.group(1)} {self._magnitude(m.group(2))}", text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _magnitude(self, char: str) -> str:
        """Convert magnitude character to word"""
        magnitudes = {'K': 'thousand', 'M': 'million', 'B': 'billion', 'T': 'trillion'}
        return magnitudes.get(char.upper(), '')
    
    def _currency(self, symbol: str) -> str:
        """Convert currency symbol to word"""
        currencies = {'$': 'dollars', '€': 'euros', '£': 'pounds'}
        return currencies.get(symbol, 'units')
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 5.0


# ============================================================================
# WHISPER TRANSCRIBER (Word-Level Timestamps)
# ============================================================================

class WhisperTranscriber:
    """Transcribe audio with word-level timestamps using Whisper"""
    
    def __init__(self, model_size: str = "turbo"):
        self.model_size = model_size
        self._model = None
    
    @property
    def model(self):
        """Lazy load Whisper model"""
        if self._model is None:
            logger.info(f"Loading Whisper ({self.model_size}) model...")
            self._model = whisper.load_model(self.model_size)
        return self._model
    
    def transcribe(self, audio_path: Path) -> List[WordTimestamp]:
        """
        Transcribe audio and return word-level timestamps.
        """
        try:
            result = self.model.transcribe(
                str(audio_path),
                word_timestamps=True
            )
            
            words = []
            for segment in result.get('segments', []):
                for word_info in segment.get('words', []):
                    words.append(WordTimestamp(
                        word=word_info['word'].strip(),
                        start=word_info['start'],
                        end=word_info['end']
                    ))
            
            logger.info(f"Transcribed {len(words)} words from {audio_path.name}")
            return words
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []
    
    def transcribe_multiple(self, audio_files: List[Path], parallel: bool = True) -> List[List[WordTimestamp]]:
        """Transcribe multiple audio files with optional parallel processing"""
        use_parallel = parallel and config.PARALLEL_TRANSCRIPTION and len(audio_files) > 1
        if not use_parallel:
            # Sequential processing
            results = []
            for audio_path in audio_files:
                if audio_path and audio_path.exists():
                    words = self.transcribe(audio_path)
                    results.append(words)
                else:
                    results.append([])
            return results
        
        # Parallel transcription with progress bar
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        # Pre-load model in main thread to avoid concurrent loading
        _ = self.model
        
        # Create indexed tasks to maintain order
        tasks = [(i, path) for i, path in enumerate(audio_files) if path and path.exists()]
        results = [[] for _ in audio_files]
        
        if not tasks:
            return results
        
        # Use configured workers for Whisper (GPU memory constrained)
        max_workers = min(config.MAX_TRANSCRIBE_WORKERS, len(tasks))
        
        pbar = tqdm(
            total=len(tasks),
            desc="📝 Transcribing Audio",
            unit="file",
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='yellow',
            ncols=100
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.transcribe, path): (i, path) 
                      for i, path in tasks}
            
            for future in as_completed(futures):
                idx, path = futures[future]
                try:
                    words = future.result()
                    results[idx] = words
                    pbar.set_postfix({'words': len(words)})
                except Exception as e:
                    logger.warning(f"Transcription failed for {path.name}: {e}")
                    results[idx] = []
                pbar.update(1)
        
        pbar.close()
        return results


# ============================================================================
# ANIMATED CAPTIONS RENDERER
# ============================================================================

class CaptionRenderer:
    """Render animated word-by-word captions"""
    
    # Styling
    FONT_SIZE = 80
    FONT_COLOR = (255, 255, 0)  # Yellow
    STROKE_COLOR = (0, 0, 0)    # Black outline
    STROKE_WIDTH = 4
    BG_COLOR = (0, 0, 0, 180)   # Semi-transparent black
    POSITION_Y = 0.72           # 72% from top
    
    def __init__(self, width: int = None, height: int = None):
        self.width = width or config.VIDEO_WIDTH
        self.height = height or config.VIDEO_HEIGHT
        self.font = self._load_font()
    
    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Load a bold font for captions"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
        for path in font_paths:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, self.FONT_SIZE)
                except:
                    continue
        return ImageFont.load_default()
    
    def create_caption_frame(self, words: List[WordTimestamp], 
                            current_time: float) -> np.ndarray:
        """Create a frame with the current word highlighted"""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Find current word
        current_word = None
        for word in words:
            if word.start <= current_time <= word.end:
                current_word = word
                break
        
        if not current_word:
            return np.array(img)
        
        # Draw the word
        display_text = current_word.word.upper().strip()
        
        # Get text size
        bbox = draw.textbbox((0, 0), display_text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center position
        x = (self.width - text_width) // 2
        y = int(self.height * self.POSITION_Y) - text_height // 2
        
        # Draw background pill
        padding_x, padding_y = 40, 20
        bg_rect = [x - padding_x, y - padding_y, 
                   x + text_width + padding_x, y + text_height + padding_y]
        draw.rounded_rectangle(bg_rect, radius=15, fill=self.BG_COLOR)
        
        # Draw text with outline
        for dx in range(-self.STROKE_WIDTH, self.STROKE_WIDTH + 1):
            for dy in range(-self.STROKE_WIDTH, self.STROKE_WIDTH + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), display_text, 
                             font=self.font, fill=self.STROKE_COLOR)
        
        # Draw main text
        draw.text((x, y), display_text, font=self.font, fill=self.FONT_COLOR)
        
        return np.array(img)
    
    def create_caption_clip(self, words: List[WordTimestamp], 
                           duration: float) -> VideoClip:
        """Create a video clip with animated captions"""
        def make_frame(t):
            frame = self.create_caption_frame(words, t)
            return frame[:, :, :3]
        
        def make_mask(t):
            frame = self.create_caption_frame(words, t)
            return frame[:, :, 3] / 255.0
        
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.with_fps(config.FPS)
        
        mask = VideoClip(make_mask, duration=duration, is_mask=True)
        mask = mask.with_fps(config.FPS)
        
        return clip.with_mask(mask)


# ============================================================================
# VIDEO COMPOSER
# ============================================================================

class VideoComposer:
    """Compose final video from scenes, audio, and captions"""
    
    def __init__(self):
        self.caption_renderer = CaptionRenderer()
    
    def create_video(self, scenes: List[SceneInfo], 
                    audio_files: List[Tuple[Path, float]],
                    word_timestamps: List[List[WordTimestamp]],
                    output_path: Path,
                    add_captions: bool = True) -> Optional[Path]:
        """
        Create the final video by combining scenes, audio, and captions.
        """
        logger.info("Composing final video...")
        
        clips = []
        current_time = 0
        
        # Progress bar for video composition
        try:
            from tqdm import tqdm
            pbar = tqdm(
                total=len(audio_files),
                desc="🎬 Composing Video",
                unit="segment",
                bar_format='{l_bar}{bar:30}{r_bar}',
                colour='magenta',
                ncols=100
            )
        except ImportError:
            pbar = None
        
        for i, ((audio_path, audio_duration), scene, words) in enumerate(
            zip(audio_files, scenes, word_timestamps)):
            
            if audio_path is None:
                if pbar:
                    pbar.update(1)
                continue
            
            # Load and process video scene
            try:
                video_clip = self._prepare_scene_clip(scene, audio_duration)
                
                # Add audio
                audio_clip = AudioFileClip(str(audio_path))
                video_clip = video_clip.with_audio(audio_clip)
                
                # Add captions if enabled
                if add_captions and words:
                    # Adjust word timestamps for this segment
                    adjusted_words = [
                        WordTimestamp(w.word, w.start, w.end) 
                        for w in words
                    ]
                    caption_clip = self.caption_renderer.create_caption_clip(
                        adjusted_words, audio_duration
                    )
                    video_clip = CompositeVideoClip([video_clip, caption_clip])
                
                clips.append(video_clip)
                current_time += audio_duration
                
                if pbar:
                    pbar.set_postfix({'duration': f'{current_time:.1f}s'})
                
            except Exception as e:
                logger.error(f"Error processing segment {i}: {e}")
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        if not clips:
            logger.error("No clips to compose!")
            return None
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Export with optimized settings
        logger.info("Exporting final video (optimized)...")
        try:
            # Use faster preset and more threads for speed
            # 'fast' preset is ~2x faster than 'medium' with minimal quality loss
            import os
            cpu_count = os.cpu_count() or 4
            optimal_threads = min(cpu_count, 8)  # Cap at 8 threads
            
            final_video.write_videofile(
                str(output_path),
                fps=config.FPS,
                codec='libx264',
                audio_codec='aac',
                preset='fast',  # Changed from 'medium' for faster encoding
                threads=optimal_threads,  # Use more CPU cores
                ffmpeg_params=[
                    '-movflags', '+faststart',  # Web optimization
                    '-pix_fmt', 'yuv420p',  # Compatibility
                ],
                logger=None
            )
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
        finally:
            # Cleanup
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            try:
                final_video.close()
            except:
                pass
        
        # Add background music and effects
        output_path = self._add_background_audio(output_path)
        
        return output_path
    
    def _prepare_scene_clip(self, scene: SceneInfo, target_duration: float, 
                            remove_text: bool = None) -> VideoClip:
        """Prepare a scene clip - remove text, resize, crop, and adjust duration"""
        remove_text = remove_text if remove_text is not None else config.REMOVE_TEXT
        
        clip = VideoFileClip(str(scene.video_path))
        
        # Get subclip
        start = scene.start_time
        end = min(scene.end_time, start + target_duration + 1)
        clip = clip.subclipped(start, end)
        
        # Remove text from video frames if enabled
        if remove_text and PADDLEOCR_AVAILABLE:
            clip = self._remove_text_from_clip(clip)
        
        # Resize to fit 9:16
        clip = self._fit_to_vertical(clip)
        
        # Adjust duration
        if clip.duration < target_duration:
            # Loop if too short
            loops = int(target_duration / clip.duration) + 1
            clips = [clip] * loops
            clip = concatenate_videoclips(clips)
        
        clip = clip.subclipped(0, target_duration)
        
        return clip
    
    def _remove_text_from_clip(self, clip: VideoClip) -> VideoClip:
        """
        Remove text from video clip using PaddleOCR.
        Processes frames and returns a new clip with text removed.
        """
        if not PADDLEOCR_AVAILABLE:
            return clip
        
        sample_rate = config.TEXT_REMOVAL_SAMPLE_RATE
        frame_cache = {}  # Cache processed frames
        
        def process_frame(get_frame, t):
            """Process a single frame to remove text"""
            # Round time to reduce unique frames to process
            frame_key = round(t * clip.fps) // sample_rate
            
            if frame_key in frame_cache:
                return frame_cache[frame_key]
            
            frame = get_frame(t)
            
            # Only process every Nth frame for speed
            if round(t * clip.fps) % sample_rate == 0:
                try:
                    processed = text_remover.remove_text_from_image(frame)
                    frame_cache[frame_key] = processed
                    return processed
                except Exception as e:
                    logger.warning(f"Frame text removal failed: {e}")
                    return frame
            
            return frame
        
        # Create new clip with processed frames
        processed_clip = clip.transform(
            lambda get_frame, t: process_frame(get_frame, t)
        )
        
        return processed_clip
    
    def _fit_to_vertical(self, clip: VideoClip) -> VideoClip:
        """Resize and crop video to 9:16 aspect ratio"""
        target_w = config.VIDEO_WIDTH
        target_h = config.VIDEO_HEIGHT
        target_aspect = target_w / target_h
        
        clip_aspect = clip.w / clip.h
        
        if clip_aspect > target_aspect:
            # Video is wider - scale to height and crop sides
            new_h = target_h
            new_w = int(clip.w * (target_h / clip.h))
            clip = clip.resized(height=target_h)
            x_center = clip.w // 2
            clip = clip.cropped(
                x1=x_center - target_w // 2,
                x2=x_center + target_w // 2,
                y1=0, y2=target_h
            )
        else:
            # Video is taller - scale to width and crop top/bottom
            new_w = target_w
            clip = clip.resized(width=target_w)
            # Favor upper portion for faces
            y_start = int((clip.h - target_h) * 0.3)
            clip = clip.cropped(
                x1=0, x2=target_w,
                y1=y_start, y2=y_start + target_h
            )
        
        return clip
    
    def _add_background_audio(self, video_path: Path) -> Path:
        """Add background music to the video"""
        music_path = config.get_music()
        if not music_path:
            return video_path
        
        output_path = video_path.parent / f"final_{video_path.name}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(music_path),
            "-filter_complex",
            f"[0:a]volume=1.0[voice];"
            f"[1:a]volume={config.MUSIC_VOLUME},aloop=loop=-1:size=2e9[music];"
            f"[voice][music]amix=inputs=2:duration=first:weights=1 0.3[out]",
            "-map", "0:v", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0 and output_path.exists():
            video_path.unlink()
            return output_path
        
        return video_path


# ============================================================================
# THUMBNAIL GENERATOR
# ============================================================================

class ThumbnailGenerator:
    """Generate attractive thumbnails using g4f AI"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or config.THUMBNAILS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
    
    def generate(self, topic: str, title: str, 
                video_path: Path = None) -> Optional[Path]:
        """Generate a thumbnail for the video"""
        output_path = self.output_dir / f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Try AI generation first
        if G4F_AVAILABLE:
            try:
                if self._client is None:
                    self._client = G4FClient()
                
                prompt = self._create_prompt(topic, title)
                
                response = self._client.images.generate(
                    model="sdxl",
                    prompt=prompt,
                    response_format="url"
                )
                
                if response.data:
                    image_url = response.data[0].url
                    img_response = requests.get(image_url, timeout=30)
                    
                    with open(output_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    # Resize to YouTube thumbnail size
                    self._resize_thumbnail(output_path)
                    
                    logger.info(f"Generated AI thumbnail: {output_path}")
                    return output_path
                    
            except Exception as e:
                logger.warning(f"AI thumbnail generation failed: {e}")
        
        # Fallback: extract frame from video
        if video_path and video_path.exists():
            return self._extract_frame(video_path, output_path)
        
        return None
    
    def _create_prompt(self, topic: str, title: str) -> str:
        """Create a prompt for thumbnail generation"""
        return f"""Create a professional YouTube thumbnail for a video about {topic}.
        Style: Bold, vibrant colors, high contrast, eye-catching
        Mood: Exciting and engaging
        Format: 16:9 horizontal, clean composition
        NO text in the image
        Subject: Visual representation of {topic}
        Quality: High resolution, professional photography style"""
    
    def _resize_thumbnail(self, path: Path):
        """Resize to YouTube thumbnail dimensions"""
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                img = img.resize((1280, 720), Image.LANCZOS)
                img.save(path, 'JPEG', quality=95)
        except Exception as e:
            logger.warning(f"Thumbnail resize failed: {e}")
    
    def _extract_frame(self, video_path: Path, output_path: Path) -> Optional[Path]:
        """Extract a frame from video as thumbnail"""
        try:
            clip = VideoFileClip(str(video_path))
            frame_time = min(2.0, clip.duration / 2)
            frame = clip.get_frame(frame_time)
            clip.close()
            
            img = Image.fromarray(frame)
            
            # Crop to 16:9 if needed
            if img.height > img.width * 9 / 16:
                new_h = int(img.width * 9 / 16)
                top = (img.height - new_h) // 2
                img = img.crop((0, top, img.width, top + new_h))
            
            img = img.resize((1280, 720), Image.LANCZOS)
            img.save(output_path, 'JPEG', quality=95)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None


# ============================================================================
# YOUTUBE UPLOADER
# ============================================================================

class YouTubeUploader:
    """Upload videos to YouTube with multi-credential support"""
    
    def __init__(self):
        self.credentials_files = self._get_credentials()
    
    def _get_credentials(self) -> List[Path]:
        """Find OAuth credential files"""
        cred_dir = config.GOOGLE_CONSOLE_DIR
        if not cred_dir.exists():
            return []
        
        creds = list(cred_dir.glob("client_secret*.json"))
        creds.extend([f for f in cred_dir.glob("*.json") 
                      if "token" not in f.name.lower()])
        return creds
    
    def upload(self, video_path: Path, script: VideoScript,
              thumbnail_path: Path = None) -> Tuple[bool, str]:
        """Upload video to YouTube"""
        if not YOUTUBE_API_AVAILABLE:
            return False, "YouTube API not available"
        
        if not self.credentials_files:
            return False, "No credentials found"
        
        # Prepare metadata
        title = script.youtube_title[:100]
        description = script.description[:5000]
        
        # Sanitize tags for YouTube (max 30 tags, each max 30 chars)
        tags = []
        for tag in script.tags[:30]:
            sanitized = sanitize_tag(tag)
            if sanitized and sanitized not in tags:  # Avoid duplicates
                tags.append(sanitized)
        
        # Log tags info
        tags_str = ", ".join(tags)
        logger.info(f"Uploading: {title}")
        logger.info(f"Tags ({len(tags_str)} chars, {len(tags)} tags): {tags_str[:100]}...")
        
        for cred_file in self.credentials_files:
            try:
                service = self._get_service(cred_file)
                if not service:
                    continue
                
                # Upload
                body = {
                    'snippet': {
                        'title': title,
                        'description': description,
                        'tags': tags,
                        'categoryId': '22'
                    },
                    'status': {
                        'privacyStatus': 'public',
                        'selfDeclaredMadeForKids': False
                    }
                }
                
                media = MediaFileUpload(
                    str(video_path),
                    mimetype='video/mp4',
                    resumable=True
                )
                
                request = service.videos().insert(
                    part=','.join(body.keys()),
                    body=body,
                    media_body=media
                )
                
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        logger.info(f"Upload progress: {int(status.progress() * 100)}%")
                
                video_id = response.get('id', '')
                
                # Upload thumbnail
                if thumbnail_path and thumbnail_path.exists():
                    try:
                        logger.info(f"Uploading thumbnail: {thumbnail_path}")
                        service.thumbnails().set(
                            videoId=video_id,
                            media_body=MediaFileUpload(
                                str(thumbnail_path),
                                mimetype='image/jpeg'
                            )
                        ).execute()
                        logger.info("✅ Thumbnail uploaded successfully")
                    except HttpError as e:
                        logger.warning(f"Thumbnail upload failed (HTTP): {e}")
                        # Try with different mimetype
                        try:
                            service.thumbnails().set(
                                videoId=video_id,
                                media_body=MediaFileUpload(
                                    str(thumbnail_path),
                                    mimetype='image/png'
                                )
                            ).execute()
                            logger.info("✅ Thumbnail uploaded (PNG format)")
                        except Exception as e2:
                            logger.warning(f"Thumbnail retry failed: {e2}")
                    except Exception as e:
                        logger.warning(f"Thumbnail upload failed: {e}")
                else:
                    logger.warning(f"Thumbnail not found or path invalid: {thumbnail_path}")
                
                return True, video_id
                
            except HttpError as e:
                logger.warning(f"Upload failed with {cred_file.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Upload error: {e}")
                continue
        
        return False, "All credentials failed"
    
    def _get_service(self, cred_file: Path):
        """Get authenticated YouTube service"""
        token_file = cred_file.parent / f"token_{cred_file.stem}.pickle"
        creds = None
        
        if token_file.exists():
            with open(token_file, 'rb') as f:
                creds = pickle.load(f)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(cred_file), config.YOUTUBE_SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            with open(token_file, 'wb') as f:
                pickle.dump(creds, f)
        
        return build('youtube', 'v3', credentials=creds)


# ============================================================================
# MAIN REEL GENERATOR
# ============================================================================

class ReelGenerator:
    """Main orchestrator for the reel generation pipeline"""
    
    def __init__(self, gemini_api_key: str = None):
        config.setup_dirs()
        
        self.script_gen = GeminiScriptGenerator(gemini_api_key)
        self.downloader = VideoDownloader()
        self.scene_analyzer = SceneAnalyzer()
        self.tts = KokoroTTSEngine()  # Use Kokoro TTS instead of Edge TTS
        self.transcriber = WhisperTranscriber()
        self.composer = VideoComposer()
        self.thumbnail_gen = ThumbnailGenerator()
        self.uploader = YouTubeUploader()
    
    def _select_or_generate_thumbnail(self, topic: str, title: str, 
                                       video_path: Path) -> Optional[Path]:
        """
        Let user select a thumbnail from the thumbnails folder or generate a new one.
        
        Args:
            topic: The video topic
            title: The video title
            video_path: Path to the generated video
        
        Returns:
            Path to the selected/generated thumbnail
        """
        thumbnails_dir = config.THUMBNAILS_DIR
        
        # Get list of existing thumbnails
        existing_thumbs = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            existing_thumbs.extend(thumbnails_dir.glob(ext))
        
        # Sort by modification time (newest first)
        existing_thumbs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print("\n   📁 Thumbnails folder:", thumbnails_dir)
        print("\n   Choose an option:")
        print("   [1] Generate new AI thumbnail")
        print("   [2] Extract frame from video")
        
        if existing_thumbs:
            print(f"   [3] Select from existing thumbnails ({len(existing_thumbs)} available)")
            print("\n   📷 Recent thumbnails:")
            for i, thumb in enumerate(existing_thumbs[:10], 1):
                size_kb = thumb.stat().st_size / 1024
                mtime = datetime.fromtimestamp(thumb.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                print(f"      {i:2d}. {thumb.name} ({size_kb:.1f} KB, {mtime})")
            if len(existing_thumbs) > 10:
                print(f"      ... and {len(existing_thumbs) - 10} more")
        
        print("\n   [s] Skip thumbnail")
        
        choice = input("\n   Enter choice [1/2/3/s] (default: 1): ").strip().lower()
        
        if choice == 's':
            print("   ⏭️  Skipping thumbnail")
            return None
        
        elif choice == '2':
            # Extract frame from video
            print("   🎬 Extracting frame from video...")
            output_path = thumbnails_dir / f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            return self.thumbnail_gen._extract_frame(video_path, output_path)
        
        elif choice == '3' and existing_thumbs:
            # Select from existing
            print(f"\n   Enter thumbnail number (1-{min(len(existing_thumbs), 10)}) or filename:")
            selection = input("   > ").strip()
            
            try:
                # Try as number first
                idx = int(selection) - 1
                if 0 <= idx < len(existing_thumbs):
                    selected = existing_thumbs[idx]
                    print(f"   ✓ Selected: {selected.name}")
                    return selected
            except ValueError:
                # Try as filename
                for thumb in existing_thumbs:
                    if selection.lower() in thumb.name.lower():
                        print(f"   ✓ Selected: {thumb.name}")
                        return thumb
            
            print("   ⚠️  Invalid selection, generating new thumbnail...")
        
        # Default: Generate new AI thumbnail
        print("   🎨 Generating AI thumbnail...")
        return self.thumbnail_gen.generate(topic, title, video_path)
    
    def generate(self, topic: str, duration: int = 45,
                auto_upload: bool = False) -> Optional[ReelMetadata]:
        """
        Generate a complete reel from topic to final video.
        
        Args:
            topic: The video topic
            duration: Target duration in seconds
            auto_upload: Whether to upload to YouTube
        
        Returns:
            ReelMetadata with all video information
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_')[:30]
        video_name = f"reel_{timestamp}_{safe_topic}"
        
        print("\n" + "="*60)
        print(f"🎬 GENERATING REEL: {topic}")
        print(f"⏱️  Target Duration: {duration} seconds")
        print("="*60)
        
        # Step 1: Generate Script
        print("\n📝 Step 1: Generating Script...")
        script = self.script_gen.generate_script(topic, duration)
        if not script:
            logger.error("Script generation failed!")
            return None
        
        print(f"   ✓ Generated {len(script.segments)} segments")
        
        # Save script
        script_path = config.SCRIPTS_DIR / f"{video_name}.json"
        with open(script_path, 'w') as f:
            json.dump(script.to_dict(), f, indent=2)
        
        # Step 2: Download Videos
        print("\n📥 Step 2: Downloading Source Videos...")
        videos = self.downloader.download_for_script(script)
        total_videos = sum(len(v) for v in videos.values())
        print(f"   ✓ Downloaded {total_videos} videos")
        
        # Step 3: Analyze Scenes
        print("\n🎬 Step 3: Analyzing Scenes with PySceneDetect...")
        all_scenes = self.scene_analyzer.analyze_videos(videos)
        total_scenes = sum(len(s) for s in all_scenes.values())
        print(f"   ✓ Detected {total_scenes} scenes")
        
        # Step 4: Generate Audio
        print("\n🎤 Step 4: Generating Voice Narration...")
        audio_dir = config.AUDIO_DIR / video_name
        audio_results = self.tts.generate_for_script(script, audio_dir)
        print(f"   ✓ Generated {len(audio_results)} audio segments")
        
        # Step 5: Transcribe for Word Timestamps
        print("\n📝 Step 5: Transcribing for Word-Level Captions...")
        audio_paths = [ar[0] for ar in audio_results if ar[0]]
        word_timestamps = self.transcriber.transcribe_multiple(audio_paths)
        total_words = sum(len(wt) for wt in word_timestamps)
        print(f"   ✓ Transcribed {total_words} words")
        
        # Step 6: Select Best Scenes
        print("\n🎯 Step 6: Selecting Best Scenes...")
        segment_durations = [ar[1] for ar in audio_results]
        selected_scenes = self.scene_analyzer.select_best_scenes(
            all_scenes, segment_durations
        )
        print(f"   ✓ Selected {len(selected_scenes)} scenes")
        
        # Step 7: Compose Video
        print("\n🎥 Step 7: Composing Final Video...")
        output_path = config.OUTPUT_DIR / f"{video_name}.mp4"
        final_video = self.composer.create_video(
            selected_scenes,
            audio_results,
            word_timestamps,
            output_path,
            add_captions=True
        )
        
        if not final_video:
            logger.error("Video composition failed!")
            return None
        
        print(f"   ✓ Video saved: {final_video}")
        
        # Step 8: Select or Generate Thumbnail
        print("\n🖼️  Step 8: Thumbnail Selection...")
        thumbnail = self._select_or_generate_thumbnail(topic, script.title, final_video)
        if thumbnail:
            print(f"   ✓ Thumbnail ready: {thumbnail}")
        
        # Create metadata
        metadata = ReelMetadata(
            video_path=str(final_video),
            script=script.to_dict(),
            audio_duration=sum(segment_durations),
            word_timestamps=[
                [{"word": w.word, "start": w.start, "end": w.end} for w in wt]
                for wt in word_timestamps
            ],
            scenes_used=[
                {"video": str(s.video_path), "start": s.start_time, 
                 "end": s.end_time, "duration": s.duration}
                for s in selected_scenes
            ],
            thumbnail_path=str(thumbnail) if thumbnail else ""
        )
        
        # Save metadata
        metadata_path = config.SCRIPTS_DIR / f"{video_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Step 9: Upload to YouTube (optional)
        if auto_upload:
            print("\n📤 Step 9: Uploading to YouTube...")
            success, result = self.uploader.upload(
                final_video, script, thumbnail
            )
            
            if success:
                metadata.uploaded = True
                metadata.youtube_video_id = result
                metadata.youtube_url = f"https://youtube.com/shorts/{result}"
                metadata.upload_date = datetime.now().isoformat()
                
                # Update saved metadata
                with open(metadata_path, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
                
                print(f"   ✓ Uploaded: {metadata.youtube_url}")
            else:
                print(f"   ✗ Upload failed: {result}")
        
        print("\n" + "="*60)
        print("✅ REEL GENERATION COMPLETE!")
        print("="*60)
        print(f"📹 Video: {final_video}")
        print(f"📝 Script: {script_path}")
        print(f"📊 Metadata: {metadata_path}")
        if metadata.youtube_url:
            print(f"🔗 YouTube: {metadata.youtube_url}")
        print("="*60 + "\n")
        
        return metadata
    
    def generate_bulk(self, count: int = 5, duration: int = 45,
                     auto_upload: bool = False, category: str = None) -> List[ReelMetadata]:
        """
        Generate multiple reels based on trending topics.
        
        Args:
            count: Number of videos to generate
            duration: Target duration for each video
            auto_upload: Whether to upload to YouTube
            category: Optional category filter for topics
        
        Returns:
            List of ReelMetadata for all generated videos
        """
        print("\n" + "="*60)
        print("🚀 BULK REEL GENERATION MODE")
        print("="*60)
        print(f"🎯 Videos to generate: {count}")
        print(f"⏱️  Duration per video: {duration} seconds")
        print(f"📤 Auto-upload: {'Yes' if auto_upload else 'No'}")
        if category:
            print(f"📁 Category: {category}")
        print("="*60)
        
        # Step 1: Get trending topics from Gemini
        print("\n🔥 Fetching trending topics from Google Trends...")
        topics = self.script_gen.get_trending_topics(count, category)
        
        if not topics:
            logger.error("Failed to get trending topics!")
            return []
        
        print(f"\n📊 Found {len(topics)} trending topics:")
        for i, topic in enumerate(topics, 1):
            print(f"   {i}. {topic}")
        
        # Confirm with user
        print("\n" + "-"*60)
        confirm = input("❓ Proceed with these topics? (y/n) [y]: ").strip().lower()
        if confirm == 'n':
            print("\n❌ Bulk generation cancelled.")
            return []
        
        # Step 2: Generate videos for each topic
        results = []
        successful = 0
        failed = 0
        
        for i, topic in enumerate(topics, 1):
            print("\n" + "="*60)
            print(f"🎬 VIDEO {i}/{len(topics)}")
            print("="*60)
            
            try:
                metadata = self.generate(topic, duration, auto_upload)
                if metadata:
                    results.append(metadata)
                    successful += 1
                    print(f"\n✅ Video {i} completed successfully!")
                else:
                    failed += 1
                    print(f"\n❌ Video {i} failed!")
            except Exception as e:
                logger.error(f"Error generating video for '{topic}': {e}")
                failed += 1
                print(f"\n❌ Video {i} failed with error: {e}")
            
            # Small delay between videos
            if i < len(topics):
                print("\n⏳ Waiting 5 seconds before next video...")
                time.sleep(5)
        
        # Summary
        print("\n" + "="*60)
        print("🎉 BULK GENERATION COMPLETE!")
        print("="*60)
        print(f"✅ Successful: {successful}/{len(topics)}")
        print(f"❌ Failed: {failed}/{len(topics)}")
        
        if results:
            print("\n📹 Generated Videos:")
            for i, meta in enumerate(results, 1):
                print(f"   {i}. {meta.video_path}")
                if meta.youtube_url:
                    print(f"      🔗 {meta.youtube_url}")
        
        print("="*60 + "\n")
        
        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🎬 Ultimate Viral Reel Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reel_generator.py -t "Bitcoin investing tips"
  python reel_generator.py -t "Morning routines for success" -d 60
  python reel_generator.py -t "AI revolution" --upload
  python reel_generator.py -t "Tech news" --no-text-removal  # Skip text removal for speed
  python reel_generator.py -t "Topic" --fast               # Fast mode (lower quality, faster)
  python reel_generator.py -t "Topic" --no-parallel        # Disable parallel processing
  python reel_generator.py --interactive
  
  # Bulk mode - generate multiple videos from trending topics:
  python reel_generator.py --bulk 5                    # Generate 5 videos
  python reel_generator.py --bulk 3 --category tech    # 3 tech-related videos
  python reel_generator.py --bulk 10 --upload          # Generate and upload 10 videos
        """
    )
    
    parser.add_argument("-t", "--topic", type=str, help="Video topic")
    parser.add_argument("-d", "--duration", type=int, default=45,
                       help="Target duration (30/45/60 seconds)")
    parser.add_argument("--upload", action="store_true",
                       help="Auto-upload to YouTube")
    parser.add_argument("--api-key", type=str, help="Gemini API key")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--no-text-removal", action="store_true",
                       help="Disable text removal from source videos (faster processing)")
    parser.add_argument("--text-sample-rate", type=int, default=5,
                       help="Text removal frame sample rate (1=all frames, higher=faster)")
    
    # Bulk mode arguments
    parser.add_argument("--bulk", type=int, metavar="N",
                       help="Bulk mode: generate N videos from trending topics")
    parser.add_argument("--category", type=str,
                       help="Category filter for bulk mode (tech, entertainment, sports, etc.)")
    
    # Performance arguments
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing (use sequential mode)")
    parser.add_argument("--fast", action="store_true",
                       help="Enable fast mode (less quality, faster processing)")
    
    args = parser.parse_args()
    
    # Apply text removal settings
    if args.no_text_removal:
        config.REMOVE_TEXT = False
    if args.text_sample_rate:
        config.TEXT_REMOVAL_SAMPLE_RATE = args.text_sample_rate
    
    # Apply parallelization settings
    if args.no_parallel:
        config.PARALLEL_DOWNLOADS = False
        config.PARALLEL_SCENE_DETECTION = False
        config.PARALLEL_TRANSCRIPTION = False
        print("⚠️  Parallel processing disabled - using sequential mode")
    
    # Fast mode settings
    if args.fast:
        config.TEXT_REMOVAL_SAMPLE_RATE = 10  # Process fewer frames
        config.VIDEOS_PER_SEARCH = 2  # Download fewer videos
        print("⚡ Fast mode enabled - reduced quality for faster processing")
    
    generator = ReelGenerator(args.api_key)
    
    # Bulk mode
    if args.bulk:
        generator.generate_bulk(
            count=args.bulk,
            duration=args.duration,
            auto_upload=args.upload,
            category=args.category
        )
    elif args.interactive or not args.topic:
        print("\n" + "="*60)
        print("🎬 ULTIMATE VIRAL REEL GENERATOR")
        print("="*60)
        
        # Ask for mode
        print("\n📝 Select mode:")
        print("   1. Single video (enter topic manually)")
        print("   2. Bulk mode (generate from trending topics)")
        mode = input("\nEnter choice (1/2) [1]: ").strip()
        
        if mode == '2':
            # Bulk mode
            count_input = input("\n📊 How many videos to generate? [5]: ").strip()
            count = int(count_input) if count_input.isdigit() else 5
            
            category = input("📌 Category filter (tech/entertainment/sports/etc.) [any]: ").strip() or None
            
            duration_input = input("⏱️  Duration per video (30/45/60) [45]: ").strip()
            duration = int(duration_input) if duration_input.isdigit() else 45
            
            upload = input("📤 Upload to YouTube? (y/n) [n]: ").strip().lower() == 'y'
            
            if PADDLEOCR_AVAILABLE:
                remove_text = input("🔤 Remove text from source videos? (y/n) [y]: ").strip().lower()
                config.REMOVE_TEXT = remove_text != 'n'
            
            generator.generate_bulk(count, duration, upload, category)
        else:
            # Single video mode
            topic = input("\n📝 Enter your topic: ").strip()
            if not topic:
                topic = "productivity tips for success"
                print(f"   Using default: {topic}")
            
            duration_input = input("⏱️  Duration (30/45/60) [45]: ").strip()
            duration = int(duration_input) if duration_input.isdigit() else 45
            
            upload = input("📤 Upload to YouTube? (y/n) [n]: ").strip().lower() == 'y'
            
            if PADDLEOCR_AVAILABLE:
                remove_text = input("🔤 Remove text from source videos? (y/n) [y]: ").strip().lower()
                config.REMOVE_TEXT = remove_text != 'n'
            
            generator.generate(topic, duration, upload)
    else:
        generator.generate(args.topic, args.duration, args.upload)


if __name__ == "__main__":
    main()

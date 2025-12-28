# 🎬 Ultimate Viral Reel Generator

A professional-grade automated YouTube Shorts/Reels generator that creates viral content using AI.

## ✨ Features

- 🤖 **Gemini AI Script Generation** - Creates engaging, viral-optimized scripts with SEO metadata
- 📥 **yt-dlp Video Downloads** - Downloads source videos from YouTube based on search terms
- 🎬 **PySceneDetect** - Intelligent scene detection and selection from downloaded videos
- 🎤 **Edge TTS** - High-quality voice narration
- 📝 **Whisper AI** - Word-level timestamp captions for engaging text overlays
- 🖼️ **AI Thumbnails** - Generates eye-catching thumbnails using g4f
- 📤 **YouTube Upload** - Multi-credential support with automatic failover
- 📊 **Complete Metadata** - Comprehensive JSON output with all video details

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this repository
cd reel_generator

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required)
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg
```

### 2. Configure API Keys

Edit `key.txt` and add your Gemini API key:

```
geminikey="YOUR_GEMINI_API_KEY"
```

Get your key from: https://aistudio.google.com/app/apikey

### 3. Run the Generator

```bash
# Interactive mode
python reel_generator.py --interactive

# Quick generation
python reel_generator.py -t "Bitcoin investing tips" -d 45

# With auto-upload
python reel_generator.py -t "Morning routines for success" --upload
```

## 📁 Output Structure

```
reel_output/
├── source_videos/     # Downloaded source videos
├── scenes/            # Extracted scene clips
├── audio/             # Generated voice narration
├── final/             # Final rendered videos
├── thumbnails/        # Generated thumbnails
└── scripts/           # Script JSON files and metadata
```

## 🎯 How It Works

### Pipeline Overview

```
User Topic
    ↓
┌─────────────────────────────────┐
│  1. GEMINI AI SCRIPT GENERATOR  │
│  - Creates viral script         │
│  - SEO-optimized metadata       │
│  - Search terms for each segment│
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  2. YT-DLP VIDEO DOWNLOADER     │
│  - Downloads source videos      │
│  - Based on segment search terms│
│  - 2-3 min videos per term      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  3. PYSCENEDETECT ANALYZER      │
│  - Detects scene boundaries     │
│  - Selects best scenes          │
│  - Matches to segment durations │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  4. EDGE TTS VOICE GENERATOR    │
│  - Generates narration          │
│  - Natural speech with emotion  │
│  - Per-segment audio files      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  5. WHISPER TRANSCRIBER         │
│  - Word-level timestamps        │
│  - Powers animated captions     │
│  - Precise synchronization      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  6. VIDEO COMPOSER              │
│  - Combines scenes + audio      │
│  - Adds word-by-word captions   │
│  - Background music             │
│  - Ken Burns effects            │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  7. THUMBNAIL GENERATOR         │
│  - AI-generated thumbnails      │
│  - Or video frame extraction    │
│  - YouTube-optimized 16:9       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  8. YOUTUBE UPLOADER            │
│  - Multi-credential support     │
│  - Automatic quota handling     │
│  - Thumbnail upload             │
└─────────────────────────────────┘
    ↓
Final Reel + Metadata JSON
```

## 📋 Command Line Options

| Option | Description |
|--------|-------------|
| `-t, --topic` | Video topic (required in non-interactive mode) |
| `-d, --duration` | Target duration: 30, 45, or 60 seconds (default: 45) |
| `--upload` | Auto-upload to YouTube after generation |
| `--api-key` | Gemini API key (overrides key.txt) |
| `--interactive` | Run in interactive mode |

## 🎥 YouTube Upload Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the YouTube Data API v3
4. Create OAuth 2.0 credentials (Desktop App)
5. Download the client secret JSON
6. Place it in `google-console/` folder

## 🎵 Background Audio (Optional)

Create these folders for background audio:
```
background-sounds/
├── music/     # Background music files (.mp3, .wav)
└── clicks/    # Transition click sounds (.mp3, .wav)
```

## 📊 Output Metadata

Each generated reel produces a detailed JSON file:

```json
{
  "video_path": "/path/to/final/video.mp4",
  "script": {
    "title": "The Truth About Bitcoin",
    "youtube_title": "🔥 Bitcoin Secrets They Don't Tell You #shorts",
    "description": "Discover the truth about Bitcoin...",
    "segments": [...],
    "hashtags": ["shorts", "viral", "crypto"],
    "tags": ["bitcoin", "investing", "finance"]
  },
  "audio_duration": 45.2,
  "word_timestamps": [...],
  "scenes_used": [...],
  "thumbnail_path": "/path/to/thumbnail.jpg",
  "youtube_video_id": "abc123",
  "youtube_url": "https://youtube.com/shorts/abc123",
  "uploaded": true,
  "created_at": "2024-12-28T10:30:00"
}
```

## 🔧 Advanced Configuration

Edit the `Config` class in `reel_generator.py`:

```python
@dataclass
class Config:
    # Video settings
    VIDEO_WIDTH: int = 1080
    VIDEO_HEIGHT: int = 1920
    FPS: int = 30
    
    # TTS Settings
    TTS_VOICE: str = "en-US-AriaNeural"
    
    # Audio levels
    MUSIC_VOLUME: float = 0.12
    
    # Scene detection
    SCENE_THRESHOLD: float = 27.0
    MIN_SCENE_DURATION: float = 1.0
```

## 🎤 Available TTS Voices

The generator uses Edge TTS. Popular voices:
- `en-US-AriaNeural` (Female, US)
- `en-US-GuyNeural` (Male, US)
- `en-GB-SoniaNeural` (Female, UK)
- `en-AU-NatashaNeural` (Female, Australia)

## ⚠️ Troubleshooting

### Common Issues

1. **"No video downloaded"**
   - Check your internet connection
   - Try a more specific search term
   - Some videos may be age-restricted or private

2. **"Whisper model loading slow"**
   - First run downloads the model
   - Use `base` model for faster processing
   - GPU acceleration requires CUDA

3. **"YouTube upload failed"**
   - Check your OAuth credentials
   - Ensure YouTube API is enabled
   - Daily upload quota may be exceeded

4. **"ffmpeg not found"**
   - Install ffmpeg: `sudo apt install ffmpeg` or `brew install ffmpeg`

## 📜 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Pull requests welcome! Please read the contributing guidelines first.

---

Made with ❤️ for content creators

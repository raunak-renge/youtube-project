# 🎬 Ultimate Viral Reel Generator

A professional-grade automated reel/shorts generator that creates viral content using AI, stock footage, and advanced video processing.

## ✨ Features

- 🤖 **Gemini AI** for script generation with SEO optimization
- 📥 **yt-dlp** for downloading source videos from YouTube
- 🎬 **PySceneDetect** for intelligent scene selection
- 🎤 **Kokoro TTS** for natural voice narration
- 📝 **Whisper** for word-level timestamp captions
- 🖼️ **g4f** for AI thumbnail generation
- 🔤 **PaddleOCR** for removing text/watermarks from source videos
- 📤 **YouTube upload** with multi-credential support
- 📊 Comprehensive JSON metadata output

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- FFmpeg installed on your system
- Kokoro TTS model files (optional, for voice generation)

### Setup

```bash
# Clone the repository
git clone https://github.com/raunak-renge/youtube-project.git
cd youtube-project

# Install dependencies (auto-installed on first run)
pip install -r requirements.txt

# Add your API keys to key.txt
echo 'geminikey="YOUR_GEMINI_API_KEY"' > key.txt
```

### Required Files

1. **key.txt** - API keys file:
```
geminikey="YOUR_GEMINI_API_KEY"
pexelkey="YOUR_PEXELS_API_KEY"
```

2. **Kokoro TTS Models** (for voice generation):
   - `kokoro-v1.0.onnx`
   - `voices-v1.0.bin`

3. **Google OAuth Credentials** (for YouTube upload):
   - Place `client_secret_*.json` files in `./google-console/`

---

## 🚀 Usage

This project has two main scripts:

| Script | Description | Best For |
|--------|-------------|----------|
| `reel_generator.py` | Downloads YouTube videos, extracts scenes, adds narration | Compilation/reaction style content |
| `main.py` | Uses stock images/videos with Ken Burns effects | Educational/fact-based content |

---

## 📹 reel_generator.py - YouTube Video Compiler

Creates reels by downloading relevant YouTube videos, detecting scenes, and compositing them with AI-generated narration.

### Single Video Generation

```bash
# Basic usage - generate a single video
python reel_generator.py -t "Bitcoin investing tips"

# Specify duration (30, 45, or 60 seconds)
python reel_generator.py -t "Morning routines for success" -d 60

# Generate and auto-upload to YouTube
python reel_generator.py -t "AI revolution" --upload

# Skip text removal for faster processing
python reel_generator.py -t "Tech news" --no-text-removal

# Custom text removal sample rate (higher = faster, lower = better quality)
python reel_generator.py -t "Crypto update" --text-sample-rate 10
```

### Bulk Generation (Trending Topics)

Generate multiple videos automatically from trending topics:

```bash
# Generate 5 videos from trending topics
python reel_generator.py --bulk 5

# Generate 3 tech-related trending videos
python reel_generator.py --bulk 3 --category tech --upload

# Generate 10 videos and upload all to YouTube
python reel_generator.py --bulk 10 --upload

# Bulk with custom duration
python reel_generator.py --bulk 5 -d 60 --upload

# Bulk with specific category
python reel_generator.py --bulk 5 --category entertainment
```

**Available Categories for Bulk Mode:**
- `tech` - Technology, AI, gadgets
- `entertainment` - Movies, TV, celebrities
- `sports` - Sports news, athletes
- `finance` - Crypto, stocks, money
- `gaming` - Video games, esports
- `science` - Scientific discoveries
- `lifestyle` - Health, fitness, productivity

### Interactive Mode

```bash
# Run in interactive mode (prompts for all options)
python reel_generator.py --interactive

# Or just run without arguments
python reel_generator.py
```

Interactive mode menu:
```
📝 Select mode:
   1. Single video (enter topic manually)
   2. Bulk mode (generate from trending topics)
```

### All Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-t, --topic` | Video topic | None |
| `-d, --duration` | Duration in seconds (30/45/60) | 45 |
| `--upload` | Auto-upload to YouTube | False |
| `--api-key` | Gemini API key | From key.txt |
| `--interactive` | Interactive mode | False |
| `--no-text-removal` | Disable OCR text removal | False |
| `--text-sample-rate` | Frame sample rate for text removal | 5 |
| `--bulk N` | Generate N videos from trending topics | None |
| `--category` | Category filter for bulk mode | Any |

---

## 🎨 main.py - Stock Media Reel Designer

Creates reels using stock images/videos from Pexels and Bing with Ken Burns effects.

### Basic Usage

```bash
# Interactive mode (recommended for first use)
python main.py

# Generate with specific topic
python main.py --topic "Why Elon Musk is successful"

# Specify duration
python main.py --topic "Morning productivity tips" --duration 60

# Generate and upload
python main.py --topic "Bitcoin price prediction" --upload
```

### Media Priority

The script intelligently chooses media sources:
- **Celebrities/Specific People** → Bing Images first
- **Generic Concepts** → Pexels Videos/Photos first
- **Action Scenes** → Pexels Videos preferred
- **Static Subjects** → Pexels Photos preferred

### All Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--topic` | Video topic | Interactive |
| `--duration` | Duration (30/45/60) | 45 |
| `--upload` | Upload to YouTube | False |
| `--no-captions` | Disable word captions | False |
| `--no-music` | Disable background music | False |

---

## 📁 Output Structure

```
reel_output_youtube/          # reel_generator.py output
├── source_videos/            # Downloaded YouTube videos
├── scenes/                   # Extracted scenes
├── audio/                    # Generated TTS audio
├── final/                    # Final rendered videos
├── thumbnails/               # AI-generated thumbnails
└── scripts/                  # JSON scripts and metadata

reel_output/                  # main.py output
├── images/                   # Downloaded images
├── audio/                    # Generated audio
├── video/                    # Final videos
├── scripts/                  # JSON scripts
└── thumbnails/               # Generated thumbnails
```

---

## ⚙️ Configuration

### Channel Branding

Edit the `Config` class in either script to customize:

```python
# In reel_generator.py
CHANNEL_NAME: str = "Your Channel Name"
DEFAULT_KEYWORDS: List[str] = [
    "your channel", "your brand",
    # Add your SEO keywords...
]
```

### Video Settings

```python
VIDEO_WIDTH: int = 1080      # YouTube Shorts width
VIDEO_HEIGHT: int = 1920     # YouTube Shorts height (9:16)
FPS: int = 30                # Frame rate
```

### TTS Settings

```python
TTS_VOICE: str = "af_bella"  # Kokoro voice
TTS_SPEED: float = 1.0       # Speech speed
```

### Audio Levels

```python
MUSIC_VOLUME: float = 0.12   # Background music (0-1)
CLICK_VOLUME: float = 0.25   # Transition sounds (0-1)
```

---

## 🔑 API Keys Setup

### Gemini AI (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Add to `key.txt`: `geminikey="YOUR_KEY"`

### Pexels (For main.py)
1. Go to [Pexels API](https://www.pexels.com/api/)
2. Create an account and get API key
3. Add to `key.txt`: `pexelkey="YOUR_KEY"`

### YouTube Upload
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project and enable YouTube Data API v3
3. Create OAuth 2.0 credentials
4. Download JSON and place in `./google-console/`

---

## 🎯 Example Workflows

### Daily Trending Content (Automated)

```bash
# Generate 5 trending videos every day
python reel_generator.py --bulk 5 --upload -d 45
```

### Topic-Specific Batch

```bash
# Tech news compilation
python reel_generator.py --bulk 3 --category tech --upload

# Entertainment/Celebrity content  
python reel_generator.py --bulk 3 --category entertainment --upload
```

### High-Quality Single Video

```bash
# Full quality with text removal
python reel_generator.py -t "iPhone 16 review" -d 60 --text-sample-rate 1 --upload
```

### Quick Draft (No Text Removal)

```bash
# Fast generation for testing
python reel_generator.py -t "Quick test topic" --no-text-removal
```

### Educational Content with Stock Media

```bash
# Use main.py for cleaner educational content
python main.py --topic "How compound interest works" --duration 45 --upload
```

---

## 🛠️ Troubleshooting

### Common Issues

**PaddleOCR not working:**
```bash
pip install paddleocr paddlepaddle
```

**Kokoro TTS not found:**
- Ensure `kokoro-v1.0.onnx` and `voices-v1.0.bin` are in the project root
- Check the binary path in Config

**YouTube upload fails:**
- Ensure OAuth credentials are valid
- Check daily quota limits
- Try different credential files in `./google-console/`

**FFmpeg errors:**
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

---

## 📝 License

This project is for educational purposes. Ensure you comply with:
- YouTube's Terms of Service
- Content licensing for downloaded videos
- API usage limits and terms

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Support

For issues and feature requests, please open a GitHub issue.

---

**Made with ❤️ for content creators**

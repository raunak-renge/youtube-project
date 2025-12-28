#!/usr/bin/env python3
"""
YouTube Simple Browser Uploader
================================
A simpler approach using webbrowser + manual interaction.
Opens YouTube Studio and guides you through the upload process.

This is the SAFEST method - it just opens the browser and provides
copy-paste ready metadata for each video.

Usage:
    python youtube_simple_upload.py
    python youtube_simple_upload.py --limit 5
"""

import json
import webbrowser
import time
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# For clipboard
try:
    import pyperclip
except ImportError:
    print("Installing pyperclip for clipboard support...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip", "-q"])
    import pyperclip


class YouTubeSimpleUploader:
    """
    Simple uploader that opens YouTube Studio and provides
    copy-paste ready metadata for manual upload.
    """
    
    YOUTUBE_UPLOAD_URL = "https://www.youtube.com/upload"
    
    def __init__(self, scripts_dir: Path, video_dir: Path):
        self.scripts_dir = scripts_dir
        self.video_dir = video_dir
        
    def get_pending_videos(self) -> List[Dict]:
        """Get list of pending videos with their metadata"""
        pending = []
        
        for script_file in sorted(self.scripts_dir.glob("reel_*.json")):
            try:
                with open(script_file, 'r') as f:
                    script = json.load(f)
                
                if script.get('uploaded', False):
                    continue
                
                # Find video file
                video_path = script.get('video_file_path')
                if video_path and Path(video_path).exists():
                    video_file = Path(video_path)
                else:
                    # Try to find by filename
                    video_filename = script.get('video_filename')
                    if video_filename:
                        video_file = self.video_dir / video_filename
                    else:
                        # Try matching pattern
                        script_stem = script_file.stem
                        matches = list(self.video_dir.glob(f"{script_stem}*.mp4"))
                        video_file = matches[0] if matches else None
                
                if not video_file or not video_file.exists():
                    continue
                
                # Get thumbnail
                thumbnail_path = script.get('thumbnail_path')
                if thumbnail_path and not Path(thumbnail_path).exists():
                    thumbnail_path = None
                
                pending.append({
                    'script_file': script_file,
                    'script': script,
                    'video_path': video_file,
                    'thumbnail_path': thumbnail_path,
                    'title': script.get('video_title', script.get('title', 'Untitled'))[:100],
                    'description': script.get('description', ''),
                    'tags': self._parse_tags(script.get('hashtags_text', '')),
                    'visibility': script.get('visibility', 'public')
                })
                
            except Exception as e:
                print(f"⚠️ Error reading {script_file}: {e}")
                
        return pending
    
    def _parse_tags(self, hashtags_text: str) -> str:
        """Parse hashtags into comma-separated tags"""
        if not hashtags_text:
            return ""
        
        tags = []
        for tag in hashtags_text.split():
            clean_tag = tag.strip().replace('#', '')
            if clean_tag and len(clean_tag) > 1:
                tags.append(clean_tag)
        
        # YouTube allows max 500 chars for tags
        result = []
        total_len = 0
        for tag in tags:
            if total_len + len(tag) + 1 > 500:
                break
            result.append(tag)
            total_len += len(tag) + 1
        
        return ", ".join(result)
    
    def mark_uploaded(self, script_file: Path, video_url: str = None):
        """Mark a video as uploaded in its script file"""
        try:
            with open(script_file, 'r') as f:
                script = json.load(f)
            
            script['uploaded'] = True
            script['upload_date'] = datetime.now().isoformat()
            script['upload_method'] = 'manual_browser'
            if video_url:
                script['youtube_url'] = video_url
            
            with open(script_file, 'w') as f:
                json.dump(script, f, indent=2, ensure_ascii=False)
                
            return True
        except Exception as e:
            print(f"⚠️ Error updating script: {e}")
            return False
    
    def upload_interactive(self, limit: int = None):
        """Interactive upload process - guides user through each video"""
        pending = self.get_pending_videos()
        
        if not pending:
            print("\n✅ No pending videos to upload!")
            return
        
        if limit:
            pending = pending[:limit]
        
        print(f"\n📋 Found {len(pending)} pending video(s)")
        print("\n" + "="*60)
        print("📖 HOW THIS WORKS:")
        print("="*60)
        print("1. For each video, the browser will open YouTube upload page")
        print("2. Video path will be copied to clipboard - paste in file dialog")
        print("3. Title and description will be shown - copy them manually")
        print("4. After uploading, confirm and we'll mark it as uploaded")
        print("="*60)
        
        input("\nPress ENTER to start...")
        
        uploaded = 0
        failed = 0
        
        for i, video in enumerate(pending, 1):
            print(f"\n{'='*60}")
            print(f"📹 VIDEO {i}/{len(pending)}: {video['title'][:50]}...")
            print("="*60)
            
            # Show video info
            print(f"\n📁 VIDEO FILE:")
            print(f"   {video['video_path']}")
            
            if video['thumbnail_path']:
                print(f"\n🖼️  THUMBNAIL:")
                print(f"   {video['thumbnail_path']}")
            
            print(f"\n📺 TITLE (copy this):")
            print("-"*40)
            print(video['title'])
            print("-"*40)
            
            print(f"\n📄 DESCRIPTION (copy this):")
            print("-"*40)
            print(video['description'][:500] + ("..." if len(video['description']) > 500 else ""))
            print("-"*40)
            
            print(f"\n🏷️  TAGS (copy this):")
            print("-"*40)
            print(video['tags'][:200] + ("..." if len(video['tags']) > 200 else ""))
            print("-"*40)
            
            # Copy video path to clipboard
            try:
                pyperclip.copy(str(video['video_path'].absolute()))
                print(f"\n✅ Video path copied to clipboard!")
            except:
                print(f"\n⚠️ Could not copy to clipboard. Copy path manually.")
            
            # Open YouTube upload
            print("\n🌐 Opening YouTube upload page...")
            webbrowser.open(self.YOUTUBE_UPLOAD_URL)
            
            # Wait for user
            print("\n" + "="*60)
            print("📌 INSTRUCTIONS:")
            print("="*60)
            print("1. Click 'SELECT FILES' in YouTube")
            print("2. Press Cmd+Shift+G (Go to folder) in file dialog")
            print("3. Paste (Cmd+V) the video path and press Enter")
            print("4. Copy-paste the TITLE and DESCRIPTION shown above")
            print("5. Set 'Not made for kids'")
            print("6. Set visibility to 'Public' and click 'PUBLISH'")
            print("="*60)
            
            while True:
                response = input("\n⏳ When done, type 'done' (or 'skip' to skip, 'quit' to exit): ").strip().lower()
                
                if response == 'done':
                    video_url = input("📎 Paste the video URL (or press Enter to skip): ").strip()
                    self.mark_uploaded(video['script_file'], video_url if video_url else None)
                    uploaded += 1
                    print("✅ Marked as uploaded!")
                    break
                elif response == 'skip':
                    print("⏭️ Skipping this video")
                    failed += 1
                    break
                elif response == 'quit':
                    print("\n👋 Exiting...")
                    print(f"\n📊 Session Summary:")
                    print(f"   ✅ Uploaded: {uploaded}")
                    print(f"   ⏭️ Skipped/Failed: {failed}")
                    return
                else:
                    print("❓ Please type 'done', 'skip', or 'quit'")
            
            if i < len(pending):
                print("\n⏳ Waiting 5 seconds before next video...")
                time.sleep(5)
        
        print("\n" + "="*60)
        print("📊 UPLOAD SESSION COMPLETE")
        print("="*60)
        print(f"   ✅ Uploaded: {uploaded}")
        print(f"   ⏭️ Skipped/Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple YouTube uploader with copy-paste workflow"
    )
    parser.add_argument("--limit", "-l", type=int, help="Maximum number of videos")
    parser.add_argument("--scripts-dir", type=str, default="./reel_output/scripts")
    parser.add_argument("--video-dir", type=str, default="./reel_output/video")
    parser.add_argument("--list", action="store_true", help="Just list pending videos")
    
    args = parser.parse_args()
    
    scripts_dir = Path(args.scripts_dir)
    video_dir = Path(args.video_dir)
    
    if not scripts_dir.exists():
        print(f"❌ Scripts directory not found: {scripts_dir}")
        return
    
    uploader = YouTubeSimpleUploader(scripts_dir, video_dir)
    
    if args.list:
        pending = uploader.get_pending_videos()
        print(f"\n📋 Found {len(pending)} pending video(s):\n")
        for i, v in enumerate(pending, 1):
            print(f"  {i}. {v['title'][:60]}...")
            print(f"     📁 {v['video_path'].name}")
            print()
        return
    
    print("\n" + "="*60)
    print("🎬 YOUTUBE SIMPLE UPLOADER")
    print("="*60)
    print("This tool helps you manually upload videos to YouTube")
    print("by providing copy-paste ready metadata for each video.")
    print("="*60)
    
    uploader.upload_interactive(limit=args.limit)


if __name__ == "__main__":
    main()

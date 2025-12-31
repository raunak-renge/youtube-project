#!/usr/bin/env python3
"""
Bulk upload all videos from a folder to YouTube.
Automatically finds corresponding script files and thumbnails.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List
import os

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we need
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle

@dataclass
class VideoScript:
    title: str
    youtube_title: str
    description: str
    segments: List = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    topic: str = ""
    duration_target: int = 45

class YouTubeUploader:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
        self.CLIENT_SECRETS_FILE = self._find_client_secret()
        self.TOKEN_FILE = 'token.pickle'
        
    def _find_client_secret(self):
        """Find the first available client secret file."""
        secret_dir = Path('google-console')
        if secret_dir.exists():
            secrets = list(secret_dir.glob('client_secret_*.json'))
            if secrets:
                return str(secrets[0])
        return 'client_secret.json'
    
    def get_authenticated_service(self):
        """Authenticate and return YouTube service."""
        creds = None
        
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.CLIENT_SECRETS_FILE, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('youtube', 'v3', credentials=creds)
    
    def sanitize_tag(self, tag):
        """Sanitize a tag to meet YouTube requirements."""
        tag = tag.strip()
        tag = ''.join(c for c in tag if c not in '<>\'"')
        if len(tag) > 30:
            tag = tag[:30]
        return tag.strip()
    
    def upload(self, video_path, script, thumbnail_path=None):
        """Upload video to YouTube."""
        try:
            youtube = self.get_authenticated_service()
            
            # Sanitize tags
            tags = [self.sanitize_tag(tag) for tag in script.tags if self.sanitize_tag(tag)]
            
            body = {
                'snippet': {
                    'title': script.youtube_title[:100],
                    'description': script.description[:5000],
                    'tags': tags[:500],
                    'categoryId': '22'
                },
                'status': {
                    'privacyStatus': 'public',
                    'selfDeclaredMadeForKids': False
                }
            }
            
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            request = youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = request.execute()
            video_id = response['id']
            
            # Upload thumbnail if provided
            if thumbnail_path and Path(thumbnail_path).exists():
                try:
                    youtube.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
                    ).execute()
                except Exception as e:
                    print(f"⚠️  Warning: Thumbnail upload failed: {e}")
            
            return True, video_id
            
        except Exception as e:
            return False, str(e)

def find_script_for_video(video_path, script_dir):
    """Find the script JSON file matching the video timestamp."""
    # Extract timestamp from video filename (e.g., final_reel_20251230_110041_...)
    parts = video_path.stem.split('_')
    
    # Look for date and time pattern
    for i in range(len(parts) - 1):
        if len(parts[i]) == 8 and len(parts[i+1]) == 6:  # YYYYMMDD_HHMMSS
            date_str = parts[i]
            time_str = parts[i+1]
            pattern = f"reel_{date_str}_{time_str}*.json"
            
            scripts = list(script_dir.glob(pattern))
            if scripts:
                return scripts[0]
    
    return None

def find_thumbnail_for_video(video_path, thumbnail_dir):
    """Find the thumbnail matching the video timestamp."""
    # Extract timestamp from video filename
    parts = video_path.stem.split('_')
    
    # Look for date and time pattern
    for i in range(len(parts) - 1):
        if len(parts[i]) == 8 and len(parts[i+1]) == 6:
            date_str = parts[i]
            time_str = parts[i+1]
            pattern = f"*{date_str}_{time_str}*"
            
            thumbs = list(thumbnail_dir.glob(pattern))
            if thumbs:
                return thumbs[0]
    
    # Fallback: find most recent thumbnail
    thumbs = sorted(thumbnail_dir.glob('*.jpg'), key=lambda x: x.stat().st_mtime, reverse=True)
    return thumbs[0] if thumbs else None

def upload_video(video_path, script_dir, thumbnail_dir):
    """Upload a single video with its metadata."""
    print(f"\n{'='*80}")
    print(f"🎬 Processing: {video_path.name}")
    print(f"{'='*80}")
    
    # Find script file
    script_path = find_script_for_video(video_path, script_dir)
    if not script_path:
        print(f"❌ Script file not found for {video_path.name}")
        return False, "Script not found"
    
    print(f"📝 Found script: {script_path.name}")
    
    # Load script
    try:
        with open(script_path) as f:
            script_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading script: {e}")
        return False, str(e)
    
    # Create VideoScript object
    script = VideoScript(
        title=script_data.get('title', ''),
        youtube_title=script_data.get('youtube_title', ''),
        description=script_data.get('description', ''),
        segments=[],
        hashtags=script_data.get('hashtags', []),
        tags=script_data.get('tags', []),
        topic=script_data.get('topic', ''),
        duration_target=script_data.get('duration_target', 45)
    )
    
    # Find thumbnail
    thumb = find_thumbnail_for_video(video_path, thumbnail_dir)
    
    print(f"🖼️  Thumbnail: {thumb.name if thumb else 'None'}")
    print(f"📺 Title: {script.youtube_title[:80]}...")
    print(f"🏷️  Tags: {len(script.tags)} tags")
    print()
    
    # Upload
    try:
        uploader = YouTubeUploader()
        success, result = uploader.upload(video_path, script, thumb)
        
        if success:
            print(f"✅ Upload successful!")
            print(f"🔗 Video ID: {result}")
            print(f"🔗 URL: https://youtube.com/shorts/{result}")
            return True, result
        else:
            print(f"❌ Upload failed: {result}")
            return False, result
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False, str(e)

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_all_videos.py <video_folder_path>")
        print("\nExample:")
        print("  python upload_all_videos.py reel_output_youtube/final")
        sys.exit(1)
    
    # Get folder path
    video_folder = Path(sys.argv[1])
    
    if not video_folder.exists():
        print(f"❌ Folder not found: {video_folder}")
        sys.exit(1)
    
    if not video_folder.is_dir():
        print(f"❌ Not a directory: {video_folder}")
        sys.exit(1)
    
    # Find all video files
    video_files = sorted(video_folder.glob('*.mp4'))
    
    if not video_files:
        print(f"❌ No MP4 files found in {video_folder}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"📁 Found {len(video_files)} video(s) in {video_folder}")
    print(f"{'='*80}")
    
    for i, video in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {video.name}")
    
    # Confirm
    response = input(f"\n⚠️  Upload all {len(video_files)} videos? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Upload cancelled")
        sys.exit(0)
    
    # Set paths
    script_dir = Path('reel_output_youtube/scripts')
    thumbnail_dir = Path('reel_output_youtube/thumbnails')
    
    if not script_dir.exists():
        script_dir = Path('reel_output/scripts')
    
    if not thumbnail_dir.exists():
        thumbnail_dir = Path('reel_output/thumbnails')
    
    # Upload all videos
    results = []
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'#'*80}")
        print(f"VIDEO {i}/{len(video_files)}")
        print(f"{'#'*80}")
        
        success, result = upload_video(video_path, script_dir, thumbnail_dir)
        results.append({
            'video': video_path.name,
            'success': success,
            'result': result
        })
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 UPLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}/{len(video_files)}")
    print(f"❌ Failed: {failed}/{len(video_files)}")
    print()
    
    if successful > 0:
        print("✅ Successfully uploaded:")
        for r in results:
            if r['success']:
                print(f"  • {r['video']}")
                print(f"    🔗 https://youtube.com/shorts/{r['result']}")
    
    if failed > 0:
        print("\n❌ Failed uploads:")
        for r in results:
            if not r['success']:
                print(f"  • {r['video']}")
                print(f"    Error: {r['result']}")

if __name__ == '__main__':
    main()

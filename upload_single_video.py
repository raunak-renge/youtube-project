#!/usr/bin/env python3
"""
Simple script to upload a single video to YouTube
Usage: python upload_single_video.py <video_path>
"""

import sys
import json
from pathlib import Path

# Import from reel-generator-raunak
sys.path.insert(0, '.')
exec(open('reel-generator-raunak.py').read())

def upload_video(video_path: str):
    """Upload a single video to YouTube"""
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False
    
    # Extract timestamp from filename (e.g., reel_20251230_110041)
    timestamp = None
    parts = video_path.stem.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 8 and i+1 < len(parts):
            timestamp = f"{part}_{parts[i+1]}"
            break
    
    if not timestamp:
        print(f"❌ Could not extract timestamp from filename: {video_path.name}")
        return False
    
    # Find the corresponding metadata file
    metadata_files = list(Path('reel_output_youtube/scripts').glob(f'*{timestamp}*_metadata.json'))
    
    if not metadata_files:
        print(f"❌ Metadata file not found for timestamp: {timestamp}")
        print(f"   Searched in: reel_output_youtube/scripts/*{timestamp}*_metadata.json")
        return False
    
    metadata_path = metadata_files[0]
    print(f"📝 Found metadata: {metadata_path.name}")
    
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Extract script data
    script_data = metadata.get('script', {})
    
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
    thumbnail_path = Path('reel_output_youtube/thumbnails')
    thumbs = list(thumbnail_path.glob(f'*{timestamp}*'))
    if not thumbs:
        thumbs = sorted(thumbnail_path.glob('*.jpg'), key=lambda x: x.stat().st_mtime, reverse=True)
    thumb = thumbs[0] if thumbs else None
    
    print(f"🎬 Video: {video_path.name}")
    print(f"🖼️  Thumbnail: {thumb.name if thumb else 'None'}")
    print(f"📺 Title: {script.youtube_title[:80]}...")
    print(f"🏷️  Tags: {len(script.tags)} tags")
    print()
    
    # Upload
    print("⏳ Uploading to YouTube...")
    uploader = YouTubeUploader()
    success, result = uploader.upload(video_path, script, thumb)
    
    if success:
        print(f'\n✅ Upload successful!')
        print(f'🔗 Video ID: {result}')
        print(f'🔗 URL: https://youtube.com/shorts/{result}')
        
        # Update metadata with upload info
        metadata['uploaded'] = True
        metadata['youtube_video_id'] = result
        metadata['youtube_url'] = f'https://youtube.com/shorts/{result}'
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    else:
        print(f'\n❌ Upload failed: {result}')
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_single_video.py <video_path>")
        print("\nExample:")
        print("  python upload_single_video.py reel_output_youtube/final/final_reel_20251230_121210_LeBron_James_surpasses_Kareem_.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    success = upload_video(video_path)
    sys.exit(0 if success else 1)

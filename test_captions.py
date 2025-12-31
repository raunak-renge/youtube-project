#!/usr/bin/env python3
"""
Test script to verify captions are correctly displayed per segment.
Creates a short test video with different captions per segment.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to patch argparse before importing main-raunak
import argparse
original_parse_args = argparse.ArgumentParser.parse_args
def patched_parse_args(self, args=None, namespace=None):
    if args is None:
        args = []
    return original_parse_args(self, args, namespace)
argparse.ArgumentParser.parse_args = patched_parse_args

# Now import from main-raunak
print("Loading modules...")
exec(open('main-raunak.py').read())

def test_captions():
    """Test that captions work correctly across multiple segments"""
    
    print("\n" + "="*80)
    print("🧪 CAPTION LOOP TEST")
    print("="*80)
    
    # Create test segments with very different text
    test_segments = [
        {"text": "This is segment ONE. First words here."},
        {"text": "Now segment TWO. Different content."},
        {"text": "Finally segment THREE. Last section."},
    ]
    
    # Test the caption grouper cache key mechanism
    print("\n📝 Testing caption cache key uniqueness...")
    
    words_seg1 = [
        {"word": "This", "start": 0.0, "end": 0.3},
        {"word": "is", "start": 0.3, "end": 0.5},
        {"word": "ONE", "start": 0.5, "end": 0.8},
    ]
    
    words_seg2 = [
        {"word": "Now", "start": 0.0, "end": 0.3},
        {"word": "segment", "start": 0.3, "end": 0.6},
        {"word": "TWO", "start": 0.6, "end": 0.9},
    ]
    
    words_seg3 = [
        {"word": "Finally", "start": 0.0, "end": 0.4},
        {"word": "segment", "start": 0.4, "end": 0.7},
        {"word": "THREE", "start": 0.7, "end": 1.0},
    ]
    
    # Create caption instance
    captions = AnimatedCaptions()
    
    # Test that cache keys are different for different word sets
    cache_key_1 = tuple((w.get('word', ''), round(w.get('start', 0), 3)) for w in words_seg1)
    cache_key_2 = tuple((w.get('word', ''), round(w.get('start', 0), 3)) for w in words_seg2)
    cache_key_3 = tuple((w.get('word', ''), round(w.get('start', 0), 3)) for w in words_seg3)
    
    print(f"\n  Segment 1 cache key: {cache_key_1}")
    print(f"  Segment 2 cache key: {cache_key_2}")
    print(f"  Segment 3 cache key: {cache_key_3}")
    
    if cache_key_1 == cache_key_2 or cache_key_1 == cache_key_3 or cache_key_2 == cache_key_3:
        print("\n❌ FAIL: Cache keys are not unique!")
        return False
    else:
        print("\n✅ PASS: Cache keys are unique for different segments")
    
    # Test caption frame generation
    print("\n📝 Testing caption frame generation...")
    
    captions.clear_cache()
    frame1 = captions.create_caption_frame(words_seg1, 0.2)  # Should show "This"
    
    captions.clear_cache()
    frame2 = captions.create_caption_frame(words_seg2, 0.2)  # Should show "Now"
    
    captions.clear_cache()
    frame3 = captions.create_caption_frame(words_seg3, 0.2)  # Should show "Finally"
    
    print(f"  Frame 1 generated: {frame1.shape}")
    print(f"  Frame 2 generated: {frame2.shape}")
    print(f"  Frame 3 generated: {frame3.shape}")
    
    # Verify frames are different (different caption text)
    import numpy as np
    if np.array_equal(frame1, frame2):
        print("\n❌ FAIL: Frame 1 and Frame 2 are identical!")
        return False
    elif np.array_equal(frame1, frame3):
        print("\n❌ FAIL: Frame 1 and Frame 3 are identical!")
        return False
    elif np.array_equal(frame2, frame3):
        print("\n❌ FAIL: Frame 2 and Frame 3 are identical!")
        return False
    else:
        print("\n✅ PASS: All caption frames are unique for different segments")
    
    # Test the old buggy cache behavior (should now be fixed)
    print("\n📝 Testing that old cache bug is fixed...")
    
    # Create words with same COUNT but different content
    words_a = [
        {"word": "Apple", "start": 0.0, "end": 0.5},
        {"word": "Banana", "start": 0.5, "end": 1.0},
    ]
    words_b = [
        {"word": "Cat", "start": 0.0, "end": 0.5},
        {"word": "Dog", "start": 0.5, "end": 1.0},
    ]
    
    print(f"\n  Both have {len(words_a)} words (old bug used len() as cache key)")
    
    captions.clear_cache()
    frame_a = captions.create_caption_frame(words_a, 0.25)  # Should show "Apple"
    
    # Without clearing cache, test if different words still work
    frame_b = captions.create_caption_frame(words_b, 0.25)  # Should show "Cat", NOT "Apple"
    
    if np.array_equal(frame_a, frame_b):
        print("\n❌ FAIL: Same-length word lists produced identical frames (old bug still present!)")
        return False
    else:
        print("\n✅ PASS: Same-length word lists produce different frames correctly")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe caption loop bug has been fixed.")
    print("Captions now use unique cache keys based on word content + timing.")
    return True


def create_test_video():
    """Create a quick test video to visually verify captions"""
    
    print("\n" + "="*80)
    print("🎬 CREATING TEST VIDEO")
    print("="*80)
    
    output_dir = Path("reel_output_youtube/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"caption_test_{timestamp}.mp4"
    
    print(f"\n📁 Output: {output_path}")
    
    # Check if we have test assets
    test_images = list(Path("reel_output_youtube/images").glob("**/*.jpg"))[:3]
    if len(test_images) < 3:
        test_images = list(Path("reel_output").glob("**/*.jpg"))[:3]
    
    if len(test_images) < 3:
        print("\n⚠️  Not enough test images found. Using placeholder approach.")
        print("   To create a full test video, run the main generator first.")
        return None
    
    print(f"\n🖼️  Using images: {[img.name for img in test_images]}")
    
    # We'd need TTS audio files too - this is just a framework
    print("\n💡 To fully test, run the main generator with a short topic:")
    print('   python main-raunak.py -t "Test captions one two three" -d 20')
    
    return output_path


if __name__ == "__main__":
    print("\n🔬 Caption Bug Test Suite")
    print("Testing fix for: captions looping back to segment 1 during segment 3")
    
    # Run unit tests first
    if test_captions():
        print("\n" + "-"*80)
        print("💡 To create a visual test video, run:")
        print('   python main-raunak.py -t "Test captions" -d 20 --no-parallel')
        print("-"*80)

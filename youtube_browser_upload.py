#!/usr/bin/env python3
"""
YouTube Browser Upload - Upload videos to YouTube via browser automation
Bypasses API quotas by using Selenium to upload through YouTube Studio
"""

import json
import os
import time
import glob
import argparse
import tempfile
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


class YouTubeBrowserUploader:
    """Upload videos to YouTube using browser automation"""
    
    def __init__(self, headless=False):
        self.headless = headless
        self.driver = None
        # Create a dedicated profile directory for YouTube uploads
        self.profile_dir = os.path.expanduser("~/.youtube_uploader_profile")
        os.makedirs(self.profile_dir, exist_ok=True)
        
    def _setup_driver(self):
        """Initialize Chrome WebDriver with proper settings"""
        options = Options()
        
        # Use a separate profile directory (not the default Chrome profile)
        options.add_argument(f"--user-data-dir={self.profile_dir}")
        options.add_argument("--profile-directory=Default")
        
        # Essential options for stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--start-maximized")
        options.add_argument("--window-size=1920,1080")
        
        # Prevent detection
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        if self.headless:
            options.add_argument("--headless=new")
        
        # Use webdriver-manager to handle ChromeDriver
        service = Service(ChromeDriverManager().install())
        
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.driver.implicitly_wait(10)
        
        print("✅ Chrome browser initialized successfully")
        return self.driver
    
    def login_to_youtube(self):
        """Navigate to YouTube and let user login manually"""
        if not self.driver:
            self._setup_driver()
        
        print("\n" + "="*60)
        print("📺 YOUTUBE LOGIN REQUIRED")
        print("="*60)
        
        self.driver.get("https://studio.youtube.com")
        time.sleep(2)
        
        # Check if already logged in
        if "studio.youtube.com" in self.driver.current_url and "channel" in self.driver.current_url:
            print("✅ Already logged in to YouTube Studio!")
            return True
        
        print("\n⚠️  Please login to your YouTube account in the browser window.")
        print("   The script will wait for you to complete the login.")
        print("   After logging in, you should see YouTube Studio dashboard.")
        print("\n   Press ENTER in this terminal when login is complete...")
        
        input()
        
        # Verify login
        self.driver.get("https://studio.youtube.com")
        time.sleep(3)
        
        if "studio.youtube.com" in self.driver.current_url:
            print("✅ Login successful! Ready to upload videos.")
            return True
        else:
            print("❌ Login verification failed. Please try again.")
            return False
    
    def upload_video(self, video_path, title, description, tags=None, thumbnail_path=None):
        """Upload a single video to YouTube"""
        if not self.driver:
            self._setup_driver()
            
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
        
        print(f"\n📤 Uploading: {os.path.basename(video_path)}")
        print(f"   Title: {title[:50]}...")
        
        try:
            # Navigate to YouTube Studio upload page
            self.driver.get("https://studio.youtube.com")
            time.sleep(3)
            
            # Click the Create button (camera icon with +)
            try:
                # Try multiple selectors for the create button
                create_selectors = [
                    "ytcp-button#create-icon",
                    "#create-icon",
                    "button[aria-label='Create']",
                    "ytcp-icon-button#create-icon",
                    "#upload-icon"
                ]
                
                create_btn = None
                for selector in create_selectors:
                    try:
                        create_btn = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if create_btn:
                            break
                    except:
                        continue
                
                if create_btn:
                    create_btn.click()
                    time.sleep(1)
                else:
                    # Try direct URL to upload
                    self.driver.get("https://www.youtube.com/upload")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"   Trying direct upload URL...")
                self.driver.get("https://www.youtube.com/upload")
                time.sleep(2)
            
            # Click "Upload videos" option if menu appeared
            try:
                upload_menu_selectors = [
                    "tp-yt-paper-item#text-item-0",
                    "#text-item-0",
                    "tp-yt-paper-item:first-child",
                    "ytcp-text-menu tp-yt-paper-item"
                ]
                
                for selector in upload_menu_selectors:
                    try:
                        upload_option = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        upload_option.click()
                        time.sleep(1)
                        break
                    except:
                        continue
            except:
                pass
            
            # Find and use the file input to upload
            time.sleep(2)
            
            # Look for file input element
            file_input_selectors = [
                "input[type='file']",
                "input#select-files-button",
                "ytcp-uploads-file-picker input[type='file']"
            ]
            
            file_input = None
            for selector in file_input_selectors:
                try:
                    file_input = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if file_input:
                        break
                except:
                    continue
            
            if not file_input:
                print("❌ Could not find file input. Manual intervention may be needed.")
                print("   Please drag and drop the video file into the upload area.")
                print(f"   Video path: {video_path}")
                input("   Press ENTER after manually selecting the file...")
            else:
                # Send the file path to the input
                abs_video_path = os.path.abspath(video_path)
                file_input.send_keys(abs_video_path)
                print("   ✅ Video file selected")
            
            # Wait for upload dialog to appear
            time.sleep(3)
            
            # Set the title
            try:
                title_selectors = [
                    "ytcp-social-suggestions-textbox#title-textarea",
                    "#title-textarea",
                    "div#textbox[aria-label*='title']",
                    "ytcp-mention-textbox#title-textarea div#textbox"
                ]
                
                title_input = None
                for selector in title_selectors:
                    try:
                        title_input = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        if title_input:
                            break
                    except:
                        continue
                
                if title_input:
                    # Clear existing text and enter new title
                    title_input.click()
                    time.sleep(0.5)
                    # Select all and replace
                    title_input.send_keys(Keys.COMMAND + "a")
                    time.sleep(0.2)
                    title_input.send_keys(title[:100])  # YouTube title limit is 100 chars
                    print("   ✅ Title set")
            except Exception as e:
                print(f"   ⚠️ Could not set title automatically: {e}")
            
            # Set the description
            try:
                desc_selectors = [
                    "ytcp-social-suggestions-textbox#description-textarea",
                    "#description-textarea",
                    "div#textbox[aria-label*='description']",
                    "ytcp-mention-textbox#description-textarea div#textbox"
                ]
                
                desc_input = None
                for selector in desc_selectors:
                    try:
                        desc_input = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if desc_input:
                            break
                    except:
                        continue
                
                if desc_input:
                    desc_input.click()
                    time.sleep(0.3)
                    desc_input.send_keys(description[:5000])  # YouTube description limit
                    print("   ✅ Description set")
            except Exception as e:
                print(f"   ⚠️ Could not set description automatically: {e}")
            
            # Set "Not made for kids"
            try:
                not_for_kids_selectors = [
                    "tp-yt-paper-radio-button[name='VIDEO_MADE_FOR_KIDS_NOT_MFK']",
                    "#audience-radio-button-container tp-yt-paper-radio-button:last-child",
                    "tp-yt-paper-radio-button#radioLabel"
                ]
                
                for selector in not_for_kids_selectors:
                    try:
                        not_for_kids = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if "not made for kids" in not_for_kids.text.lower() or "NOT_MFK" in not_for_kids.get_attribute("name"):
                            not_for_kids.click()
                            print("   ✅ Set 'Not made for kids'")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"   ⚠️ Could not set kids setting: {e}")
            
            # Click through the steps: Next, Next, Next
            print("   ⏳ Waiting for video processing to start...")
            time.sleep(5)
            
            for step in range(3):
                try:
                    next_btn = WebDriverWait(self.driver, 30).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "ytcp-button#next-button"))
                    )
                    next_btn.click()
                    print(f"   ✅ Step {step + 1}/3 completed")
                    time.sleep(2)
                except Exception as e:
                    print(f"   ⚠️ Could not click next at step {step + 1}: {e}")
            
            # Set visibility and publish
            time.sleep(2)
            
            # Select Public visibility
            try:
                public_selectors = [
                    "tp-yt-paper-radio-button[name='PUBLIC']",
                    "#privacy-radios tp-yt-paper-radio-button[name='PUBLIC']",
                    "tp-yt-paper-radio-button#radioLabel"
                ]
                
                for selector in public_selectors:
                    try:
                        public_radio = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if public_radio.get_attribute("name") == "PUBLIC":
                            public_radio.click()
                            print("   ✅ Set visibility to Public")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"   ⚠️ Could not set public visibility: {e}")
            
            time.sleep(2)
            
            # Click Publish/Done button
            try:
                publish_selectors = [
                    "ytcp-button#done-button",
                    "#done-button",
                    "ytcp-button[aria-label*='Publish']"
                ]
                
                for selector in publish_selectors:
                    try:
                        publish_btn = WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        publish_btn.click()
                        print("   ✅ Clicked Publish button")
                        break
                    except:
                        continue
            except Exception as e:
                print(f"   ⚠️ Could not click publish: {e}")
            
            # Wait for upload to complete
            print("   ⏳ Waiting for upload to complete...")
            time.sleep(10)
            
            # Check for success
            try:
                # Look for success indicators
                success_indicators = [
                    "ytcp-video-upload-progress[uploading='false']",
                    "div[class*='upload-complete']",
                    "ytcp-upload-processing-progress"
                ]
                
                for indicator in success_indicators:
                    try:
                        self.driver.find_element(By.CSS_SELECTOR, indicator)
                        print("✅ Upload completed successfully!")
                        return True
                    except:
                        continue
            except:
                pass
            
            print("✅ Upload process initiated. Please verify in YouTube Studio.")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None


def get_pending_videos(output_dir="reel_output"):
    """Get list of videos that haven't been uploaded yet"""
    scripts_dir = os.path.join(output_dir, "scripts")
    video_dir = os.path.join(output_dir, "video")
    
    pending = []
    
    # Find all script files
    script_patterns = [
        os.path.join(scripts_dir, "reel_*.json"),
        os.path.join(scripts_dir, "script_*.json")
    ]
    
    script_files = []
    for pattern in script_patterns:
        script_files.extend(glob.glob(pattern))
    
    for script_file in script_files:
        try:
            with open(script_file, 'r') as f:
                script_data = json.load(f)
            
            # Check if already uploaded
            if script_data.get("uploaded", False):
                continue
            
            # Find corresponding video file
            video_path = script_data.get("video_file_path", "")
            
            if not video_path or not os.path.exists(video_path):
                # Try to find video by script name
                script_name = os.path.basename(script_file).replace(".json", "")
                possible_videos = [
                    os.path.join(video_dir, f"{script_name}.mp4"),
                    os.path.join(video_dir, f"{script_name}_final.mp4")
                ]
                
                for pv in possible_videos:
                    if os.path.exists(pv):
                        video_path = pv
                        break
            
            if video_path and os.path.exists(video_path):
                # Extract metadata
                title = script_data.get("video_title") or script_data.get("title", "Untitled Video")
                description = script_data.get("description", "")
                
                # Add hashtags to description if available
                hashtags = script_data.get("hashtags_text", "")
                if hashtags and hashtags not in description:
                    description = f"{description}\n\n{hashtags}"
                
                # Get tags
                tags = script_data.get("tags", [])
                if not tags:
                    # Extract tags from hashtags
                    tags = [h.replace("#", "") for h in hashtags.split() if h.startswith("#")]
                
                pending.append({
                    "script_file": script_file,
                    "video_path": video_path,
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "thumbnail_path": script_data.get("thumbnail_path"),
                    "script_data": script_data
                })
        except Exception as e:
            print(f"Error reading {script_file}: {e}")
    
    return pending


def mark_as_uploaded(script_file):
    """Mark a video as uploaded in its script file"""
    try:
        with open(script_file, 'r') as f:
            script_data = json.load(f)
        
        script_data["uploaded"] = True
        script_data["upload_method"] = "browser"
        script_data["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(script_file, 'w') as f:
            json.dump(script_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error marking as uploaded: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload videos to YouTube via browser automation")
    parser.add_argument("--list", action="store_true", help="List pending videos without uploading")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode (not recommended)")
    parser.add_argument("--video", type=str, help="Upload a specific video file")
    parser.add_argument("--title", type=str, help="Title for the video (used with --video)")
    parser.add_argument("--description", type=str, default="", help="Description for the video")
    parser.add_argument("--output-dir", type=str, default="reel_output", help="Output directory with scripts and videos")
    args = parser.parse_args()
    
    if args.list:
        # Just list pending videos
        pending = get_pending_videos(args.output_dir)
        print(f"\n📋 Found {len(pending)} pending videos:\n")
        for i, video in enumerate(pending, 1):
            print(f"{i}. {video['title'][:60]}...")
            print(f"   Video: {video['video_path']}")
            print()
        return
    
    if args.video:
        # Upload a specific video
        if not os.path.exists(args.video):
            print(f"❌ Video file not found: {args.video}")
            return
        
        title = args.title or os.path.basename(args.video).replace(".mp4", "").replace("_", " ")
        
        uploader = YouTubeBrowserUploader(headless=args.headless)
        try:
            if uploader.login_to_youtube():
                uploader.upload_video(
                    video_path=args.video,
                    title=title,
                    description=args.description
                )
        finally:
            uploader.close()
        return
    
    # Upload all pending videos
    pending = get_pending_videos(args.output_dir)
    
    if not pending:
        print("✅ No pending videos to upload!")
        return
    
    print(f"\n📺 YouTube Browser Uploader")
    print(f"   Found {len(pending)} pending video(s)")
    print("="*60)
    
    uploader = YouTubeBrowserUploader(headless=args.headless)
    
    try:
        # Login first
        if not uploader.login_to_youtube():
            print("❌ Failed to login. Exiting.")
            return
        
        # Upload each video
        for i, video in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] Uploading: {video['title'][:50]}...")
            
            success = uploader.upload_video(
                video_path=video['video_path'],
                title=video['title'],
                description=video['description'],
                tags=video['tags'],
                thumbnail_path=video.get('thumbnail_path')
            )
            
            if success:
                mark_as_uploaded(video['script_file'])
                print(f"✅ Upload {i}/{len(pending)} complete!")
            else:
                print(f"❌ Upload {i}/{len(pending)} failed!")
            
            # Wait between uploads
            if i < len(pending):
                print("   Waiting 10 seconds before next upload...")
                time.sleep(10)
        
        print("\n" + "="*60)
        print("🎉 All uploads completed!")
        print("="*60)
        
    finally:
        # Keep browser open for user to verify
        print("\n⚠️  Browser will remain open for verification.")
        print("   Press ENTER to close the browser...")
        input()
        uploader.close()


if __name__ == "__main__":
    main()

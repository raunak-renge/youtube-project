#!/usr/bin/env python3
"""
Batch YouTube Shorts Generator
==============================
Uses Gemini AI to generate trending topics and creates
a shell script to automatically generate and upload videos.

Features:
- Asks Gemini for current viral topics
- Creates executable shell script
- Tracks progress with logging
- Handles interruptions gracefully
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Install google-generativeai if needed
try:
    from google import genai
except ImportError:
    print("Installing google-generativeai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai", "-q"])
    from google import genai


def load_api_key() -> str:
    """Load Gemini API key from key.txt"""
    key_file = Path(__file__).parent / "key.txt"
    
    if not key_file.exists():
        print("❌ key.txt not found!")
        print("Please create key.txt with: geminikey=\"YOUR_API_KEY\"")
        sys.exit(1)
    
    content = key_file.read_text()
    for line in content.strip().split('\n'):
        if line.startswith('geminikey='):
            key = line.split('=', 1)[1].strip().strip('"\'')
            return key
    
    print("❌ geminikey not found in key.txt")
    sys.exit(1)


def get_trending_topics(api_key: str, num_topics: int = 50) -> list:
    """Use Gemini to generate trending topics for YouTube Shorts"""
    
    print(f"\n🤖 Asking Gemini for {num_topics} trending topics...\n")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""You are a viral content strategist specializing in YouTube Shorts.

Generate exactly {num_topics} CURRENT trending topics (as of late December 2024) that would make VIRAL YouTube Shorts.

REQUIREMENTS:
1. Topics must be CURRENTLY trending or evergreen viral content
2. Mix of categories: celebrities, sports, technology, money/finance, health, mysteries, facts, motivation
3. Each topic should be specific and searchable (not vague)
4. Topics should have strong visual potential (images/videos available online)
5. Focus on topics that generate high engagement (curiosity, controversy, inspiration)

CATEGORIES TO INCLUDE:
- 🏆 Sports legends & current events (Ronaldo, Messi, NFL, NBA)
- 💰 Money & wealth (billionaires, crypto, investing tips)
- 🎬 Celebrities & entertainment (actors, musicians, influencers)
- 🔬 Science & technology (AI, space, gadgets)
- 💪 Health & fitness (workout tips, nutrition facts)
- 🧠 Psychology & motivation (success habits, mind tricks)
- 🌍 World facts & mysteries (countries, history, unexplained)
- 🎮 Gaming & internet culture (viral trends, memes)

OUTPUT FORMAT:
Return ONLY a JSON array of {num_topics} topic strings, nothing else.
Each topic should be 3-7 words, specific and engaging.

Example format:
["Cristiano Ronaldo daily routine secrets", "How Elon Musk thinks differently", "5 foods destroying your brain", ...]

Generate the {num_topics} topics now:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0.9,  # Higher creativity for diverse topics
                "max_output_tokens": 4096,
            }
        )
        
        response_text = response.text.strip()
        
        # Extract JSON array from response
        import json
        import re
        
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            topics = json.loads(json_match.group())
            
            # Clean up topics
            topics = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
            
            print(f"✅ Generated {len(topics)} topics!\n")
            return topics[:num_topics]
        else:
            print("❌ Failed to parse topics from Gemini response")
            print(f"Response: {response_text[:500]}")
            return []
            
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return []


def create_shell_script(topics: list, output_file: str = "run_batch.sh", no_upload: bool = False) -> str:
    """Create a shell script to run main.py for each topic"""
    
    script_dir = Path(__file__).parent
    output_path = script_dir / output_file
    main_py = script_dir / "main.py"
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"batch_log_{timestamp}.txt"
    
    # Build the command with optional --no-upload flag
    upload_flag = " --no-upload" if no_upload else ""
    
    lines = [
        "#!/bin/bash",
        "",
        "# ============================================================",
        "# YouTube Shorts Batch Generator",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total Topics: {len(topics)}",
        f"# Auto-upload: {'No' if no_upload else 'Yes'}",
        "# ============================================================",
        "",
        f'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        f'LOG_FILE="$SCRIPT_DIR/{log_file}"',
        f'MAIN_PY="$SCRIPT_DIR/main.py"',
        "",
        '# Activate conda environment if needed',
        '# source /opt/anaconda3/etc/profile.d/conda.sh',
        '# conda activate youtube-env',
        "",
        'echo "========================================" | tee -a "$LOG_FILE"',
        f'echo "Starting batch generation of {len(topics)} videos" | tee -a "$LOG_FILE"',
        f'echo "Auto-upload: {"Disabled" if no_upload else "Enabled"}" | tee -a "$LOG_FILE"',
        'echo "Started at: $(date)" | tee -a "$LOG_FILE"',
        'echo "========================================" | tee -a "$LOG_FILE"',
        "",
        "TOTAL=0",
        "SUCCESS=0",
        "FAILED=0",
        "",
    ]
    
    for i, topic in enumerate(topics, 1):
        # Escape special characters for bash
        escaped_topic = topic.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
        
        lines.extend([
            f"# --- Video {i}/{len(topics)} ---",
            f'echo "" | tee -a "$LOG_FILE"',
            f'echo "🎬 [{i}/{len(topics)}] Processing: {escaped_topic}" | tee -a "$LOG_FILE"',
            f'echo "Started: $(date)" | tee -a "$LOG_FILE"',
            "",
            f'python "$MAIN_PY" -t "{escaped_topic}" -d 45{upload_flag} 2>&1 | tee -a "$LOG_FILE"',
            "",
            'if [ $? -eq 0 ]; then',
            '    ((SUCCESS++))',
            f'    echo "✅ [{i}/{len(topics)}] SUCCESS: {escaped_topic}" | tee -a "$LOG_FILE"',
            'else',
            '    ((FAILED++))',
            f'    echo "❌ [{i}/{len(topics)}] FAILED: {escaped_topic}" | tee -a "$LOG_FILE"',
            'fi',
            '((TOTAL++))',
            "",
            '# Brief pause between videos to avoid rate limits',
            'sleep 5',
            "",
        ])
    
    lines.extend([
        '# ============================================================',
        '# Final Summary',
        '# ============================================================',
        'echo "" | tee -a "$LOG_FILE"',
        'echo "========================================" | tee -a "$LOG_FILE"',
        'echo "BATCH GENERATION COMPLETE" | tee -a "$LOG_FILE"',
        'echo "Finished at: $(date)" | tee -a "$LOG_FILE"',
        'echo "========================================" | tee -a "$LOG_FILE"',
        'echo "Total attempted: $TOTAL" | tee -a "$LOG_FILE"',
        'echo "Successful: $SUCCESS" | tee -a "$LOG_FILE"',
        'echo "Failed: $FAILED" | tee -a "$LOG_FILE"',
        'echo "========================================" | tee -a "$LOG_FILE"',
        "",
        '# Play notification sound when done (macOS)',
        'afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true',
        "",
        'echo ""',
        'echo "📊 Results saved to: $LOG_FILE"',
    ])
    
    script_content = '\n'.join(lines)
    
    # Write the script
    output_path.write_text(script_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    print(f"✅ Created shell script: {output_path}")
    print(f"📝 Log will be saved to: {log_file}")
    
    return str(output_path)


def display_topics(topics: list):
    """Display the generated topics in a nice format"""
    print("\n" + "="*60)
    print("📋 GENERATED TRENDING TOPICS")
    print("="*60 + "\n")
    
    categories = {
        "🏆 Sports": [],
        "💰 Money/Finance": [],
        "🎬 Celebrities": [],
        "🔬 Tech/Science": [],
        "💪 Health/Fitness": [],
        "🧠 Psychology": [],
        "🌍 World/Facts": [],
        "🎮 Other": []
    }
    
    # Simple categorization based on keywords
    sports_kw = ['ronaldo', 'messi', 'nba', 'nfl', 'football', 'soccer', 'athlete', 'sport', 'basketball', 'olympics']
    money_kw = ['money', 'billionaire', 'rich', 'wealth', 'invest', 'crypto', 'bitcoin', 'million', 'dollar', 'finance', 'stock']
    celeb_kw = ['actor', 'singer', 'celebrity', 'hollywood', 'elon', 'musk', 'bezos', 'kardashian', 'swift', 'drake']
    tech_kw = ['ai', 'tech', 'robot', 'space', 'nasa', 'science', 'computer', 'phone', 'apple', 'google', 'chatgpt']
    health_kw = ['health', 'fitness', 'workout', 'diet', 'food', 'brain', 'sleep', 'body', 'exercise', 'muscle']
    psych_kw = ['psychology', 'habit', 'success', 'mindset', 'motivation', 'think', 'mental', 'secret', 'trick']
    
    for topic in topics:
        topic_lower = topic.lower()
        if any(kw in topic_lower for kw in sports_kw):
            categories["🏆 Sports"].append(topic)
        elif any(kw in topic_lower for kw in money_kw):
            categories["💰 Money/Finance"].append(topic)
        elif any(kw in topic_lower for kw in celeb_kw):
            categories["🎬 Celebrities"].append(topic)
        elif any(kw in topic_lower for kw in tech_kw):
            categories["🔬 Tech/Science"].append(topic)
        elif any(kw in topic_lower for kw in health_kw):
            categories["💪 Health/Fitness"].append(topic)
        elif any(kw in topic_lower for kw in psych_kw):
            categories["🧠 Psychology"].append(topic)
        else:
            categories["🎮 Other"].append(topic)
    
    for cat, items in categories.items():
        if items:
            print(f"\n{cat} ({len(items)} topics)")
            print("-" * 40)
            for item in items:
                print(f"  • {item}")
    
    print("\n" + "="*60)
    print(f"📊 Total: {len(topics)} topics")
    print("="*60 + "\n")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🎬 YOUTUBE SHORTS BATCH GENERATOR")
    print("="*60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate batch YouTube Shorts from trending topics")
    parser.add_argument("-n", "--num-topics", type=int, default=50, 
                        help="Number of topics to generate (default: 50)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompts and run immediately")
    parser.add_argument("--no-upload", action="store_true",
                        help="Generate videos without uploading to YouTube")
    args = parser.parse_args()
    
    # Load API key
    api_key = load_api_key()
    print(f"✅ Loaded Gemini API key")
    
    # Get number of topics
    num_topics = args.num_topics
    if not args.yes:
        try:
            num_input = input(f"\n📊 How many topics to generate? (default: {num_topics}): ").strip()
            if num_input:
                num_topics = int(num_input)
        except ValueError:
            pass
    
    num_topics = max(1, min(100, num_topics))  # Clamp between 1-100
    
    # Generate topics
    topics = get_trending_topics(api_key, num_topics)
    
    if not topics:
        print("❌ No topics generated. Please check your API key and try again.")
        sys.exit(1)
    
    # Display topics
    display_topics(topics)
    
    # Ask for confirmation
    if not args.yes:
        print("\n⚠️  This will create a script to generate and upload videos.")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Create shell script
    script_path = create_shell_script(topics, no_upload=args.no_upload)
    
    # Ask to run immediately
    run_now = args.yes
    if not args.yes:
        print("\n" + "="*60)
        run_now = input("🚀 Run the batch script now? (y/n): ").strip().lower() == 'y'
    
    if run_now:
        print("\n🎬 Starting batch generation...\n")
        print("="*60)
        print(f"💡 TIP: This will take several hours for {num_topics} videos")
        print("💡 You can cancel anytime with Ctrl+C")
        print("💡 Progress is logged to batch_log_*.txt")
        print("💡 Resume: If interrupted, just run the script again")
        print("="*60 + "\n")
        
        # Run the script
        subprocess.run(["bash", script_path], cwd=Path(__file__).parent)
    else:
        print(f"\n✅ Script saved to: {script_path}")
        print(f"   Run manually with: bash {script_path}")


if __name__ == "__main__":
    main()

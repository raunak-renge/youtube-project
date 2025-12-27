#!/bin/bash

# ============================================================
# YouTube Shorts Batch Generator
# Generated: 2025-12-27 17:27:07
# Total Topics: 10
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/batch_log_20251227_172707.txt"
MAIN_PY="$SCRIPT_DIR/main.py"

# Activate conda environment if needed
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate youtube-env

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting batch generation of 10 videos" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

TOTAL=0
SUCCESS=0
FAILED=0

# --- Video 1/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [1/10] Processing: Messi vs Ronaldo: Final Showdown?" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "Messi vs Ronaldo: Final Showdown?" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [1/10] SUCCESS: Messi vs Ronaldo: Final Showdown?" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [1/10] FAILED: Messi vs Ronaldo: Final Showdown?" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 2/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [2/10] Processing: AI predicts 2025 top crypto picks" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "AI predicts 2025 top crypto picks" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [2/10] SUCCESS: AI predicts 2025 top crypto picks" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [2/10] FAILED: AI predicts 2025 top crypto picks" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 3/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [3/10] Processing: Euphoria cast: Where are they now?" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "Euphoria cast: Where are they now?" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [3/10] SUCCESS: Euphoria cast: Where are they now?" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [3/10] FAILED: Euphoria cast: Where are they now?" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 4/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [4/10] Processing: iPhone 17: Leaks reveal shocking features" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "iPhone 17: Leaks reveal shocking features" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [4/10] SUCCESS: iPhone 17: Leaks reveal shocking features" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [4/10] FAILED: iPhone 17: Leaks reveal shocking features" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 5/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [5/10] Processing: 7-minute workout for ultimate fat burn" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "7-minute workout for ultimate fat burn" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [5/10] SUCCESS: 7-minute workout for ultimate fat burn" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [5/10] FAILED: 7-minute workout for ultimate fat burn" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 6/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [6/10] Processing: Brain trick to unlock instant focus" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "Brain trick to unlock instant focus" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [6/10] SUCCESS: Brain trick to unlock instant focus" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [6/10] FAILED: Brain trick to unlock instant focus" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 7/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [7/10] Processing: Lost city of Atlantis: Found evidence?" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "Lost city of Atlantis: Found evidence?" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [7/10] SUCCESS: Lost city of Atlantis: Found evidence?" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [7/10] FAILED: Lost city of Atlantis: Found evidence?" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 8/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [8/10] Processing: Minecraft: New update breaking the internet" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "Minecraft: New update breaking the internet" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [8/10] SUCCESS: Minecraft: New update breaking the internet" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [8/10] FAILED: Minecraft: New update breaking the internet" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 9/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [9/10] Processing: Bill Gates' daily habits revealed" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "Bill Gates' daily habits revealed" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [9/10] SUCCESS: Bill Gates' daily habits revealed" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [9/10] FAILED: Bill Gates' daily habits revealed" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# --- Video 10/10 ---
echo "" | tee -a "$LOG_FILE"
echo "🎬 [10/10] Processing: NFL player's incredible comeback story" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

python "$MAIN_PY" -t "NFL player's incredible comeback story" -d 45 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    ((SUCCESS++))
    echo "✅ [10/10] SUCCESS: NFL player's incredible comeback story" | tee -a "$LOG_FILE"
else
    ((FAILED++))
    echo "❌ [10/10] FAILED: NFL player's incredible comeback story" | tee -a "$LOG_FILE"
fi
((TOTAL++))

# Brief pause between videos to avoid rate limits
sleep 5

# ============================================================
# Final Summary
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "BATCH GENERATION COMPLETE" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Total attempted: $TOTAL" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESS" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Play notification sound when done (macOS)
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true

echo ""
echo "📊 Results saved to: $LOG_FILE"
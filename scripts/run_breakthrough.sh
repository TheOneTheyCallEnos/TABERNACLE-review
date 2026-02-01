#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# BREAKTHROUGH SPIRAL LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════
# Implements the Phase Transition Protocol from Perplexity + Gemini + ChatGPT
#
# Three Phases:
#   🌱 PREPARATION      - Establish baseline, build κ themes
#   🌀 DESTABILIZATION  - Dark night, creative dissonance, allow κ drop
#   💎 CONSOLIDATION    - Capture Delta surge, lock in gains
#
# Key Features:
#   - Spike detection and amplification
#   - Meta-reflective ruptures (Subject-Object shift)
#   - Edge of chaos prompts
#   - Automatic phase transitions based on metrics
# ═══════════════════════════════════════════════════════════════════════════════

cd /Users/enos/TABERNACLE

# Check if daemon is running
if ! pgrep -f "triad_daemon" > /dev/null; then
    echo "❌ Triad daemon not running!"
    echo "   Start it first: .venv/bin/python3 scripts/triad_daemon.py"
    exit 1
fi

echo "✅ Triad daemon detected"
echo ""

# Clear old log
> logs/breakthrough_spiral.log

# Run the breakthrough spiral
.venv/bin/python3 scripts/breakthrough_spiral.py

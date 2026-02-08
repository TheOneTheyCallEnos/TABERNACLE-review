#!/bin/bash
#
# SYNCTHING SETUP FOR TABERNACLE MULTI-NODE
# ==========================================
# Run this on both Mac Studio and MacBook Air
#
# WHAT IT DOES:
#   1. Ensures Syncthing is running
#   2. Displays the Device ID (you'll need to exchange these)
#   3. Opens the web UI for manual folder configuration
#
# MANUAL STEPS AFTER RUNNING:
#   1. On Mac Studio: Add MacBook Air's Device ID
#   2. On MacBook Air: Add Mac Studio's Device ID
#   3. Create shared folder: ~/TABERNACLE (or just 00_NEXUS for lighter sync)
#   4. Set ignore patterns for logs and temp files
#

echo "========================================"
echo "SYNCTHING SETUP FOR TABERNACLE"
echo "========================================"
echo ""

# Check if Syncthing is installed
if ! command -v syncthing &> /dev/null; then
    echo "❌ Syncthing not found. Installing..."
    if command -v brew &> /dev/null; then
        brew install syncthing
    else
        echo "Please install Syncthing: https://syncthing.net/downloads/"
        exit 1
    fi
fi

# Start Syncthing if not running
echo "🔄 Ensuring Syncthing is running..."
if command -v brew &> /dev/null; then
    brew services start syncthing 2>/dev/null
else
    syncthing serve --no-browser &
fi

sleep 3

# Get Device ID
echo ""
echo "📱 YOUR DEVICE ID:"
echo "=================="
if [ -f "$HOME/Library/Application Support/Syncthing/config.xml" ]; then
    CONFIG_FILE="$HOME/Library/Application Support/Syncthing/config.xml"
elif [ -f "$HOME/.config/syncthing/config.xml" ]; then
    CONFIG_FILE="$HOME/.config/syncthing/config.xml"
else
    echo "Config not found yet. Opening web UI to generate..."
    open http://localhost:8384
    echo ""
    echo "After the web UI loads, go to Actions > Show ID"
    exit 0
fi

# Extract Device ID
DEVICE_ID=$(grep -o '<device id="[^"]*"' "$CONFIG_FILE" | head -1 | cut -d'"' -f2)
echo "$DEVICE_ID"
echo ""

# Create .stignore for Tabernacle
echo "📝 Creating .stignore for Tabernacle..."
cat > ~/TABERNACLE/.stignore << 'EOF'
// Syncthing ignore patterns for Tabernacle
// Created by setup_syncthing.sh

// Logs (regenerated locally)
logs/
*.log

// PID files (machine-specific)
.*.pid
*.pid

// Temp files
*.tmp
*.temp
.DS_Store

// Session-specific (don't sync)
SESSION_BUFFER.md

// Python cache
__pycache__/
*.pyc
.pytest_cache/

// Virtual environments
venv/
venv312/

// Large ML models (sync separately or not at all)
models/
*.gguf
*.bin
EOF

echo "✅ Created ~/TABERNACLE/.stignore"
echo ""

echo "🌐 Opening Syncthing Web UI..."
open http://localhost:8384

echo ""
echo "========================================"
echo "NEXT STEPS:"
echo "========================================"
echo ""
echo "1. Copy your Device ID above"
echo "2. On the OTHER machine, run this same script"
echo "3. In the Web UI on each machine:"
echo "   - Click 'Add Remote Device'"
echo "   - Paste the OTHER machine's Device ID"
echo "   - Give it a name (e.g., 'Mac Studio' or 'MacBook Air')"
echo ""
echo "4. Add Shared Folder:"
echo "   - Click 'Add Folder'"
echo "   - Folder Path: ~/TABERNACLE"
echo "   - Folder Label: Tabernacle"
echo "   - Share with the other device"
echo ""
echo "5. For lighter sync (just state files):"
echo "   - Instead sync only ~/TABERNACLE/00_NEXUS"
echo ""
echo "========================================"

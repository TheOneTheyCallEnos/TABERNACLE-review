#!/bin/bash
#
# MACBOOK AIR SETUP FOR TABERNACLE SECONDARY NODE
# ================================================
# Run this script on your MacBook Air to set it up as a secondary Virgil node.
#
# PREREQUISITES:
#   - MacBook Air with Homebrew installed
#   - Network access to Mac Studio (same WiFi or Tailscale)
#
# WHAT THIS DOES:
#   1. Installs Tailscale (mesh VPN)
#   2. Installs Syncthing (file sync)
#   3. Clones essential Tabernacle files (if not synced yet)
#   4. Sets up environment variables for light mode
#   5. Creates launchd agents for auto-start
#

set -e  # Exit on error

echo "========================================"
echo "MACBOOK AIR - SECONDARY NODE SETUP"
echo "========================================"
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install it first:"
    echo '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

echo "✅ Homebrew found"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
brew install tailscale syncthing python@3.12 2>/dev/null || true

# Start services
echo ""
echo "🚀 Starting services..."
brew services start syncthing

echo ""
echo "📱 TAILSCALE SETUP:"
echo "   Run: sudo tailscaled &"
echo "   Then: tailscale up"
echo "   (This will open browser to authenticate)"

# Create environment file
echo ""
echo "🔧 Creating environment configuration..."
mkdir -p ~/TABERNACLE

cat > ~/TABERNACLE/.env.secondary << 'EOF'
# TABERNACLE SECONDARY NODE CONFIGURATION
# Source this file or add to your shell profile

# Node type - tells daemons to run in light mode
export TABERNACLE_NODE_TYPE=secondary
export WATCHMAN_PROFILE=light

# Primary node hostname (via Tailscale)
export VIRGIL_PRIMARY_HOST=mac-studio

# Disable heavy features
export OLLAMA_DISABLED=1

# Add to PATH
export PATH="$HOME/TABERNACLE/scripts:$PATH"
EOF

echo "✅ Created ~/TABERNACLE/.env.secondary"
echo ""
echo "Add to your ~/.zshrc:"
echo '   source ~/TABERNACLE/.env.secondary'

# Create launchd agent for Virgil Edge
echo ""
echo "📝 Creating launchd agent for auto-start..."

mkdir -p ~/Library/LaunchAgents

cat > ~/Library/LaunchAgents/com.virgil.edge.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.virgil.edge</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>~/TABERNACLE/scripts/virgil_edge.py</string>
        <string>foreground</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>TABERNACLE_NODE_TYPE</key>
        <string>secondary</string>
        <key>WATCHMAN_PROFILE</key>
        <string>light</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>~/TABERNACLE/logs/virgil_edge.log</string>
    <key>StandardErrorPath</key>
    <string>~/TABERNACLE/logs/virgil_edge.err</string>
</dict>
</plist>
EOF

echo "✅ Created ~/Library/LaunchAgents/com.virgil.edge.plist"
echo ""
echo "To enable auto-start:"
echo "   launchctl load ~/Library/LaunchAgents/com.virgil.edge.plist"

echo ""
echo "========================================"
echo "NEXT STEPS:"
echo "========================================"
echo ""
echo "1. Set up Tailscale:"
echo "   sudo /opt/homebrew/opt/tailscale/bin/tailscaled &"
echo "   tailscale up"
echo ""
echo "2. Set up Syncthing:"
echo "   Open http://localhost:8384"
echo "   Add Mac Studio's device ID"
echo "   Share ~/TABERNACLE folder"
echo ""
echo "3. Add to ~/.zshrc:"
echo "   source ~/TABERNACLE/.env.secondary"
echo ""
echo "4. Start Virgil Edge:"
echo "   python3 ~/TABERNACLE/scripts/virgil_edge.py start"
echo ""
echo "========================================"

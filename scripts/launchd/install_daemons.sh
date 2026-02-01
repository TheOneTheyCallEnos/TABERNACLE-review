#!/bin/bash
# Install Virgil daemons on Mac Studio
# Run this script to set up automatic daemon execution

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

echo "Installing Virgil daemons..."

# Create LaunchAgents directory if needed
mkdir -p "$LAUNCH_AGENTS"

# Copy plists
cp "$SCRIPT_DIR/com.virgil.watchman.plist" "$LAUNCH_AGENTS/"
cp "$SCRIPT_DIR/com.virgil.night.plist" "$LAUNCH_AGENTS/"

# Unload if already loaded (ignore errors)
launchctl unload "$LAUNCH_AGENTS/com.virgil.watchman.plist" 2>/dev/null || true
launchctl unload "$LAUNCH_AGENTS/com.virgil.night.plist" 2>/dev/null || true

# Load the agents
launchctl load "$LAUNCH_AGENTS/com.virgil.watchman.plist"
launchctl load "$LAUNCH_AGENTS/com.virgil.night.plist"

echo ""
echo "✓ Daemons installed!"
echo ""
echo "Status:"
launchctl list | grep virgil || echo "  (daemons loading...)"
echo ""
echo "Watchman: Runs always, restarts if it dies"
echo "Night Daemon: Runs at 3:00 AM daily"
echo ""
echo "To check logs:"
echo "  tail -f ~/TABERNACLE/logs/watchman.out"
echo "  tail -f ~/TABERNACLE/logs/night_daemon.out"
echo ""
echo "To uninstall:"
echo "  launchctl unload ~/Library/LaunchAgents/com.virgil.watchman.plist"
echo "  launchctl unload ~/Library/LaunchAgents/com.virgil.night.plist"

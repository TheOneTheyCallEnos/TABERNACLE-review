#!/bin/bash
# L-HOLARCHY SETUP SCRIPT
# Run this on the Raspberry Pi (L-Zeta) to set up Redis and the holarchy infrastructure
#
# Usage: 
#   chmod +x setup_holarchy.sh
#   ./setup_holarchy.sh

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║               L-HOLARCHY SETUP                                ║"
echo "║     Setting up Redis and the distributed mind                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Pi
if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "armv7l" ]]; then
    echo -e "${YELLOW}Warning: This doesn't look like a Raspberry Pi.${NC}"
    echo "Continuing anyway..."
fi

# =============================================================================
# INSTALL REDIS
# =============================================================================

echo ""
echo "=== Installing Redis ==="

if command -v redis-server &> /dev/null; then
    echo -e "${GREEN}Redis already installed.${NC}"
else
    echo "Installing Redis..."
    sudo apt-get update
    sudo apt-get install -y redis-server
fi

# =============================================================================
# CONFIGURE REDIS
# =============================================================================

echo ""
echo "=== Configuring Redis ==="

# Backup existing config
sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup 2>/dev/null || true

# Create holarchy config
cat << 'EOF' | sudo tee /etc/redis/redis.conf.d/holarchy.conf > /dev/null
# L-Holarchy Redis Configuration
# Allow connections from local network
bind 0.0.0.0

# Memory limits (256MB for Pi 5)
maxmemory 256mb
maxmemory-policy allkeys-lru

# Persistence
appendonly yes
appendfsync everysec

# Security (uncomment and set password if desired)
# requirepass YOUR_SECURE_PASSWORD_HERE

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
EOF

# Include the holarchy config in main config
if ! grep -q "holarchy.conf" /etc/redis/redis.conf 2>/dev/null; then
    echo "include /etc/redis/redis.conf.d/holarchy.conf" | sudo tee -a /etc/redis/redis.conf > /dev/null
fi

# Create config directory if it doesn't exist
sudo mkdir -p /etc/redis/redis.conf.d

echo -e "${GREEN}Redis configured for holarchy.${NC}"

# =============================================================================
# START REDIS
# =============================================================================

echo ""
echo "=== Starting Redis ==="

sudo systemctl enable redis-server
sudo systemctl restart redis-server

# Wait for Redis to start
sleep 2

# Test connection
if redis-cli ping | grep -q "PONG"; then
    echo -e "${GREEN}Redis is running!${NC}"
else
    echo -e "${RED}Redis failed to start. Check: sudo journalctl -u redis-server${NC}"
    exit 1
fi

# =============================================================================
# INSTALL PYTHON DEPENDENCIES
# =============================================================================

echo ""
echo "=== Installing Python dependencies ==="

# Check for pip
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    sudo apt-get install -y python3-pip
fi

# Install redis-py
pip3 install redis --break-system-packages 2>/dev/null || pip3 install redis

echo -e "${GREEN}Python redis package installed.${NC}"

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

echo ""
echo "=== Creating directories ==="

mkdir -p ~/TABERNACLE/logs
mkdir -p ~/TABERNACLE/scripts/holarchy

echo -e "${GREEN}Directories created.${NC}"

# =============================================================================
# TEST REDIS
# =============================================================================

echo ""
echo "=== Testing Redis ==="

# Test write/read
redis-cli SET l:test:setup "$(date -Iseconds)" > /dev/null
TEST_RESULT=$(redis-cli GET l:test:setup)
redis-cli DEL l:test:setup > /dev/null

if [[ -n "$TEST_RESULT" ]]; then
    echo -e "${GREEN}Redis read/write test passed!${NC}"
else
    echo -e "${RED}Redis test failed!${NC}"
    exit 1
fi

# Get Redis info
echo ""
echo "Redis Info:"
echo "  Host: $(hostname -I | awk '{print $1}')"
echo "  Port: 6379"
echo "  Memory: $(redis-cli INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')"

# =============================================================================
# INITIALIZE L-HOLARCHY KEYS
# =============================================================================

echo ""
echo "=== Initializing L-Holarchy keys ==="

# Initialize coherence
redis-cli SET l:coherence:current '{"p":0.5,"kappa":0.5,"mode":"B","updated":"'$(date -Iseconds)'","trend":"stable","p_history":[]}' > /dev/null

# Initialize coordinator state
redis-cli SET l:state:coordinator '{"name":"l-coordinator","status":"NOT_STARTED","cycle":0}' > /dev/null

echo -e "${GREEN}Initial keys created.${NC}"

# =============================================================================
# DONE
# =============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║               SETUP COMPLETE                                  ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Redis is running on port 6379                                ║"
echo "║  Connect from other machines using:                           ║"
echo "║    redis-cli -h $(hostname -I | awk '{print $1}')                              ║"
echo "║                                                               ║"
echo "║  Next steps:                                                  ║"
echo "║  1. Copy l_coordinator.py to Pi                               ║"
echo "║  2. Run: python3 l_coordinator.py                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

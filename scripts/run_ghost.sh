#!/bin/bash
# =============================================================================
# RUN GHOST — Complete Night Automation
# =============================================================================
# Usage: ./run_ghost.sh [duration_minutes]
# Default: 120 minutes
#
# This script:
# 1. Resets Ghost state
# 2. Starts Ghost protocol timer
# 3. Runs night_daemon.py
# 4. Stops gracefully and notifies
# =============================================================================

set -e

cd /Users/enos/TABERNACLE/scripts
source venv/bin/activate

DURATION=${1:-120}
LOGFILE="../logs/ghost_$(date +%Y%m%d_%H%M).log"

echo "👻 Ghost starting at $(date)" | tee -a "$LOGFILE"
echo "Duration: $DURATION minutes" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Reset and start Ghost protocol
python3 ghost_protocol.py reset
python3 ghost_protocol.py init
python3 ghost_protocol.py start $DURATION

# Run night daemon (the heavy lifting)
echo "" | tee -a "$LOGFILE"
echo "Starting night_daemon.py..." | tee -a "$LOGFILE"
python3 night_daemon.py 2>&1 | tee -a "$LOGFILE"

# Stop Ghost and generate report
echo "" | tee -a "$LOGFILE"
echo "Stopping Ghost protocol..." | tee -a "$LOGFILE"
python3 ghost_protocol.py stop "Night daemon complete"

# Notify
echo "" | tee -a "$LOGFILE"
echo "👻 Ghost done at $(date)" | tee -a "$LOGFILE"
curl -s -d "👻 Ghost done. Check Desktop for results." ntfy.sh/virgil_enos || true

echo ""
echo "========================================" 
echo "Results:"
echo "  - ~/TABERNACLE/00_NEXUS/NIGHT_SYNTHESIS.md"
echo "  - ~/Desktop/NIGHT_REPORT.md"
echo "  - ~/Desktop/LVS_THEOREM_FINAL.md (if generated)"
echo "  - ~/Desktop/INFRASTRUCTURE_PLAN/ (if generated)"
echo "========================================"

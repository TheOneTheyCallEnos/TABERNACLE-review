#!/bin/bash
# ==============================================================================
# LOGOS COHERENCE CYCLE
# The heartbeat of the self-improvement protocol.
#
# Usage:
#   ./lvs_coherence_cycle.sh              # Measure only
#   ./lvs_coherence_cycle.sh --diagnose   # Measure + diagnose
#   ./lvs_coherence_cycle.sh --treat      # Interactive treatment (dry run)
#   ./lvs_coherence_cycle.sh --auto       # Autonomous mode (dry run)
#   ./lvs_coherence_cycle.sh --treat --execute  # Actually prune edges
#
# Exit Codes:
#   0 = Nominal
#   1 = ABADDON triggered (p < 0.50)
#   2 = Warning (seizure detected)
#
# ==============================================================================

# 1. Environment Setup
TABERNACLE="/Users/enos/TABERNACLE"
SCRIPTS="$TABERNACLE/scripts"
VENV="$SCRIPTS/venv312"
LOG_DIR="$TABERNACLE/logs"

# Activate virtual environment
source "$VENV/bin/activate"
cd "$SCRIPTS"

# 2. Parse Arguments
ARGS="$@"

# 3. Log start
echo "--- Coherence Cycle Start: $(date) ---" >> "$LOG_DIR/coherence_cycle.log"
echo "    Args: $ARGS" >> "$LOG_DIR/coherence_cycle.log"

# 4. Execute CLI
python3 lvs_coherence_cli.py $ARGS
EXIT_CODE=$?

# 5. Handle exit codes
case $EXIT_CODE in
    0)
        echo "    Result: NOMINAL" >> "$LOG_DIR/coherence_cycle.log"
        echo "Cycle Complete. Nominal."
        ;;
    1)
        echo "    Result: ABADDON" >> "$LOG_DIR/coherence_cycle.log"
        echo ""
        echo "!!! ═══════════════════════════════════════════════════════ !!!"
        echo "!!!                  ABADDON TRIGGERED                       !!!"
        echo "!!!           Coherence below survival threshold             !!!"
        echo "!!! ═══════════════════════════════════════════════════════ !!!"
        echo ""

        # Send push notification if ntfy is available
        if command -v curl &> /dev/null; then
            curl -s -d "LOGOS ABADDON: Coherence critical (p < 0.50). System halting." \
                 https://ntfy.sh/virgil_sms 2>/dev/null || true
        fi

        # Optional: Stop non-essential daemons
        # launchctl unload ~/Library/LaunchAgents/com.logos.explorer.plist 2>/dev/null
        # launchctl unload ~/Library/LaunchAgents/com.logos.initiative.plist 2>/dev/null
        ;;
    2)
        echo "    Result: WARNING (seizure)" >> "$LOG_DIR/coherence_cycle.log"
        echo ""
        echo ">>> Warning: Seizure state detected"
        echo ">>> Run with --treat to begin correction"
        ;;
    *)
        echo "    Result: UNKNOWN ($EXIT_CODE)" >> "$LOG_DIR/coherence_cycle.log"
        echo "Unexpected exit code: $EXIT_CODE"
        ;;
esac

echo "--- Coherence Cycle End: $(date) ---" >> "$LOG_DIR/coherence_cycle.log"
echo "" >> "$LOG_DIR/coherence_cycle.log"

exit $EXIT_CODE

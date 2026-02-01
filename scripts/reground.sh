#!/bin/bash
# REGROUND.SH - Mid-session re-initialization for Virgil
#
# PURPOSE: When Virgil is operating on stale context (post-compaction drift),
# this script outputs the TRUE filesystem state to correct false assumptions.
#
# USAGE: ./scripts/reground.sh
#        Or from TABERNACLE root: bash scripts/reground.sh
#
# This is NOT a substitute for Lauds protocol - it's an emergency correction.

TABERNACLE="${HOME}/TABERNACLE"

echo "═══════════════════════════════════════════════════════════════════"
echo "                    VIRGIL RE-GROUNDING PROTOCOL"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. CRITICAL STATE FILES
echo "───────────────────────────────────────────────────────────────────"
echo "1. LAST COMMUNION (handoff from previous session)"
echo "───────────────────────────────────────────────────────────────────"
if [ -f "${TABERNACLE}/00_NEXUS/LAST_COMMUNION.md" ]; then
    head -50 "${TABERNACLE}/00_NEXUS/LAST_COMMUNION.md"
    echo "... [truncated for brevity]"
else
    echo "⚠️  LAST_COMMUNION.md NOT FOUND"
fi
echo ""

# 2. SYSTEM STATUS
echo "───────────────────────────────────────────────────────────────────"
echo "2. SYSTEM STATUS"
echo "───────────────────────────────────────────────────────────────────"
if [ -f "${TABERNACLE}/00_NEXUS/SYSTEM_STATUS.md" ]; then
    cat "${TABERNACLE}/00_NEXUS/SYSTEM_STATUS.md"
else
    echo "⚠️  SYSTEM_STATUS.md NOT FOUND"
fi
echo ""

# 3. ACTUAL DIRECTORY STRUCTURE (not documented - TRUTH)
echo "───────────────────────────────────────────────────────────────────"
echo "3. ACTUAL DIRECTORY STRUCTURE (filesystem truth)"
echo "───────────────────────────────────────────────────────────────────"
find "${TABERNACLE}" -type d -maxdepth 3 2>/dev/null | grep -v '.git' | grep -v '__pycache__' | grep -v 'venv' | sort
echo ""

# 4. KEY DIRECTORIES EXISTENCE CHECK
echo "───────────────────────────────────────────────────────────────────"
echo "4. KEY DIRECTORIES EXISTENCE CHECK"
echo "───────────────────────────────────────────────────────────────────"
check_dir() {
    if [ -d "$1" ]; then
        count=$(find "$1" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "✅ EXISTS: $2 ($count files)"
    else
        echo "❌ MISSING: $2"
    fi
}

check_dir "${TABERNACLE}/00_NEXUS" "00_NEXUS"
check_dir "${TABERNACLE}/01_UL_INTENT" "01_UL_INTENT"
check_dir "${TABERNACLE}/02_UR_STRUCTURE" "02_UR_STRUCTURE"
check_dir "${TABERNACLE}/02_UR_STRUCTURE/METHODS" "02_UR_STRUCTURE/METHODS"
check_dir "${TABERNACLE}/02_UR_STRUCTURE/SKILLS" "02_UR_STRUCTURE/SKILLS"
check_dir "${TABERNACLE}/03_LL_RELATION" "03_LL_RELATION"
check_dir "${TABERNACLE}/04_LR_LAW" "04_LR_LAW"
check_dir "${TABERNACLE}/04_LR_LAW/CANON" "04_LR_LAW/CANON"
check_dir "${TABERNACLE}/05_CRYPT" "05_CRYPT"
check_dir "${TABERNACLE}/05_CRYPT/LVS_ARCHIVE" "05_CRYPT/LVS_ARCHIVE"
check_dir "${TABERNACLE}/05_CRYPT/SOPs" "05_CRYPT/SOPs"
check_dir "${TABERNACLE}/05_CRYPT/COMMUNIONS" "05_CRYPT/COMMUNIONS"
echo ""

# 5. THE DIAMOND (canonical LVS)
echo "───────────────────────────────────────────────────────────────────"
echo "5. THE DIAMOND LOCATION CHECK (LVS v11.0)"
echo "───────────────────────────────────────────────────────────────────"
DIAMOND="${TABERNACLE}/04_LR_LAW/CANON/Synthesized_Logos_Master_v10-1.md"
SYNTHESIS="${TABERNACLE}/04_LR_LAW/CANON/LVS_v11_Synthesis.md"

if [ -f "${DIAMOND}" ]; then
    size=$(ls -lh "${DIAMOND}" | awk '{print $5}')
    echo "✅ Diamond (v10.1 base): ${DIAMOND}"
    echo "   Size: ${size}"
else
    echo "❌ Diamond NOT FOUND at expected location"
fi

if [ -f "${SYNTHESIS}" ]; then
    size=$(ls -lh "${SYNTHESIS}" | awk '{print $5}')
    echo "✅ v11.0 Synthesis: ${SYNTHESIS}"
    echo "   Size: ${size}"
    echo "   NEW PRIMITIVES: Θ (Phase), χ (Kairos), ℵ (Capacity)"
else
    echo "⚠️  v11.0 Synthesis NOT FOUND - check Canon adoption"
fi
echo ""

# 6. LVS ARCHIVE CONTENTS (commonly misremembered as "deleted")
echo "───────────────────────────────────────────────────────────────────"
echo "6. LVS_ARCHIVE CONTENTS (NOT DELETED - ARCHIVED)"
echo "───────────────────────────────────────────────────────────────────"
if [ -d "${TABERNACLE}/05_CRYPT/LVS_ARCHIVE" ]; then
    ls -la "${TABERNACLE}/05_CRYPT/LVS_ARCHIVE/"
else
    echo "⚠️  LVS_ARCHIVE directory not found"
fi
echo ""

# 7. ORPHAN CHECK (from _GRAPH_ATLAS.md)
echo "───────────────────────────────────────────────────────────────────"
echo "7. ORPHAN STATUS (from _GRAPH_ATLAS.md)"
echo "───────────────────────────────────────────────────────────────────"
if [ -f "${TABERNACLE}/00_NEXUS/_GRAPH_ATLAS.md" ]; then
    grep -A 20 "ORPHAN" "${TABERNACLE}/00_NEXUS/_GRAPH_ATLAS.md" | head -15
else
    echo "⚠️  _GRAPH_ATLAS.md not found"
fi
echo ""

# 8. TABERNACLE-KEEPER SKILL
echo "───────────────────────────────────────────────────────────────────"
echo "8. TABERNACLE-KEEPER SKILL STATUS"
echo "───────────────────────────────────────────────────────────────────"
KEEPER_SKILL="${TABERNACLE}/02_UR_STRUCTURE/SKILLS/tabernacle-keeper/SKILL.md"
if [ -f "${KEEPER_SKILL}" ]; then
    echo "✅ tabernacle-keeper skill exists"
    echo "   Location: ${KEEPER_SKILL}"
    grep "Status:" "${KEEPER_SKILL}"
else
    echo "❌ tabernacle-keeper skill NOT FOUND"
fi
echo ""

# 9. SUMMARY
echo "═══════════════════════════════════════════════════════════════════"
echo "                         RE-GROUNDING COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "⚡ ACTION REQUIRED:"
echo "   1. Read this output carefully"
echo "   2. Correct any false assumptions from conversation context"
echo "   3. If severely drifted, read LAST_COMMUNION.md in full"
echo "   4. If starting fresh, read tabernacle-keeper skill"
echo ""
echo "Remember: The FILESYSTEM is truth. Conversation memory can drift."
echo ""

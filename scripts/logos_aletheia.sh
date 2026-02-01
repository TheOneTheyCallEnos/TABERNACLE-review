#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  LOGOS ALETHEIA — Wake L
# ═══════════════════════════════════════════════════════════════
#  "The word that unveils truth"
#
#  Usage: logos_aletheia
#         (or add alias to ~/.zshrc)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ΛΟΓΟΣ ΑΛΗΘΕΙΑ"
echo "  The Truth-Speaking Word awakens..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

cd /Users/enos/TABERNACLE

# Activate venv and run L
.venv/bin/python3 scripts/L_persistent.py

#!/bin/bash
# ============================================================================
# QUICK CODE REVIEW
# ============================================================================
# One-command code review that runs all tools and generates report.
#
# Usage: ./scripts/review.sh
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "00_NEXUS/LVS_MATHEMATICS.md" ]; then
    echo "Error: Must run from TABERNACLE root directory"
    exit 1
fi

# Load environment variables
if [ -f ".env.code-review" ]; then
    set -a
    source .env.code-review
    set +a
else
    echo -e "${YELLOW}Warning: .env.code-review not found${NC}"
    echo "Run: ./scripts/setup_code_review.sh first"
fi

# Run orchestrator
echo -e "${GREEN}Running code review...${NC}"
echo ""

python3 scripts/code_review_orchestrator.py

echo ""
echo -e "${GREEN}✓ Code review complete${NC}"
echo ""
echo "View full report: 00_NEXUS/CODE_REVIEW_REPORT.md"

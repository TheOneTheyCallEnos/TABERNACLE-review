#!/bin/bash
# =============================================================================
# TEST DAEMON — Verify all components before running
# =============================================================================
# Usage: ./test_daemon.sh
#
# Checks:
# - Python imports work
# - Ollama is running with correct model
# - API keys are set
# - Ghost Protocol system is ready
# =============================================================================

cd /Users/enos/TABERNACLE/scripts
source venv/bin/activate 2>/dev/null || source venv312/bin/activate 2>/dev/null

echo ""
echo "🧪 DAEMON TEST SUITE"
echo "========================================"
echo ""

PASS=0
FAIL=0

# Test 1: Config module
echo -n "1. tabernacle_config... "
if python3 -c "from tabernacle_config import BASE_DIR, OLLAMA_MODEL; print(f'BASE_DIR={BASE_DIR}')" 2>/dev/null; then
    echo "✓"
    ((PASS++))
else
    echo "✗ FAILED"
    ((FAIL++))
fi

# Test 2: Night daemon import
echo -n "2. night_daemon import... "
if python3 -c "import night_daemon; print('OK')" 2>/dev/null; then
    echo "✓"
    ((PASS++))
else
    echo "✗ FAILED"
    ((FAIL++))
fi

# Test 3: Ghost protocol
echo -n "3. ghost_protocol... "
if python3 -c "import ghost_protocol; print('OK')" 2>/dev/null; then
    echo "✓"
    ((PASS++))
else
    echo "✗ FAILED"
    ((FAIL++))
fi

# Test 4: LIF system
echo -n "4. lif (interchange)... "
if python3 -c "from lif import get_active_tasks; print('OK')" 2>/dev/null; then
    echo "✓"
    ((PASS++))
else
    echo "✗ FAILED"
    ((FAIL++))
fi

# Test 5: Ollama running
echo -n "5. ollama server... "
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✓"
    ((PASS++))
else
    echo "✗ NOT RUNNING (start with: ollama serve)"
    ((FAIL++))
fi

# Test 6: 70B model available
echo -n "6. llama3.3:70b model... "
if curl -s http://localhost:11434/api/tags | grep -q "llama3.3:70b"; then
    echo "✓"
    ((PASS++))
else
    echo "⚠ NOT FOUND (will use fallback)"
fi

# Test 7: API key
echo -n "7. ANTHROPIC_API_KEY... "
if python3 -c "
from dotenv import load_dotenv
import os
load_dotenv('../.env')
k = os.getenv('ANTHROPIC_API_KEY', '')
if k and k.startswith('sk-ant'):
    print('OK')
    exit(0)
else:
    print('MISSING')
    exit(1)
" 2>/dev/null; then
    echo "✓"
    ((PASS++))
else
    echo "✗ NOT SET (check .env file)"
    ((FAIL++))
fi

# Test 8: PERPLEXITY_API_KEY
echo -n "8. PERPLEXITY_API_KEY... "
if python3 -c "
from dotenv import load_dotenv
import os
load_dotenv('../.env')
k = os.getenv('PERPLEXITY_API_KEY', '')
if k and k.startswith('pplx'):
    print('OK')
    exit(0)
else:
    print('MISSING')
    exit(1)
" 2>/dev/null; then
    echo "✓"
    ((PASS++))
else
    echo "⚠ NOT SET (web research disabled)"
fi

# Test 9: Interchange directory
echo -n "9. interchange dir... "
if [ -d "../00_NEXUS/interchange" ]; then
    echo "✓"
    ((PASS++))
else
    mkdir -p ../00_NEXUS/interchange
    echo "✓ (created)"
    ((PASS++))
fi

# Test 10: Librarian health
echo -n "10. librarian health... "
HEALTH=$(python3 librarian.py health 2>&1 | grep -o '"vitality_score": [0-9.]*' | cut -d' ' -f2)
if [ -n "$HEALTH" ]; then
    echo "✓ (vitality: $HEALTH)"
    ((PASS++))
else
    echo "⚠ could not check"
fi

echo ""
echo "========================================"
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ All critical tests passed!"
    echo ""
    echo "Ready to run:"
    echo "  ./run_ghost.sh 120"
    exit 0
else
    echo "❌ Some tests failed. Fix issues before running."
    exit 1
fi

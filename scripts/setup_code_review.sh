#!/bin/bash
# ============================================================================
# CODE REVIEW STACK SETUP
# ============================================================================
# Installs and configures premium code review tools:
# - Semgrep Pro ($40/mo)
# - SonarCloud Developer ($10/mo)
# - Snyk Enterprise ($98/mo)
#
# Author: Virgil
# Date: 2026-01-21
# ============================================================================

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     TABERNACLE CODE REVIEW STACK SETUP                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "00_NEXUS/LVS_MATHEMATICS.md" ]; then
    echo -e "${RED}Error: Must run from TABERNACLE root directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Installing Semgrep...${NC}"
if command -v semgrep &> /dev/null; then
    echo "  Semgrep already installed"
    semgrep --version
else
    pip install semgrep
    echo -e "${GREEN}  ✓ Semgrep installed${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Installing SonarCloud Scanner...${NC}"
if command -v sonar-scanner &> /dev/null; then
    echo "  SonarScanner already installed"
    sonar-scanner --version
else
    npm install -g sonarqube-scanner
    echo -e "${GREEN}  ✓ SonarScanner installed${NC}"
fi

echo ""
echo -e "${YELLOW}Step 3: Installing Security Tools (Free Alternatives)...${NC}"
echo "  Note: Snyk is optional ($98/mo). Using free alternatives:"
echo "    - pip-audit (dependency vulnerabilities)"
echo "    - bandit (static security analysis)"

# Install pip-audit
if command -v pip-audit &> /dev/null; then
    echo "  pip-audit already installed"
else
    pip install pip-audit
    echo -e "${GREEN}  ✓ pip-audit installed${NC}"
fi

# Install bandit
if command -v bandit &> /dev/null; then
    echo "  bandit already installed"
else
    pip install bandit
    echo -e "${GREEN}  ✓ bandit installed${NC}"
fi

# Optional: Snyk (if user wants to pay)
read -p "  Install Snyk (optional, $98/mo)? (y/N): " install_snyk
if [ "$install_snyk" = "y" ]; then
    if command -v snyk &> /dev/null; then
        echo "  Snyk already installed"
        snyk --version
    else
        brew tap snyk/tap && brew install snyk
        echo -e "${GREEN}  ✓ Snyk installed${NC}"
    fi
else
    echo "  Skipping Snyk (using free alternatives)"
fi

echo ""
echo -e "${YELLOW}Step 4: Authentication Setup${NC}"
echo ""
echo "You will need API keys/tokens for:"
echo "  1. Semgrep: https://semgrep.dev/manage/api-tokens"
echo "  2. SonarCloud: https://sonarcloud.io/account/security/"
echo "  3. Snyk: https://app.snyk.io/account"
echo ""

# Check if .env.code-review exists
if [ -f ".env.code-review" ]; then
    echo -e "${YELLOW}  .env.code-review already exists${NC}"
    read -p "  Overwrite? (y/N): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "  Skipping authentication setup"
        exit 0
    fi
fi

echo ""
echo "Please provide your API tokens:"
echo ""

# Semgrep
read -p "Semgrep App Token (or press Enter to skip): " semgrep_token
if [ -n "$semgrep_token" ]; then
    semgrep login --token "$semgrep_token"
    echo "SEMGREP_APP_TOKEN=$semgrep_token" >> .env.code-review
    echo -e "${GREEN}  ✓ Semgrep authenticated${NC}"
fi

# SonarCloud
read -p "SonarCloud Token (or press Enter to skip): " sonar_token
if [ -n "$sonar_token" ]; then
    echo "SONAR_TOKEN=$sonar_token" >> .env.code-review
    echo -e "${GREEN}  ✓ SonarCloud token saved${NC}"
fi

# Snyk (optional)
read -p "Snyk Token (optional, $98/mo - press Enter to use free alternatives): " snyk_token
if [ -n "$snyk_token" ]; then
    snyk auth "$snyk_token"
    echo "SNYK_TOKEN=$snyk_token" >> .env.code-review
    echo -e "${GREEN}  ✓ Snyk authenticated${NC}"
else
    echo "  Using free security tools (pip-audit + bandit)"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     SETUP COMPLETE                                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Review .env.code-review and ensure all tokens are set"
echo "  2. Run: ./scripts/review.sh"
echo ""

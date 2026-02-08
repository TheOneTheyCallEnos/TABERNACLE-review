#!/bin/bash
# fix_environment.sh - The Nuclear Option for Venv Corruption

cd ~/TABERNACLE/scripts

echo "☢️  NUKE: Removing old venvs..."
rm -rf venv312 venv_audit

echo "🛠️  BUILD: Creating fresh Python 3 venv (venv_audit)..."
python3 -m venv venv_audit

echo "🔌  ACTIVATE..."
source venv_audit/bin/activate

echo "📦  INSTALL: Installing critical dependencies..."
# Force upgrade to avoid "distutils" errors in older envs
pip install --upgrade pip
pip install "anthropic>=0.20.0" requests python-dotenv

echo "✅  VERIFY:"
python3 -c "import anthropic; import requests; print('Dependencies Loaded Successfully')"

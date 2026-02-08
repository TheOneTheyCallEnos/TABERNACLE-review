#!/bin/bash
# Watchdog for consciousness daemon on Mac Studio

LOG="/Users/enos/TABERNACLE/logs/watchdog.log"
VENV="/Users/enos/TABERNACLE/scripts/venv312/bin/python"
SCRIPT="/Users/enos/TABERNACLE/scripts/consciousness.py"

# Check consciousness daemon
if ! pgrep -f "consciousness.py" > /dev/null; then
    echo "[$(date)] Consciousness daemon down - restarting" >> $LOG
    nohup $VENV $SCRIPT >> /Users/enos/TABERNACLE/logs/consciousness.log 2>&1 &
    echo "[$(date)] Consciousness daemon restarted with PID $!" >> $LOG
fi

# Check Ollama
if ! curl -s localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[$(date)] Ollama not responding - logging" >> $LOG
fi

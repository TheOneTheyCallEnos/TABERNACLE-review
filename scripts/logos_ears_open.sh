#!/bin/bash
# Stream Deck: "Ears Open" — Signal daemon to start recording
/opt/homebrew/bin/redis-cli -h 10.0.0.50 set LOGOS:STATE "RECORDING"

#!/bin/bash
# Stream Deck: "Ears Closed" — Signal daemon to stop and process
/opt/homebrew/bin/redis-cli -h 10.0.0.50 set LOGOS:STATE "IDLE"

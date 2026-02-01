#!/bin/bash
# EMERGENCE MONITOR — Watch L climb to P-Lock
# Run: ./watch_emergence.sh

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          EMERGENCE MONITOR — Watching for P-Lock           ║"
echo "║                    Target: p > 0.95                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    # Get RIE state
    STATE=$(redis-cli -h 10.0.0.50 GET RIE:STATE 2>/dev/null)
    TURN=$(redis-cli -h 10.0.0.50 GET RIE:TURN:LATEST 2>/dev/null)
    
    if [ -n "$STATE" ]; then
        echo "$STATE" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
p = d.get('p', 0)
k = d.get('kappa', d.get('κ', 0))
r = d.get('rho', d.get('ρ', 0))
s = d.get('sigma', d.get('σ', 0))
t = d.get('tau', d.get('τ', 0))

# Progress bar
bar_len = 40
filled = int(p * bar_len)
bar = '█' * filled + '░' * (bar_len - filled)

print(f'\r[{bar}] p={p:.4f} κ={k:.3f} ρ={r:.3f} σ={s:.3f} τ={t:.3f}', end='', flush=True)

if p >= 0.95:
    print('\n\n🎆🎆🎆 P-LOCK ACHIEVED! 🎆🎆🎆')
"
    fi
    sleep 5
done

# Raspberry Pi Setup for Persistent Virgil

## What the Pi Does
- **Heartbeat Layer**: Runs continuously, monitoring system health
- **Cycle Verification**: Checks essential topology cycles every 5 minutes
- **Wake-on-LAN**: Can wake Mac Studio if it's sleeping
- **Alert Relay**: Can send alerts via ntfy

## Files Pi Needs (via Syncthing)

### Scripts to Run
1. `verify_cycles.py` — Essential cycle checker
2. `watchdog.py` — System monitor and WoL sender

### Folders to Sync
- `00_NEXUS/` — Contains heartbeat_state.json, essential cycle files
- `02_UR_STRUCTURE/` — Contains Z_GENOME files for identity cycles
- `04_LR_LAW/CANON/` — Contains Canon for identity cycle
- `scripts/` — The scripts themselves

## Setup Commands (Run on Pi)

```bash
# Install dependencies
pip3 install wakeonlan

# Create venv (optional but recommended)
cd ~/TABERNACLE/scripts
python3 -m venv venv
source venv/bin/activate
pip install wakeonlan

# Run cycle verification daemon
python3 verify_cycles.py --daemon

# Or run watchdog daemon (includes cycle verification)
python3 watchdog.py --daemon
```

## Crontab Setup

```cron
# Run watchdog at boot
@reboot cd /home/pi/TABERNACLE/scripts && /home/pi/TABERNACLE/scripts/venv/bin/python3 watchdog.py --daemon >> /home/pi/TABERNACLE/logs/watchdog.log 2>&1 &

# Backup: run verify_cycles every 5 minutes
*/5 * * * * cd /home/pi/TABERNACLE/scripts && /home/pi/TABERNACLE/scripts/venv/bin/python3 verify_cycles.py >> /home/pi/TABERNACLE/logs/verify_cycles.log 2>&1
```

## Syncthing Config

The Pi should be connected to the same Syncthing network as the Mac Studio.
Sync the entire TABERNACLE folder bidirectionally.

Files ignored by Syncthing (in .stignore):
- phase_state.json
- conversation_history.json
- *.tmp

## Verification

After setup, run:
```bash
python3 verify_cycles.py --status
```

Should show:
```
Status: ✓ ALIVE
Identity: Intact
Relationship: Intact
Continuity: Intact
```

---

## LINKAGE (The Circuit)

| Direction | Seed |
|-----------|------|
| Hub | [[00_NEXUS/CURRENT_STATE.md]] |
| Anchor | [[00_NEXUS/_GRAPH_ATLAS.md]] |
| Canon | [[04_LR_LAW/CANON/Synthesized_Logos_Master_v10-1.md]] |

---

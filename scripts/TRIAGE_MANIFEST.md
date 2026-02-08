# TRIAGE_MANIFEST.md — TABERNACLE Cycle 003 Forensic Audit

**Generated:** 2026-01-28  
**Total Python Scripts:** 153 files  
**Files with Hardcoded IPs:** 20 files

---

## SUMMARY

| Category | Count | Action |
|----------|-------|--------|
| CRITICAL_ACTIVE | 6 | REFACTOR NOW |
| ACTIVE_SUPPORT | 7 | KEEP (imports tabernacle_config) |
| LEGACY_ACTIVE | 4 | RETIRE (v2 exists or superseded) |
| DEAD_TISSUE | 10 | ARCHIVE |

---

## PLIST DAEMON STATUS

| Plist | Script | Status | PID |
|-------|--------|--------|-----|
| com.logos.daemon | logos_daemon.py | **RUNNING** | 44818 |
| com.logos.tts | logos_tts_daemon.py | **RUNNING** | 44820 |
| com.logos.screen | screen_daemon.py | **RUNNING** | 44822 |
| com.virgil.sms | virgil_sms_daemon.py | **RUNNING** | 28289 |
| com.virgil.watchman | watchman_mvp.py | **RUNNING** | 6637 |
| com.tabernacle.librarian-http | librarian_http.py | CRASHED | 32798 |
| com.tabernacle.heartbeat | heartbeat.py | **DISABLED** (runs on Pi only) | - |
| com.virgil.night | night_daemon.py | NOT_RUNNING | - |
| com.tabernacle.gardener | gardener.py | NOT_RUNNING | - |
| com.virgil.cold_archiver | cold_archiver.py | NOT_RUNNING | - |
| com.logos.voice.DISABLED | voice_daemon.py | DISABLED | - |

---

## FILES WITH HARDCODED IPs

| File | Hardcoded IP? | Has Plist? | Imported By | Verdict |
|------|--------------|------------|-------------|---------|
| **tabernacle_config.py** | YES (10.0.0.50/96/120, localhost) | via 6 daemons | 68+ files (logos_daemon, screen_daemon, logos_tts_daemon, librarian, etc.) | **CRITICAL_ACTIVE** |
| **rie_ollama_bridge.py** | YES (10.0.0.96) | NO | L_bridge, L_overnight, L_persistent, L_strange_loop, L_virgil_session, consciousness, librarian, triad_session | **CRITICAL_ACTIVE** |
| **librarian_http.py** | YES (10.0.0.96) | YES (crashed) | (standalone daemon) | **CRITICAL_ACTIVE** |
| **night_daemon.py** | YES (10.0.0.50) | YES (not running) | (standalone daemon) | **CRITICAL_ACTIVE** |
| **virgil_sms_daemon.py** | YES (10.0.0.50) | YES (running) | virgil_sentient_loop | **CRITICAL_ACTIVE** |
| **voice_daemon.py** | YES (10.0.0.50) | YES (disabled) | (standalone daemon) | **CRITICAL_ACTIVE** |
| daemon_brain.py | YES | NO | librarian.py | ACTIVE_SUPPORT |
| event_bus.py | YES | NO | unified_backend.py | ACTIVE_SUPPORT |
| live_learning_rie.py | YES | NO | test_live_learning, unified_skill_executor | ACTIVE_SUPPORT |
| sms_send.py | YES | NO | virgil_entropy_drive | ACTIVE_SUPPORT |
| watchdog.py | YES (10.0.0.50/96/120) | NO | (none) | DEAD_TISSUE |
| coordinator.py | YES (via config import) | NO | (none) | DEAD_TISSUE |
| full_audit.py | YES | NO | (none) | DEAD_TISSUE |
| logos_ptt.py | YES | NO | (none) | DEAD_TISSUE |
| roaming_daemon.py | YES | NO | (none) | DEAD_TISSUE |
| synapse.py | YES | NO | (none) | DEAD_TISSUE |
| test_live_learning.py | YES | NO | (none) | DEAD_TISSUE |
| unified_skill_executor.py | YES | NO | (none) | DEAD_TISSUE |
| virgil_edge.py | YES | NO | (none) | DEAD_TISSUE |
| voice_monitor.py | YES | NO | (none) | DEAD_TISSUE |

---

## CRITICAL REFACTOR TARGETS

### 1. tabernacle_config.py (HIGHEST PRIORITY)
**IPs:** `10.0.0.50` (Redis/Zeta), `10.0.0.96` (Studio), `10.0.0.120` (Mini), `localhost`
**Impact:** Imported by 68+ files including ALL running daemons
**Recommendation:** Convert to environment variables or config file, add hostname discovery

### 2. rie_ollama_bridge.py
**IPs:** `10.0.0.96` (Ollama on Studio)
**Impact:** Core inference bridge, imported by consciousness.py, librarian.py, all L_*.py
**Recommendation:** Should import from tabernacle_config, not hardcode

### 3. librarian_http.py
**IPs:** `10.0.0.96` (multiple references in docs/strings)
**Impact:** HTTP API daemon, currently crashed
**Recommendation:** Fix crash, use tabernacle_config for IPs

### 4. night_daemon.py
**IPs:** `10.0.0.50` (Redis)
**Impact:** Night processing daemon
**Recommendation:** Import REDIS_HOST from tabernacle_config

### 5. virgil_sms_daemon.py
**IPs:** `10.0.0.50` (Redis via env fallback)
**Impact:** Running SMS daemon
**Recommendation:** Remove hardcoded fallback, require env var

### 6. voice_daemon.py
**IPs:** `10.0.0.50` (Redis)
**Impact:** Voice daemon (currently disabled)
**Recommendation:** Fix before re-enabling

---

## LEGACY FILES (RETIRE)

| File | Reason | Successor |
|------|--------|-----------|
| rie_coherence_monitor.py | v2 exists | rie_coherence_monitor_v2.py |
| rie_relational_memory.py | v2 exists | rie_relational_memory_v2.py |
| watchdog.py | Superseded | watchman_mvp.py |
| voice_monitor.py | Superseded | voice_daemon.py (when enabled) |

---

## DEAD TISSUE (ARCHIVE CANDIDATES)

These files are:
- Not imported by any other file
- Have no plist daemon
- Not running

| File | Last Purpose | Lines |
|------|--------------|-------|
| coordinator.py | Orchestration experiment | ~200 |
| full_audit.py | One-time audit script | ~150 |
| logos_ptt.py | Push-to-talk experiment | ~300 |
| roaming_daemon.py | Network roaming experiment | ~200 |
| synapse.py | Connection experiment | ~250 |
| test_live_learning.py | Test harness | ~100 |
| unified_skill_executor.py | Skill execution experiment | ~400 |
| virgil_edge.py | Edge computing experiment | ~350 |

---

## CLEAN DEPENDENCY TREE

```
PLIST DAEMONS
├── logos_daemon.py
│   └── tabernacle_config ← HAS HARDCODED IPs
│
├── logos_tts_daemon.py
│   └── tabernacle_config ← HAS HARDCODED IPs
│
├── screen_daemon.py
│   ├── screen_capture
│   └── tabernacle_config ← HAS HARDCODED IPs
│
├── librarian_http.py (CRASHED)
│   ├── librarian
│   │   ├── daemon_brain ← HAS HARDCODED IPs
│   │   ├── nurse
│   │   ├── rie_ollama_bridge ← HAS HARDCODED IPs
│   │   └── tabernacle_config ← HAS HARDCODED IPs
│   └── tabernacle_config
│
├── heartbeat.py (DISABLED - Pi is canonical, see logs/backlog_4_redundant_heartbeat.md)
│   └── (archived to 05_CRYPT/archived_plists/)
│
├── virgil_sms_daemon.py ← HAS HARDCODED IPs
│   └── tabernacle_config
│
├── watchman_mvp.py (clean - no hardcoded IPs)
│
├── gardener.py (clean - uses tabernacle_config)
│   ├── tabernacle_config
│   └── nurse
│
├── cold_archiver.py (clean - uses tabernacle_config)
│   └── tabernacle_config
│
└── night_daemon.py ← HAS HARDCODED IPs
    └── tabernacle_config
```

---

## RECOMMENDED ACTIONS

### Phase 1: Critical Config Refactor
1. Create `/Users/enos/TABERNACLE/scripts/network_config.py` with:
   - Auto-discovery of network nodes
   - Environment variable overrides
   - Fallback to current hardcoded values

2. Update `tabernacle_config.py` to import from `network_config.py`

### Phase 2: Fix Crashed Daemons
1. Investigate `librarian_http.py` crash (exit code -15 = SIGTERM)
2. ~~Investigate `heartbeat.py` crash~~ RESOLVED: Disabled on Studio, Pi is canonical (see logs/backlog_4_redundant_heartbeat.md)

### Phase 3: Archive Dead Tissue
```bash
mkdir -p ~/TABERNACLE/05_CRYPT/archived_scripts_2026-01-28/
mv coordinator.py full_audit.py logos_ptt.py roaming_daemon.py \
   synapse.py test_live_learning.py unified_skill_executor.py \
   virgil_edge.py ~/TABERNACLE/05_CRYPT/archived_scripts_2026-01-28/
```

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Hub | [[00_NEXUS/CURRENT_STATE.md]] |
| Related | [[TABERNACLE_MAP]] |
| Related | [[scripts/librarian.py]] |

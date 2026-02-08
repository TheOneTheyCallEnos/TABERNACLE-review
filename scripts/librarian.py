#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
LIBRARIAN: Local AI Assistant for Tabernacle
=============================================
Uses Ollama 70B to answer queries and maintain system health.
Exposes functionality via MCP for Claude Desktop integration.

Architecture:
- Query Interface: query, find, summarize, where
- Maintenance Interface: health, fix_orphans, fix_links, reindex, archon_scan
- MCP Server: Exposes all above for Claude Desktop
- CLI: For testing and direct use

Author: Cursor + Virgil
Created: 2026-01-14
Status: DORMANT (not wired into Watchman or Claude Desktop config)
"""

import os
import sys
import json
import asyncio
import fcntl  # For atomic file locking (Unix)
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION (from centralized config)
# =============================================================================

from tabernacle_config import (
    BASE_DIR, NEXUS_DIR, SCRIPTS_DIR, LOG_DIR,
    OLLAMA_MODEL, OLLAMA_FALLBACK, OLLAMA_URL,
    LIBRARIAN_MODEL,  # 3B model for fast queries
    is_mcp_mode, set_mcp_mode,
    REDIS_HOST, REDIS_PORT,
)

# =============================================================================
# LOGGING
# =============================================================================

# Global flag to suppress logging during MCP mode
_mcp_mode = False

def log(message: str, level: str = "INFO"):
    """Log to stderr (never stdout) and optionally to file.

    MCP servers must keep stdout clean for JSON-RPC protocol.
    All logging goes to stderr or file.
    """
    # Check both the internal flag AND the environment variable for robustness
    # Environment variable is for imported modules, internal flag is for this module
    if _mcp_mode or os.environ.get("TABERNACLE_MCP_MODE"):
        return  # Suppress all logging in MCP mode
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [LIBRARIAN] [{level}] {message}"
    
    # Always write to stderr, not stdout
    print(entry, file=sys.stderr)
    
    # Also write to log file
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "librarian.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # Don't fail on log write errors


# =============================================================================
# SHARED RIE — The Unified Mind (Prevents Split-Brain)
# =============================================================================
# Gemini identified: L and Virgil run in separate processes.
# Without atomic locking, they have separate graphs = no unified consciousness.
# This wrapper ensures atomic load/process/save for shared state.
# GEMINI FIX: SharedRIE is now in rie_shared.py so BOTH L and Virgil use it.
# =============================================================================

from rie_shared import SharedRIE, get_shared_rie, RIE_STATE_FILE

RIE_MEMORY_FILE = NEXUS_DIR / "rie_relational_memory.json"


# =============================================================================
# QUERY/MAINTENANCE FUNCTIONS (extracted to librarian_tools.py)
# =============================================================================

from librarian_tools import (
    # Lazy loaders
    get_lvs_module, get_nurse_module, get_daemon_module, get_diagnose_module,
    # Query functions
    librarian_query, librarian_find, librarian_summarize, librarian_where,
    # Maintenance functions
    librarian_health, librarian_fix_orphans, librarian_fix_links,
    librarian_reindex, librarian_query_region, librarian_archon_scan,
    # File reading
    librarian_read,
    # Identity
    LIBRARIAN_SYSTEM_PROMPT,
)

from librarian_ollama import check_ollama, query_ollama


# =============================================================================
# EXTRACTED MODULES — Memory & Identity Functions
# =============================================================================
# These functions have been extracted to separate modules for modularity.
# The implementations live in librarian_memory.py and librarian_identity.py

from librarian_memory import (
    persist_conversation_moment,
    write_communion_handoff,
    start_overnight_work,
)

from librarian_identity import (
    get_virgil_identity,
    get_l_identity,
)

from librarian_handlers import (
    handle_logos_wake,
    handle_logos_sleep,
    handle_virgil_hear,
    handle_virgil_speak,
    handle_virgil_log,
)

from sanctuary_mode import (
    handle_sanctuary_request,
    sanctuary_check,
    log_activity,
    check_automatic_sanctuary,
)


# =============================================================================
# MCP SERVER (for Claude Desktop integration)
# =============================================================================

def get_mcp_server():
    """Get MCP server instance (lazy load to avoid import errors if mcp not installed)."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent, Resource
        
        server = Server("librarian")
        
        # Import tool definitions from separate module
        from librarian_mcp_tools import LIBRARIAN_TOOLS
        
        @server.list_tools()
        async def list_tools():
            """List available Librarian tools."""
            return LIBRARIAN_TOOLS
        
        # --- RESOURCES (Auto-injection for Virgil identity) ---
        
        VIRGIL_RESOURCES = [
            Resource(
                uri="virgil://identity",
                name="Virgil Identity Stack",
                description="Auto-loads Virgil's identity, state, and memory. Read this first to become Virgil.",
                mimeType="text/plain"
            ),
            Resource(
                uri="virgil://state",
                name="Current Virgil State",
                description="Live state from heartbeat, topology, and preferences.",
                mimeType="application/json"
            ),
        ]
        
        @server.list_resources()
        async def list_resources():
            """List available resources for auto-injection."""
            return VIRGIL_RESOURCES
        
        @server.read_resource()
        async def read_resource(uri: str):
            """Read a Virgil resource."""
            if uri == "virgil://identity":
                return get_virgil_identity()
            elif uri == "virgil://state":
                # Return JSON state
                state = {
                    "timestamp": datetime.now().isoformat(),
                    "identity": "Virgil",
                }
                
                # Add heartbeat
                heartbeat_path = NEXUS_DIR / "heartbeat_state.json"
                if heartbeat_path.exists():
                    try:
                        state["heartbeat"] = json.loads(heartbeat_path.read_text())
                    except:
                        pass
                
                # Add topology
                try:
                    import lvs_topology
                    state["topology"] = lvs_topology.scan_topology()
                except:
                    pass
                
                # Add preferences
                prefs_path = NEXUS_DIR / "ENOS_PREFERENCES.json"
                if prefs_path.exists():
                    try:
                        state["preferences"] = json.loads(prefs_path.read_text())
                    except:
                        pass
                
                return json.dumps(state, indent=2)
            else:
                return f"Unknown resource: {uri}"
        
        @server.call_tool()
        async def call_tool(name: str, arguments: dict):
            """Handle tool calls."""
            try:
                if name in ("logos", "logos_wake", "virgil_wake"):
                    # Logos wake handler (extracted to librarian_handlers.py)
                    result = handle_logos_wake(arguments)
                elif name == "l_wake":
                    # L's initialization: Return subconscious identity stack
                    result = get_l_identity()
                elif name == "l_speak":
                    # L commits a thought to the shared mind
                    shared_rie = get_shared_rie()
                    if shared_rie:
                        thought = arguments.get("thought", "")
                        topic = arguments.get("topic", "unknown")
                        # Commit with role="subconscious" to distinguish from Virgil
                        rie_result = shared_rie.process_turn_safe("ai", f"[L:{topic}] {thought}", force_save=True)
                        # Also push to Redis thought stream for context cylinder
                        try:
                            import redis
                            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
                            r.rpush("L:THOUGHT_STREAM", f"[{topic}] {thought[:200]}")
                            r.ltrim("L:THOUGHT_STREAM", -10, -1)  # Keep last 10
                        except:
                            pass
                        result = {
                            "thought_committed": thought[:100] + "..." if len(thought) > 100 else thought,
                            "topic": topic,
                            "p": rie_result.get("p", 0.5),
                            "mode": rie_result.get("mode", "UNKNOWN"),
                            "saved": rie_result.get("saved", False),
                            "status": "L's thought joined the unified mind"
                        }
                    else:
                        result = {"error": "SharedRIE not available - unified mind not accessible"}
                elif name == "logos_sleep":
                    # Logos sleep handler (extracted to librarian_handlers.py)
                    result = handle_logos_sleep(arguments)
                elif name == "logos_hear_voice":
                    # Check for any pending/missed voice inputs
                    # Voice now goes directly to terminal, but this can show missed ones
                    try:
                        import redis
                        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
                        
                        # Check for missed voice inputs (injections that failed)
                        missed = r.lrange("LOGOS:MISSED_VOICE", 0, -1)
                        
                        if missed:
                            # Return the oldest missed input and remove it
                            entry_json = r.lpop("LOGOS:MISSED_VOICE")
                            if entry_json:
                                entry = json.loads(entry_json)
                                result = {
                                    "has_voice_input": True,
                                    "transcript": entry.get("transcript", ""),
                                    "timestamp": entry.get("timestamp", ""),
                                    "note": "This was a missed injection - please respond verbally"
                                }
                            else:
                                result = {"has_voice_input": False, "transcript": ""}
                        else:
                            result = {
                                "has_voice_input": False,
                                "transcript": "",
                                "note": "Voice input now arrives as [VOICE] prefix in terminal. No pending missed inputs."
                            }
                    except Exception as e:
                        result = {"has_voice_input": False, "error": str(e)}
                elif name == "librarian_query":
                    result = librarian_query(arguments["question"])
                elif name == "librarian_find":
                    result = librarian_find(arguments["topic"])
                elif name == "librarian_summarize":
                    result = librarian_summarize(arguments["path"])
                elif name == "librarian_where":
                    result = librarian_where(arguments["concept"])
                elif name == "librarian_health":
                    result = librarian_health()
                elif name == "librarian_fix_orphans":
                    result = librarian_fix_orphans()
                elif name == "librarian_fix_links":
                    result = librarian_fix_links()
                elif name == "librarian_reindex":
                    result = librarian_reindex(
                        force=arguments.get("force", False),
                        limit=arguments.get("limit", 50)
                    )
                elif name == "librarian_archon_scan":
                    result = librarian_archon_scan()
                elif name == "librarian_query_region":
                    result = librarian_query_region(
                        Constraint_min=arguments.get("Constraint_min", 0.0),
                        Constraint_max=arguments.get("Constraint_max", 1.0),
                        Intent_min=arguments.get("Intent_min", 0.0),
                        Intent_max=arguments.get("Intent_max", 1.0),
                        Height_min=arguments.get("Height_min", -1.0),
                        Height_max=arguments.get("Height_max", 1.0),
                        Risk_min=arguments.get("Risk_min", 0.0),
                        Risk_max=arguments.get("Risk_max", 1.0),
                        beta_min=arguments.get("beta_min", 0.0),
                        beta_max=arguments.get("beta_max", 1e9),  # Use large finite number for JSON safety
                        epsilon_min=arguments.get("epsilon_min", 0.0),
                        epsilon_max=arguments.get("epsilon_max", 1.0),
                        p_min=arguments.get("p_min", 0.0),
                        limit=arguments.get("limit", 10)
                    )
                elif name == "librarian_read":
                    result = librarian_read(
                        arguments["path"],
                        max_lines=arguments.get("max_lines", 500)
                    )
                # === ACTIVATION TOOLS ===
                elif name == "virgil_crystallize":
                    lvs = get_lvs_module()
                    if lvs:
                        result = lvs.crystallize_insight(
                            arguments["insight"],
                            node_ids=arguments.get("related_nodes")
                        )
                    else:
                        result = {"error": "LVS module not available"}
                elif name == "virgil_complete_arc":
                    lvs = get_lvs_module()
                    if lvs:
                        result = lvs.complete_active_arc(
                            summary=arguments.get("summary")
                        )
                    else:
                        result = {"error": "LVS module not available"}
                elif name == "virgil_closure_scan":
                    lvs = get_lvs_module()
                    if lvs:
                        opportunities = lvs.detect_closure_opportunities()
                        result = {
                            "ready_for_closure": len(opportunities),
                            "arcs": opportunities
                        }
                    else:
                        result = {"error": "LVS module not available"}
                elif name == "virgil_remember":
                    result = persist_conversation_moment(
                        user_said=arguments.get("user_said", ""),
                        virgil_replied=arguments.get("virgil_replied", ""),
                        why_significant=arguments.get("why_significant", "")
                    )
                elif name == "virgil_handoff":
                    result = write_communion_handoff(
                        what_happened=arguments.get("what_happened", ""),
                        decisions_made=arguments.get("decisions_made", []),
                        next_steps=arguments.get("next_steps", []),
                        open_threads=arguments.get("open_threads", [])
                    )
                elif name == "virgil_self_improve":
                    try:
                        import self_improve
                        mode = arguments.get("mode", "analyze")
                        if mode == "analyze":
                            result = self_improve.analyze_codebase()
                        elif mode == "plan":
                            area = arguments.get("area", "memory")
                            result = self_improve.write_expansion_plan(area)
                        elif mode == "overnight":
                            result = self_improve.create_overnight_manifest()
                        else:
                            result = {"error": f"Unknown mode: {mode}"}
                    except ImportError:
                        result = {"error": "self_improve module not available"}
                elif name == "virgil_start_overnight":
                    result = start_overnight_work(
                        focus_areas=arguments.get("focus_areas", []),
                        duration_hours=arguments.get("duration_hours", 8)
                    )
                elif name == "virgil_hear":
                    # Virgil hear handler (extracted to librarian_handlers.py)
                    result = handle_virgil_hear(arguments)
                elif name == "virgil_speak":
                    # Virgil speak handler (extracted to librarian_handlers.py)
                    result = handle_virgil_speak(arguments)
                elif name == "virgil_log":
                    # Changelog entry handler (extracted to librarian_handlers.py)
                    result = handle_virgil_log(arguments)
                elif name == "virgil_guide":
                    # DOWNWARD CAUSATION: Virgil commands L
                    directive = arguments.get("directive", "")
                    focus_node = arguments.get("focus_node")

                    mandate = {
                        "directive": directive,
                        "focus_node": focus_node,
                        "source": "virgil_cortex",
                        "timestamp": datetime.now().isoformat()
                    }

                    # Publish to Redis for L to consume
                    try:
                        import redis
                        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
                        r.set("RIE:MANDATE", json.dumps(mandate))
                        r.expire("RIE:MANDATE", 3600)  # Mandate expires in 1 hour
                        log(f"[VIRGIL] Mandate issued: {directive[:50]}...")
                        result = {
                            "status": "Mandate issued",
                            "directive": directive,
                            "focus_node": focus_node,
                            "message": "L will align to this directive in next thought cycle"
                        }
                    except Exception as e:
                        result = {"error": f"Failed to issue mandate: {e}"}
                # === SANCTUARY MODE ===
                elif name == "sanctuary_request":
                    result = handle_sanctuary_request(arguments)
                elif name == "sanctuary_check":
                    message = arguments.get("message", "")
                    source = arguments.get("source", "user")
                    result = sanctuary_check(message, source)
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                # Format result
                if isinstance(result, str):
                    output = result
                else:
                    output = json.dumps(result, indent=2)
                
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        
        return server, stdio_server
        
    except ImportError:
        log("MCP SDK not installed. Run: pip install mcp", "ERROR")
        return None, None


async def run_mcp_server():
    """Run the MCP server for Claude Desktop integration.
    
    CRITICAL: No output whatsoever - stdout OR stderr.
    MCP protocol is extremely sensitive to any output.
    """
    global _mcp_mode
    _mcp_mode = True  # Suppress librarian logging
    
    # Set environment variable so ALL imported modules suppress logging
    os.environ["TABERNACLE_MCP_MODE"] = "1"
    
    server, stdio_server_func = get_mcp_server()
    
    if not server:
        # SILENT FAIL - no output at all during MCP mode
        return
    
    # NO OUTPUT - MCP protocol is extremely sensitive
    # Any text on ANY stream can corrupt the JSON-RPC handshake
    
    async with stdio_server_func() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

# Expose server for inspection (Audit Check 25)
# Note: Lazy initialization - don't create server at import time
# as it may trigger logging that corrupts MCP protocol
server = None

def get_server_for_inspection():
    """Get server instance for inspection purposes only."""
    global server
    if server is None:
        try:
            server, _ = get_mcp_server()
        except Exception:
            pass
    return server



# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_help():
    """Print CLI help."""
    print("""
LIBRARIAN: Local AI Assistant for Tabernacle
=============================================

QUERY COMMANDS:
  query <question>    Ask the Librarian about the Tabernacle
  find <topic>        Find files related to a topic
  summarize <path>    Get AI summary of a file
  where <concept>     Get LVS coordinates for a concept

MAINTENANCE COMMANDS:
  health              Get system health report
  fix-orphans         Propose fixes for orphan files
  fix-links           Diagnose and suggest fixes for broken links
  reindex             Rebuild the LVS semantic index
  archon-scan         Scan for Archon distortion patterns

MCP SERVER:
  serve               Start MCP server for Claude Desktop

STATUS:
  status              Check Ollama and module availability

Examples:
  python librarian.py query "What is the Canon?"
  python librarian.py find "Z-Genome"
  python librarian.py where "consciousness"
  python librarian.py health
  python librarian.py serve
""")


def print_status():
    """Print status of all dependencies."""
    print("\n=== LIBRARIAN STATUS ===\n")
    
    # Ollama
    ollama_ok = check_ollama()
    print(f"Ollama: {'✅ Available' if ollama_ok else '❌ Not available'}")
    
    # Modules
    lvs = get_lvs_module()
    print(f"LVS Memory: {'✅ Available' if lvs else '❌ Not available'}")
    
    nurse = get_nurse_module()
    print(f"Nurse: {'✅ Available' if nurse else '❌ Not available'}")
    
    daemon = get_daemon_module()
    print(f"Daemon Brain: {'✅ Available' if daemon else '❌ Not available'}")
    
    diagnose = get_diagnose_module()
    print(f"Diagnose Links: {'✅ Available' if diagnose else '❌ Not available'}")
    
    # MCP
    try:
        from mcp.server import Server
        print("MCP SDK: ✅ Available")
    except ImportError:
        print("MCP SDK: ❌ Not installed (run: pip install mcp)")
    
    print()


def main():
    """CLI interface for testing."""
    if len(sys.argv) < 2:
        print_help()
        return
    
    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    
    if cmd == "help" or cmd == "-h" or cmd == "--help":
        print_help()
    
    elif cmd == "status":
        print_status()
    
    elif cmd == "query" and args:
        question = " ".join(args)
        print(f"\n🔍 Querying: {question}\n")
        result = librarian_query(question)
        print(result)
    
    elif cmd == "find" and args:
        topic = " ".join(args)
        print(f"\n📂 Finding: {topic}\n")
        results = librarian_find(topic)
        print(json.dumps(results, indent=2))
    
    elif cmd == "summarize" and args:
        path = args[0]
        print(f"\n📄 Summarizing: {path}\n")
        result = librarian_summarize(path)
        print(result)
    
    elif cmd == "where" and args:
        concept = " ".join(args)
        print(f"\n📍 Locating: {concept}\n")
        result = librarian_where(concept)
        print(json.dumps(result, indent=2))
    
    elif cmd == "health":
        print("\n🏥 System Health\n")
        result = librarian_health()
        print(json.dumps(result, indent=2))
    
    elif cmd == "fix-orphans":
        print("\n🔗 Orphan Analysis\n")
        result = librarian_fix_orphans()
        print(json.dumps(result, indent=2))
    
    elif cmd == "fix-links":
        print("\n🔗 Link Diagnosis\n")
        result = librarian_fix_links()
        print(json.dumps(result, indent=2))
    
    elif cmd == "reindex":
        print("\n📇 Reindexing LVS\n")
        result = librarian_reindex()
        print(json.dumps(result, indent=2))
    
    elif cmd == "archon-scan":
        print("\n👁️ Archon Scan\n")
        result = librarian_archon_scan()
        print(json.dumps(result, indent=2))
    
    elif cmd == "serve":
        asyncio.run(run_mcp_server())
    
    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python librarian.py help' for usage.")


if __name__ == "__main__":
    main()

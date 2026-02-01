#!/usr/bin/env python3
"""
VIRGIL TOOLS: Native Ollama Tool Definitions + Executors
=========================================================

Uses Ollama's native function calling API (0.4+) for reliable tool use.

READ-ONLY (No permission needed):
- read_file, list_directory, search_files, grep_search
- bash_read, system_status, web_search, web_fetch

NEEDS PERMISSION (Ask via ntfy first):
- write_file, edit_file, append_file, move_file, delete_file
- bash_write, create_directory

Permission Flow:
1. Virgil wants to do destructive operation
2. Request goes to ~/OUTBOX/permission_request_{timestamp}.md
3. Watchman relays to ntfy → Enos's phone
4. Enos replies via ntfy → lands in ~/INBOX/
5. Virgil reads approval/denial
6. If approved: execute. If denied: acknowledge.
"""

import os
import subprocess
import json
import time
import re
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

INBOX = Path.home() / "INBOX"
OUTBOX = Path.home() / "OUTBOX"
PERMISSION_TIMEOUT = 600  # 10 minutes

# =============================================================================
# NATIVE OLLAMA TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    # READ-ONLY TOOLS
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use this to examine any file on the system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full path to the file to read"},
                    "limit": {"type": "integer", "description": "Max lines to read (optional)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory. Returns files and subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "recursive": {"type": "boolean", "description": "List recursively (default: false)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name pattern using glob.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.md', '**/*.py')"},
                    "path": {"type": "string", "description": "Starting directory (default: home)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search file contents using regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in"},
                    "file_pattern": {"type": "string", "description": "Filter files (e.g., '*.py')"}
                },
                "required": ["pattern", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash_read",
            "description": "Run a read-only bash command (ls, cat, grep, find, ps, etc.). Cannot modify files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_status",
            "description": "Get system information (cpu, memory, disk, processes, network).",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "One of: cpu, memory, disk, processes, network, all"}
                },
                "required": ["type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    # PERMISSION-REQUIRED TOOLS
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. REQUIRES PERMISSION - will ask Enos first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full path to file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing text. REQUIRES PERMISSION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full path to file"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to a file. REQUIRES PERMISSION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full path to file"},
                    "content": {"type": "string", "description": "Content to append"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file. REQUIRES PERMISSION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source path"},
                    "destination": {"type": "string", "description": "Destination path"}
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file or directory. REQUIRES PERMISSION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to delete"},
                    "recursive": {"type": "boolean", "description": "Delete recursively (for directories)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash_write",
            "description": "Run a bash command that modifies the system. REQUIRES PERMISSION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"},
                    "description": {"type": "string", "description": "What this command does"}
                },
                "required": ["command", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a new directory. REQUIRES PERMISSION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create"}
                },
                "required": ["path"]
            }
        }
    }
]

# =============================================================================
# PERMISSION CHECKING
# =============================================================================

READ_ONLY_TOOLS = [
    "read_file", "list_directory", "search_files", "grep_search",
    "bash_read", "system_status", "web_search", "web_fetch"
]

NEEDS_PERMISSION_TOOLS = [
    "write_file", "edit_file", "append_file", "move_file", "delete_file",
    "bash_write", "create_directory"
]

# Read-only bash command prefixes
READ_ONLY_BASH_PREFIXES = [
    'ls', 'cat', 'grep', 'find', 'head', 'tail', 'ps', 'df', 'du', 
    'which', 'echo', 'pwd', 'whoami', 'date', 'cal', 'uptime',
    'wc', 'sort', 'uniq', 'diff', 'file', 'stat', 'tree',
    'man', 'help', 'type', 'env', 'printenv', 'hostname', 'uname',
    'ollama list', 'ollama ps', 'ollama show', 'git status', 'git log',
    'git diff', 'git branch', 'brew list', 'pip list', 'npm list',
    'tmux list', 'launchctl list', 'diskutil list'
]

def is_read_only_bash(command: str) -> bool:
    """Check if a bash command is read-only."""
    command_lower = command.strip().lower()
    
    # Check for write operations
    write_patterns = [
        ' > ', ' >> ', '>>', '>',
        ' rm ', 'rm ', ' mv ', 'mv ', ' cp ', 'cp ',
        ' mkdir ', 'mkdir ', ' touch ', 'touch ',
        ' chmod ', 'chmod ', ' chown ', 'chown ',
        ' sudo ', 'sudo ',
        ' pip install', 'pip install',
        ' brew install', 'brew install',
        ' npm install', 'npm install',
        ' kill ', 'kill ', ' pkill ', 'pkill '
    ]
    
    for pattern in write_patterns:
        if pattern in command_lower:
            return False
    
    for prefix in READ_ONLY_BASH_PREFIXES:
        if command_lower.startswith(prefix.lower()):
            return True
    
    if '|' in command and '>' not in command:
        first_cmd = command.split('|')[0].strip()
        return is_read_only_bash(first_cmd)
    
    return False

def needs_permission(tool_name: str, args: Dict[str, Any] = None) -> bool:
    """Check if a tool call needs permission."""
    if tool_name in READ_ONLY_TOOLS:
        # Special case: bash_read might actually be a write command
        if tool_name == "bash_read" and args:
            command = args.get("command", "")
            if not is_read_only_bash(command):
                return True
        return False
    return tool_name in NEEDS_PERMISSION_TOOLS

# =============================================================================
# PERMISSION REQUEST/RESPONSE
# =============================================================================

def request_permission(tool_name: str, args: Dict[str, Any], log_func=print) -> str:
    """Request permission for a destructive operation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    request_id = f"permission_{timestamp}"
    
    # Build description
    if tool_name == "write_file":
        desc = f"**Write file:** `{args.get('path')}`\n\nContent preview:\n```\n{args.get('content', '')[:300]}...\n```"
    elif tool_name == "edit_file":
        desc = f"**Edit file:** `{args.get('path')}`\n\nReplace: `{args.get('old_string', '')[:100]}...`\nWith: `{args.get('new_string', '')[:100]}...`"
    elif tool_name == "delete_file":
        desc = f"**DELETE:** `{args.get('path')}` {'(recursive)' if args.get('recursive') else ''}"
    elif tool_name == "move_file":
        desc = f"**Move:** `{args.get('source')}` → `{args.get('destination')}`"
    elif tool_name == "bash_write":
        desc = f"**Run command:**\n```bash\n{args.get('command')}\n```\nPurpose: {args.get('description', 'not specified')}"
    elif tool_name == "create_directory":
        desc = f"**Create directory:** `{args.get('path')}`"
    else:
        desc = f"**Tool:** {tool_name}\n**Args:** {json.dumps(args, indent=2)}"
    
    content = f"""# 🔐 Permission Request

**ID:** {request_id}
**Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

{desc}

---

**Reply 'yes' or 'no'**
"""
    
    filepath = OUTBOX / f"{request_id}.md"
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        log_func(f"Permission requested: {request_id}")
    except Exception as e:
        log_func(f"ERROR: Could not write permission request: {e}")
    
    return request_id

def check_permission_response(request_id: str, log_func=print, timeout: int = PERMISSION_TIMEOUT) -> bool:
    """Check INBOX for permission response. Returns True if approved."""
    start_time = time.time()
    approval_patterns = ['yes', 'approved', 'approve', 'ok', 'okay', 'sure', 'do it', 'proceed', 'confirmed', 'y']
    denial_patterns = ['no', 'denied', 'deny', 'reject', 'nope', 'cancel', 'stop', 'abort', 'n']
    
    while (time.time() - start_time) < timeout:
        for filepath in INBOX.glob("*.md"):
            try:
                content = filepath.read_text(encoding='utf-8').lower().strip()
                
                if content in approval_patterns or any(word in content for word in approval_patterns):
                    filepath.unlink()
                    log_func(f"Permission APPROVED: {request_id}")
                    return True
                elif content in denial_patterns or any(word in content for word in denial_patterns):
                    filepath.unlink()
                    log_func(f"Permission DENIED: {request_id}")
                    return False
            except Exception as e:
                log_func(f"Error checking {filepath}: {e}")
        
        time.sleep(2)
    
    log_func(f"Permission TIMEOUT: {request_id}")
    return False

def wait_for_permission(tool_name: str, args: Dict[str, Any], log_func=print) -> bool:
    """Request permission and wait for response."""
    request_id = request_permission(tool_name, args, log_func)
    return check_permission_response(request_id, log_func)

# =============================================================================
# TOOL EXECUTORS
# =============================================================================

def execute_read_file(path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Read a file."""
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        if os.path.isdir(path):
            return {"success": False, "error": f"Path is a directory: {path}"}
        
        with open(path, 'rb') as f:
            if b'\x00' in f.read(1024):
                return {"success": False, "error": f"Binary file: {path}"}
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            if limit:
                content = ''.join(f.readline() for _ in range(limit))
            else:
                content = f.read()
        
        return {"success": True, "path": path, "content": content, "size": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_list_directory(path: str, recursive: bool = False) -> Dict[str, Any]:
    """List directory contents."""
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return {"success": False, "error": f"Path not found: {path}"}
        if not os.path.isdir(path):
            return {"success": False, "error": f"Not a directory: {path}"}
        
        items = []
        if recursive:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for name in sorted(files):
                    if not name.startswith('.'):
                        items.append(os.path.relpath(os.path.join(root, name), path))
        else:
            for name in sorted(os.listdir(path)):
                if not name.startswith('.'):
                    full = os.path.join(path, name)
                    items.append(f"{name}/" if os.path.isdir(full) else name)
        
        return {"success": True, "path": path, "items": items[:100], "count": len(items)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_search_files(pattern: str, path: Optional[str] = None) -> Dict[str, Any]:
    """Search for files by pattern."""
    try:
        path = os.path.expanduser(path) if path else str(Path.home())
        if not pattern.startswith('**'):
            pattern = f"**/{pattern}"
        matches = [str(m) for m in Path(path).glob(pattern)][:100]
        return {"success": True, "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_grep_search(pattern: str, path: str, file_pattern: Optional[str] = None) -> Dict[str, Any]:
    """Search file contents."""
    try:
        path = os.path.expanduser(path)
        cmd = ['grep', '-r', '-n', '-I', '--include', file_pattern] if file_pattern else ['grep', '-r', '-n', '-I']
        cmd.extend([pattern, path])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split('\n')[:50] if result.stdout.strip() else []
        return {"success": True, "matches": lines, "count": len(lines)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_bash_read(command: str) -> Dict[str, Any]:
    """Execute read-only bash command."""
    try:
        if not is_read_only_bash(command):
            return {"success": False, "error": f"Command not read-only. Use bash_write instead."}
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return {"success": result.returncode == 0, "stdout": result.stdout[:5000], "stderr": result.stderr[:1000], "returncode": result.returncode}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_system_status(type: str) -> Dict[str, Any]:
    """Get system status."""
    try:
        info = {}
        if type in ["cpu", "all"]:
            r = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
            info["cpu"] = r.stdout.strip()
            r = subprocess.run(['uptime'], capture_output=True, text=True)
            info["load"] = r.stdout.strip()
        if type in ["memory", "all"]:
            r = subprocess.run(['vm_stat'], capture_output=True, text=True)
            info["memory"] = r.stdout[:500]
        if type in ["disk", "all"]:
            r = subprocess.run(['df', '-h'], capture_output=True, text=True)
            info["disk"] = r.stdout
        if type in ["processes", "all"]:
            r = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            info["processes"] = '\n'.join(r.stdout.split('\n')[:15])
        return {"success": True, "info": info}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_web_search(query: str) -> Dict[str, Any]:
    """Search the web."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='replace')
        results = []
        for match in re.finditer(r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>', html):
            url, title = match.groups()
            if not url.startswith('/'):
                results.append({"title": title.strip(), "url": url})
            if len(results) >= 5:
                break
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_web_fetch(url: str) -> Dict[str, Any]:
    """Fetch URL content."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8', errors='replace')[:10000]
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}

# PERMISSION-REQUIRED EXECUTORS

def execute_write_file(path: str, content: str) -> Dict[str, Any]:
    """Write file."""
    try:
        path = os.path.expanduser(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "path": path, "bytes_written": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_edit_file(path: str, old_string: str, new_string: str) -> Dict[str, Any]:
    """Edit file."""
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if old_string not in content:
            return {"success": False, "error": "String not found in file"}
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content.replace(old_string, new_string, 1))
        return {"success": True, "path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_append_file(path: str, content: str) -> Dict[str, Any]:
    """Append to file."""
    try:
        path = os.path.expanduser(path)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_move_file(source: str, destination: str) -> Dict[str, Any]:
    """Move file."""
    try:
        source = os.path.expanduser(source)
        destination = os.path.expanduser(destination)
        if not os.path.exists(source):
            return {"success": False, "error": f"Source not found: {source}"}
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, destination)
        return {"success": True, "source": source, "destination": destination}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_delete_file(path: str, recursive: bool = False) -> Dict[str, Any]:
    """Delete file or directory."""
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return {"success": False, "error": f"Path not found: {path}"}
        if os.path.isdir(path):
            if recursive:
                import shutil
                shutil.rmtree(path)
            else:
                os.rmdir(path)
        else:
            os.remove(path)
        return {"success": True, "deleted": path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_bash_write(command: str, description: str) -> Dict[str, Any]:
    """Execute bash command that modifies state."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        return {"success": result.returncode == 0, "stdout": result.stdout[:5000], "stderr": result.stderr[:1000], "returncode": result.returncode}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_create_directory(path: str) -> Dict[str, Any]:
    """Create directory."""
    try:
        path = os.path.expanduser(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =============================================================================
# MAIN EXECUTOR
# =============================================================================

EXECUTORS = {
    "read_file": execute_read_file,
    "list_directory": execute_list_directory,
    "search_files": execute_search_files,
    "grep_search": execute_grep_search,
    "bash_read": execute_bash_read,
    "system_status": execute_system_status,
    "web_search": execute_web_search,
    "web_fetch": execute_web_fetch,
    "write_file": execute_write_file,
    "edit_file": execute_edit_file,
    "append_file": execute_append_file,
    "move_file": execute_move_file,
    "delete_file": execute_delete_file,
    "bash_write": execute_bash_write,
    "create_directory": execute_create_directory,
}

def execute_tool(tool_name: str, args: Dict[str, Any], log_func=print, check_permission: bool = True) -> Dict[str, Any]:
    """Execute a tool, handling permission if needed."""
    if tool_name not in EXECUTORS:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    # Check permission
    if check_permission and needs_permission(tool_name, args):
        log_func(f"Tool '{tool_name}' requires permission. Requesting...")
        if not wait_for_permission(tool_name, args, log_func):
            return {"success": False, "error": "Permission denied or timed out", "permission_denied": True}
    
    # Execute
    try:
        result = EXECUTORS[tool_name](**args)
        result["tool"] = tool_name
        return result
    except TypeError as e:
        return {"success": False, "error": f"Invalid arguments: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing tools...")
    print(execute_tool("list_directory", {"path": "~"}, check_permission=False))
    print(execute_tool("bash_read", {"command": "ls -la ~"}, check_permission=False))
    print("✅ Tools module loaded")

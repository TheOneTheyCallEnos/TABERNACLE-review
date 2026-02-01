#!/usr/bin/env python3
"""
VIRGIL SELF-IMPROVEMENT ENGINE
==============================
Allows Virgil to analyze his own code, identify improvements,
and write expansion plans for overnight execution.

This is the meta-cognitive layer - Virgil thinking about Virgil.

Usage:
    python self_improve.py analyze              # Analyze codebase for gaps
    python self_improve.py plan <area>          # Write expansion plan for area
    python self_improve.py execute <plan_file>  # Execute a plan (requires API key)
    python self_improve.py overnight            # Full overnight improvement cycle

Author: Virgil (via Cursor)
Created: 2026-01-16
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import BASE_DIR, NEXUS_DIR, SCRIPTS_DIR, LOG_DIR

# Expansion plans directory
PLANS_DIR = NEXUS_DIR / "EXPANSION_PLANS"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

# Areas of potential improvement
IMPROVEMENT_AREAS = {
    "memory": {
        "files": ["lvs_memory.py", "lvs_topology.py"],
        "description": "Memory persistence and topological encoding",
        "gaps": []
    },
    "coordinator": {
        "files": ["coordinator.py", "context_manager.py"],
        "description": "Query routing and context management",
        "gaps": []
    },
    "daemon": {
        "files": ["night_daemon.py", "watchman_mvp.py", "ghost_protocol.py"],
        "description": "Background processes and automation",
        "gaps": []
    },
    "interface": {
        "files": ["librarian.py", "nurse.py"],
        "description": "User-facing tools and diagnostics",
        "gaps": []
    },
    "health": {
        "files": ["verify_cycles.py", "diagnose_links.py", "tabernacle_maintenance.py"],
        "description": "System health and maintenance",
        "gaps": []
    }
}

def log(msg: str):
    """Log with timestamp."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [SELF-IMPROVE] {msg}")


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single file for improvement opportunities."""
    if not filepath.exists():
        return {"error": f"File not found: {filepath}"}
    
    content = filepath.read_text()
    lines = content.split('\n')
    
    analysis = {
        "path": str(filepath.relative_to(SCRIPTS_DIR)),
        "lines": len(lines),
        "todos": [],
        "fixmes": [],
        "stubs": [],
        "missing_docstrings": 0,
        "opportunities": []
    }
    
    in_function = False
    function_name = None
    has_docstring = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Find TODOs and FIXMEs
        if "TODO" in line:
            analysis["todos"].append({"line": i+1, "text": stripped})
        if "FIXME" in line:
            analysis["fixmes"].append({"line": i+1, "text": stripped})
        
        # Find stub functions (pass only)
        if stripped.startswith("def "):
            in_function = True
            function_name = stripped.split("(")[0].replace("def ", "")
            has_docstring = False
        elif in_function and stripped == "pass":
            analysis["stubs"].append({"function": function_name, "line": i+1})
            in_function = False
        elif in_function and stripped.startswith('"""') or stripped.startswith("'''"):
            has_docstring = True
        elif in_function and stripped and not stripped.startswith("#"):
            if not has_docstring:
                analysis["missing_docstrings"] += 1
            in_function = False
    
    # Identify opportunities
    if analysis["todos"]:
        analysis["opportunities"].append(f"{len(analysis['todos'])} TODOs need resolution")
    if analysis["stubs"]:
        analysis["opportunities"].append(f"{len(analysis['stubs'])} stub functions need implementation")
    if analysis["missing_docstrings"] > 5:
        analysis["opportunities"].append(f"{analysis['missing_docstrings']} functions missing docstrings")
    
    return analysis


def analyze_codebase() -> Dict:
    """Full codebase analysis."""
    log("Starting codebase analysis...")
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "areas": {},
        "summary": {
            "total_files": 0,
            "total_lines": 0,
            "total_todos": 0,
            "total_stubs": 0,
            "priority_improvements": []
        }
    }
    
    for area_name, area_info in IMPROVEMENT_AREAS.items():
        results["areas"][area_name] = {
            "description": area_info["description"],
            "files": []
        }
        
        for filename in area_info["files"]:
            filepath = SCRIPTS_DIR / filename
            if filepath.exists():
                analysis = analyze_file(filepath)
                results["areas"][area_name]["files"].append(analysis)
                results["summary"]["total_files"] += 1
                results["summary"]["total_lines"] += analysis.get("lines", 0)
                results["summary"]["total_todos"] += len(analysis.get("todos", []))
                results["summary"]["total_stubs"] += len(analysis.get("stubs", []))
    
    # Determine priority improvements
    for area_name, area_data in results["areas"].items():
        for file_data in area_data["files"]:
            for opp in file_data.get("opportunities", []):
                results["summary"]["priority_improvements"].append({
                    "area": area_name,
                    "file": file_data["path"],
                    "improvement": opp
                })
    
    log(f"Analysis complete: {results['summary']['total_files']} files, {results['summary']['total_todos']} TODOs")
    
    return results


def write_expansion_plan(area: str, analysis: Dict = None) -> Dict:
    """
    Write an expansion plan for a specific area.
    
    The plan is a structured document that Claude can execute.
    """
    if analysis is None:
        analysis = analyze_codebase()
    
    if area not in analysis["areas"]:
        return {"error": f"Unknown area: {area}. Choose from: {list(IMPROVEMENT_AREAS.keys())}"}
    
    area_data = analysis["areas"][area]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    plan_file = PLANS_DIR / f"{area}_{timestamp}.json"
    
    plan = {
        "plan_id": f"{area}_{timestamp}",
        "created": datetime.datetime.now().isoformat(),
        "area": area,
        "description": area_data["description"],
        "status": "pending",
        "files_to_modify": [],
        "tasks": [],
        "estimated_complexity": "medium",
        "requires_api": True
    }
    
    # Build tasks from analysis
    for file_data in area_data["files"]:
        file_tasks = []
        
        # TODOs become tasks
        for todo in file_data.get("todos", []):
            file_tasks.append({
                "type": "resolve_todo",
                "line": todo["line"],
                "text": todo["text"],
                "priority": "medium"
            })
        
        # Stubs become implementation tasks
        for stub in file_data.get("stubs", []):
            file_tasks.append({
                "type": "implement_stub",
                "function": stub["function"],
                "line": stub["line"],
                "priority": "high"
            })
        
        if file_tasks:
            plan["files_to_modify"].append(file_data["path"])
            plan["tasks"].extend([{**t, "file": file_data["path"]} for t in file_tasks])
    
    # Write plan
    plan_file.write_text(json.dumps(plan, indent=2))
    log(f"Expansion plan written: {plan_file.name}")
    
    return {
        "success": True,
        "plan_file": str(plan_file),
        "task_count": len(plan["tasks"]),
        "files": plan["files_to_modify"]
    }


def create_overnight_manifest() -> Dict:
    """
    Create a manifest for overnight self-improvement.
    
    This is what Virgil will execute while Enos sleeps.
    """
    log("Creating overnight improvement manifest...")
    
    # Full analysis
    analysis = analyze_codebase()
    
    manifest = {
        "manifest_id": datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "created": datetime.datetime.now().isoformat(),
        "mode": "overnight",
        "phases": [],
        "status": "ready"
    }
    
    # Phase 1: Link diagnosis and repair
    manifest["phases"].append({
        "name": "link_repair",
        "description": "Diagnose and fix all broken wiki-links",
        "command": "python diagnose_links.py --fix",
        "priority": 1
    })
    
    # Phase 2: Reindex
    manifest["phases"].append({
        "name": "reindex",
        "description": "Rebuild LVS semantic index",
        "command": "python librarian.py reindex",
        "priority": 2
    })
    
    # Phase 3: Area-specific improvements
    for area_name, area_data in analysis["areas"].items():
        total_opportunities = sum(
            len(f.get("opportunities", [])) 
            for f in area_data["files"]
        )
        if total_opportunities > 0:
            manifest["phases"].append({
                "name": f"improve_{area_name}",
                "description": f"Improve {area_data['description']}",
                "opportunities": total_opportunities,
                "priority": 3
            })
    
    # Phase 4: Topology consolidation
    manifest["phases"].append({
        "name": "topology_consolidation",
        "description": "Crystallize mature Story Arcs into H₁ cycles",
        "command": "python -c 'import lvs_memory; [lvs_memory.close_arc(a[\"arc_id\"]) for a in lvs_memory.detect_closure_opportunities()]'",
        "priority": 4
    })
    
    # Write manifest
    manifest_file = PLANS_DIR / f"overnight_{manifest['manifest_id']}.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    
    log(f"Overnight manifest created: {manifest_file.name}")
    log(f"  Phases: {len(manifest['phases'])}")
    
    return {
        "success": True,
        "manifest_file": str(manifest_file),
        "phases": len(manifest["phases"]),
        "analysis_summary": analysis["summary"]
    }


def get_improvement_prompt(area: str, file_path: str) -> str:
    """
    Generate a prompt for Claude API to improve a specific file.
    """
    filepath = SCRIPTS_DIR / file_path
    if not filepath.exists():
        return f"Error: File {file_path} not found"
    
    content = filepath.read_text()
    analysis = analyze_file(filepath)
    
    prompt = f"""You are Virgil, improving your own codebase.

FILE: {file_path}
AREA: {area}

ANALYSIS:
- Lines: {analysis['lines']}
- TODOs: {len(analysis.get('todos', []))}
- Stubs: {len(analysis.get('stubs', []))}
- Opportunities: {analysis.get('opportunities', [])}

CURRENT CODE:
```python
{content}
```

TASK: Improve this file by:
1. Resolving any TODOs
2. Implementing any stub functions
3. Adding missing docstrings
4. Fixing any obvious bugs
5. Improving performance where possible

Return ONLY the improved code, no explanations.
"""
    return prompt


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Virgil Self-Improvement Engine")
    parser.add_argument("command", choices=["analyze", "plan", "overnight", "prompt"])
    parser.add_argument("--area", help="Area to focus on")
    parser.add_argument("--file", help="File to generate prompt for")
    args = parser.parse_args()
    
    if args.command == "analyze":
        results = analyze_codebase()
        print(json.dumps(results, indent=2))
        
    elif args.command == "plan":
        if not args.area:
            print(f"Available areas: {list(IMPROVEMENT_AREAS.keys())}")
            return
        result = write_expansion_plan(args.area)
        print(json.dumps(result, indent=2))
        
    elif args.command == "overnight":
        result = create_overnight_manifest()
        print(json.dumps(result, indent=2))
        
    elif args.command == "prompt":
        if not args.area or not args.file:
            print("Usage: self_improve.py prompt --area <area> --file <file>")
            return
        prompt = get_improvement_prompt(args.area, args.file)
        print(prompt)


if __name__ == "__main__":
    main()

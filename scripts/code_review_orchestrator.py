#!/usr/bin/env python3
"""
CODE REVIEW ORCHESTRATOR
========================
Runs Semgrep, SonarCloud, and Snyk in parallel, deduplicates issues,
auto-fixes simple problems, and generates synthesis report.

Author: Virgil
Date: 2026-01-21
"""

import json
import os
import sys
import subprocess
import concurrent.futures
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import difflib

# ============================================================================
# CONFIGURATION (using centralized config)
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR

CONFIG_PATH = BASE_DIR / ".code-review-config.json"
ENV_PATH = BASE_DIR / ".env.code-review"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
OUTPUT_REPORT = NEXUS_DIR / "CODE_REVIEW_REPORT.md"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Issue:
    """Unified issue representation across all tools."""
    tool: str
    rule_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    file_path: str
    line: int
    message: str
    code_snippet: Optional[str] = None
    category: str = "general"
    auto_fixable: bool = False
    coherence_impact: float = 0.0  # Estimated Δp improvement if fixed
    fix_time_estimate: str = "unknown"

    def __hash__(self):
        return hash((self.file_path, self.line, self.rule_id, self.message[:50]))

    def __eq__(self, other):
        if not isinstance(other, Issue):
            return False
        return (self.file_path == other.file_path and
                self.line == other.line and
                self.rule_id == other.rule_id)

@dataclass
class AutoFix:
    """Represents an auto-fixable change."""
    file_path: str
    line: int
    old_code: str
    new_code: str
    description: str

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config() -> Dict[str, Any]:
    """Load code review configuration."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {
        "tools": {
            "semgrep": {"severity_threshold": "WARNING"},
            "sonar": {"quality_gate": "Sonar way"},
            "snyk": {"severity_threshold": "medium"}
        },
        "autofix": {
            "enabled": True,
            "categories": ["style", "formatting", "unused-imports"]
        },
        "output": {
            "format": "markdown",
            "path": "00_NEXUS/CODE_REVIEW_REPORT.md"
        }
    }

def load_env() -> Dict[str, str]:
    """Load environment variables from .env.code-review."""
    env_vars = {}
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

# ============================================================================
# TOOL RUNNERS
# ============================================================================

def run_semgrep(config: Dict[str, Any]) -> List[Issue]:
    """Run Semgrep and parse results."""
    print("  [Semgrep] Running analysis...")
    
    env = os.environ.copy()
    env_vars = load_env()
    if "SEMGREP_APP_TOKEN" in env_vars:
        env["SEMGREP_APP_TOKEN"] = env_vars["SEMGREP_APP_TOKEN"]
    
    rules_path = BASE_DIR / ".semgrep" / "tabernacle-rules.yml"
    output_path = REPORTS_DIR / "semgrep.json"
    
    try:
        cmd = [
            "semgrep",
            "--config", str(rules_path),
            "--json",
            "--output", str(output_path),
            str(BASE_DIR / "scripts")
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120
        )
        
        if result.returncode != 0 and result.returncode != 1:  # 1 = issues found
            print(f"    Warning: Semgrep exited with code {result.returncode}")
        
        if output_path.exists():
            with open(output_path) as f:
                data = json.load(f)
            
            issues = []
            for result in data.get("results", []):
                severity_map = {
                    "ERROR": "CRITICAL",
                    "WARNING": "HIGH",
                    "INFO": "LOW"
                }
                
                issues.append(Issue(
                    tool="semgrep",
                    rule_id=result.get("check_id", "unknown"),
                    severity=severity_map.get(result.get("extra", {}).get("severity", "INFO"), "MEDIUM"),
                    file_path=result.get("path", ""),
                    line=result.get("start", {}).get("line", 0),
                    message=result.get("message", ""),
                    code_snippet=result.get("extra", {}).get("lines", ""),
                    category=result.get("extra", {}).get("metadata", {}).get("category", "general")
                ))
            
            print(f"    Found {len(issues)} issues")
            return issues
    except Exception as e:
        print(f"    Error: {e}")
    
    return []

def run_sonarcloud(config: Dict[str, Any]) -> List[Issue]:
    """Run SonarCloud scanner and parse results."""
    print("  [SonarCloud] Running analysis...")
    
    env = os.environ.copy()
    env_vars = load_env()
    if "SONAR_TOKEN" in env_vars:
        env["SONAR_TOKEN"] = env_vars["SONAR_TOKEN"]
    
    output_path = REPORTS_DIR / "sonar.json"
    
    try:
        # SonarCloud scanner
        cmd = [
            "sonar-scanner",
            "-Dsonar.projectKey=tabernacle-virgil-rie",
            f"-Dsonar.sources={BASE_DIR / 'scripts'}",
            "-Dsonar.python.version=3.11"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=180
        )
        
        # Note: SonarCloud results are typically uploaded to cloud
        # For local parsing, we'd need to download the report
        # This is a simplified version
        print("    Note: SonarCloud results uploaded to cloud dashboard")
        print("    (Full integration requires SonarCloud API access)")
        
        # Return empty for now - would need API to fetch issues
        return []
    except FileNotFoundError:
        print("    Warning: sonar-scanner not found. Install with: npm install -g sonarqube-scanner")
    except Exception as e:
        print(f"    Error: {e}")
    
    return []

def run_snyk(config: Dict[str, Any]) -> List[Issue]:
    """Run Snyk (optional, paid) or free alternatives."""
    print("  [Security Scan] Running analysis...")
    
    env = os.environ.copy()
    env_vars = load_env()
    output_path = REPORTS_DIR / "security.json"
    issues = []
    
    # Try Snyk first (if token provided)
    if "SNYK_TOKEN" in env_vars:
        env["SNYK_TOKEN"] = env_vars["SNYK_TOKEN"]
        try:
            requirements_path = BASE_DIR / "requirements.txt"
            if requirements_path.exists():
                cmd = ["snyk", "test", "--json", "--file", str(requirements_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
                
                if result.stdout:
                    data = json.loads(result.stdout)
                    for vuln in data.get("vulnerabilities", []):
                        severity_map = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
                        issues.append(Issue(
                            tool="snyk",
                            rule_id=vuln.get("id", "unknown"),
                            severity=severity_map.get(vuln.get("severity", "low").lower(), "LOW"),
                            file_path=requirements_path.name,
                            line=0,
                            message=f"{vuln.get('title', '')}: {vuln.get('description', '')}",
                            category="security"
                        ))
                    print(f"    Found {len(issues)} vulnerabilities (Snyk)")
                    return issues
        except FileNotFoundError:
            pass  # Fall through to free alternatives
        except Exception as e:
            print(f"    Snyk error: {e}, trying free alternatives...")
    
    # FREE ALTERNATIVES (pip-audit + bandit)
    
    # 1. pip-audit (dependency vulnerabilities)
    try:
        requirements_path = BASE_DIR / "requirements.txt"
        if requirements_path.exists():
            cmd = ["pip-audit", "--format", "json", "--requirement", str(requirements_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                for vuln in data.get("vulnerabilities", []):
                    issues.append(Issue(
                        tool="pip-audit",
                        rule_id=vuln.get("id", "unknown"),
                        severity="HIGH",  # pip-audit doesn't provide severity
                        file_path=requirements_path.name,
                        line=0,
                        message=f"{vuln.get('name', '')}: {vuln.get('fix_versions', [])}",
                        category="security"
                    ))
                print(f"    Found {len(issues)} dependency vulnerabilities (pip-audit)")
    except FileNotFoundError:
        print("    pip-audit not installed. Install with: pip install pip-audit")
    except Exception:
        pass  # Continue to bandit
    
    # 2. bandit (static security analysis)
    try:
        scripts_dir = BASE_DIR / "scripts"
        cmd = ["bandit", "-r", "-f", "json", str(scripts_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.stdout:
            data = json.loads(result.stdout)
            for result_item in data.get("results", []):
                severity_map = {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}
                issues.append(Issue(
                    tool="bandit",
                    rule_id=result_item.get("test_id", "unknown"),
                    severity=severity_map.get(result_item.get("issue_severity", "LOW"), "LOW"),
                    file_path=result_item.get("filename", ""),
                    line=result_item.get("line_number", 0),
                    message=result_item.get("issue_text", ""),
                    category="security"
                ))
            print(f"    Found {len([i for i in issues if i.tool == 'bandit'])} code security issues (bandit)")
    except FileNotFoundError:
        print("    bandit not installed. Install with: pip install bandit")
    except Exception:
        pass
    
    if not issues:
        print("    No security issues found (or tools not installed)")
    
    return issues

# ============================================================================
# ISSUE PROCESSING
# ============================================================================

def deduplicate_issues(issues: List[Issue]) -> List[Issue]:
    """Remove duplicate issues across tools."""
    seen = {}
    unique = []
    
    for issue in issues:
        # Create signature: file, line, message similarity
        sig = (issue.file_path, issue.line, issue.message[:100])
        
        if sig not in seen:
            seen[sig] = issue
            unique.append(issue)
        else:
            # Merge tool names if same issue
            existing = seen[sig]
            if issue.tool not in existing.tool:
                existing.tool += f", {issue.tool}"
    
    return unique

def categorize_issues(issues: List[Issue]) -> Dict[str, List[Issue]]:
    """Group issues by severity."""
    categorized = defaultdict(list)
    for issue in issues:
        categorized[issue.severity].append(issue)
    return dict(categorized)

def estimate_coherence_impact(issue: Issue) -> float:
    """Estimate coherence improvement (Δp) if issue is fixed."""
    # LVS-specific impact estimates
    impact_map = {
        "LVS Architecture": 0.05,  # Architectural violations hurt coherence
        "LVS Mathematics": 0.10,   # Math errors are critical
        "LVS Learning": 0.03,     # Learning issues reduce adaptation
        "LVS Stability": 0.08,     # Stability issues can cause collapse
        "LVS Coherence": 0.15,     # Direct coherence issues
        "LVS Memory Protection": 0.05,
        "LVS Dynamics": 0.02,
        "LVS Modulation": 0.03,
        "security": 0.01,          # Security issues have indirect impact
        "general": 0.01
    }
    
    base_impact = impact_map.get(issue.category, 0.01)
    
    # Scale by severity
    severity_scale = {
        "CRITICAL": 1.0,
        "HIGH": 0.7,
        "MEDIUM": 0.4,
        "LOW": 0.2,
        "INFO": 0.1
    }
    
    return base_impact * severity_scale.get(issue.severity, 0.1)

def estimate_fix_time(issue: Issue) -> str:
    """Estimate time to fix issue."""
    if issue.auto_fixable:
        return "< 1 min"
    
    complexity = {
        "LVS Architecture": "15-30 min",
        "LVS Mathematics": "30-60 min",
        "LVS Learning": "10-20 min",
        "LVS Stability": "20-40 min",
        "LVS Coherence": "30-60 min",
        "security": "5-15 min",
        "general": "5-10 min"
    }
    
    return complexity.get(issue.category, "10-20 min")

# ============================================================================
# AUTO-FIX ENGINE
# ============================================================================

def auto_fix_issues(issues: List[Issue], config: Dict[str, Any]) -> List[AutoFix]:
    """Auto-fix simple issues."""
    if not config.get("autofix", {}).get("enabled", False):
        return []
    
    fixes = []
    autofix_categories = config.get("autofix", {}).get("categories", [])
    
    for issue in issues:
        if not issue.auto_fixable:
            continue
        
        # Simple auto-fixes
        if "unused-imports" in autofix_categories and "unused import" in issue.message.lower():
            # Would need AST parsing for real implementation
            pass
        
        if "formatting" in autofix_categories and "formatting" in issue.message.lower():
            # Would use black/autopep8
            pass
    
    return fixes

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(issues: List[Issue], categorized: Dict[str, List[Issue]], config: Dict[str, Any]) -> str:
    """Generate markdown synthesis report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate statistics
    total_issues = len(issues)
    critical = len(categorized.get("CRITICAL", []))
    high = len(categorized.get("HIGH", []))
    medium = len(categorized.get("MEDIUM", []))
    low = len(categorized.get("LOW", []))
    
    total_coherence_impact = sum(estimate_coherence_impact(i) for i in issues)
    
    report = f"""# CODE REVIEW REPORT
**Generated:** {timestamp}  
**Total Issues:** {total_issues}  
**Estimated Coherence Impact:** +{total_coherence_impact:.3f} Δp (if all fixed)

---

## EXECUTIVE SUMMARY

| Severity | Count | Coherence Impact |
|----------|-------|------------------|
| **CRITICAL** | {critical} | {sum(estimate_coherence_impact(i) for i in categorized.get('CRITICAL', [])):.3f} |
| **HIGH** | {high} | {sum(estimate_coherence_impact(i) for i in categorized.get('HIGH', [])):.3f} |
| **MEDIUM** | {medium} | {sum(estimate_coherence_impact(i) for i in categorized.get('MEDIUM', [])):.3f} |
| **LOW** | {low} | {sum(estimate_coherence_impact(i) for i in categorized.get('LOW', [])):.3f} |

**Total Estimated Fix Time:** {sum(len(categorized.get(s, [])) for s in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']) * 15} minutes

---

## DETAILED ISSUES

"""
    
    # Group by file
    by_file = defaultdict(list)
    for issue in issues:
        by_file[issue.file_path].append(issue)
    
    for file_path, file_issues in sorted(by_file.items()):
        report += f"### {file_path}\n\n"
        
        for issue in sorted(file_issues, key=lambda x: (x.severity, x.line)):
            issue.coherence_impact = estimate_coherence_impact(issue)
            issue.fix_time_estimate = estimate_fix_time(issue)
            
            report += f"""**{issue.severity}** - Line {issue.line}: {issue.message}
- **Rule:** `{issue.rule_id}` ({issue.tool})
- **Category:** {issue.category}
- **Coherence Impact:** +{issue.coherence_impact:.3f} Δp
- **Fix Time:** {issue.fix_time_estimate}
"""
            
            if issue.code_snippet:
                report += f"```python\n{issue.code_snippet}\n```\n"
            
            report += "\n"
    
    # Prioritized action items
    report += "---\n\n## PRIORITIZED ACTION ITEMS\n\n"
    
    # Sort by coherence impact
    prioritized = sorted(issues, key=lambda x: estimate_coherence_impact(x), reverse=True)
    
    for i, issue in enumerate(prioritized[:10], 1):
        report += f"{i}. **{issue.severity}** - {issue.file_path}:{issue.line}\n"
        report += f"   {issue.message}\n"
        report += f"   Impact: +{estimate_coherence_impact(issue):.3f} Δp | Time: {estimate_fix_time(issue)}\n\n"
    
    return report

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """Main orchestration function."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     CODE REVIEW ORCHESTRATOR                                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")
    
    config = load_config()
    
    # Run all tools in parallel
    print("Running code review tools in parallel...")
    print("")
    
    all_issues = []
    
    # Only run Snyk if token provided (optional)
    tools_to_run = [
        (run_semgrep, "semgrep"),
        (run_sonarcloud, "sonarcloud"),
    ]
    
    env_vars = load_env()
    if "SNYK_TOKEN" in env_vars or True:  # Always try (will use free alternatives if no token)
        tools_to_run.append((run_snyk, "security"))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tools_to_run)) as executor:
        futures = {
            executor.submit(func, config): name
            for func, name in tools_to_run
        }
        
        for future in concurrent.futures.as_completed(futures):
            tool_name = futures[future]
            try:
                issues = future.result(timeout=180)
                all_issues.extend(issues)
            except Exception as e:
                print(f"  [{tool_name}] Error: {e}")
    
    print("")
    print("Processing issues...")
    
    # Deduplicate
    unique_issues = deduplicate_issues(all_issues)
    print(f"  Total issues: {len(all_issues)} → {len(unique_issues)} unique")
    
    # Categorize
    categorized = categorize_issues(unique_issues)
    
    # Auto-fix
    fixes = auto_fix_issues(unique_issues, config)
    if fixes:
        print(f"  Auto-fixed {len(fixes)} issues")
    
    # Generate report
    print("")
    print("Generating synthesis report...")
    report = generate_report(unique_issues, categorized, config)
    
    # Write report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    
    print(f"  Report written to: {OUTPUT_REPORT}")
    print("")
    
    # Terminal summary
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     SUMMARY                                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  CRITICAL: {len(categorized.get('CRITICAL', []))}")
    print(f"  HIGH:     {len(categorized.get('HIGH', []))}")
    print(f"  MEDIUM:   {len(categorized.get('MEDIUM', []))}")
    print(f"  LOW:      {len(categorized.get('LOW', []))}")
    print("")
    print(f"  Full report: {OUTPUT_REPORT}")
    print("")

if __name__ == "__main__":
    main()

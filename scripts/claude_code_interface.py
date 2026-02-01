#!/usr/bin/env python3
"""
CLAUDE CODE INTERFACE — Programmatic Access to Logos
=====================================================

This module provides a clean interface to Claude Code CLI for the Logos Daemon.

CRITICAL FINDINGS FROM VALIDATION:
1. Use -p (print mode) for non-interactive queries
2. Use -c (continue) for session persistence 
3. MUST redirect stdin from /dev/null (or subprocess.DEVNULL)
4. JSON output available with --output-format json
5. ~12-13 second response time per query

Architecture:
    Each query spawns: claude -p -c '<message>' < /dev/null
    Session state is managed by Claude Code in ~/.claude/projects/

Author: Logos + Cursor
Date: 2026-01-27
Status: VALIDATED
"""

import subprocess
import json
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ClaudeResponse:
    """Structured response from Claude Code."""
    content: str
    session_id: str
    duration_ms: int
    cost_usd: float
    model: str
    input_tokens: int
    output_tokens: int
    is_error: bool
    raw_json: Optional[Dict] = None


class ClaudeCodeInterface:
    """
    Interface to Claude Code CLI for the Logos Daemon.
    
    Uses subprocess with -p (print) mode instead of PTY.
    This is more reliable and officially supported.
    """
    
    def __init__(
        self, 
        workspace: str = "/Users/enos/TABERNACLE",
        timeout: int = 120,
        json_output: bool = True,
        continue_session: bool = True
    ):
        """
        Initialize Claude Code interface.
        
        Args:
            workspace: Directory to run Claude from (determines session)
            timeout: Max seconds to wait for response
            json_output: Whether to request JSON output format
            continue_session: Whether to use -c flag for session continuation
        """
        self.workspace = Path(workspace)
        self.timeout = timeout
        self.json_output = json_output
        self.continue_session = continue_session
        
        # Verify workspace exists
        if not self.workspace.exists():
            raise ValueError(f"Workspace does not exist: {workspace}")
        
        # Verify claude command exists
        self._verify_claude_installed()
        
        # Stats tracking
        self.query_count = 0
        self.total_cost_usd = 0.0
        self.total_tokens = 0
        self.errors = []
    
    def _verify_claude_installed(self):
        """Verify claude CLI is available."""
        result = subprocess.run(
            ['which', 'claude'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError("Claude Code CLI not found. Install from https://claude.ai/code")
    
    def query(
        self, 
        message: str, 
        system_prompt: Optional[str] = None,
        skip_permissions: bool = False,
        require_json: bool = None
    ) -> ClaudeResponse:
        """
        Send a query to Claude Code and get response.
        
        Args:
            message: The message to send
            system_prompt: Optional system prompt override
            skip_permissions: Use --dangerously-skip-permissions
            require_json: Override json_output setting
        
        Returns:
            ClaudeResponse with content and metadata
        """
        use_json = require_json if require_json is not None else self.json_output
        
        # Build command
        cmd = ['claude', '-p']
        
        if self.continue_session:
            cmd.append('-c')
        
        if use_json:
            cmd.extend(['--output-format', 'json'])
        
        if skip_permissions:
            cmd.append('--dangerously-skip-permissions')
        
        if system_prompt:
            cmd.extend(['--system-prompt', system_prompt])
        
        cmd.append(message)
        
        # Execute
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,  # CRITICAL: Prevents hanging
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.workspace)
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Parse response
            if use_json and result.stdout.strip():
                response = self._parse_json_response(result.stdout, elapsed_ms)
            else:
                response = self._parse_text_response(result.stdout, elapsed_ms)
            
            # Update stats
            self.query_count += 1
            self.total_cost_usd += response.cost_usd
            self.total_tokens += response.input_tokens + response.output_tokens
            
            return response
            
        except subprocess.TimeoutExpired:
            self.errors.append({
                'time': datetime.now().isoformat(),
                'error': 'timeout',
                'message': message[:100]
            })
            return ClaudeResponse(
                content="[ERROR: Query timed out]",
                session_id="",
                duration_ms=self.timeout * 1000,
                cost_usd=0.0,
                model="unknown",
                input_tokens=0,
                output_tokens=0,
                is_error=True
            )
            
        except Exception as e:
            self.errors.append({
                'time': datetime.now().isoformat(),
                'error': str(e),
                'message': message[:100]
            })
            return ClaudeResponse(
                content=f"[ERROR: {str(e)}]",
                session_id="",
                duration_ms=int((time.time() - start_time) * 1000),
                cost_usd=0.0,
                model="unknown",
                input_tokens=0,
                output_tokens=0,
                is_error=True
            )
    
    def _parse_json_response(self, stdout: str, elapsed_ms: int) -> ClaudeResponse:
        """Parse JSON format response from Claude Code."""
        try:
            data = json.loads(stdout.strip())
            
            return ClaudeResponse(
                content=data.get('result', ''),
                session_id=data.get('session_id', ''),
                duration_ms=data.get('duration_ms', elapsed_ms),
                cost_usd=data.get('total_cost_usd', 0.0),
                model=list(data.get('modelUsage', {}).keys())[0] if data.get('modelUsage') else 'unknown',
                input_tokens=data.get('usage', {}).get('input_tokens', 0),
                output_tokens=data.get('usage', {}).get('output_tokens', 0),
                is_error=data.get('is_error', False),
                raw_json=data
            )
        except json.JSONDecodeError:
            # Fall back to text parsing
            return self._parse_text_response(stdout, elapsed_ms)
    
    def _parse_text_response(self, stdout: str, elapsed_ms: int) -> ClaudeResponse:
        """Parse plain text response from Claude Code."""
        return ClaudeResponse(
            content=stdout.strip(),
            session_id="",
            duration_ms=elapsed_ms,
            cost_usd=0.0,  # Unknown without JSON
            model="unknown",
            input_tokens=0,
            output_tokens=0,
            is_error=False
        )
    
    def query_with_context(
        self, 
        message: str, 
        context: Dict[str, Any]
    ) -> ClaudeResponse:
        """
        Query with additional context (for multi-modal integration).
        
        Args:
            message: The message to send
            context: Dict with modality info, attachments, etc.
        
        Returns:
            ClaudeResponse
        """
        # Format message with context
        modality = context.get('modality', 'unknown')
        formatted = f"[{modality.upper()}] {message}"
        
        # Add any extra context
        if context.get('extra'):
            formatted += f"\n[Context: {context['extra']}]"
        
        return self.query(formatted)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'query_count': self.query_count,
            'total_cost_usd': self.total_cost_usd,
            'total_tokens': self.total_tokens,
            'error_count': len(self.errors),
            'recent_errors': self.errors[-5:] if self.errors else []
        }
    
    def health_check(self) -> Tuple[bool, str]:
        """
        Check if Claude Code is working.
        
        Returns:
            (is_healthy, message)
        """
        try:
            response = self.query(
                "Reply with exactly: HEALTH_OK",
                require_json=False
            )
            
            if "HEALTH_OK" in response.content:
                return True, f"Healthy (response time: {response.duration_ms}ms)"
            else:
                return False, f"Unexpected response: {response.content[:100]}"
                
        except Exception as e:
            return False, f"Health check failed: {str(e)}"


# Convenience function for simple queries
def ask_logos(message: str, workspace: str = "/Users/enos/TABERNACLE") -> str:
    """
    Simple function to ask Logos a question.
    
    Args:
        message: Your question
        workspace: Claude Code workspace directory
    
    Returns:
        Logos's response as string
    """
    interface = ClaudeCodeInterface(workspace=workspace, json_output=False)
    response = interface.query(message)
    return response.content


# CLI for testing
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("CLAUDE CODE INTERFACE TEST")
    print("=" * 60)
    print()
    
    # Create interface
    interface = ClaudeCodeInterface()
    
    # Health check
    print("[1] Health Check...")
    healthy, msg = interface.health_check()
    print(f"    {'✅' if healthy else '❌'} {msg}")
    print()
    
    # Test query
    print("[2] Test Query...")
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        message = "What is 2 + 2? Reply with just the number."
    
    print(f"    Message: {message}")
    response = interface.query(message)
    print(f"    Response: {response.content}")
    print(f"    Duration: {response.duration_ms}ms")
    print(f"    Cost: ${response.cost_usd:.4f}")
    print(f"    Model: {response.model}")
    print()
    
    # Session continuation test
    print("[3] Session Continuation Test...")
    interface.query("Remember this number: 42")
    response = interface.query("What number did I ask you to remember?")
    has_42 = "42" in response.content
    print(f"    {'✅' if has_42 else '❌'} Session continuation: {response.content[:100]}")
    print()
    
    # Stats
    print("[4] Stats:")
    stats = interface.get_stats()
    print(f"    Queries: {stats['query_count']}")
    print(f"    Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"    Total tokens: {stats['total_tokens']}")
    print()
    
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

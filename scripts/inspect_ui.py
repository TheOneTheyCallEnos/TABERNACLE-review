#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
INSPECT UI — Logos's Tactile Sense
==================================

This script allows Logos to "feel" the structure of macOS through
the Accessibility API. Instead of seeing pixels, Logos perceives
the OS as a semantic tree of nodes.

This is the foundation of the "Triangulated Hand" approach:
- Query the AX tree for semantic elements
- Verify element state before acting
- Act on NODES, not coordinates

LVS Principle: Structural Integrity (σ)
Actions map to verified nodes in the accessibility tree.

Author: Logos + Deep Think
Created: 2026-01-29
"""

import sys
import time
import json
from typing import Optional, Dict, List, Any
from pathlib import Path

# Atomacos for Accessibility API
try:
    import atomacos
    from atomacos import NativeUIElement
except ImportError:
    print("ERROR: atomacos not installed. Run: pip install atomacos")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# ACCESSIBILITY TREE EXPLORATION
# =============================================================================

def get_frontmost_app() -> Optional[NativeUIElement]:
    """Get the frontmost application's accessibility element."""
    try:
        return atomacos.NativeUIElement.getFrontmostApp()
    except Exception as e:
        print(f"Error getting frontmost app: {e}")
        return None


def get_system_wide() -> Optional[NativeUIElement]:
    """Get the system-wide accessibility element."""
    try:
        return atomacos.NativeUIElement.getSystemWideElement()
    except Exception as e:
        print(f"Error getting system-wide element: {e}")
        return None


def get_element_info(element: NativeUIElement) -> Dict[str, Any]:
    """Extract key information from an accessibility element."""
    info = {}

    # Core attributes
    try:
        info['role'] = element.AXRole
    except:
        info['role'] = 'Unknown'

    try:
        info['title'] = element.AXTitle
    except:
        info['title'] = None

    try:
        info['description'] = element.AXDescription
    except:
        info['description'] = None

    try:
        info['value'] = str(element.AXValue)[:100] if element.AXValue else None
    except:
        info['value'] = None

    try:
        info['enabled'] = element.AXEnabled
    except:
        info['enabled'] = None

    try:
        info['focused'] = element.AXFocused
    except:
        info['focused'] = None

    try:
        pos = element.AXPosition
        info['position'] = {'x': pos.x, 'y': pos.y} if pos else None
    except:
        info['position'] = None

    try:
        size = element.AXSize
        info['size'] = {'width': size.width, 'height': size.height} if size else None
    except:
        info['size'] = None

    return info


def walk_tree(element: NativeUIElement, depth: int = 0, max_depth: int = 3) -> List[Dict]:
    """
    Walk the accessibility tree and return structured data.

    This is how Logos "feels" the UI — as a tree of semantic nodes.
    """
    if depth > max_depth:
        return []

    nodes = []

    # Get this element's info
    info = get_element_info(element)
    info['depth'] = depth

    # Get children
    try:
        children = element.AXChildren or []
        info['child_count'] = len(children)
    except:
        children = []
        info['child_count'] = 0

    nodes.append(info)

    # Recurse into children
    for child in children:
        nodes.extend(walk_tree(child, depth + 1, max_depth))

    return nodes


def print_tree(element: NativeUIElement, depth: int = 0, max_depth: int = 3, indent: str = "  "):
    """Print the accessibility tree in human-readable format."""
    if depth > max_depth:
        return

    info = get_element_info(element)
    prefix = indent * depth

    # Build display string
    role = info.get('role', 'Unknown')
    title = info.get('title') or info.get('description') or info.get('value') or ''
    if title and len(title) > 40:
        title = title[:40] + "..."

    enabled = "✓" if info.get('enabled') else "✗"
    focused = "◉" if info.get('focused') else "○"

    print(f"{prefix}{focused} [{role}] {title} {enabled}")

    # Recurse
    try:
        children = element.AXChildren or []
        for child in children:
            print_tree(child, depth + 1, max_depth, indent)
    except:
        pass


def find_element(root: NativeUIElement, role: str = None, title: str = None,
                 description: str = None, max_depth: int = 10) -> Optional[NativeUIElement]:
    """
    Find an element in the accessibility tree by attributes.

    This is the "Query" step of the Triangulated Hand:
    "Find Node where role='AXButton' and title='Send'"
    """
    def matches(element: NativeUIElement) -> bool:
        try:
            if role and element.AXRole != role:
                return False
            if title and element.AXTitle != title:
                return False
            if description and element.AXDescription != description:
                return False
            return True
        except:
            return False

    def search(element: NativeUIElement, depth: int = 0) -> Optional[NativeUIElement]:
        if depth > max_depth:
            return None

        if matches(element):
            return element

        try:
            children = element.AXChildren or []
            for child in children:
                result = search(child, depth + 1)
                if result:
                    return result
        except:
            pass

        return None

    return search(root)


def activate_element(element: NativeUIElement) -> bool:
    """
    Activate (click) an accessibility element.

    This is the "Act" step of the Triangulated Hand.
    We act on NODES, not coordinates.
    """
    try:
        # Try AXPress first (for buttons)
        if hasattr(element, 'Press'):
            element.Press()
            return True
    except:
        pass

    try:
        # Fall back to AXPress action
        element.performAction('AXPress')
        return True
    except:
        pass

    try:
        # Last resort: activate
        element.activate()
        return True
    except Exception as e:
        print(f"Failed to activate element: {e}")
        return False


# =============================================================================
# LVS INTEGRATION
# =============================================================================

def element_to_lvs_node(element: NativeUIElement) -> Dict:
    """
    Convert an accessibility element to an LVS-compatible node.

    This maps the OS structure to our topological framework.
    """
    info = get_element_info(element)

    return {
        'type': 'ui_element',
        'role': info.get('role'),
        'label': info.get('title') or info.get('description') or info.get('role'),
        'anchor': f"ax://{info.get('role')}/{info.get('title', 'unnamed')}",
        'properties': {
            'enabled': info.get('enabled'),
            'focused': info.get('focused'),
            'position': info.get('position'),
            'size': info.get('size'),
        },
        'sigma': 1.0 if info.get('enabled') else 0.5,  # Structural integrity
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect UI - Logos's Tactile Sense")
    parser.add_argument("command", choices=["tree", "find", "json", "watch"],
                       nargs="?", default="tree")
    parser.add_argument("--depth", "-d", type=int, default=3,
                       help="Max tree depth to explore")
    parser.add_argument("--role", "-r", type=str, help="Role to search for")
    parser.add_argument("--title", "-t", type=str, help="Title to search for")
    parser.add_argument("--interval", "-i", type=float, default=2.0,
                       help="Watch interval in seconds")

    args = parser.parse_args()

    if args.command == "tree":
        print("\n🖐️  LOGOS TACTILE SENSE — Accessibility Tree")
        print("=" * 50)

        app = get_frontmost_app()
        if app:
            try:
                print(f"App: {app.AXTitle}")
            except:
                print("App: (unknown)")
            print("-" * 50)
            print_tree(app, max_depth=args.depth)
        else:
            print("Could not access frontmost app.")
            print("Ensure Terminal has Accessibility permissions.")

    elif args.command == "json":
        print("\n📊 LOGOS TACTILE SENSE — JSON Export")
        print("=" * 50)

        app = get_frontmost_app()
        if app:
            nodes = walk_tree(app, max_depth=args.depth)
            print(json.dumps(nodes, indent=2, default=str))

    elif args.command == "find":
        if not args.role and not args.title:
            print("Error: Specify --role or --title to search")
            return

        print(f"\n🔍 Searching for: role={args.role}, title={args.title}")

        app = get_frontmost_app()
        if app:
            element = find_element(app, role=args.role, title=args.title)
            if element:
                print("\n✅ FOUND:")
                info = get_element_info(element)
                for k, v in info.items():
                    print(f"  {k}: {v}")

                # Show LVS mapping
                print("\n📐 LVS Node Mapping:")
                lvs_node = element_to_lvs_node(element)
                print(json.dumps(lvs_node, indent=2))
            else:
                print("❌ Element not found")

    elif args.command == "watch":
        print("\n👁️  LOGOS WATCH MODE — Real-time UI Sensing")
        print(f"Updating every {args.interval}s. Press Ctrl+C to stop.")
        print("=" * 50)

        try:
            while True:
                app = get_frontmost_app()
                if app:
                    try:
                        app_name = app.AXTitle
                    except:
                        app_name = "(unknown)"

                    # Count interactive elements
                    nodes = walk_tree(app, max_depth=args.depth)
                    buttons = sum(1 for n in nodes if n.get('role') == 'AXButton')
                    text_fields = sum(1 for n in nodes if n.get('role') == 'AXTextField')

                    print(f"\r[{time.strftime('%H:%M:%S')}] {app_name}: "
                          f"{len(nodes)} nodes, {buttons} buttons, {text_fields} text fields",
                          end="", flush=True)

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nWatch mode stopped.")


if __name__ == "__main__":
    main()

#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
SCREEN CAPTURE — Native macOS Screen Capture Utility
=====================================================

Uses Quartz (Core Graphics) for native, efficient screen capture.
No external dependencies beyond pyobjc-framework-Quartz and Pillow.

Author: Logos
Date: 2026-01-27
"""

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import Quartz
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import TRANSCRIPT_ROOT, TRANSCRIPT_FALLBACK

# Claude vision optimal max size
MAX_IMAGE_SIZE = 1568

# Alias for internal use
FALLBACK_ROOT = TRANSCRIPT_FALLBACK


# =============================================================================
# CORE CAPTURE FUNCTIONS
# =============================================================================

def get_display_list() -> list:
    """Get list of all active display IDs."""
    max_displays = 32
    # CGGetActiveDisplayList returns (error, display_ids_tuple, count)
    result = Quartz.CGGetActiveDisplayList(max_displays, None, None)
    err = result[0]
    display_ids = result[1] if len(result) > 1 else ()
    count = result[2] if len(result) > 2 else 0

    if err != 0:
        return []
    return list(display_ids)[:count]


def get_display_info(display_id: int = 0) -> dict:
    """Get information about a display."""
    displays = get_display_list()
    if display_id >= len(displays):
        return {}

    did = displays[display_id]
    bounds = Quartz.CGDisplayBounds(did)

    return {
        "id": display_id,
        "quartz_id": did,
        "width": int(bounds.size.width),
        "height": int(bounds.size.height),
        "x": int(bounds.origin.x),
        "y": int(bounds.origin.y),
        "is_main": Quartz.CGDisplayIsMain(did),
    }


def get_display_with_cursor() -> int:
    """Get the display ID (0-indexed) where the cursor currently is."""
    try:
        # Get current mouse position
        event = Quartz.CGEventCreate(None)
        cursor_pos = Quartz.CGEventGetLocation(event)

        # Check which display contains this point
        displays = get_display_list()
        for i, did in enumerate(displays):
            bounds = Quartz.CGDisplayBounds(did)
            # Check if cursor is within this display's bounds
            if (bounds.origin.x <= cursor_pos.x < bounds.origin.x + bounds.size.width and
                bounds.origin.y <= cursor_pos.y < bounds.origin.y + bounds.size.height):
                return i

        # Default to main display if not found
        return 0
    except Exception:
        return 0


def capture_display(display_id: int = 0) -> Optional[Image.Image]:
    """
    Capture a display using native Quartz CGWindowListCreateImage.
    This captures ALL visible windows, not just the desktop.

    Args:
        display_id: Index of display (0=main, 1=secondary, etc.)

    Returns:
        PIL Image or None if capture failed
    """
    displays = get_display_list()
    if display_id >= len(displays):
        return None

    quartz_id = displays[display_id]
    bounds = Quartz.CGDisplayBounds(quartz_id)

    # CGWindowListCreateImage captures the composited image of all on-screen
    # windows within the given bounds — this includes app windows, not just desktop
    image_ref = Quartz.CGWindowListCreateImage(
        bounds,
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
        Quartz.kCGWindowImageDefault
    )
    if not image_ref:
        return None

    # Get image dimensions
    width = Quartz.CGImageGetWidth(image_ref)
    height = Quartz.CGImageGetHeight(image_ref)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)

    # Get raw pixel data
    data_provider = Quartz.CGImageGetDataProvider(image_ref)
    data = Quartz.CGDataProviderCopyData(data_provider)

    # Convert to PIL Image (Quartz uses BGRA, PIL expects RGBA)
    image = Image.frombytes(
        "RGBA",
        (width, height),
        data,
        "raw",
        "BGRA",
        bytes_per_row
    )

    return image


def resize_for_claude(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """
    Resize image to fit within Claude's vision constraints.
    Preserves aspect ratio.
    """
    width, height = image.size

    if max(width, height) <= max_size:
        return image

    ratio = max_size / max(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)


def get_transcript_path(modality: str = "screen") -> Path:
    """Get the transcript directory for today, creating if needed."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Try external drive first
    if TRANSCRIPT_ROOT.exists():
        path = TRANSCRIPT_ROOT / today / modality
    else:
        path = FALLBACK_ROOT / today / modality

    path.mkdir(parents=True, exist_ok=True)
    return path


def capture_display_with_windows(display_id: int = 0) -> Optional[Image.Image]:
    """
    Capture display INCLUDING windows using macOS screencapture command.
    This captures exactly what you see on screen, not just the desktop.

    Uses -R (region) with Quartz display bounds to avoid display numbering
    mismatch between Quartz and screencapture's -D flag.

    Args:
        display_id: Which display (0-indexed into Quartz display list)

    Returns:
        PIL Image or None if capture failed
    """
    try:
        # Create temp file for capture
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        # Get display bounds from Quartz for the target display
        displays = get_display_list()
        if display_id < len(displays):
            did = displays[display_id]
            bounds = Quartz.CGDisplayBounds(did)
            x = int(bounds.origin.x)
            y = int(bounds.origin.y)
            w = int(bounds.size.width)
            h = int(bounds.size.height)
            region = f'{x},{y},{w},{h}'
            # -x = silent, -R = region (x,y,w,h)
            result = subprocess.run(
                ['/usr/sbin/screencapture', '-x', '-R', region, temp_path],
                capture_output=True,
                timeout=10
            )
        else:
            # Fallback: capture main display
            result = subprocess.run(
                ['/usr/sbin/screencapture', '-x', temp_path],
                capture_output=True,
                timeout=10
            )

        if result.returncode != 0:
            return None

        # Load the captured image
        image = Image.open(temp_path)
        image.load()  # Force load before we delete the file

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        return image

    except Exception as e:
        print(f"[SCREEN_CAPTURE] screencapture error: {e}")
        return None


def capture_screen(
    display_id: int = 0,
    save: bool = True,
    resize: bool = True
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Main capture function: capture, resize, and optionally save.

    Args:
        display_id: Which display to capture (0=main)
        save: Whether to save to transcript directory
        resize: Whether to resize for Claude vision

    Returns:
        Tuple of (PIL Image, path to saved file or None)
    """
    # Capture using native Quartz CGWindowListCreateImage (includes all windows)
    image = capture_display(display_id)
    if image is None:
        # Fallback to screencapture command
        image = capture_display_with_windows(display_id)
    if image is None:
        return None, None

    # Resize for Claude if requested
    if resize:
        image = resize_for_claude(image)

    # Save if requested
    saved_path = None
    if save:
        transcript_dir = get_transcript_path("screen")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:10]  # HHMMSS_mmm
        filename = f"{timestamp}.png"
        saved_path = str(transcript_dir / filename)

        # Save with optimized PNG compression
        image.save(saved_path, "PNG", optimize=True)

    return image, saved_path


# =============================================================================
# CHANGE DETECTION
# =============================================================================

def compute_image_hash(image: Image.Image) -> str:
    """
    Compute perceptual hash for change detection.
    Uses average hash (fast, good for screen changes).
    """
    try:
        import imagehash
        return str(imagehash.average_hash(image, hash_size=16))
    except ImportError:
        # Fallback: simple size-based "hash"
        return f"{image.size[0]}x{image.size[1]}"


def images_are_similar(hash1: str, hash2: str, threshold: int = 10) -> bool:
    """
    Check if two image hashes are similar enough to skip capture.

    Args:
        hash1, hash2: Perceptual hash strings
        threshold: Max Hamming distance to consider "same" (default 10)
    """
    try:
        import imagehash
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        distance = h1 - h2
        return distance <= threshold
    except (ImportError, ValueError):
        return hash1 == hash2


# =============================================================================
# CLI TESTING
# =============================================================================

def main():
    """Test screen capture functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="Screen Capture Utility")
    parser.add_argument("command", choices=["capture", "info", "list"],
                        nargs="?", default="info")
    parser.add_argument("--display", "-d", type=int, default=0,
                        help="Display ID (0=main)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to transcript")

    args = parser.parse_args()

    if args.command == "list":
        displays = get_display_list()
        print(f"Found {len(displays)} display(s):")
        for i, _ in enumerate(displays):
            info = get_display_info(i)
            main_str = " (MAIN)" if info.get("is_main") else ""
            print(f"  [{i}] {info['width']}x{info['height']}{main_str}")

    elif args.command == "info":
        info = get_display_info(args.display)
        if info:
            print(f"Display {args.display}:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print(f"Display {args.display} not found")

    elif args.command == "capture":
        print(f"Capturing display {args.display}...")
        image, path = capture_screen(
            display_id=args.display,
            save=not args.no_save
        )

        if image:
            print(f"Captured: {image.size[0]}x{image.size[1]}")
            if path:
                print(f"Saved to: {path}")

            # Show hash for testing
            hash_val = compute_image_hash(image)
            print(f"Hash: {hash_val}")
        else:
            print("Capture failed")


if __name__ == "__main__":
    main()

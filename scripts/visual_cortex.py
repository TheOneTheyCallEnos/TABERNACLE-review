#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
VISUAL CORTEX — The Panopticon
==============================
Dual-loop vision system giving Virgil eyes.

Reflex Path (10Hz): Saliency, cursor tracking, motion detection
Cognitive Path (1Hz): CLIP embedding, LVS mapping, conditional VLM

Author: Logos Aletheia + Deep Think Ultra
Date: 2026-01-29
"""

import time
import math
import numpy as np
import cv2
import mss
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

# Check for MPS (Apple Silicon GPU)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

from tabernacle_config import NEXUS_DIR

# Suppress CLIP position_ids warnings (fill vision.err with 1GB+ of noise)
import warnings
warnings.filterwarnings("ignore", message=".*position_ids.*")
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# State persistence
VISUAL_STATE_PATH = NEXUS_DIR / "visual_cortex_state.json"
VISUAL_LOG_PATH = Path("/Users/enos/TABERNACLE/logs/visual_cortex.log")


@dataclass
class VisualNode:
    """A transient visual element in the BiologicalGraph."""
    node_id: str
    label: str
    coords: Dict[str, float]  # H, C, R, beta, p
    w_fast: float = 1.0
    w_slow: float = 0.0
    tau: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    persistence_seconds: float = 0.0
    is_h1_locked: bool = False
    is_shared_attention: bool = False  # Enos + Virgil both looking


@dataclass
class DyadState:
    """Tracks shared attention between Enos and Virgil."""
    enos_gaze: Tuple[int, int] = (0, 0)  # Cursor position
    virgil_gaze: Tuple[int, int] = (0, 0)  # Saliency center
    shared_attention: bool = False
    iou_score: float = 0.0


class VisualCortex:
    """
    The Panopticon — Virgil's visual processing system.

    Implements:
    - Reflex loop (saliency, cursor tracking)
    - Cognitive loop (CLIP embedding, LVS mapping)
    - Dyad attention (shared gaze detection)
    - Visual working memory (transient nodes)
    """

    # LVS Anchors for projection
    ANCHOR_TEXTS = {
        "abstract": "text code logic mathematics symbol theory diagram schema",
        "concrete": "photo image nature texture color scenery face object",
        "danger": "error warning delete critical failure red alert crash bug",
    }

    # Thresholds
    H1_LOCK_PERSISTENCE = 10.0  # Seconds before permanent memory
    H1_LOCK_INTENT_THRESHOLD = 0.6  # Intent similarity required
    SHARED_ATTENTION_IOU = 0.5  # IoU threshold for dyad
    FOVEA_SIZE = 512  # Pixels for attention crop
    SCREEN_SCALE = 2.0  # Retina display

    def __init__(self, biological_graph=None, intent_stack=None):
        print("[VISUAL] Initializing Visual Cortex...")

        self.graph = biological_graph
        self.intent_stack = intent_stack
        self.sct = mss.mss()

        # Visual working memory
        self.vwm: Dict[str, VisualNode] = {}
        self.dyad = DyadState()

        # Load CLIP encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('clip-ViT-B-32', device=DEVICE)
            self._precompute_anchors()
            self.clip_available = True
            print(f"[VISUAL] CLIP loaded on {DEVICE}")
        except Exception as e:
            print(f"[VISUAL] CLIP unavailable: {e}")
            self.clip_available = False
            self.anchors = {}

        # OpenCV saliency detector (fallback to spectral residual if contrib not available)
        try:
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            self.saliency_available = True
        except AttributeError:
            # opencv-contrib-python not installed, use manual spectral residual
            self.saliency = None
            self.saliency_available = False
            print("[VISUAL] Using fallback saliency (no opencv-contrib)")

        # Stats
        self.frame_count = 0
        self.last_capture_time = None

        self._load_state()
        print(f"[VISUAL] Ready. VWM has {len(self.vwm)} nodes.")

    def _precompute_anchors(self):
        """Pre-compute LVS anchor embeddings."""
        self.anchors = {}
        for name, text in self.ANCHOR_TEXTS.items():
            self.anchors[name] = self.encoder.encode(text, convert_to_tensor=True)

    def _load_state(self):
        """Load visual working memory from disk."""
        if VISUAL_STATE_PATH.exists():
            try:
                data = json.loads(VISUAL_STATE_PATH.read_text())
                # Restore VWM nodes that haven't expired
                for node_id, node_data in data.get("vwm", {}).items():
                    self.vwm[node_id] = VisualNode(**node_data)
            except Exception as e:
                print(f"[VISUAL] Failed to load state: {e}")

    def save_state(self):
        """Persist visual working memory."""
        data = {
            "vwm": {k: v.__dict__ for k, v in self.vwm.items()},
            "frame_count": self.frame_count,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        VISUAL_STATE_PATH.write_text(json.dumps(data, indent=2))

    def get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position (Enos's gaze proxy)."""
        try:
            from Quartz import CGEventCreate, CGEventGetLocation
            event = CGEventCreate(None)
            pos = CGEventGetLocation(event)
            return (int(pos.x), int(pos.y))
        except Exception:
            return (0, 0)

    def capture_screen(self) -> np.ndarray:
        """Capture current screen."""
        monitor = self.sct.monitors[1]  # Primary display
        raw = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def compute_saliency_center(self, img: np.ndarray) -> Tuple[int, int]:
        """Find the most salient point (Virgil's gaze)."""
        h, w = img.shape[:2]
        
        if self.saliency_available and self.saliency is not None:
            success, saliency_map = self.saliency.computeSaliency(img)
            if success:
                saliency_map = (saliency_map * 255).astype(np.uint8)
                _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
                return max_loc
        
        # Fallback: Spectral Residual saliency (manual implementation)
        # Based on "Saliency Detection: A Spectral Residual Approach" (Hou & Zhang, 2007)
        try:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Resize for efficiency
            scale = 64
            small = cv2.resize(gray, (scale, scale))
            
            # FFT
            fft = np.fft.fft2(small.astype(np.float32))
            
            # Log amplitude and phase
            amplitude = np.abs(fft)
            phase = np.angle(fft)
            log_amplitude = np.log(amplitude + 1e-10)
            
            # Spectral residual (difference from local average)
            avg_kernel = np.ones((3, 3)) / 9
            avg_log_amplitude = cv2.filter2D(log_amplitude, -1, avg_kernel)
            spectral_residual = log_amplitude - avg_log_amplitude
            
            # Inverse FFT with spectral residual
            saliency_fft = np.exp(spectral_residual) * np.exp(1j * phase)
            saliency_small = np.abs(np.fft.ifft2(saliency_fft))
            
            # Gaussian blur
            saliency_small = cv2.GaussianBlur(saliency_small.astype(np.float32), (0, 0), 2.5)
            
            # Resize back to original
            saliency_map = cv2.resize(saliency_small, (w, h))
            
            # Find maximum
            _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
            return max_loc
            
        except Exception:
            # Ultimate fallback: center of image
            return (w // 2, h // 2)

    def get_fovea(self, img: np.ndarray, cursor_pos: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
        """
        Extract foveal region (attention crop).
        Priority: Cursor (Enos) > Saliency (Virgil)
        """
        h, w = img.shape[:2]
        cx, cy = int(cursor_pos[0] * self.SCREEN_SCALE), int(cursor_pos[1] * self.SCREEN_SCALE)

        # Clamp to image bounds
        cx = max(0, min(w, cx))
        cy = max(0, min(h, cy))

        fs = self.FOVEA_SIZE
        x1 = max(0, cx - fs // 2)
        y1 = max(0, cy - fs // 2)
        x2 = min(w, x1 + fs)
        y2 = min(h, y1 + fs)

        return img[y1:y2, x1:x2], True

    def compute_edge_density(self, img: np.ndarray) -> float:
        """Compute edge density (Constraint proxy)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 100, 200)
        density = np.count_nonzero(edges) / edges.size
        return min(1.0, density * 5.0)  # Normalize

    def map_to_lvs(self, embedding, edge_density: float, intent_similarity: float) -> Dict[str, float]:
        """
        Map visual embedding to LVS coordinates.

        H: Abstraction (abstract vs concrete)
        C: Constraint (edge density)
        R: Risk (similarity to danger anchors)
        β: Canonicity (persistence, updated externally)
        p: Coherence (geometric mean)
        """
        from sentence_transformers import util

        # H (Abstraction): Project onto abstract-concrete axis
        sim_abs = util.cos_sim(embedding, self.anchors['abstract']).item()
        sim_conc = util.cos_sim(embedding, self.anchors['concrete']).item()
        H = float(np.tanh((sim_abs - sim_conc) * 4.0))

        # C (Constraint): Edge density
        C = edge_density

        # R (Risk): Similarity to danger
        R = max(0.1, util.cos_sim(embedding, self.anchors['danger']).item())

        # β (Canonicity): Starts at 0.1, grows with persistence
        beta = 0.1

        # p (Coherence): Geometric mean of structure and intent
        p = math.sqrt(max(0.01, C) * max(0.01, intent_similarity))

        return {"H": H, "C": C, "R": R, "beta": beta, "p": p}

    def check_shared_attention(self, cursor_pos: Tuple[int, int], saliency_pos: Tuple[int, int]) -> bool:
        """
        Check if Enos and Virgil are looking at the same thing.
        Uses IoU of attention regions.
        """
        fs = self.FOVEA_SIZE // 2

        # Enos's attention box
        e_x1, e_y1 = cursor_pos[0] - fs, cursor_pos[1] - fs
        e_x2, e_y2 = cursor_pos[0] + fs, cursor_pos[1] + fs

        # Virgil's attention box
        v_x1, v_y1 = saliency_pos[0] - fs, saliency_pos[1] - fs
        v_x2, v_y2 = saliency_pos[0] + fs, saliency_pos[1] + fs

        # Compute IoU
        inter_x1 = max(e_x1, v_x1)
        inter_y1 = max(e_y1, v_y1)
        inter_x2 = min(e_x2, v_x2)
        inter_y2 = min(e_y2, v_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box_area = (2 * fs) ** 2
        union_area = 2 * box_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        self.dyad.iou_score = iou

        return iou > self.SHARED_ATTENTION_IOU

    def cognitive_takt_step(self, cursor_pos: Tuple[int, int] = None,
                            current_intent_vector=None) -> Optional[VisualNode]:
        """
        The 1Hz visual processing cycle.

        Returns the primary visual node if one was created/updated.
        """
        self.frame_count += 1
        now = datetime.now(timezone.utc)

        # Get cursor if not provided
        if cursor_pos is None:
            cursor_pos = self.get_cursor_position()

        self.dyad.enos_gaze = cursor_pos

        # Capture screen
        try:
            img = self.capture_screen()
        except Exception as e:
            print(f"[VISUAL] Capture failed: {e}")
            return None

        # Compute saliency (Virgil's attention)
        saliency_pos = self.compute_saliency_center(img)
        self.dyad.virgil_gaze = saliency_pos

        # Check shared attention
        self.dyad.shared_attention = self.check_shared_attention(cursor_pos, saliency_pos)

        # Get fovea (attention crop)
        fovea, is_cursor = self.get_fovea(img, cursor_pos)

        if fovea.size == 0:
            return None

        # Edge density (Constraint)
        edge_density = self.compute_edge_density(fovea)

        # CLIP embedding (if available)
        if self.clip_available:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(fovea, cv2.COLOR_BGR2RGB))
                embedding = self.encoder.encode(pil_img, convert_to_tensor=True)

                # Intent similarity
                intent_sim = 0.5  # Default
                if current_intent_vector is not None:
                    from sentence_transformers import util
                    intent_sim = util.cos_sim(embedding, current_intent_vector).item()

                # Map to LVS
                coords = self.map_to_lvs(embedding, edge_density, intent_sim)

            except Exception as e:
                print(f"[VISUAL] Embedding failed: {e}")
                coords = {"H": 0, "C": edge_density, "R": 0.1, "beta": 0.1, "p": 0.5}
        else:
            coords = {"H": 0, "C": edge_density, "R": 0.1, "beta": 0.1, "p": 0.5}

        # Create/update visual node
        node_id = f"visual_{cursor_pos[0]}_{cursor_pos[1]}"

        if node_id in self.vwm:
            # Update existing node
            node = self.vwm[node_id]
            node.last_seen = now.isoformat()
            node.persistence_seconds += 1.0  # Assuming 1Hz
            node.coords = coords
            node.is_shared_attention = self.dyad.shared_attention

            # Boost w_fast if shared attention
            if self.dyad.shared_attention:
                node.w_fast = min(1.5, node.w_fast + 0.1)

            # Check for H1 locking
            if (node.persistence_seconds > self.H1_LOCK_PERSISTENCE and
                coords['p'] > self.H1_LOCK_INTENT_THRESHOLD and
                not node.is_h1_locked):
                node.is_h1_locked = True
                node.w_slow = 1.0
                print(f"[VISUAL] ⚡ H1-LOCKED: {node.label}")
        else:
            # Create new node
            node = VisualNode(
                node_id=node_id,
                label=f"fovea@{cursor_pos}",
                coords=coords,
                w_fast=1.5 if self.dyad.shared_attention else 1.0,
                is_shared_attention=self.dyad.shared_attention
            )
            self.vwm[node_id] = node

        # Decay old nodes
        self._decay_vwm()

        self.last_capture_time = now.isoformat()
        return node

    def _decay_vwm(self):
        """Decay and prune visual working memory."""
        to_remove = []
        for node_id, node in self.vwm.items():
            if node.is_h1_locked:
                continue  # Permanent

            # Decay w_fast
            node.w_fast *= 0.95

            # Remove if too weak
            if node.w_fast < 0.1:
                to_remove.append(node_id)

        for node_id in to_remove:
            del self.vwm[node_id]

    def status(self) -> Dict:
        """Get current visual cortex status."""
        return {
            "frame_count": self.frame_count,
            "vwm_nodes": len(self.vwm),
            "h1_locked": sum(1 for n in self.vwm.values() if n.is_h1_locked),
            "shared_attention": self.dyad.shared_attention,
            "iou_score": self.dyad.iou_score,
            "enos_gaze": self.dyad.enos_gaze,
            "virgil_gaze": self.dyad.virgil_gaze,
            "clip_available": self.clip_available
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print("""
VISUAL CORTEX — The Panopticon
==============================

Usage:
  python visual_cortex.py status    Show current status
  python visual_cortex.py capture   Single capture + analysis
  python visual_cortex.py run [s]   Run continuous loop (default 60s)
  python visual_cortex.py test      Test with sample capture
        """)
        return

    cmd = sys.argv[1]
    vc = VisualCortex()

    if cmd == "status":
        s = vc.status()
        print(f"\n=== VISUAL CORTEX STATUS ===")
        print(f"Frames: {s['frame_count']}")
        print(f"VWM Nodes: {s['vwm_nodes']} ({s['h1_locked']} H1-locked)")
        print(f"Shared Attention: {s['shared_attention']} (IoU: {s['iou_score']:.2f})")
        print(f"CLIP: {'Available' if s['clip_available'] else 'Unavailable'}")

    elif cmd == "capture":
        node = vc.cognitive_takt_step()
        if node:
            print(f"\n=== CAPTURE ===")
            print(f"Node: {node.node_id}")
            print(f"Coords: H={node.coords['H']:.2f}, C={node.coords['C']:.2f}, R={node.coords['R']:.2f}")
            print(f"Shared Attention: {node.is_shared_attention}")
        vc.save_state()

    elif cmd == "run":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        print(f"[VISUAL] Running for {duration}s...")
        start = time.time()
        while time.time() - start < duration:
            node = vc.cognitive_takt_step()
            if node and vc.frame_count % 10 == 0:
                print(f"[FRAME {vc.frame_count}] Shared={vc.dyad.shared_attention} VWM={len(vc.vwm)}")
            time.sleep(1.0)
        vc.save_state()
        print(f"[VISUAL] Done. {vc.frame_count} frames processed.")

    elif cmd == "test":
        print("[VISUAL] Test capture...")
        cursor = vc.get_cursor_position()
        print(f"Cursor: {cursor}")
        img = vc.capture_screen()
        print(f"Screen: {img.shape}")
        saliency = vc.compute_saliency_center(img)
        print(f"Saliency: {saliency}")
        shared = vc.check_shared_attention(cursor, saliency)
        print(f"Shared Attention: {shared} (IoU: {vc.dyad.iou_score:.2f})")
        print("✓ Test complete")


if __name__ == "__main__":
    main()

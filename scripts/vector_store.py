#!/usr/bin/env python3
"""
VECTOR STORE — Lightweight Semantic Search
==========================================
Uses sentence-transformers + numpy for semantic search.
Persists embeddings to disk for fast reload.

Author: Virgil
Date: 2026-01-19
Status: Phase II-B (Path B Implementation)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CONFIGURATION (using centralized config)
# =============================================================================

from tabernacle_config import BASE_DIR

TABERNACLE = BASE_DIR  # Alias for backwards compatibility
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"
INDEX_FILE = DATA_DIR / "vector_index.json"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model for embeddings (loads on first use)
_model = None
_model_name = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions

def get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        print(f"[VECTOR] Loading model: {_model_name}")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        print(f"[VECTOR] Model loaded")
    return _model


# =============================================================================
# VECTOR STORE
# =============================================================================

class VectorStore:
    """Lightweight vector store with persistence."""

    def __init__(self):
        self.documents: List[Dict] = []  # [{id, path, text, metadata}]
        self.embeddings: Optional[np.ndarray] = None  # Shape: (n_docs, embedding_dim)
        self.id_to_idx: Dict[str, int] = {}

        # Load existing data
        self.load()

    def load(self) -> bool:
        """Load embeddings from disk."""
        if EMBEDDINGS_FILE.exists() and INDEX_FILE.exists():
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    self.embeddings = pickle.load(f)
                with open(INDEX_FILE) as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.id_to_idx = {d["id"]: i for i, d in enumerate(self.documents)}
                print(f"[VECTOR] Loaded {len(self.documents)} documents")
                return True
            except Exception as e:
                print(f"[VECTOR] Error loading: {e}")
        return False

    def save(self) -> bool:
        """Save embeddings to disk."""
        try:
            if self.embeddings is not None:
                with open(EMBEDDINGS_FILE, 'wb') as f:
                    pickle.dump(self.embeddings, f)

            index_data = {
                "documents": self.documents,
                "model": _model_name,
                "dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
                "count": len(self.documents),
                "saved_at": datetime.now().isoformat()
            }
            with open(INDEX_FILE, 'w') as f:
                json.dump(index_data, f, indent=2)

            print(f"[VECTOR] Saved {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"[VECTOR] Error saving: {e}")
            return False

    def add(self, doc_id: str, text: str, path: str = "", metadata: Dict = None) -> bool:
        """Add or update a document."""
        model = get_model()

        # Generate embedding
        embedding = model.encode([text[:2000]])[0]  # Limit text size

        if doc_id in self.id_to_idx:
            # Update existing
            idx = self.id_to_idx[doc_id]
            self.documents[idx] = {
                "id": doc_id,
                "path": path,
                "text_preview": text[:200],
                "metadata": metadata or {},
                "updated_at": datetime.now().isoformat()
            }
            self.embeddings[idx] = embedding
        else:
            # Add new
            self.documents.append({
                "id": doc_id,
                "path": path,
                "text_preview": text[:200],
                "metadata": metadata or {},
                "added_at": datetime.now().isoformat()
            })
            self.id_to_idx[doc_id] = len(self.documents) - 1

            if self.embeddings is None:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])

        return True

    def search(self, query: str, limit: int = 10) -> List[Tuple[Dict, float]]:
        """Search for similar documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        model = get_model()

        # Encode query
        query_embedding = model.encode([query])[0]

        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only positive similarity
                results.append((self.documents[idx], float(similarities[idx])))

        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents."""
        # Normalize
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(b_norm, a_norm)

    def get_stats(self) -> Dict:
        """Get store statistics."""
        return {
            "document_count": len(self.documents),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model": _model_name,
            "storage_path": str(DATA_DIR)
        }

    def index_tabernacle(self) -> int:
        """Index all markdown files in Tabernacle."""
        md_files = list(TABERNACLE.rglob("*.md"))
        indexed = 0

        for f in md_files:
            # Skip irrelevant directories
            if any(skip in str(f) for skip in [".git", "venv", "node_modules"]):
                continue

            try:
                rel_path = str(f.relative_to(TABERNACLE))
                content = f.read_text()

                # Use path hash as ID
                doc_id = rel_path.replace("/", "_").replace(".", "_")

                self.add(
                    doc_id=doc_id,
                    text=content,
                    path=rel_path,
                    metadata={"type": "markdown", "size": len(content)}
                )
                indexed += 1

                if indexed % 50 == 0:
                    print(f"[VECTOR] Indexed {indexed} files...")

            except Exception as e:
                print(f"[VECTOR] Error indexing {f}: {e}")

        self.save()
        print(f"[VECTOR] Total indexed: {indexed}")
        return indexed


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vector Store")
    parser.add_argument("command", choices=["index", "search", "stats"],
                        help="Command to execute")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Result limit")

    args = parser.parse_args()

    store = VectorStore()

    if args.command == "index":
        count = store.index_tabernacle()
        print(f"Indexed {count} files")

    elif args.command == "search":
        if not args.query:
            print("Error: --query required")
            return
        results = store.search(args.query, limit=args.limit)
        print(f"\nTop {len(results)} results for: '{args.query}'\n")
        for doc, score in results:
            print(f"  [{score:.3f}] {doc['path']}")
            print(f"          {doc['text_preview'][:80]}...")
            print()

    elif args.command == "stats":
        print(json.dumps(store.get_stats(), indent=2))


if __name__ == "__main__":
    main()

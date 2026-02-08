"""
HoloTower Vectors — ChromaDB Vector Storage for Semantic Search

Implements:
    - ChromaDB storage in holotower/vectors/
    - Ollama nomic-embed-text embeddings
    - Semantic similarity search across holons
    - Vector store hashing for snapshot versioning
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import requests
import typer
from rich.console import Console
from rich.table import Table

from .core import get_holotower_root

# CLI app for standalone execution
app = typer.Typer(
    name="vectors",
    help="HoloTower Vector Storage — Semantic search for holons"
)
console = Console()

# Ollama embedding endpoint
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Collection name
COLLECTION_NAME = "holons"


def get_vectors_dir() -> Path:
    """Get the vectors directory path."""
    root = get_holotower_root()
    return root / "vectors"


def get_chroma_client():
    """
    Get or create ChromaDB client with persistent storage.

    Returns:
        chromadb.Client (with persistence)
    """
    import chromadb
    from chromadb.config import Settings

    vectors_dir = get_vectors_dir()
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Use chromadb 0.3.x API for pydantic v1 compatibility
    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(vectors_dir),
        anonymized_telemetry=False
    ))


def get_collection():
    """
    Get or create the holons collection.
    
    Returns:
        chromadb.Collection
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def init_vector_store() -> Path:
    """
    Initialize the vector store.
    
    Creates holotower/vectors/ directory and initializes ChromaDB.
    
    Returns:
        Path to the vectors directory
    """
    vectors_dir = get_vectors_dir()
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB (creates chroma.sqlite3)
    client = get_chroma_client()
    
    # Create the collection
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    return vectors_dir


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding via Ollama nomic-embed-text.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats (embedding vector)
        
    Raises:
        ConnectionError: If Ollama is not available
        ValueError: If embedding fails
    """
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text
            },
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        embedding = data.get("embedding")
        
        if not embedding:
            raise ValueError(f"No embedding returned: {data}")
        
        return embedding
        
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_EMBED_URL}. "
            "Ensure Ollama is running with nomic-embed-text model."
        )
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Embedding request failed: {e}")


def embed_holon(file_path: str, content: str) -> str:
    """
    Generate embedding and store in ChromaDB.
    
    Args:
        file_path: Path to the file (used as ID)
        content: File content to embed
        
    Returns:
        Embedding ID (the file_path)
    """
    # Generate embedding
    embedding = get_embedding(content)
    
    # Get collection
    collection = get_collection()
    
    # Upsert (update or insert)
    collection.upsert(
        ids=[file_path],
        embeddings=[embedding],
        metadatas=[{
            "file_path": file_path,
            "content_preview": content[:500] if len(content) > 500 else content,
            "char_count": len(content)
        }],
        documents=[content]
    )
    
    return file_path


def query_similar(text: str, k: int = 5) -> List[dict]:
    """
    Find k most similar holons to the query text.
    
    Args:
        text: Query text
        k: Number of results to return (default 5)
        
    Returns:
        List of dicts with keys: id, score, metadata, document
    """
    # Generate query embedding
    query_embedding = get_embedding(text)
    
    # Get collection
    collection = get_collection()
    
    # Query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "documents", "distances"]
    )
    
    # Format results
    formatted = []
    
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns L2 distance; convert to similarity score
            # For cosine space, distance is 1 - similarity
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = 1 - distance
            
            formatted.append({
                "id": doc_id,
                "score": similarity,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "document": results["documents"][0][i] if results["documents"] else ""
            })
    
    return formatted


def get_vector_hash() -> str:
    """
    Get SHA-256 hash of the ChromaDB database file.
    
    Used for snapshot versioning to track vector store state.
    
    Returns:
        SHA-256 hex digest of chroma.sqlite3, or empty string if not found
    """
    vectors_dir = get_vectors_dir()
    db_path = vectors_dir / "chroma.sqlite3"
    
    if not db_path.exists():
        return ""
    
    # Hash the file contents
    sha256 = hashlib.sha256()
    
    with open(db_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def get_vector_stats() -> dict:
    """
    Get statistics about the vector store.
    
    Returns:
        Dict with count, hash, and size_bytes
    """
    vectors_dir = get_vectors_dir()
    db_path = vectors_dir / "chroma.sqlite3"
    
    stats = {
        "count": 0,
        "hash": "",
        "size_bytes": 0,
        "initialized": False
    }
    
    if not db_path.exists():
        return stats
    
    stats["initialized"] = True
    stats["hash"] = get_vector_hash()
    stats["size_bytes"] = db_path.stat().st_size
    
    try:
        collection = get_collection()
        stats["count"] = collection.count()
    except Exception:
        pass
    
    return stats


# =============================================================================
# CLI Commands
# =============================================================================

@app.command()
def init():
    """Initialize the vector store."""
    
    console.print("[bold cyan]HoloTower Vectors[/bold cyan] — Initializing...")
    
    try:
        vectors_dir = init_vector_store()
        console.print(f"  [green]✓[/green] Created: {vectors_dir}")
        
        # Verify chroma.sqlite3 was created
        db_path = vectors_dir / "chroma.sqlite3"
        if db_path.exists():
            console.print(f"  [green]✓[/green] Database: chroma.sqlite3 ({db_path.stat().st_size} bytes)")
        
        console.print()
        console.print("[bold green]Vector store initialized.[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def embed(
    file_path: str = typer.Argument(..., help="Path to file to embed"),
    show_preview: bool = typer.Option(False, "--preview", "-p", help="Show content preview")
):
    """Embed a file into the vector store."""
    
    # Resolve path
    path = Path(file_path)
    if not path.is_absolute():
        # Try relative to repo root
        root = get_holotower_root().parent
        path = root / file_path
    
    if not path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Read content
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        console.print(f"[red]Cannot read file (not UTF-8): {file_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[dim]Embedding: {file_path}[/dim]")
    
    if show_preview:
        preview = content[:200] + "..." if len(content) > 200 else content
        console.print(f"[dim]{preview}[/dim]")
    
    try:
        doc_id = embed_holon(file_path, content)
        console.print(f"[green]✓[/green] Embedded: {doc_id}")
        console.print(f"  Characters: {len(content)}")
        
    except ConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    text: str = typer.Argument(..., help="Search query text"),
    k: int = typer.Option(5, "-k", "--limit", help="Number of results")
):
    """Query for similar holons."""
    
    console.print(f"[dim]Searching for: {text[:50]}...[/dim]" if len(text) > 50 else f"[dim]Searching for: {text}[/dim]")
    console.print()
    
    try:
        results = query_similar(text, k=k)
        
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        # Display as table
        table = Table(title=f"Top {len(results)} Similar Holons")
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", justify="right", width=8)
        table.add_column("File", style="cyan")
        table.add_column("Preview", style="dim", max_width=40)
        
        for i, result in enumerate(results, 1):
            score = f"{result['score']:.3f}"
            file_id = result["id"]
            
            # Get preview from metadata or document
            preview = result.get("metadata", {}).get("content_preview", "")
            if not preview and result.get("document"):
                preview = result["document"][:100]
            preview = preview.replace("\n", " ")[:40] + "..." if len(preview) > 40 else preview
            
            table.add_row(str(i), score, file_id, preview)
        
        console.print(table)
        
    except ConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("hash")
def show_hash():
    """Show the vector store hash for snapshot versioning."""
    
    stats = get_vector_stats()
    
    if not stats["initialized"]:
        console.print("[yellow]Vector store not initialized. Run 'vectors init' first.[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]Vector Store Hash[/bold cyan]")
    console.print(f"  Hash:    {stats['hash'][:12]}..." if stats['hash'] else "  Hash:    (empty)")
    console.print(f"  Count:   {stats['count']} documents")
    console.print(f"  Size:    {stats['size_bytes']:,} bytes")


@app.command("stats")
def show_stats():
    """Show vector store statistics."""
    
    stats = get_vector_stats()
    
    console.print("[bold cyan]Vector Store Statistics[/bold cyan]")
    console.print()
    
    if not stats["initialized"]:
        console.print("[yellow]Not initialized. Run 'vectors init' first.[/yellow]")
        return
    
    console.print(f"  Status:      [green]Initialized[/green]")
    console.print(f"  Documents:   {stats['count']}")
    console.print(f"  Size:        {stats['size_bytes']:,} bytes")
    console.print(f"  Hash:        {stats['hash'][:16]}..." if stats['hash'] else "  Hash:        (empty)")


if __name__ == "__main__":
    app()

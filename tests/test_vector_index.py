"""Test vector indexing functionality."""
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import build_vector_index, load_index, INDEXES_DIR


def test_build_and_load_index():
    """Test building and loading a vector index."""
    # Find an existing repo that has docs.jsonl
    repo_id = None
    docs_file = None
    
    # Search for any repo with docs.jsonl
    if INDEXES_DIR.exists():
        for repo_dir in INDEXES_DIR.iterdir():
            if repo_dir.is_dir():
                candidate_docs = repo_dir / "docs.jsonl"
                if candidate_docs.exists():
                    repo_id = repo_dir.name
                    docs_file = candidate_docs
                    break
    
    if not repo_id or not docs_file:
        print("No repository with docs.jsonl found in data/indexes/")
        print("Skipping test - need to run ingestion first")
        print(f"Searched in: {INDEXES_DIR}")
        return
    
    index_dir = INDEXES_DIR / repo_id
    
    print(f"\n=== Testing Vector Index Build ===")
    print(f"Repo ID: {repo_id}")
    print(f"Docs file: {docs_file}")
    
    # Count documents
    with open(docs_file, 'r', encoding='utf-8') as f:
        num_docs = sum(1 for line in f if line.strip())
    print(f"Number of documents: {num_docs}")
    
    # Build index
    print("\nBuilding vector index...")
    num_vectors, embed_dim = build_vector_index(repo_id, docs_file)
    
    print(f"\n✓ Index built successfully!")
    print(f"  Vectors: {num_vectors}")
    print(f"  Embedding dimension: {embed_dim}")
    
    # Verify files were created
    faiss_index_file = index_dir / "faiss.index"
    meta_file = index_dir / "meta.jsonl"
    
    assert faiss_index_file.exists(), "faiss.index file not created"
    assert meta_file.exists(), "meta.jsonl file not created"
    print(f"\n✓ Files created:")
    print(f"  {faiss_index_file}")
    print(f"  {meta_file}")
    
    # Load index
    print("\nLoading index...")
    index, metadata, success = load_index(repo_id)
    
    assert success, "Failed to load index"
    assert index is not None, "Index is None"
    assert metadata is not None, "Metadata is None"
    print(f"\n✓ Index loaded successfully!")
    print(f"  Index vectors: {index.ntotal}")
    print(f"  Metadata entries: {len(metadata)}")
    
    # Verify metadata structure
    if metadata:
        sample_meta = metadata[0]
        print(f"\n✓ Sample metadata entry:")
        print(f"  {json.dumps(sample_meta, indent=2)}")
    
    print("\n=== All tests passed! ===\n")


if __name__ == "__main__":
    test_build_and_load_index()

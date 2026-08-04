"""
build_index.py
Embeds data/chunks.jsonl with a sentence-transformer model and builds a
FAISS index for retrieval. Saves the index and chunk metadata to disk.

Usage (from project root, works fine locally, no GPU needed):
    python src/build_index.py
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.jsonl")
INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, CPU-friendly


def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks loaded")

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embeddings = embedder.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    # Inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = INDEX_DIR / "chunks.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path} ({index.ntotal} vectors, dim={dim})")

    metadata_path = INDEX_DIR / "chunk_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk metadata to {metadata_path}")

    # Save the embedding model name too, so agent.py always uses the same one
    with open(INDEX_DIR / "embedding_model.txt", "w") as f:
        f.write(EMBEDDING_MODEL)


if __name__ == "__main__":
    main()
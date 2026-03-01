"""Vector indexer using HNSW."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

from ..embedding.embedder import get_embedder


class VectorIndexer:
    """Vector indexer using HNSW."""

    def __init__(
        self,
        index_path: Path,
        embedding_dim: int = 384,  # all-MiniLM-L6-v2 dimension
        max_elements: int = 1000000,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,  # Increased from 50 for better recall on multi-word queries
    ):
        """
        Initialize the vector indexer.

        Args:
            index_path: Path to save/load the index
            embedding_dim: Dimension of embeddings
            max_elements: Maximum number of elements in the index
            m: HNSW M parameter
            ef_construction: HNSW ef_construction parameter
            ef_search: HNSW ef_search parameter
        """
        if not HNSW_AVAILABLE:
            raise ImportError("hnswlib is not installed. Install it with: pip install hnswlib")

        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.max_elements = max_elements
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self._index = None
        self._id_mapping: Dict[int, int] = {}  # vector_id -> chunk_id
        self._reverse_mapping: Dict[int, int] = {}  # chunk_id -> vector_id
        self._next_id = 0
        self._embedder = None

    @property
    def embedder(self):
        """Lazy load the embedder."""
        if self._embedder is None:
            self._embedder = get_embedder()
            self.embedding_dim = self._embedder.embedding_dim
        return self._embedder

    @property
    def index(self):
        """Lazy initialize the HNSW index."""
        if self._index is None:
            index_file = self.index_path.with_suffix(".bin")
            meta_file = self.index_path.with_suffix(".meta.json")

            # Try to load existing index first
            if index_file.exists() and meta_file.exists():
                try:
                    self._index = hnswlib.Index(
                        space="cosine",
                        dim=self.embedding_dim,
                    )
                    self._index.load_index(str(index_file))
                    self._index.set_ef(self.ef_search)
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                        self._id_mapping = {int(k): v for k, v in meta.get("id_mapping", {}).items()}
                        self._reverse_mapping = {int(k): v for k, v in meta.get("reverse_mapping", {}).items()}
                        self._next_id = meta.get("next_id", 0)
                except Exception as e:
                    import sys
                    print(f"Warning: Failed to load index: {e}", file=sys.stderr)
                    self._index = None

            # Create new index if not loaded
            if self._index is None:
                self._index = hnswlib.Index(
                    space="cosine",
                    dim=self.embedding_dim,
                )
                self._index.init_index(
                    max_elements=self.max_elements,
                    ef_construction=self.ef_construction,
                    M=self.m,
                )
                self._index.set_ef(self.ef_search)

        return self._index

    def add_vectors(
        self,
        chunk_ids: List[int],
        texts: Optional[List[str]] = None,
        vectors: Optional[np.ndarray] = None,
        num_threads: int = -1,
    ) -> None:
        """
        Add vectors to the index.

        Args:
            chunk_ids: List of chunk IDs
            texts: List of texts to embed (if vectors not provided)
            vectors: Pre-computed vectors (optional)
            num_threads: Number of threads for parallel index construction.
                        -1 means use all available cores. Default is -1.
        """
        if not chunk_ids:
            return

        if vectors is None:
            vectors = self.embedder.embed(texts)

        if len(vectors) == 0:
            return

        # Ensure vectors is 2D
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # Generate vector IDs
        start_id = self._next_id
        end_id = start_id + len(chunk_ids)
        vector_ids = list(range(start_id, end_id))

        # Add to index with multi-threading
        self.index.add_items(vectors, vector_ids, num_threads=num_threads)

        # Update mappings
        for vector_id, chunk_id in zip(vector_ids, chunk_ids):
            self._id_mapping[vector_id] = chunk_id
            self._reverse_mapping[chunk_id] = vector_id

        self._next_id = end_id

    def search(
        self,
        query: str,
        k: int = 10,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.

        Args:
            query: Query text
            k: Number of results
            query_vector: Pre-computed query vector (optional)

        Returns:
            List of (chunk_id, distance) tuples
        """
        # Handle empty index case
        if self.get_vector_count() == 0:
            return []

        if query_vector is None:
            query_vector = self.embedder.embed_single(query)

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        labels, distances = self.index.knn_query(query_vector, k=k)

        results = []
        for label, distance in zip(labels[0], distances[0]):
            chunk_id = self._id_mapping.get(label)
            if chunk_id is not None:
                results.append((chunk_id, float(distance)))

        return results

    def remove_by_chunk_id(self, chunk_id: int) -> bool:
        """
        Mark a vector as deleted (HNSW doesn't support true deletion).

        Args:
            chunk_id: Chunk ID to remove

        Returns:
            True if the chunk was found and marked
        """
        if chunk_id not in self._reverse_mapping:
            return False

        vector_id = self._reverse_mapping[chunk_id]
        self.index.mark_deleted(vector_id)
        del self._id_mapping[vector_id]
        del self._reverse_mapping[chunk_id]
        return True

    def remove_by_chunk_ids(self, chunk_ids: List[int]) -> int:
        """
        Remove multiple vectors by chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to remove

        Returns:
            Number of vectors successfully removed
        """
        removed = 0
        for chunk_id in chunk_ids:
            if self.remove_by_chunk_id(chunk_id):
                removed += 1
        return removed

    def save(self) -> None:
        """Save the index and metadata to disk."""
        self.index.save_index(str(self.index_path.with_suffix(".bin")))

        meta = {
            "id_mapping": self._id_mapping,
            "reverse_mapping": self._reverse_mapping,
            "next_id": self._next_id,
        }
        with open(self.index_path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)

    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._id_mapping = {}
        self._reverse_mapping = {}
        self._next_id = 0

        # Delete files
        index_file = self.index_path.with_suffix(".bin")
        meta_file = self.index_path.with_suffix(".meta.json")

        if index_file.exists():
            index_file.unlink()
        if meta_file.exists():
            meta_file.unlink()

    def get_vector_count(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._id_mapping)
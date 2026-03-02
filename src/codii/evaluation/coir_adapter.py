"""CoIR adapter for evaluating codii using the CoIR benchmark.

This module provides an adapter that wraps codii's hybrid search functionality
to work with the CoIR/MTEB evaluation framework for code retrieval tasks.
"""

import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..chunkers.text_chunker import CodeChunk
from ..indexers.bm25_indexer import BM25Indexer
from ..indexers.hybrid_search import HybridSearch
from ..indexers.vector_indexer import VectorIndexer
from ..storage.database import Database
from ..utils.config import CodiiConfig, get_config


@dataclass
class CorpusDocument:
    """A document in the corpus for indexing."""
    doc_id: str
    content: str
    language: str = "python"  # Default language
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A search result from the adapter."""
    doc_id: str
    score: float
    content: str


class CodiiCoIRAdapter:
    """Adapter to evaluate codii using CoIR benchmark.

    This adapter:
    1. Accepts a corpus from CoIR datasets
    2. Indexes the corpus using codii's indexing pipeline
    3. Executes searches using HybridSearch
    4. Returns results in CoIR's expected format

    The adapter creates temporary directories for indexing and cleans up
    after evaluation unless keep_index is set to True.

    Example:
        >>> adapter = CodiiCoIRAdapter()
        >>> adapter.index_corpus({"doc1": "def hello(): pass", "doc2": "class World: pass"})
        >>> results = adapter.search("hello function")
        >>> print(results[0].doc_id)
        'doc1'
    """

    def __init__(
        self,
        config: Optional[CodiiConfig] = None,
        cleanup_index: bool = True,
        rerank: bool = False,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rerank_threshold: Optional[float] = None,
    ):
        """Initialize the CoIR adapter.

        Args:
            config: CodiiConfig instance (uses global config if None)
            cleanup_index: Whether to delete temp index after use
            rerank: Enable/disable cross-encoder re-ranking
            bm25_weight: Weight for BM25 scores in hybrid search
            vector_weight: Weight for vector scores in hybrid search
            rerank_threshold: Override rerank threshold for benchmarking (None uses config default)
        """
        self.config = config or get_config()
        self.cleanup_index = cleanup_index
        self.rerank = rerank
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rerank_threshold = rerank_threshold

        self.temp_dir: Optional[Path] = None
        self.index_dir: Optional[Path] = None
        self.db_path: Optional[Path] = None
        self.vector_path: Optional[Path] = None

        # Mapping from chunk IDs to document IDs
        self._chunk_to_doc: Dict[int, str] = {}
        self._doc_to_chunks: Dict[str, List[int]] = {}

        # Components
        self._db: Optional[Database] = None
        self._bm25_indexer: Optional[BM25Indexer] = None
        self._vector_indexer: Optional[VectorIndexer] = None
        self._hybrid_search: Optional[HybridSearch] = None

    def _setup_temp_dirs(self) -> None:
        """Set up temporary directories for indexing."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="codii_coir_"))
            self.index_dir = self.temp_dir / "index"
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.index_dir / "chunks.db"
            self.vector_path = self.index_dir / "vectors"

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self.cleanup_index and self.temp_dir and self.temp_dir.exists():
            # Close connections first
            if self._hybrid_search:
                self._hybrid_search.close()
            if self._bm25_indexer:
                self._bm25_indexer.close()
            if self._db:
                self._db.close()

            # Clean up singletons to allow fresh instances
            import codii.embedding.embedder as embedder_module
            import codii.embedding.cross_encoder as cross_encoder_module
            embedder_module.Embedder._instance = None
            cross_encoder_module.CrossEncoderWrapper._instance = None

            # Remove temp directory
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
            self.index_dir = None
            self.db_path = None
            self.vector_path = None

    def _detect_language(self, content: str, doc_id: str = "") -> str:
        """Detect language from content or doc_id.

        Args:
            content: The document content
            doc_id: Document ID (may contain extension hints)

        Returns:
            Detected language string
        """
        # Try to detect from doc_id extension
        if doc_id:
            ext_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".jsx": "javascript",
                ".go": "go",
                ".rs": "rust",
                ".java": "java",
                ".c": "c",
                ".cpp": "cpp",
                ".cc": "cpp",
                ".cxx": "cpp",
                ".h": "c",
                ".hpp": "cpp",
            }
            for ext, lang in ext_map.items():
                if doc_id.endswith(ext):
                    return lang

        # Try to detect from content
        # Python: def, import, class keywords
        if ("def " in content or "class " in content) and ("import " in content or "from " in content or ":" in content):
            return "python"
        # Python also: just def or class with Python-style syntax
        if ("def " in content and ":" in content) or ("class " in content and ":" in content):
            return "python"
        # JavaScript: function, const, let keywords
        if "function " in content and ("{" in content or "=>" in content):
            return "javascript"
        # TypeScript: similar to JS but with type annotations
        if ("function " in content or "const " in content) and (": " in content or ":{" in content):
            return "typescript"
        # Go: func, package keywords
        if "func " in content and "package " in content:
            return "go"
        # Rust: fn, let keywords
        if "fn " in content and ("let " in content or "impl " in content):
            return "rust"
        # Java: public class, private keywords
        if "public class " in content or "private " in content or "public static void main" in content:
            return "java"
        # C/C++: #include, int main
        if "#include" in content or "int main(" in content:
            return "c"

        return "text"

    def _chunk_document(self, doc_id: str, content: str, language: str) -> List[CodeChunk]:
        """Chunk a document into smaller pieces.

        Uses simple text chunking for benchmark documents to ensure
        all content is indexed.

        Args:
            doc_id: Document ID
            content: Document content
            language: Detected language

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        if not content or not content.strip():
            return chunks

        # For benchmark documents, use simple chunking
        max_chunk_size = self.config.max_chunk_size
        min_chunk_size = self.config.min_chunk_size

        lines = content.split("\n")
        current_chunk_lines = []
        current_start_line = 1
        current_size = 0

        for i, line in enumerate(lines, start=1):
            line_size = len(line) + 1

            if current_size + line_size > max_chunk_size and current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines)
                if len(chunk_content) >= min_chunk_size:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        path=doc_id,
                        start_line=current_start_line,
                        end_line=i - 1,
                        language=language,
                        chunk_type="text_block",
                    ))

                current_chunk_lines = []
                current_start_line = i
                current_size = 0

            current_chunk_lines.append(line)
            current_size += line_size

        # Last chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            if len(chunk_content) >= min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    path=doc_id,
                    start_line=current_start_line,
                    end_line=len(lines),
                    language=language,
                    chunk_type="text_block",
                ))

        # If no chunks were created but content exists, create one chunk
        if not chunks and content.strip():
            chunks.append(CodeChunk(
                content=content,
                path=doc_id,
                start_line=1,
                end_line=len(lines),
                language=language,
                chunk_type="text_block",
            ))

        return chunks

    def index_corpus(self, corpus: Dict[str, str]) -> None:
        """Index corpus documents for evaluation.

        Args:
            corpus: Dictionary mapping doc_id to document content
        """
        self._setup_temp_dirs()

        # Clear any existing mappings
        self._chunk_to_doc.clear()
        self._doc_to_chunks.clear()

        # Initialize database
        self._db = Database(self.db_path)
        self._bm25_indexer = BM25Indexer(self.db_path)

        # Initialize vector indexer with config settings
        self._vector_indexer = VectorIndexer(
            self.vector_path,
            embedding_dim=self.config.embedding_model == "all-MiniLM-L6-v2" and 384 or 768,
            m=self.config.hnsw_m,
            ef_construction=self.config.hnsw_ef_construction,
            ef_search=self.config.hnsw_ef_search,
        )

        # Process all documents
        all_chunks: List[CodeChunk] = []

        for doc_id, content in corpus.items():
            language = self._detect_language(content, doc_id)
            chunks = self._chunk_document(doc_id, content, language)

            # Track chunk-to-doc mapping
            start_idx = len(all_chunks)
            for i, chunk in enumerate(chunks):
                self._doc_to_chunks.setdefault(doc_id, []).append(start_idx + i)

            all_chunks.extend(chunks)

        if not all_chunks:
            return

        # Add chunks to BM25 index
        self._bm25_indexer.add_chunks(all_chunks)

        # Get chunk IDs (they are assigned sequentially)
        chunk_ids = self._bm25_indexer.get_all_chunk_ids()
        new_chunk_ids = chunk_ids[-len(all_chunks):]

        # Update mapping with actual chunk IDs
        for doc_id, indices in self._doc_to_chunks.items():
            self._doc_to_chunks[doc_id] = [new_chunk_ids[i] for i in indices]
            for chunk_id in self._doc_to_chunks[doc_id]:
                self._chunk_to_doc[chunk_id] = doc_id

        # Embed and index vectors
        texts = [chunk.content for chunk in all_chunks]
        vectors = self._vector_indexer.embedder.embed(texts)

        # Add vectors with multi-threading
        import os
        num_threads = os.cpu_count() or 4
        self._vector_indexer.add_vectors(new_chunk_ids, vectors=vectors, num_threads=num_threads)
        self._vector_indexer.save()

        # Initialize hybrid search
        self._hybrid_search = HybridSearch(
            self.db_path,
            self.vector_path,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
        )

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search and return results with document IDs.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of SearchResult objects with doc_id, score, and content
        """
        if not self._hybrid_search:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")

        # Use hybrid search with re-ranking
        results = self._hybrid_search.search(
            query,
            limit=top_k,
            rerank=self.rerank,
            rerank_threshold=self.rerank_threshold,
        )

        # Convert to SearchResult with doc_id
        search_results = []
        seen_docs = set()

        for result in results:
            doc_id = self._chunk_to_doc.get(result.id)
            if doc_id and doc_id not in seen_docs:
                # Use rerank_score if available, otherwise combined_score
                score = result.rerank_score if result.rerank_score > 0 else result.combined_score
                search_results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    content=result.content,
                ))
                seen_docs.add(doc_id)

        return search_results

    def get_scores(self, query: str, doc_ids: List[str]) -> List[float]:
        """Get relevance scores for specific documents.

        Args:
            query: Search query
            doc_ids: List of document IDs to score

        Returns:
            List of scores corresponding to doc_ids
        """
        if not self._hybrid_search:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")

        # Get all chunks for the requested documents
        chunk_ids = []
        for doc_id in doc_ids:
            chunk_ids.extend(self._doc_to_chunks.get(doc_id, []))

        if not chunk_ids:
            return [0.0] * len(doc_ids)

        # Search with high limit to get more candidates
        results = self._hybrid_search.search(
            query,
            limit=len(chunk_ids) * 2,
            rerank=self.rerank,
            rerank_threshold=self.rerank_threshold,
        )

        # Build score map from chunk_id to score
        chunk_scores = {}
        for result in results:
            score = result.rerank_score if result.rerank_score > 0 else result.combined_score
            chunk_scores[result.id] = score

        # Aggregate scores per document (max score across chunks)
        doc_scores = {}
        for doc_id in doc_ids:
            chunks = self._doc_to_chunks.get(doc_id, [])
            if chunks:
                scores = [chunk_scores.get(cid, 0.0) for cid in chunks]
                doc_scores[doc_id] = max(scores) if scores else 0.0
            else:
                doc_scores[doc_id] = 0.0

        return [doc_scores.get(doc_id, 0.0) for doc_id in doc_ids]

    def get_doc_count(self) -> int:
        """Get the number of indexed documents."""
        return len(self._doc_to_chunks)

    def get_chunk_count(self) -> int:
        """Get the total number of chunks."""
        return len(self._chunk_to_doc)

    def __del__(self):
        """Clean up on destruction."""
        self._cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
        return False
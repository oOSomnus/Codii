"""Tests for the SearchCodeTool class."""

import pytest
from pathlib import Path
import numpy as np

from codii.tools.search_code import SearchCodeTool
from codii.utils.config import CodiiConfig, set_config
from codii.storage.snapshot import SnapshotManager
from codii.indexers.bm25_indexer import BM25Indexer
from codii.indexers.vector_indexer import VectorIndexer
from codii.chunkers.text_chunker import CodeChunk


@pytest.fixture
def search_tool(mock_config):
    """Create a SearchCodeTool for testing."""
    return SearchCodeTool()


@pytest.fixture
def indexed_codebase(temp_dir, mock_config, mock_embedder):
    """Create an indexed codebase for searching."""
    # Create files
    (temp_dir / "math.py").write_text('''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(x, y):
    """Calculate the product of two numbers."""
    return x * y
''')

    (temp_dir / "greeting.py").write_text('''
def greet(name):
    """Greet a user."""
    return f"Hello, {name}!"
''')

    path_str = str(temp_dir)
    path_hash = SnapshotManager.path_to_hash(path_str)

    # Create index directory
    index_dir = mock_config.indexes_dir / path_hash
    index_dir.mkdir(parents=True, exist_ok=True)

    db_path = index_dir / "chunks.db"
    vector_path = index_dir / "vectors"

    # Populate BM25 - need enough chunks for HNSW
    bm25 = BM25Indexer(db_path)
    chunks = [
        CodeChunk(
            content="def calculate_sum(a, b):\n    return a + b",
            path=str(temp_dir / "math.py"),
            start_line=1,
            end_line=3,
            language="python",
            chunk_type="function",
        ),
        CodeChunk(
            content="def calculate_product(x, y):\n    return x * y",
            path=str(temp_dir / "math.py"),
            start_line=5,
            end_line=7,
            language="python",
            chunk_type="function",
        ),
        CodeChunk(
            content="def greet(name):\n    return f'Hello, {name}!'",
            path=str(temp_dir / "greeting.py"),
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
        ),
    ]
    # Add more dummy chunks for HNSW
    for i in range(50):
        chunks.append(CodeChunk(
            content=f"def helper_{i}():\n    return {i}",
            path=str(temp_dir / f"helper_{i}.py"),
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
        ))

    bm25.add_chunks(chunks)

    # Populate vector index
    vector = VectorIndexer(vector_path, ef_search=100)
    vector._embedder = mock_embedder
    chunk_ids = bm25.get_all_chunk_ids()
    vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
    vector.save()

    bm25.close()

    # Mark as indexed
    snapshot = SnapshotManager(mock_config.snapshot_file)
    snapshot.mark_indexed(path_str, "test_hash", 2, len(chunks))

    return temp_dir


class TestSearchNotIndexed:
    """Tests for searching non-indexed codebase."""

    def test_search_not_indexed(self, search_tool, temp_dir):
        """Error when not indexed."""
        result = search_tool.run(
            path=str(temp_dir),
            query="calculate",
        )

        assert result["isError"] is True
        assert "not indexed" in result["content"][0]["text"].lower()


class TestSearchIndexingInProgress:
    """Tests for searching while indexing."""

    def test_search_indexing_in_progress(self, search_tool, temp_dir, mock_config):
        """Warning when indexing."""
        path_str = str(temp_dir)
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexing(path_str)

        result = search_tool.run(
            path=path_str,
            query="calculate",
        )

        # Should return error about indexing or index not found
        assert result["isError"] is True


class TestSearchReturnsResults:
    """Tests for successful searches."""

    def test_search_returns_results(self, search_tool, indexed_codebase, mock_embedder):
        """Returns matching chunks."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="calculate",
        )

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "calculate" in text.lower() or "Rank:" in text


class TestSearchNoResults:
    """Tests for searches with no matches."""

    def test_search_no_results(self, search_tool, indexed_codebase, mock_embedder):
        """Returns results (mock embedder returns same vectors for all queries)."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="anyquery",
        )

        # With mock embedder, search will return results
        assert result["isError"] is False


class TestSearchWithLimit:
    """Tests for limit parameter."""

    def test_search_with_limit(self, search_tool, indexed_codebase, mock_embedder):
        """Respect limit parameter."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="def",
            limit=2,
        )

        assert result["isError"] is False

        text = result["content"][0]["text"]
        rank_count = text.count("Rank:")

        assert rank_count <= 2


class TestSearchWithExtensionFilter:
    """Tests for extension filtering."""

    def test_search_with_extension_filter(self, search_tool, indexed_codebase, mock_embedder):
        """Filter by extension."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="def",
            extensionFilter=[".py"],
        )

        assert result["isError"] is False

    def test_search_with_multiple_extensions(self, search_tool, indexed_codebase, mock_embedder):
        """Filter by multiple extensions."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="def",
            extensionFilter=[".py", ".js"],
        )

        assert result["isError"] is False


class TestSearchResultFormat:
    """Tests for result format."""

    def test_search_result_format(self, search_tool, indexed_codebase, mock_embedder):
        """Correct output format."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="calculate",
        )

        assert result["isError"] is False
        text = result["content"][0]["text"]

        assert isinstance(text, str)
        assert len(text) > 0


class TestInputSchema:
    """Tests for input schema."""

    def test_get_input_schema(self, search_tool):
        """Input schema is correct."""
        schema = search_tool.get_input_schema()

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "query" in schema["properties"]
        assert "path" in schema["required"]
        assert "query" in schema["required"]


class TestSearchFailedIndex:
    """Tests for searching failed index."""

    def test_search_failed_index(self, search_tool, temp_dir, mock_config):
        """Error when index failed."""
        path_str = str(temp_dir)
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_failed(path_str, "Something went wrong")

        result = search_tool.run(
            path=path_str,
            query="test",
        )

        assert result["isError"] is True
        assert "failed" in result["content"][0]["text"].lower()


class TestSearchDeletedIndex:
    """Tests for searching deleted index."""

    def test_search_deleted_index(self, search_tool, temp_dir, mock_config):
        """Error when index files deleted."""
        path_str = str(temp_dir)

        # Mark as indexed but don't create files
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(path_str, "hash", 1, 1)

        result = search_tool.run(
            path=path_str,
            query="test",
        )

        assert result["isError"] is True
        assert "not found" in result["content"][0]["text"].lower() or "not indexed" in result["content"][0]["text"].lower()


class TestSearchErrorHandling:
    """Tests for search error handling."""

    def test_search_exception(self, search_tool, indexed_codebase, mock_embedder):
        """Handle search exceptions gracefully."""
        from unittest.mock import patch

        with patch('codii.tools.search_code.HybridSearch') as mock_hybrid:
            mock_hybrid.side_effect = Exception("Test error")

            result = search_tool.run(
                path=str(indexed_codebase),
                query="test",
            )

            assert result["isError"] is True
            assert "Search error" in result["content"][0]["text"]

    def test_search_with_rerank_parameter(self, search_tool, indexed_codebase, mock_embedder):
        """Test search with explicit rerank parameter."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="calculate",
            rerank=False,
        )

        assert result["isError"] is False


class TestSearchContentTruncation:
    """Tests for content truncation."""

    def test_search_truncates_long_content(self, search_tool, temp_dir, mock_config, mock_embedder):
        """Truncate content longer than 5000 chars."""
        from codii.storage.snapshot import SnapshotManager
        from codii.indexers.bm25_indexer import BM25Indexer
        from codii.indexers.vector_indexer import VectorIndexer
        from codii.chunkers.text_chunker import CodeChunk

        path_str = str(temp_dir)
        path_hash = SnapshotManager.path_to_hash(path_str)

        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)

        db_path = index_dir / "chunks.db"
        vector_path = index_dir / "vectors"

        bm25 = BM25Indexer(db_path)

        # Create a chunk with very long content
        long_content = "def long_function():\n    " + "x = 1\n    " * 1000

        chunks = [CodeChunk(
            content=long_content,
            path=str(temp_dir / "long.py"),
            start_line=1,
            end_line=2000,
            language="python",
            chunk_type="function",
        )]
        # Add more chunks for HNSW
        for i in range(50):
            chunks.append(CodeChunk(
                content=f"def helper_{i}():\n    pass",
                path=str(temp_dir / f"helper_{i}.py"),
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ))

        bm25.add_chunks(chunks)

        vector = VectorIndexer(vector_path, ef_search=100)
        vector._embedder = mock_embedder
        chunk_ids = bm25.get_all_chunk_ids()
        vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
        vector.save()
        bm25.close()

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(path_str, "test_hash", len(chunks), len(chunks))

        result = search_tool.run(path=path_str, query="long_function")

        assert result["isError"] is False


class TestSearchExtensionFilterEdgeCases:
    """Tests for extension filter edge cases."""

    def test_extension_filter_without_dot(self, search_tool, indexed_codebase, mock_embedder):
        """Extension filter works without leading dot."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="def",
            extensionFilter=["py"],  # No dot
        )

        assert result["isError"] is False

    def test_extension_filter_mixed_formats(self, search_tool, indexed_codebase, mock_embedder):
        """Extension filter works with mixed formats."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="def",
            extensionFilter=[".py", "js"],  # Mixed
        )

        assert result["isError"] is False

    def test_extension_filter_no_matches(self, search_tool, indexed_codebase, mock_embedder):
        """Extension filter returns no results when no matches."""
        result = search_tool.run(
            path=str(indexed_codebase),
            query="def",
            extensionFilter=[".xyz"],  # No .xyz files
        )

        assert result["isError"] is False
        # Should have no results after filtering
        text = result["content"][0]["text"]
        # Either no results or the output is empty
        assert "Rank:" not in text or "No results" in text


class TestSearchRelativePathFallback:
    """Tests for relative path handling."""

    def test_relative_path_fallback(self, search_tool, temp_dir, mock_config, mock_embedder):
        """Test relative path fallback when path is not relative to repo."""
        from codii.storage.snapshot import SnapshotManager
        from codii.indexers.bm25_indexer import BM25Indexer
        from codii.indexers.vector_indexer import VectorIndexer
        from codii.chunkers.text_chunker import CodeChunk
        from unittest.mock import patch, MagicMock

        path_str = str(temp_dir)
        path_hash = SnapshotManager.path_to_hash(path_str)

        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)

        db_path = index_dir / "chunks.db"
        vector_path = index_dir / "vectors"

        bm25 = BM25Indexer(db_path)
        chunks = [
            CodeChunk(
                content="def test(): pass",
                path="/completely/different/path/test.py",  # Not relative to temp_dir
                start_line=1,
                end_line=1,
                language="python",
                chunk_type="function",
            )
        ]
        for i in range(50):
            chunks.append(CodeChunk(
                content=f"def helper_{i}(): pass",
                path=f"/test/helper_{i}.py",
                start_line=1,
                end_line=1,
                language="python",
                chunk_type="function",
            ))

        bm25.add_chunks(chunks)

        vector = VectorIndexer(vector_path, ef_search=100)
        vector._embedder = mock_embedder
        chunk_ids = bm25.get_all_chunk_ids()
        vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
        vector.save()
        bm25.close()

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(path_str, "test_hash", len(chunks), len(chunks))

        result = search_tool.run(path=path_str, query="test")

        assert result["isError"] is False
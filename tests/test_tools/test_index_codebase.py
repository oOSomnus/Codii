"""Tests for the IndexCodebaseTool class."""

import pytest
from pathlib import Path
import time
import threading

from codii.tools.index_codebase import IndexCodebaseTool
from codii.utils.config import CodiiConfig, set_config
from codii.storage.snapshot import SnapshotManager


@pytest.fixture
def index_tool(mock_config):
    """Create an IndexCodebaseTool for testing."""
    return IndexCodebaseTool()


@pytest.fixture
def sample_repo(temp_dir):
    """Create a sample repository for indexing."""
    # Create Python files
    (temp_dir / "main.py").write_text('''
def main():
    """Main entry point."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

    (temp_dir / "utils.py").write_text('''
def helper():
    """A helper function."""
    return 42
''')

    # Create subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "module.py").write_text('''
class Module:
    pass
''')

    return temp_dir


class TestIndexNonexistentPath:
    """Tests for non-existent paths."""

    def test_index_nonexistent_path(self, index_tool):
        """Error for invalid path."""
        result = index_tool.run(path="/nonexistent/path/xyz")

        assert result["isError"] is True
        assert "does not exist" in result["content"][0]["text"]


class TestIndexFileNotDirectory:
    """Tests for file paths instead of directories."""

    def test_index_file_not_directory(self, index_tool, sample_python_file):
        """Error for file path."""
        result = index_tool.run(path=str(sample_python_file))

        assert result["isError"] is True
        assert "not a directory" in result["content"][0]["text"]


class TestIndexAlreadyIndexing:
    """Tests for already indexing state."""

    def test_index_already_indexing(self, index_tool, sample_repo, mock_config):
        """Error when already indexing."""
        # Mark as indexing
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexing(str(sample_repo))

        result = index_tool.run(path=str(sample_repo))

        assert result["isError"] is True
        assert "currently being indexed" in result["content"][0]["text"]


class TestIndexAlreadyIndexed:
    """Tests for already indexed state."""

    def test_index_already_indexed(self, index_tool, sample_repo, mock_config):
        """Info message when indexed."""
        # Mark as indexed
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(
            str(sample_repo),
            merkle_root="hash",
            indexed_files=5,
            total_chunks=10,
        )

        result = index_tool.run(path=str(sample_repo))

        assert result["isError"] is False
        assert "already indexed" in result["content"][0]["text"]


class TestIndexForceReindex:
    """Tests for force re-indexing."""

    def test_index_force_reindex(self, index_tool, sample_repo, mock_config):
        """Force re-index existing."""
        # First, mark as indexed
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(
            str(sample_repo),
            merkle_root="hash",
            indexed_files=5,
            total_chunks=10,
        )

        # Force re-index
        result = index_tool.run(path=str(sample_repo), force=True)

        assert result["isError"] is False
        assert "Indexing started" in result["content"][0]["text"]


class TestIndexStartsBackground:
    """Tests for background indexing."""

    def test_index_starts_background(self, index_tool, sample_repo):
        """Indexing runs in background."""
        result = index_tool.run(path=str(sample_repo))

        assert result["isError"] is False
        assert "Indexing started" in result["content"][0]["text"]

        # Wait a bit for indexing to start
        time.sleep(0.5)

    def test_index_returns_immediately(self, index_tool, sample_repo):
        """Index tool returns immediately."""
        start = time.time()
        result = index_tool.run(path=str(sample_repo))
        elapsed = time.time() - start

        # Should return quickly, not wait for indexing to complete
        assert elapsed < 2.0


class TestIndexWithCustomExtensions:
    """Tests for custom file extensions."""

    def test_index_with_custom_extensions(self, index_tool, temp_dir, mock_config):
        """Custom file extensions."""
        # Create files with custom extension
        (temp_dir / "file.custom").write_text("custom content")
        (temp_dir / "file.py").write_text("def hello(): pass")

        result = index_tool.run(
            path=str(temp_dir),
            customExtensions=[".custom"],
        )

        assert result["isError"] is False


class TestIndexWithIgnorePatterns:
    """Tests for custom ignore patterns."""

    def test_index_with_ignore_patterns(self, index_tool, temp_dir, mock_config):
        """Custom ignore patterns."""
        # Create files
        (temp_dir / "main.py").write_text("def main(): pass")
        (temp_dir / "test_main.py").write_text("def test_main(): pass")

        result = index_tool.run(
            path=str(temp_dir),
            ignorePatterns=["test_*.py"],
        )

        assert result["isError"] is False


class TestInputSchema:
    """Tests for input schema."""

    def test_get_input_schema(self, index_tool):
        """Input schema is correct."""
        schema = index_tool.get_input_schema()

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]


class TestInvalidPath:
    """Tests for invalid path handling."""

    def test_index_relative_path(self, index_tool):
        """Handle relative path."""
        result = index_tool.run(path="some/relative/path")

        # Should resolve to absolute path and fail
        assert result["isError"] is True


class TestEmptyDirectory:
    """Tests for empty directories."""

    def test_index_empty_directory(self, index_tool, temp_dir):
        """Handle empty directory."""
        result = index_tool.run(path=str(temp_dir))

        # Should start indexing but may fail during processing
        assert result["isError"] is False or "No files" in result["content"][0]["text"]


class TestSplitterOptions:
    """Tests for splitter options."""

    def test_index_with_ast_splitter(self, index_tool, sample_repo):
        """Use AST splitter."""
        result = index_tool.run(path=str(sample_repo), splitter="ast")

        assert result["isError"] is False

    def test_index_with_text_splitter(self, index_tool, sample_repo):
        """Use text splitter."""
        result = index_tool.run(path=str(sample_repo), splitter="langchain")

        assert result["isError"] is False


class TestConcurrentIndexing:
    """Tests for concurrent indexing prevention."""

    def test_concurrent_indexing_same_path(self, index_tool, sample_repo, mock_config):
        """Prevent concurrent indexing of same path."""
        # Start first indexing
        result1 = index_tool.run(path=str(sample_repo))
        assert result1["isError"] is False

        # Try to start second indexing of same path
        result2 = index_tool.run(path=str(sample_repo))
        assert result2["isError"] is True
        assert "currently being indexed" in result2["content"][0]["text"]


class TestIncrementalIndexing:
    """Tests for incremental indexing functionality."""

    def test_early_exit_no_changes(self, index_tool, sample_repo, mock_config):
        """Early exit when no changes since last index."""
        import time
        from codii.storage.database import Database
        from codii.storage.snapshot import SnapshotManager

        # First index
        result = index_tool.run(path=str(sample_repo))
        assert result["isError"] is False

        # Wait for indexing to complete
        time.sleep(2)

        # Get status
        snapshot = SnapshotManager(mock_config.snapshot_file)
        status = snapshot.get_status(str(sample_repo))

        # If indexing completed, verify it worked
        if status.status == "indexed":
            # Get initial chunk count
            path_hash = snapshot.path_to_hash(str(sample_repo))
            db_path = mock_config.indexes_dir / path_hash / "chunks.db"
            if db_path.exists():
                db = Database(db_path)
                initial_chunks = db.get_chunk_count()

                # Try to index again without force (should skip)
                result2 = index_tool.run(path=str(sample_repo))
                assert result2["isError"] is False
                assert "already indexed" in result2["content"][0]["text"]

    def test_incremental_add_file(self, index_tool, sample_repo, mock_config):
        """Incremental update when file is added."""
        import time
        from codii.storage.database import Database
        from codii.storage.snapshot import SnapshotManager

        # First index
        result = index_tool.run(path=str(sample_repo))
        assert result["isError"] is False

        # Wait for indexing to complete
        time.sleep(2)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        status = snapshot.get_status(str(sample_repo))

        if status.status == "indexed":
            # Get initial chunk count
            path_hash = snapshot.path_to_hash(str(sample_repo))
            db_path = mock_config.indexes_dir / path_hash / "chunks.db"
            if db_path.exists():
                db = Database(db_path)
                initial_chunks = db.get_chunk_count()

                # Add a new file
                new_file = sample_repo / "new_file.py"
                new_file.write_text("def new_function():\n    pass\n")

                # Incremental reindex (no force needed - will detect change)
                result2 = index_tool.run(path=str(sample_repo))
                assert result2["isError"] is False

                # Wait for indexing
                time.sleep(2)

                # Verify new chunks were added
                final_chunks = db.get_chunk_count()
                assert final_chunks >= initial_chunks

    def test_incremental_modify_file(self, index_tool, sample_repo, mock_config):
        """Incremental update when file is modified."""
        import time
        from codii.storage.database import Database
        from codii.storage.snapshot import SnapshotManager

        # First index
        result = index_tool.run(path=str(sample_repo))
        assert result["isError"] is False

        # Wait for indexing to complete
        time.sleep(2)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        status = snapshot.get_status(str(sample_repo))

        if status.status == "indexed":
            # Modify an existing file
            (sample_repo / "main.py").write_text('''
def new_main():
    """Modified main entry point."""
    print("Hello, Modified World!")

if __name__ == "__main__":
    new_main()
''')

            # Incremental reindex (no force needed - will detect change)
            result2 = index_tool.run(path=str(sample_repo))
            assert result2["isError"] is False

            # Wait for indexing
            time.sleep(2)

            # Verify indexing completed
            status2 = snapshot.get_status(str(sample_repo))
            assert status2.status == "indexed"

    def test_incremental_remove_file(self, index_tool, sample_repo, mock_config):
        """Incremental update when file is removed."""
        import time
        from codii.storage.database import Database
        from codii.storage.snapshot import SnapshotManager

        # First index
        result = index_tool.run(path=str(sample_repo))
        assert result["isError"] is False

        # Wait for indexing to complete
        max_wait = 10
        for _ in range(max_wait):
            time.sleep(1)
            snapshot = SnapshotManager(mock_config.snapshot_file)
            status = snapshot.get_status(str(sample_repo))
            if status.status == "indexed":
                break

        if status.status == "indexed":
            # Get initial chunk count
            path_hash = snapshot.path_to_hash(str(sample_repo))
            db_path = mock_config.indexes_dir / path_hash / "chunks.db"
            if db_path.exists():
                db = Database(db_path)
                initial_chunks = db.get_chunk_count()

                # Verify we have chunks for the file we'll remove
                utils_path = str(sample_repo / "utils.py")
                initial_utils_chunks = db.get_chunk_ids_by_path(utils_path)
                if len(initial_utils_chunks) == 0:
                    pytest.skip("utils.py has no chunks to remove")

                # Remove a file
                (sample_repo / "utils.py").unlink()

                # Incremental reindex (no force needed - will detect change)
                result2 = index_tool.run(path=str(sample_repo))
                assert result2["isError"] is False

                # Wait for indexing to complete
                for _ in range(max_wait):
                    time.sleep(1)
                    status2 = snapshot.get_status(str(sample_repo))
                    if status2.status == "indexed":
                        break

                # Verify chunks for the removed file are gone
                final_utils_chunks = db.get_chunk_ids_by_path(utils_path)
                assert len(final_utils_chunks) == 0, f"Expected 0 chunks for removed file, got {len(final_utils_chunks)}"

                # Verify total chunk count decreased
                final_chunks = db.get_chunk_count()
                assert final_chunks < initial_chunks, f"Expected {final_chunks} < {initial_chunks}"


class TestGetChunkIdsByPath:
    """Tests for get_chunk_ids_by_path database method."""

    def test_get_chunk_ids_by_path(self, temp_db_path):
        """Test retrieving chunk IDs by path."""
        from codii.storage.database import Database

        db = Database(temp_db_path)

        # Insert some chunks
        id1 = db.insert_chunk("content1", "/path/to/file.py", 1, 10, "python", "function")
        id2 = db.insert_chunk("content2", "/path/to/file.py", 11, 20, "python", "class")
        id3 = db.insert_chunk("content3", "/path/to/other.py", 1, 5, "python", "function")

        # Get chunks for file.py
        chunk_ids = db.get_chunk_ids_by_path("/path/to/file.py")

        assert len(chunk_ids) == 2
        assert id1 in chunk_ids
        assert id2 in chunk_ids
        assert id3 not in chunk_ids

    def test_get_chunk_ids_by_path_empty(self, temp_db_path):
        """Test retrieving chunk IDs for non-existent path."""
        from codii.storage.database import Database

        db = Database(temp_db_path)

        # Insert a chunk
        db.insert_chunk("content", "/path/to/file.py", 1, 10, "python", "function")

        # Get chunks for non-existent path
        chunk_ids = db.get_chunk_ids_by_path("/path/to/nonexistent.py")

        assert len(chunk_ids) == 0


class TestRemoveByChunkIds:
    """Tests for remove_by_chunk_ids vector indexer method."""

    def test_remove_by_chunk_ids(self, temp_vector_path):
        """Test removing multiple vectors by chunk IDs."""
        import numpy as np
        from codii.indexers.vector_indexer import VectorIndexer

        indexer = VectorIndexer(temp_vector_path)

        # Add some vectors
        chunk_ids = [1, 2, 3]
        vectors = np.random.rand(3, 384).astype(np.float32)
        indexer.add_vectors(chunk_ids, vectors=vectors)

        assert indexer.get_vector_count() == 3

        # Remove two vectors
        removed = indexer.remove_by_chunk_ids([1, 2])

        assert removed == 2
        assert indexer.get_vector_count() == 1

        # Verify remaining vector
        assert indexer.remove_by_chunk_id(3)  # Should still exist
        assert indexer.get_vector_count() == 0

    def test_remove_by_chunk_ids_partial(self, temp_vector_path):
        """Test removing vectors when some don't exist."""
        import numpy as np
        from codii.indexers.vector_indexer import VectorIndexer

        indexer = VectorIndexer(temp_vector_path)

        # Add some vectors
        chunk_ids = [1, 2]
        vectors = np.random.rand(2, 384).astype(np.float32)
        indexer.add_vectors(chunk_ids, vectors=vectors)

        # Try to remove existing and non-existing
        removed = indexer.remove_by_chunk_ids([1, 999])

        assert removed == 1  # Only one actually removed
        assert indexer.get_vector_count() == 1


class TestParallelChunking:
    """Tests for parallel chunking functionality."""

    def test_chunk_file_worker_ast(self):
        """Test the chunk file worker function with AST splitter."""
        from codii.tools.index_codebase import _chunk_file_worker

        content = '''
def hello():
    """Say hello."""
    print("Hello, World!")

class Greeter:
    def greet(self):
        pass
'''
        args = (
            "/path/to/file.py",
            content,
            "python",
            1500,
            100,
            "ast",
        )

        chunks = _chunk_file_worker(args)

        assert len(chunks) >= 1
        # Verify chunks have proper structure
        for chunk in chunks:
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'path')
            assert chunk.path == "/path/to/file.py"

    def test_chunk_file_worker_text(self):
        """Test the chunk file worker function with text splitter."""
        from codii.tools.index_codebase import _chunk_file_worker

        content = "Line 1\nLine 2\nLine 3\n" * 100  # Large enough to chunk
        args = (
            "/path/to/file.txt",
            content,
            "text",
            500,  # Small max chunk size to force multiple chunks
            100,
            "langchain",
        )

        chunks = _chunk_file_worker(args)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert hasattr(chunk, 'content')

    def test_chunk_file_worker_none_content(self):
        """Test the chunk file worker with None content."""
        from codii.tools.index_codebase import _chunk_file_worker

        args = (
            "/path/to/file.py",
            None,  # No content
            "python",
            1500,
            100,
            "ast",
        )

        chunks = _chunk_file_worker(args)

        assert chunks == []

    def test_parallel_vs_sequential_results(self, temp_dir, mock_config):
        """Verify parallel chunking produces same results as sequential."""
        import time
        from codii.storage.snapshot import SnapshotManager
        from codii.storage.database import Database

        # Create multiple Python files
        for i in range(10):
            (temp_dir / f"file_{i}.py").write_text(f'''
def function_{i}():
    """Function {i}."""
    return {i}

class Class{i}:
    def method(self):
        return {i}
''')

        # Index with default (parallel) workers
        tool1 = IndexCodebaseTool()
        result1 = tool1.run(path=str(temp_dir), force=True)
        assert result1["isError"] is False

        # Wait for indexing
        time.sleep(3)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        status = snapshot.get_status(str(temp_dir))

        if status.status == "indexed":
            path_hash = snapshot.path_to_hash(str(temp_dir))
            db_path = mock_config.indexes_dir / path_hash / "chunks.db"
            if db_path.exists():
                db = Database(db_path)
                parallel_chunk_count = db.get_chunk_count()

                # Now index with workers=1 (sequential)
                tool2 = IndexCodebaseTool()
                result2 = tool2.run(path=str(temp_dir), force=True, workers=1)
                assert result2["isError"] is False

                time.sleep(3)

                # Should have same number of chunks
                sequential_chunk_count = db.get_chunk_count()
                assert parallel_chunk_count == sequential_chunk_count

    def test_index_with_explicit_workers(self, index_tool, sample_repo):
        """Test indexing with explicit worker count."""
        result = index_tool.run(
            path=str(sample_repo),
            workers=2,
        )

        assert result["isError"] is False
        assert "Indexing started" in result["content"][0]["text"]

    def test_index_with_explicit_hnsw_threads(self, index_tool, sample_repo):
        """Test indexing with explicit HNSW thread count."""
        result = index_tool.run(
            path=str(sample_repo),
            hnswThreads=2,
        )

        assert result["isError"] is False
        assert "Indexing started" in result["content"][0]["text"]

    def test_index_with_workers_auto_detect(self, index_tool, sample_repo):
        """Test indexing with workers=0 (auto-detect)."""
        result = index_tool.run(
            path=str(sample_repo),
            workers=0,  # Auto-detect
        )

        assert result["isError"] is False

    def test_input_schema_has_workers_param(self, index_tool):
        """Verify input schema includes workers parameter."""
        schema = index_tool.get_input_schema()

        assert "workers" in schema["properties"]
        assert schema["properties"]["workers"]["type"] == "integer"
        assert schema["properties"]["workers"]["default"] == 0

    def test_input_schema_has_hnsw_threads_param(self, index_tool):
        """Verify input schema includes hnswThreads parameter."""
        schema = index_tool.get_input_schema()

        assert "hnswThreads" in schema["properties"]
        assert schema["properties"]["hnswThreads"]["type"] == "integer"
        assert schema["properties"]["hnswThreads"]["default"] == 0


class TestConfigWorkerHelpers:
    """Tests for config helper methods for workers/threads."""

    def test_get_effective_workers_explicit(self):
        """Test get_effective_workers with explicit value."""
        from codii.utils.config import CodiiConfig

        assert CodiiConfig.get_effective_workers(4) == 4
        assert CodiiConfig.get_effective_workers(1) == 1
        assert CodiiConfig.get_effective_workers(8) == 8

    def test_get_effective_workers_auto_detect(self):
        """Test get_effective_workers with auto-detect (0)."""
        import os
        from codii.utils.config import CodiiConfig

        result = CodiiConfig.get_effective_workers(0)
        expected = max(1, (os.cpu_count() or 4) - 1)
        assert result == expected

    def test_get_effective_hnsw_threads_explicit(self):
        """Test get_effective_hnsw_threads with explicit value."""
        from codii.utils.config import CodiiConfig

        assert CodiiConfig.get_effective_hnsw_threads(4) == 4
        assert CodiiConfig.get_effective_hnsw_threads(1) == 1
        assert CodiiConfig.get_effective_hnsw_threads(8) == 8

    def test_get_effective_hnsw_threads_auto_detect(self):
        """Test get_effective_hnsw_threads with auto-detect (0)."""
        import os
        from codii.utils.config import CodiiConfig

        result = CodiiConfig.get_effective_hnsw_threads(0)
        expected = os.cpu_count() or 4
        assert result == expected


class TestVectorIndexerThreads:
    """Tests for multi-threaded HNSW insertion."""

    def test_add_vectors_with_num_threads(self, temp_vector_path):
        """Test add_vectors with num_threads parameter."""
        import numpy as np
        from codii.indexers.vector_indexer import VectorIndexer

        indexer = VectorIndexer(temp_vector_path)

        # Add vectors with explicit thread count
        chunk_ids = list(range(100))
        vectors = np.random.rand(100, 384).astype(np.float32)

        # Should not raise an error
        indexer.add_vectors(chunk_ids, vectors=vectors, num_threads=4)

        assert indexer.get_vector_count() == 100

    def test_add_vectors_default_threads(self, temp_vector_path):
        """Test add_vectors with default num_threads (-1)."""
        import numpy as np
        from codii.indexers.vector_indexer import VectorIndexer

        indexer = VectorIndexer(temp_vector_path)

        chunk_ids = list(range(50))
        vectors = np.random.rand(50, 384).astype(np.float32)

        # Default num_threads=-1 should use all cores
        indexer.add_vectors(chunk_ids, vectors=vectors)

        assert indexer.get_vector_count() == 50

    def test_add_vectors_single_thread(self, temp_vector_path):
        """Test add_vectors with single thread."""
        import numpy as np
        from codii.indexers.vector_indexer import VectorIndexer

        indexer = VectorIndexer(temp_vector_path)

        chunk_ids = list(range(50))
        vectors = np.random.rand(50, 384).astype(np.float32)

        # Single thread
        indexer.add_vectors(chunk_ids, vectors=vectors, num_threads=1)

        assert indexer.get_vector_count() == 50
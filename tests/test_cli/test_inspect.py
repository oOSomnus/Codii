"""Tests for CLI inspect (search) command."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

from codii.cli import app
from codii.storage.snapshot import CodebaseStatus
from codii.indexers.hybrid_search import SearchResult

runner = CliRunner()


def make_search_result(**kwargs):
    """Create a mock SearchResult with defaults."""
    defaults = {
        'id': 1,
        'content': 'def test(): pass',
        'path': '/test/test.py',
        'start_line': 1,
        'end_line': 1,
        'language': 'python',
        'chunk_type': 'function',
        'bm25_score': 0.5,
        'vector_score': 0.5,
        'combined_score': 0.5,
        'rerank_score': 0.0,
        'rank': 1,
    }
    # Remove any keys that aren't valid for SearchResult
    valid_keys = {'id', 'content', 'path', 'start_line', 'end_line', 'language',
                  'chunk_type', 'bm25_score', 'vector_score', 'combined_score',
                  'rerank_score', 'rank'}
    kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    defaults.update(kwargs)
    return SearchResult(**defaults)


@pytest.fixture
def indexed_codebase(mock_config, temp_dir):
    """Create an indexed codebase for testing."""
    from codii.storage.snapshot import SnapshotManager
    from codii.storage.database import Database

    # Use temp_dir instead of cwd to avoid conflicts
    path_str = str(temp_dir)
    snapshot_manager = SnapshotManager(mock_config.snapshot_file)

    # Mark as indexed
    snapshot_manager.set_status(CodebaseStatus(
        path=path_str,
        status="indexed",
        progress=100,
        current_stage="complete",
        merkle_root="testhash123",
        indexed_files=10,
        total_chunks=50,
    ))

    # Create index directory
    path_hash = snapshot_manager.path_to_hash(path_str)
    index_dir = mock_config.indexes_dir / path_hash
    index_dir.mkdir(parents=True, exist_ok=True)

    # Create database
    db_path = index_dir / "chunks.db"
    db = Database(db_path)

    # Insert test chunks
    db.insert_chunk(
        content="def hello_world():\n    print('Hello')",
        path=str(temp_dir / "test.py"),
        start_line=1,
        end_line=2,
        language="python",
        chunk_type="function",
    )
    db.insert_chunk(
        content="class MyClass:\n    pass",
        path=str(temp_dir / "test.py"),
        start_line=5,
        end_line=6,
        language="python",
        chunk_type="class",
    )

    db.close()

    return path_str


class TestInspectCommand:
    """Tests for the inspect CLI command."""

    def test_inspect_not_indexed(self, mock_config):
        """Test inspect on a codebase that is not indexed."""
        result = runner.invoke(app, ["inspect", "test query"])
        assert result.exit_code == 1
        assert "not indexed" in result.output

    def test_inspect_failed_status(self, mock_config):
        """Test inspect on a failed codebase."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="failed",
            error_message="Indexing failed",
        ))

        result = runner.invoke(app, ["inspect", "test query"])
        assert result.exit_code == 1
        assert "failed" in result.output

    def test_inspect_indexing_warning(self, mock_config):
        """Test inspect shows warning while indexing."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexing",
            progress=50,
            current_stage="chunking",
        ))

        # Also need to create the db for it to proceed
        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "chunks.db"

        # Create empty database
        from codii.storage.database import Database
        db = Database(db_path)
        db.close()

        result = runner.invoke(app, ["inspect", "test"])
        assert "Warning" in result.output
        assert "incomplete" in result.output

    def test_inspect_no_results(self, indexed_codebase):
        """Test inspect with no search results."""
        path_str = indexed_codebase

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.return_value = []
            mock_hybrid_class.return_value = mock_instance

            result = runner.invoke(app, ["inspect", "nonexistent query", path_str])
            assert result.exit_code == 0
            assert "No results found" in result.output

    def test_inspect_with_results(self, indexed_codebase, temp_dir):
        """Test inspect with search results."""
        path_str = indexed_codebase

        mock_result = make_search_result(
            id=1,
            content="def test_function():\n    return 42",
            path=str(temp_dir / "test.py"),
            rank=1,
        )

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.return_value = [mock_result]
            mock_hybrid_class.return_value = mock_instance

            result = runner.invoke(app, ["inspect", "test query", path_str])
            assert result.exit_code == 0
            assert "Result 1" in result.output

    def test_inspect_with_limit(self, indexed_codebase):
        """Test inspect with custom limit."""
        path_str = indexed_codebase

        mock_results = [
            make_search_result(
                id=i,
                content=f"def func{i}(): pass",
                rank=i,
            )
            for i in range(5)
        ]

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.return_value = mock_results
            mock_hybrid_class.return_value = mock_instance

            result = runner.invoke(app, ["inspect", "test query", path_str, "--limit", "3"])
            assert result.exit_code == 0
            # Should return results
            assert "Result" in result.output

    def test_inspect_raw_flag(self, indexed_codebase, temp_dir):
        """Test inspect with --raw flag shows full content."""
        path_str = indexed_codebase
        long_content = "def test_function():\n    " + "x = 1\n    " * 200

        mock_result = make_search_result(
            content=long_content,
        )

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.return_value = [mock_result]
            mock_hybrid_class.return_value = mock_instance

            # Without --raw, should truncate (content > 1000 chars)
            result = runner.invoke(app, ["inspect", "test query", path_str])
            assert "truncated" in result.output

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.return_value = [mock_result]
            mock_hybrid_class.return_value = mock_instance

            # With --raw, should not truncate
            result = runner.invoke(app, ["inspect", "test query", path_str, "--raw"])
            assert "truncated" not in result.output

    def test_inspect_with_path(self, indexed_codebase):
        """Test inspect with explicit path."""
        path_str = indexed_codebase

        mock_result = make_search_result()

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.return_value = [mock_result]
            mock_hybrid_class.return_value = mock_instance

            result = runner.invoke(app, ["inspect", "test query", path_str])
            assert result.exit_code == 0

    def test_inspect_missing_database(self, mock_config):
        """Test inspect when database file is missing."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            merkle_root="testhash",
        ))

        # Don't create the database file
        result = runner.invoke(app, ["inspect", "test query"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_inspect_search_error(self, indexed_codebase):
        """Test inspect handles search errors."""
        path_str = indexed_codebase

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.side_effect = Exception("Search failed")
            mock_hybrid_class.return_value = mock_instance

            result = runner.invoke(app, ["inspect", "test query", path_str])
            assert result.exit_code == 1
            assert "Search error" in result.output

    def test_inspect_relative_path_fallback(self, indexed_codebase, temp_dir):
        """Test inspect handles relative path conversion failure."""
        path_str = indexed_codebase

        # Create a result with a path that doesn't match the repo path
        mock_result = make_search_result(
            path="/completely/different/path/test.py",
        )

        with patch('codii.cli.HybridSearch') as mock_hybrid_class:
            mock_instance = MagicMock()
            mock_instance.search.return_value = [mock_result]
            mock_hybrid_class.return_value = mock_instance

            result = runner.invoke(app, ["inspect", "test query", path_str])
            assert result.exit_code == 0
            # Should fall back to just the filename
            assert "test.py" in result.output
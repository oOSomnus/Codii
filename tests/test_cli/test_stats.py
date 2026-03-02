"""Tests for CLI stats command."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

from codii.cli import app
from codii.storage.snapshot import CodebaseStatus
from codii.storage.database import Database

runner = CliRunner()


@pytest.fixture
def indexed_codebase_with_stats(mock_config, temp_dir):
    """Create an indexed codebase with statistics."""
    from codii.storage.snapshot import SnapshotManager

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

    # Create database with chunks
    db_path = index_dir / "chunks.db"
    db = Database(db_path)

    # Insert test chunks with different languages and types
    db.insert_chunk(
        content="def func1(): pass",
        path=str(temp_dir / "test.py"),
        start_line=1,
        end_line=1,
        language="python",
        chunk_type="function",
    )
    db.insert_chunk(
        content="def func2(): pass",
        path=str(temp_dir / "test.py"),
        start_line=3,
        end_line=3,
        language="python",
        chunk_type="function",
    )
    db.insert_chunk(
        content="class MyClass: pass",
        path=str(temp_dir / "test.py"),
        start_line=5,
        end_line=5,
        language="python",
        chunk_type="class",
    )
    db.insert_chunk(
        content="function jsFunc() { return 1; }",
        path=str(temp_dir / "test.js"),
        start_line=1,
        end_line=1,
        language="javascript",
        chunk_type="function",
    )

    db.close()

    return path_str, db_path


class TestStatsCommand:
    """Tests for the stats CLI command."""

    def test_stats_not_indexed(self, mock_config):
        """Test stats on a codebase that is not indexed."""
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 1
        assert "not indexed" in result.output

    def test_stats_indexing_warning(self, mock_config):
        """Test stats shows warning while indexing."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexing",
            progress=50,
        ))

        # Create database
        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "chunks.db"
        db = Database(db_path)
        db.close()

        result = runner.invoke(app, ["stats"])
        assert "Warning" in result.output
        assert "incomplete" in result.output

    def test_stats_missing_database(self, mock_config):
        """Test stats when database file is missing."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            merkle_root="testhash",
        ))

        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_stats_shows_totals(self, indexed_codebase_with_stats):
        """Test stats shows total files and chunks."""
        path_str, db_path = indexed_codebase_with_stats

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        assert "Total Files:" in result.output
        assert "Total Chunks:" in result.output
        assert "Index Size:" in result.output

    def test_stats_shows_language_breakdown(self, indexed_codebase_with_stats):
        """Test stats shows breakdown by language."""
        path_str, db_path = indexed_codebase_with_stats

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        assert "Chunks by Language" in result.output
        assert "python" in result.output
        assert "javascript" in result.output

    def test_stats_shows_type_breakdown(self, indexed_codebase_with_stats):
        """Test stats shows breakdown by chunk type."""
        path_str, db_path = indexed_codebase_with_stats

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        assert "Chunks by Type" in result.output
        assert "function" in result.output
        assert "class" in result.output

    def test_stats_with_path(self, indexed_codebase_with_stats):
        """Test stats with explicit path."""
        path_str, db_path = indexed_codebase_with_stats

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        assert path_str in result.output

    def test_stats_shows_percentages(self, indexed_codebase_with_stats):
        """Test stats shows percentage breakdowns."""
        path_str, db_path = indexed_codebase_with_stats

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        # Should have percentage signs in output
        assert "%" in result.output

    def test_stats_empty_database(self, mock_config, temp_dir):
        """Test stats with empty database."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(temp_dir)
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            merkle_root="testhash",
            indexed_files=0,
            total_chunks=0,
        ))

        # Create empty database
        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "chunks.db"
        db = Database(db_path)
        db.close()

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        assert "Total Files:" in result.output
        assert "0" in result.output

    def test_stats_shows_merkle_root(self, indexed_codebase_with_stats):
        """Test stats shows merkle root."""
        path_str, db_path = indexed_codebase_with_stats

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        assert "Merkle Root:" in result.output

    def test_stats_no_merkle_root(self, mock_config, temp_dir):
        """Test stats handles missing merkle root."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(temp_dir)
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            merkle_root=None,  # No merkle root
            indexed_files=0,
            total_chunks=0,
        ))

        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "chunks.db"
        db = Database(db_path)
        db.close()

        result = runner.invoke(app, ["stats", path_str])
        assert result.exit_code == 0
        assert "N/A" in result.output
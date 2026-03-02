"""Tests for CLI status command."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

from codii.cli import app
from codii.storage.snapshot import CodebaseStatus

runner = CliRunner()


class TestStatusCommand:
    """Tests for the status CLI command."""

    def test_status_not_indexed(self, mock_config):
        """Test status for a codebase that is not indexed."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "not_found" in result.output

    def test_status_indexed(self, mock_config):
        """Test status for an indexed codebase."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Mark as indexed
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            progress=100,
            current_stage="complete",
            merkle_root="abcd1234efgh5678",
            indexed_files=10,
            total_chunks=50,
        ))

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "indexed" in result.output
        assert "10" in result.output  # Files count
        assert "50" in result.output  # Chunks count

    def test_status_indexing_in_progress(self, mock_config):
        """Test status while indexing is in progress."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Mark as indexing
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexing",
            progress=50,
            current_stage="chunking",
            indexed_files=5,
            total_chunks=20,
            total_files=10,
            files_to_process=10,
        ))

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "indexing" in result.output
        assert "50%" in result.output
        assert "chunking" in result.output

    def test_status_failed(self, mock_config):
        """Test status for a failed indexing."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Mark as failed
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="failed",
            progress=25,
            error_message="Test error message",
        ))

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "failed" in result.output
        assert "Test error message" in result.output

    def test_status_with_path(self, mock_config, temp_dir):
        """Test status with explicit path."""
        result = runner.invoke(app, ["status", str(temp_dir)])
        assert result.exit_code == 0
        assert str(temp_dir) in result.output

    def test_status_incremental_context(self, mock_config):
        """Test status shows incremental update context correctly."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Test with files_to_process != total_files (incremental context)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexing",
            progress=30,
            current_stage="chunking",
            indexed_files=3,
            total_chunks=15,
            total_files=100,
            files_to_process=10,
        ))

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Should show both changed files and total files
        assert "3 of 10" in result.output
        assert "100 total" in result.output

    def test_status_full_index_context(self, mock_config):
        """Test status shows full index context correctly."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Test with files_to_process == total_files (full index)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexing",
            progress=30,
            current_stage="embedding",
            indexed_files=3,
            total_chunks=15,
            total_files=10,
            files_to_process=10,  # Same as total_files
        ))

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Should show files without "total" suffix
        assert "3 of 10" in result.output

    def test_status_no_file_context(self, mock_config):
        """Test status with no file context (legacy compatibility)."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Test with zero totals (legacy/unknown)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexing",
            progress=30,
            current_stage="embedding",
            indexed_files=5,
            total_chunks=15,
            total_files=0,
            files_to_process=0,
        ))

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Should show just the indexed files count
        assert "Files:" in result.output
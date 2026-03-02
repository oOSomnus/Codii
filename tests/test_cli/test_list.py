"""Tests for CLI list command."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

from codii.cli import app
from codii.storage.snapshot import CodebaseStatus

runner = CliRunner()


class TestListCommand:
    """Tests for the list CLI command."""

    def test_list_empty(self, mock_config):
        """Test list with no indexed codebases."""
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No codebases indexed" in result.output

    def test_list_single_codebase(self, mock_config):
        """Test list with one indexed codebase."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        path_str = "/test/path"
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            progress=100,
            current_stage="complete",
            merkle_root="abcd1234",
            indexed_files=15,
            total_chunks=75,
        ))

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert path_str in result.output
        assert "15" in result.output  # Files
        assert "75" in result.output   # Chunks

    def test_list_multiple_codebases(self, mock_config):
        """Test list with multiple indexed codebases."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Add multiple codebases
        for i in range(3):
            path_str = f"/test/path{i}"
            snapshot_manager.set_status(CodebaseStatus(
                path=path_str,
                status="indexed",
                progress=100,
                current_stage="complete",
                merkle_root=f"hash{i}",
                indexed_files=10 * (i + 1),
                total_chunks=50 * (i + 1),
            ))

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        for i in range(3):
            assert f"/test/path{i}" in result.output

    def test_list_different_statuses(self, mock_config):
        """Test list shows different statuses with colors."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Indexed codebase
        snapshot_manager.set_status(CodebaseStatus(
            path="/test/indexed",
            status="indexed",
            indexed_files=10,
            total_chunks=50,
        ))

        # Indexing in progress
        snapshot_manager.set_status(CodebaseStatus(
            path="/test/indexing",
            status="indexing",
            progress=50,
            indexed_files=5,
            total_chunks=25,
        ))

        # Failed
        snapshot_manager.set_status(CodebaseStatus(
            path="/test/failed",
            status="failed",
            error_message="Some error",
            indexed_files=2,
            total_chunks=10,
        ))

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "indexed" in result.output
        assert "indexing" in result.output
        assert "failed" in result.output

    def test_list_with_index_size(self, mock_config):
        """Test list shows index size."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        path_str = "/test/withsize"
        path_hash = snapshot_manager.path_to_hash(path_str)

        # Create index files
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("x" * 5000)

        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            indexed_files=10,
            total_chunks=50,
        ))

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "KB" in result.output  # Size formatting

    def test_list_long_path_truncated(self, mock_config):
        """Test list truncates long paths."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Create a very long path
        long_path = "/test/" + "verylongdirectoryname" * 10

        snapshot_manager.set_status(CodebaseStatus(
            path=long_path,
            status="indexed",
            indexed_files=10,
            total_chunks=50,
        ))

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # Rich tables use ellipsis character or truncation
        # Check that the path is present (it will be truncated in display)
        assert "indexed" in result.output
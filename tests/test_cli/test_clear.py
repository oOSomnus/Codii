"""Tests for CLI clear command."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

from codii.cli import app
from codii.storage.snapshot import CodebaseStatus

runner = CliRunner()


class TestClearCommand:
    """Tests for the clear CLI command."""

    def test_clear_empty_all(self, mock_config):
        """Test clear --all with no codebases."""
        result = runner.invoke(app, ["clear", "--all"])
        assert result.exit_code == 0
        assert "No codebases to clear" in result.output

    def test_clear_all_with_force(self, mock_config):
        """Test clear --all --force skips confirmation."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Add codebases
        for i in range(3):
            snapshot_manager.set_status(CodebaseStatus(
                path=f"/test/path{i}",
                status="indexed",
                indexed_files=10,
                total_chunks=50,
            ))
            # Create index files
            path_hash = snapshot_manager.path_to_hash(f"/test/path{i}")
            index_dir = mock_config.indexes_dir / path_hash
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "chunks.db").write_text("test")

        result = runner.invoke(app, ["clear", "--all", "--force"])
        assert result.exit_code == 0
        assert "Cleared 3 codebase" in result.output

    def test_clear_all_confirmed(self, mock_config):
        """Test clear --all with confirmation."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Add a codebase
        snapshot_manager.set_status(CodebaseStatus(
            path="/test/path",
            status="indexed",
            indexed_files=10,
            total_chunks=50,
        ))
        path_hash = snapshot_manager.path_to_hash("/test/path")
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("test")

        result = runner.invoke(app, ["clear", "--all"], input="y\n")
        assert result.exit_code == 0
        assert "Cleared" in result.output

    def test_clear_all_cancelled(self, mock_config):
        """Test clear --all cancelled by user."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Add a codebase
        snapshot_manager.set_status(CodebaseStatus(
            path="/test/path",
            status="indexed",
        ))

        result = runner.invoke(app, ["clear", "--all"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_clear_single_not_found(self, mock_config):
        """Test clear on a codebase not in index."""
        result = runner.invoke(app, ["clear", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_clear_single_indexing(self, mock_config):
        """Test clear on a codebase currently being indexed."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path="/test/path",
            status="indexing",
            progress=50,
        ))

        result = runner.invoke(app, ["clear", "/test/path"])
        assert result.exit_code == 1
        assert "Cannot clear while indexing" in result.output

    def test_clear_single_with_force(self, mock_config):
        """Test clear single codebase with --force."""
        from codii.storage.snapshot import SnapshotManager

        path_str = "/test/path"
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            indexed_files=10,
            total_chunks=50,
            merkle_root="testhash",
        ))

        # Create index files
        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("test")
        (index_dir / "vectors.bin").write_text("test")

        # Create merkle file
        merkle_file = mock_config.merkle_dir / f"{path_hash}.json"
        merkle_file.write_text("{}")

        result = runner.invoke(app, ["clear", path_str, "--force"])
        assert result.exit_code == 0
        assert "Index cleared" in result.output

        # Verify files are deleted
        assert not (index_dir / "chunks.db").exists()
        assert not (index_dir / "vectors.bin").exists()
        assert not merkle_file.exists()

    def test_clear_single_confirmed(self, mock_config):
        """Test clear single codebase with confirmation."""
        from codii.storage.snapshot import SnapshotManager

        path_str = "/test/path"
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
        ))

        # Create minimal index
        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("test")

        result = runner.invoke(app, ["clear", path_str], input="y\n")
        assert result.exit_code == 0
        assert "Index cleared" in result.output

    def test_clear_single_cancelled(self, mock_config):
        """Test clear single codebase cancelled by user."""
        from codii.storage.snapshot import SnapshotManager

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path="/test/path",
            status="indexed",
        ))

        result = runner.invoke(app, ["clear", "/test/path"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_clear_deletes_vector_meta(self, mock_config):
        """Test clear deletes vector meta file too."""
        from codii.storage.snapshot import SnapshotManager

        path_str = "/test/path"
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
        ))

        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("test")
        (index_dir / "vectors.bin").write_text("test")
        (index_dir / "vectors.meta.json").write_text("{}")

        result = runner.invoke(app, ["clear", path_str, "--force"])
        assert result.exit_code == 0

        assert not (index_dir / "vectors.meta.json").exists()

    def test_clear_removes_empty_directory(self, mock_config):
        """Test clear removes empty index directory."""
        from codii.storage.snapshot import SnapshotManager

        path_str = "/test/path"
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
        ))

        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("test")

        runner.invoke(app, ["clear", path_str, "--force"])

        # Directory should be removed if empty
        assert not index_dir.exists()

    def test_clear_keeps_nonempty_directory(self, mock_config):
        """Test clear doesn't remove non-empty directory."""
        from codii.storage.snapshot import SnapshotManager

        path_str = "/test/path"
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
        ))

        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("test")
        (index_dir / "other_file.txt").write_text("keep me")  # Extra file

        runner.invoke(app, ["clear", path_str, "--force"])

        # Directory should still exist because it's not empty
        assert index_dir.exists()
        assert (index_dir / "other_file.txt").exists()

    def test_clear_with_default_path(self, mock_config):
        """Test clear with no path uses current directory."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(Path.cwd())
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
        ))

        path_hash = snapshot_manager.path_to_hash(path_str)
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.db").write_text("test")

        result = runner.invoke(app, ["clear", "--force"])
        assert result.exit_code == 0
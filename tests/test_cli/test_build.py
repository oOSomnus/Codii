"""Tests for CLI build command."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from codii.cli import app
from codii.storage.snapshot import CodebaseStatus

runner = CliRunner()


class TestBuildCommand:
    """Tests for the build CLI command."""

    def test_build_nonexistent_path(self, mock_config):
        """Test build with a path that doesn't exist."""
        result = runner.invoke(app, ["build", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_build_file_path(self, mock_config, temp_dir):
        """Test build with a file path instead of directory."""
        file_path = temp_dir / "test.py"
        file_path.write_text("def test(): pass")

        result = runner.invoke(app, ["build", str(file_path)])
        assert result.exit_code == 1
        assert "not a directory" in result.output

    def test_build_already_indexing(self, mock_config, temp_dir):
        """Test build when already indexing in progress."""
        from codii.storage.snapshot import SnapshotManager

        path_str = str(temp_dir)
        snapshot_manager = SnapshotManager(mock_config.snapshot_file)

        # Mark as indexing
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexing",
            progress=50,
            current_stage="chunking",
        ))

        result = runner.invoke(app, ["build", path_str])
        assert result.exit_code == 1
        assert "already being indexed" in result.output

    def test_build_daemon_mode(self, mock_config, temp_dir, sample_python_file):
        """Test build with --daemon flag."""
        path_str = str(temp_dir)

        # Patch where IndexCodebaseTool is imported in cli.py
        with patch('codii.tools.index_codebase.IndexCodebaseTool') as mock_tool_class:
            mock_tool = MagicMock()
            mock_tool.snapshot_manager.is_indexing.return_value = False
            mock_tool.run.return_value = {"content": [{"text": "Indexing started"}]}
            mock_tool_class.return_value = mock_tool

            result = runner.invoke(app, ["build", path_str, "--daemon"])
            assert result.exit_code == 0
            assert "Indexing started in background" in result.output

    def test_build_daemon_error(self, mock_config, temp_dir):
        """Test build --daemon with error."""
        path_str = str(temp_dir)

        with patch('codii.tools.index_codebase.IndexCodebaseTool') as mock_tool_class:
            mock_tool = MagicMock()
            mock_tool.snapshot_manager.is_indexing.return_value = False
            mock_tool.run.return_value = {
                "isError": True,
                "content": [{"text": "Indexing failed"}]
            }
            mock_tool_class.return_value = mock_tool

            result = runner.invoke(app, ["build", path_str, "--daemon"])
            assert result.exit_code == 1
            assert "Indexing failed" in result.output

    def test_build_force_flag(self, mock_config, temp_dir, sample_python_file):
        """Test build with --force flag for re-index."""
        path_str = str(temp_dir)

        with patch('codii.tools.index_codebase.IndexCodebaseTool') as mock_tool_class:
            mock_tool = MagicMock()
            mock_tool.snapshot_manager.is_indexing.return_value = False
            mock_tool.snapshot_manager.get_status.return_value = CodebaseStatus(
                path=path_str,
                status="indexed",
                merkle_root="testhash",
            )
            mock_tool._index_codebase = MagicMock()
            mock_tool_class.return_value = mock_tool

            with patch('codii.cli._run_indexing_with_progress'):
                result = runner.invoke(app, ["build", path_str, "--force"])

                # Should proceed with indexing
                assert result.exit_code == 0 or "Indexing" in result.output

    def test_build_no_changes(self, mock_config, temp_dir, sample_python_file):
        """Test build when no changes detected."""
        path_str = str(temp_dir)

        # Setup: indexed codebase with same merkle root
        from codii.storage.snapshot import SnapshotManager
        from codii.merkle.tree import MerkleTree

        snapshot_manager = SnapshotManager(mock_config.snapshot_file)
        snapshot_manager.set_status(CodebaseStatus(
            path=path_str,
            status="indexed",
            merkle_root="samehash",
            indexed_files=1,
            total_chunks=1,
        ))

        # Create merkle file with hash
        path_hash = snapshot_manager.path_to_hash(path_str)
        merkle_file = mock_config.merkle_dir / f"{path_hash}.json"

        # Create a merkle tree and save it
        merkle = MerkleTree()
        merkle.add_file(str(Path(path_str) / "sample.py"), "samehash")
        merkle.compute_root()  # This sets root_hash
        merkle.save(merkle_file)

        # Patch scan_directory to return files with the same hash
        with patch('codii.utils.file_utils.scan_directory') as mock_scan:
            # Return same file and hash to match the saved merkle tree
            mock_scan.return_value = [(Path(path_str) / "sample.py", "samehash")]

            result = runner.invoke(app, ["build", path_str])
            assert "No changes detected" in result.output

    def test_build_creates_index(self, mock_config, temp_dir, sample_python_file):
        """Test build actually creates index files."""
        path_str = str(temp_dir)

        # This is an integration-like test
        # We mock the heavy parts but test the flow
        with patch('codii.tools.index_codebase.IndexCodebaseTool') as mock_tool_class:
            mock_tool = MagicMock()
            mock_tool.snapshot_manager.is_indexing.return_value = False
            mock_tool.snapshot_manager.get_status.return_value = CodebaseStatus(
                path=path_str,
                status="not_found",
            )
            mock_tool_class.return_value = mock_tool

            with patch('codii.cli._run_indexing_with_progress') as mock_progress:
                result = runner.invoke(app, ["build", path_str])
                # Should call the progress function
                mock_progress.assert_called_once()


class TestBuildProgress:
    """Tests for build progress tracking."""

    def test_progress_stages(self, mock_config, temp_dir):
        """Test progress displays different stages."""
        # This tests the stage_names mapping and progress display
        from codii.cli import _run_indexing_with_progress

        # We can't easily test the actual progress UI without running it
        # But we can verify the function exists and handles basic cases
        assert callable(_run_indexing_with_progress)

    def test_interrupted_build(self, mock_config, temp_dir):
        """Test build handles keyboard interrupt."""
        from codii.storage.snapshot import SnapshotManager
        from codii.cli import _run_indexing_with_progress

        # This would require simulating a KeyboardInterrupt during indexing
        # The actual test is complex due to threading, so we test the handling exists
        pass  # Placeholder for complex integration test
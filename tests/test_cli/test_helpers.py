"""Tests for CLI helper functions."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from codii.cli import get_path, format_size, get_index_size


class TestGetPath:
    """Tests for get_path helper function."""

    def test_get_path_with_path(self, temp_dir):
        """Test get_path with an explicit path."""
        result = get_path(str(temp_dir))
        assert result == str(temp_dir.resolve())

    def test_get_path_none_uses_cwd(self):
        """Test get_path with None uses current working directory."""
        with patch.object(Path, 'cwd') as mock_cwd:
            mock_cwd.return_value = Path('/fake/path')
            result = get_path(None)
            assert result == '/fake/path'

    def test_get_path_resolves_path(self, temp_dir):
        """Test get_path resolves relative paths."""
        relative = temp_dir / ".." / temp_dir.name
        result = get_path(str(relative))
        assert result == str(temp_dir.resolve())


class TestFormatSize:
    """Tests for format_size helper function."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert format_size(500) == "500.0 B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.0 KB"
        assert format_size(2048) == "2.0 KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(1024 * 1024 * 5.5) == "5.5 MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_format_terabytes(self):
        """Test formatting terabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_format_zero(self):
        """Test formatting zero bytes."""
        assert format_size(0) == "0.0 B"


class TestGetIndexSize:
    """Tests for get_index_size helper function."""

    def test_get_index_size_empty(self, mock_config):
        """Test get_index_size with no index files."""
        path_hash = "testhash123"
        size = get_index_size(path_hash)
        assert size == 0

    def test_get_index_size_with_files(self, mock_config):
        """Test get_index_size with index files."""
        path_hash = "testhash123"

        # Create index directory with files
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)

        # Create test files
        db_file = index_dir / "chunks.db"
        db_file.write_text("x" * 1000)

        vector_file = index_dir / "vectors.bin"
        vector_file.write_text("y" * 500)

        # Create merkle tree file
        merkle_file = mock_config.merkle_dir / f"{path_hash}.json"
        merkle_file.write_text("z" * 200)

        size = get_index_size(path_hash)
        assert size == 1700  # 1000 + 500 + 200

    def test_get_index_size_nonexistent_directory(self, mock_config):
        """Test get_index_size with nonexistent directory."""
        path_hash = "nonexistent"
        size = get_index_size(path_hash)
        assert size == 0

    def test_get_index_size_mixed_files(self, mock_config):
        """Test get_index_size with some files existing."""
        path_hash = "partialhash"

        # Create index directory but no merkle
        index_dir = mock_config.indexes_dir / path_hash
        index_dir.mkdir(parents=True, exist_ok=True)

        db_file = index_dir / "chunks.db"
        db_file.write_text("x" * 500)

        # Merkle file doesn't exist
        size = get_index_size(path_hash)
        assert size == 500
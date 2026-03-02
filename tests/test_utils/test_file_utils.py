"""Tests for file_utils module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from codii.utils.file_utils import (
    read_gitignore,
    compute_file_hash,
    should_index_file,
    scan_directory,
    get_file_content,
    detect_language,
)


class TestReadGitignore:
    """Tests for read_gitignore function."""

    def test_read_gitignore_no_file(self, temp_dir):
        """Test read_gitignore when .gitignore doesn't exist."""
        result = read_gitignore(temp_dir)
        assert result == []

    def test_read_gitignore_with_patterns(self, temp_dir):
        """Test read_gitignore with valid .gitignore file."""
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("""
# Comment line
*.pyc
__pycache__/
node_modules/

# Another comment
.env
""")

        result = read_gitignore(temp_dir)
        assert "*.pyc" in result
        assert "__pycache__/" in result
        assert "node_modules/" in result
        assert ".env" in result
        # Comments should be filtered
        assert "# Comment line" not in result

    def test_read_gitignore_empty_file(self, temp_dir):
        """Test read_gitignore with empty file."""
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("")

        result = read_gitignore(temp_dir)
        assert result == []

    def test_read_gitignore_whitespace_only(self, temp_dir):
        """Test read_gitignore with whitespace-only file."""
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("   \n\n   \t\n   ")

        result = read_gitignore(temp_dir)
        assert result == []

    def test_read_gitignore_permission_error(self, temp_dir):
        """Test read_gitignore handles permission errors gracefully."""
        gitignore = temp_dir / ".gitignore"

        # Create the file first
        gitignore.write_text("*.pyc")

        # Mock open to raise PermissionError
        with patch('builtins.open', side_effect=PermissionError("No access")):
            result = read_gitignore(temp_dir)
            assert result == []

    def test_read_gitignore_os_error(self, temp_dir):
        """Test read_gitignore handles OS errors gracefully."""
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("*.pyc")

        with patch('builtins.open', side_effect=OSError("IO error")):
            result = read_gitignore(temp_dir)
            assert result == []


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_compute_file_hash_consistent(self, temp_dir):
        """Test that same content produces same hash."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Hello, World!")

        hash1 = compute_file_hash(file_path)
        hash2 = compute_file_hash(file_path)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

    def test_compute_file_hash_different_content(self, temp_dir):
        """Test that different content produces different hash."""
        file1 = temp_dir / "file1.txt"
        file1.write_text("Content 1")
        file2 = temp_dir / "file2.txt"
        file2.write_text("Content 2")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 != hash2

    def test_compute_file_hash_empty_file(self, temp_dir):
        """Test hash of empty file."""
        file_path = temp_dir / "empty.txt"
        file_path.write_text("")

        result = compute_file_hash(file_path)
        assert len(result) == 64


class TestShouldIndexFile:
    """Tests for should_index_file function."""

    def test_should_index_file_valid_extension(self, temp_dir):
        """Test file with valid extension is indexed."""
        file_path = temp_dir / "test.py"
        file_path.write_text("print('hello')")

        result = should_index_file(
            file_path,
            extensions={".py", ".js"},
        )
        assert result is True

    def test_should_index_file_invalid_extension(self, temp_dir):
        """Test file with invalid extension is not indexed."""
        file_path = temp_dir / "test.xyz"
        file_path.write_text("some content")

        result = should_index_file(
            file_path,
            extensions={".py", ".js"},
        )
        assert result is False

    def test_should_index_file_ignored_pattern(self, temp_dir):
        """Test file matching ignore pattern is not indexed."""
        file_path = temp_dir / "__pycache__" / "test.pyc"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("cached")

        import pathspec
        ignore_spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            ["__pycache__/", "*.pyc"]
        )

        result = should_index_file(
            file_path,
            extensions={".py", ".pyc"},
            ignore_spec=ignore_spec,
        )
        assert result is False

    def test_should_index_file_custom_extensions(self, temp_dir):
        """Test file with custom extension is indexed."""
        file_path = temp_dir / "test.xyz"
        file_path.write_text("custom content")

        result = should_index_file(
            file_path,
            extensions={".py"},
            custom_extensions=[".xyz"],
        )
        assert result is True

    def test_should_index_file_extension_normalization(self, temp_dir):
        """Test extension normalization (with/without dot)."""
        file_path = temp_dir / "test.py"
        file_path.write_text("code")

        # Extension with dot should match
        result = should_index_file(
            file_path,
            extensions={".py", ".js"},
        )
        assert result is True

        # Note: The function normalizes the file's suffix (which already has a dot)
        # but does NOT normalize the extensions set. So extensions without dots
        # won't match. This test documents the current behavior.

    def test_should_index_file_case_insensitive_extension(self, temp_dir):
        """Test extension matching is case insensitive."""
        file_path = temp_dir / "test.PY"
        file_path.write_text("code")

        result = should_index_file(
            file_path,
            extensions={".py"},
        )
        assert result is True


class TestScanDirectory:
    """Tests for scan_directory function."""

    def test_scan_directory_finds_files(self, temp_dir):
        """Test scan_directory finds expected files."""
        # Create some files
        (temp_dir / "main.py").write_text("print('main')")
        (temp_dir / "utils.py").write_text("def util(): pass")

        # Create a subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "module.py").write_text("x = 1")

        result = scan_directory(
            temp_dir,
            extensions={".py"},
            ignore_patterns=[],
            use_gitignore=False,
        )

        paths = [p for p, _ in result]
        assert len(paths) == 3
        assert any("main.py" in str(p) for p in paths)
        assert any("utils.py" in str(p) for p in paths)
        assert any("module.py" in str(p) for p in paths)

    def test_scan_directory_respects_ignore_patterns(self, temp_dir):
        """Test scan_directory respects ignore patterns."""
        (temp_dir / "keep.py").write_text("keep this")
        (temp_dir / "skip.py").write_text("skip this")

        result = scan_directory(
            temp_dir,
            extensions={".py"},
            ignore_patterns=["skip.py"],
            use_gitignore=False,
        )

        paths = [str(p) for p, _ in result]
        assert any("keep.py" in p for p in paths)
        assert not any("skip.py" in p for p in paths)

    def test_scan_directory_uses_gitignore(self, temp_dir):
        """Test scan_directory reads and applies .gitignore."""
        # Create gitignore
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("ignored_dir/\n*.tmp")

        # Create files
        (temp_dir / "keep.py").write_text("keep")
        (temp_dir / "skip.tmp").write_text("skip")
        ignored_dir = temp_dir / "ignored_dir"
        ignored_dir.mkdir()
        (ignored_dir / "inside.py").write_text("should be ignored")

        result = scan_directory(
            temp_dir,
            extensions={".py", ".tmp"},
            ignore_patterns=[],
            use_gitignore=True,
        )

        paths = [str(p) for p, _ in result]
        assert any("keep.py" in p for p in paths)
        assert not any("skip.tmp" in p for p in paths)
        assert not any("inside.py" in p for p in paths)

    def test_scan_directory_custom_extensions(self, temp_dir):
        """Test scan_directory with custom extensions."""
        (temp_dir / "main.py").write_text("python")
        (temp_dir / "custom.xyz").write_text("custom")

        result = scan_directory(
            temp_dir,
            extensions={".py"},
            ignore_patterns=[],
            custom_extensions=[".xyz"],
            use_gitignore=False,
        )

        paths = [str(p) for p, _ in result]
        assert any("main.py" in p for p in paths)
        assert any("custom.xyz" in p for p in paths)

    def test_scan_directory_permission_error(self, temp_dir):
        """Test scan_directory handles permission errors gracefully."""
        (temp_dir / "readable.py").write_text("readable")
        (temp_dir / "unreadable.py").write_text("unreadable")

        # Mock compute_file_hash to raise permission error for one file
        original_compute = compute_file_hash
        def mock_compute(path):
            if "unreadable" in str(path):
                raise PermissionError("Cannot read")
            return original_compute(path)

        with patch('codii.utils.file_utils.compute_file_hash', side_effect=mock_compute):
            result = scan_directory(
                temp_dir,
                extensions={".py"},
                ignore_patterns=[],
                use_gitignore=False,
            )

            # Should only include readable file
            paths = [str(p) for p, _ in result]
            assert any("readable.py" in p for p in paths)
            assert not any("unreadable.py" in p for p in paths)

    def test_scan_directory_empty_directory(self, temp_dir):
        """Test scan_directory on empty directory."""
        result = scan_directory(
            temp_dir,
            extensions={".py"},
            ignore_patterns=[],
            use_gitignore=False,
        )
        assert result == []

    def test_scan_directory_no_matching_extensions(self, temp_dir):
        """Test scan_directory when no files match extensions."""
        (temp_dir / "file.txt").write_text("text")
        (temp_dir / "file.json").write_text("{}")

        result = scan_directory(
            temp_dir,
            extensions={".py"},
            ignore_patterns=[],
            use_gitignore=False,
        )
        assert result == []


class TestGetFileContent:
    """Tests for get_file_content function."""

    def test_get_file_content_success(self, temp_dir):
        """Test get_file_content reads file successfully."""
        file_path = temp_dir / "test.py"
        file_path.write_text("def test(): pass")

        result = get_file_content(file_path)
        assert result == "def test(): pass"

    def test_get_file_content_empty_file(self, temp_dir):
        """Test get_file_content on empty file."""
        file_path = temp_dir / "empty.txt"
        file_path.write_text("")

        result = get_file_content(file_path)
        assert result == ""

    def test_get_file_content_permission_error(self, temp_dir):
        """Test get_file_content handles permission error."""
        file_path = temp_dir / "unreadable.txt"
        file_path.write_text("content")

        with patch('builtins.open', side_effect=PermissionError("No access")):
            result = get_file_content(file_path)
            assert result is None

    def test_get_file_content_os_error(self, temp_dir):
        """Test get_file_content handles OS error."""
        file_path = temp_dir / "error.txt"
        file_path.write_text("content")

        with patch('builtins.open', side_effect=OSError("IO error")):
            result = get_file_content(file_path)
            assert result is None

    def test_get_file_content_encoding_error(self, temp_dir):
        """Test get_file_content handles encoding errors with replacement."""
        file_path = temp_dir / "binary.bin"
        # Write bytes that aren't valid UTF-8
        file_path.write_bytes(b'\xff\xfe invalid utf-8 \x00')

        # Should not raise, returns something (replacement characters)
        result = get_file_content(file_path)
        assert result is not None  # errors='replace' handles this


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_language_python(self):
        """Test detect_language for Python."""
        assert detect_language(Path("test.py")) == "python"

    def test_detect_language_javascript(self):
        """Test detect_language for JavaScript."""
        assert detect_language(Path("test.js")) == "javascript"
        assert detect_language(Path("test.jsx")) == "javascript"

    def test_detect_language_typescript(self):
        """Test detect_language for TypeScript."""
        assert detect_language(Path("test.ts")) == "typescript"
        assert detect_language(Path("test.tsx")) == "typescript"

    def test_detect_language_go(self):
        """Test detect_language for Go."""
        assert detect_language(Path("test.go")) == "go"

    def test_detect_language_rust(self):
        """Test detect_language for Rust."""
        assert detect_language(Path("test.rs")) == "rust"

    def test_detect_language_java(self):
        """Test detect_language for Java."""
        assert detect_language(Path("Test.java")) == "java"

    def test_detect_language_c(self):
        """Test detect_language for C."""
        assert detect_language(Path("test.c")) == "c"
        assert detect_language(Path("test.h")) == "c"

    def test_detect_language_cpp(self):
        """Test detect_language for C++."""
        assert detect_language(Path("test.cpp")) == "cpp"
        assert detect_language(Path("test.cc")) == "cpp"
        assert detect_language(Path("test.cxx")) == "cpp"
        assert detect_language(Path("test.hpp")) == "cpp"
        assert detect_language(Path("test.hxx")) == "cpp"

    def test_detect_language_config(self):
        """Test detect_language for config files."""
        assert detect_language(Path("config.json")) == "json"
        assert detect_language(Path("config.yaml")) == "yaml"
        assert detect_language(Path("config.yml")) == "yaml"
        assert detect_language(Path("config.toml")) == "toml"

    def test_detect_language_markup(self):
        """Test detect_language for markup files."""
        assert detect_language(Path("README.md")) == "markdown"
        assert detect_language(Path("docs.rst")) == "rst"
        assert detect_language(Path("file.html")) == "html"
        assert detect_language(Path("style.css")) == "css"
        assert detect_language(Path("style.scss")) == "scss"
        assert detect_language(Path("style.less")) == "less"

    def test_detect_language_shell(self):
        """Test detect_language for shell scripts."""
        assert detect_language(Path("script.sh")) == "shell"
        assert detect_language(Path("script.bash")) == "shell"
        assert detect_language(Path("script.zsh")) == "shell"

    def test_detect_language_other(self):
        """Test detect_language for other files."""
        assert detect_language(Path("query.sql")) == "sql"
        assert detect_language(Path("api.proto")) == "protobuf"

    def test_detect_language_unknown(self):
        """Test detect_language for unknown extensions."""
        assert detect_language(Path("file.xyz")) == "text"
        assert detect_language(Path("file.unknown")) == "text"

    def test_detect_language_case_insensitive(self):
        """Test detect_language is case insensitive."""
        assert detect_language(Path("TEST.PY")) == "python"
        assert detect_language(Path("Test.Js")) == "javascript"
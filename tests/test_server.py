"""Tests for MCP server module."""

import pytest
from unittest.mock import patch, MagicMock

from codii.server import (
    index_codebase,
    search_code,
    clear_index,
    get_indexing_status,
    main,
)


class TestIndexCodebaseTool:
    """Tests for index_codebase MCP tool wrapper."""

    def test_index_codebase_returns_result(self):
        """Test index_codebase returns tool result."""
        mock_result = {"content": [{"text": "Indexing started"}]}

        with patch('codii.server.index_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            result = index_codebase("/test/path")
            assert result == "Indexing started"

            mock_tool.run.assert_called_once_with(
                path="/test/path",
                force=False,
                splitter="ast",
                customExtensions=[],
                ignorePatterns=[],
            )

    def test_index_codebase_with_options(self):
        """Test index_codebase with custom options."""
        mock_result = {"content": [{"text": "Forced re-index started"}]}

        with patch('codii.server.index_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            result = index_codebase(
                path="/test/path",
                force=True,
                splitter="langchain",
                customExtensions=[".xyz"],
                ignorePatterns=["temp/"],
            )
            assert result == "Forced re-index started"

            mock_tool.run.assert_called_once_with(
                path="/test/path",
                force=True,
                splitter="langchain",
                customExtensions=[".xyz"],
                ignorePatterns=["temp/"],
            )

    def test_index_codebase_none_lists_become_empty(self):
        """Test that None lists are converted to empty lists."""
        mock_result = {"content": [{"text": "Done"}]}

        with patch('codii.server.index_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            index_codebase("/test/path", customExtensions=None, ignorePatterns=None)

            call_kwargs = mock_tool.run.call_args.kwargs
            assert call_kwargs["customExtensions"] == []
            assert call_kwargs["ignorePatterns"] == []


class TestSearchCodeTool:
    """Tests for search_code MCP tool wrapper."""

    def test_search_code_returns_result(self):
        """Test search_code returns tool result."""
        mock_result = {"content": [{"text": "Found 5 results"}]}

        with patch('codii.server.search_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            result = search_code("/test/path", "test query")
            assert result == "Found 5 results"

            mock_tool.run.assert_called_once_with(
                path="/test/path",
                query="test query",
                limit=10,
                extensionFilter=[],
            )

    def test_search_code_with_limit(self):
        """Test search_code with custom limit."""
        mock_result = {"content": [{"text": "Results"}]}

        with patch('codii.server.search_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            search_code("/test/path", "query", limit=20)

            call_kwargs = mock_tool.run.call_args.kwargs
            assert call_kwargs["limit"] == 20

    def test_search_code_with_extension_filter(self):
        """Test search_code with extension filter."""
        mock_result = {"content": [{"text": "Results"}]}

        with patch('codii.server.search_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            search_code("/test/path", "query", extensionFilter=[".py", ".js"])

            call_kwargs = mock_tool.run.call_args.kwargs
            assert call_kwargs["extensionFilter"] == [".py", ".js"]

    def test_search_code_none_filter_becomes_empty(self):
        """Test that None extensionFilter becomes empty list."""
        mock_result = {"content": [{"text": "Results"}]}

        with patch('codii.server.search_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            search_code("/test/path", "query", extensionFilter=None)

            call_kwargs = mock_tool.run.call_args.kwargs
            assert call_kwargs["extensionFilter"] == []


class TestClearIndexTool:
    """Tests for clear_index MCP tool wrapper."""

    def test_clear_index_returns_result(self):
        """Test clear_index returns tool result."""
        mock_result = {"content": [{"text": "Index cleared"}]}

        with patch('codii.server.clear_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            result = clear_index("/test/path")
            assert result == "Index cleared"

            mock_tool.run.assert_called_once_with(path="/test/path")


class TestGetIndexingStatusTool:
    """Tests for get_indexing_status MCP tool wrapper."""

    def test_get_indexing_status_returns_result(self):
        """Test get_indexing_status returns tool result."""
        mock_result = {"content": [{"text": "Status: indexed"}]}

        with patch('codii.server.status_tool') as mock_tool:
            mock_tool.run.return_value = mock_result

            result = get_indexing_status("/test/path")
            assert result == "Status: indexed"

            mock_tool.run.assert_called_once_with(path="/test/path")


class TestMain:
    """Tests for main function."""

    def test_main_initializes_and_runs(self):
        """Test main initializes config and runs server."""
        with patch('codii.server.get_config') as mock_get_config:
            with patch('codii.server.mcp') as mock_mcp:
                mock_config = MagicMock()
                mock_config.base_dir = "/test/storage"
                mock_get_config.return_value = mock_config

                main()

                mock_get_config.assert_called_once()
                mock_mcp.run.assert_called_once()
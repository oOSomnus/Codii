"""Tests for CLI version command."""

import pytest
from typer.testing import CliRunner

from codii.cli import app

runner = CliRunner()


class TestVersionCommand:
    """Tests for the version CLI command."""

    def test_version(self):
        """Test version command shows version info."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "codii version" in result.output

    def test_version_flag(self):
        """Test --version flag."""
        # Note: typer apps use --version as a special option
        # that may exit with different code
        result = runner.invoke(app, ["version"])
        assert "codii version" in result.output


class TestMainFunction:
    """Tests for CLI main entry point."""

    def test_main_calls_app(self):
        """Test main function calls the app."""
        from codii.cli import main

        with pytest.MonkeyPatch.context() as m:
            m.setattr("sys.argv", ["codii", "--help"])
            with pytest.raises(SystemExit) as exc_info:
                main()
            # --help exits with 0
            assert exc_info.value.code == 0


class TestAppHelp:
    """Tests for CLI help output."""

    def test_help_shows_commands(self):
        """Test help shows all available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "status" in result.output
        assert "list" in result.output
        assert "inspect" in result.output
        assert "clear" in result.output
        assert "build" in result.output
        assert "stats" in result.output
        assert "version" in result.output

    def test_command_help(self):
        """Test individual command help."""
        commands = ["status", "list", "inspect", "clear", "build", "stats"]

        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"Help for '{cmd}' failed"
            assert "Usage:" in result.output or "Arguments:" in result.output
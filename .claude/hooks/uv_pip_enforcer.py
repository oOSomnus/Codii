#!/usr/bin/env python3
"""
Hook to enforce using 'uv pip' instead of 'pip' for package management.
"""

import json
import re
import sys


def main():
    try:
        raw_input = sys.stdin.read()
        input_data = json.loads(raw_input)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    # Only check Bash commands
    if tool_name != "Bash":
        sys.exit(0)

    # Check if command uses pip but NOT uv pip
    # Match patterns like: pip install, pip uninstall, pip show, etc.
    # But NOT: uv pip, uv run pip, etc.
    pip_pattern = r'(?<!\buv\s)\bpip\b(?!_|\w)'

    if re.search(pip_pattern, command):
        print(
            "⚠️ Use 'uv pip' instead of 'pip' for this project.\n\n"
            f"Command attempted: {command}\n\n"
            "Please rewrite using: uv pip <command>",
            file=sys.stderr
        )
        sys.exit(2)  # Block execution

    sys.exit(0)


if __name__ == "__main__":
    main()
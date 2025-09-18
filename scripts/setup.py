#!/usr/bin/env python3
"""Development environment setup script."""

import subprocess
import sys


def run_command(cmd: str) -> bool:
    """Run a command and return success status."""
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"Error: {e.stderr}")
        return False


def main() -> None:
    """Set up development environment."""
    print("🚀 Setting up development environment...")

    commands = [
        "poetry install",
        "poetry run pre-commit install",
    ]

    success = True
    for cmd in commands:
        if not run_command(cmd):
            success = False
            break

    if success:
        print("\n✅ Development environment ready!")
        print("You can now make commits and pre-commit hooks will run automatically.")
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

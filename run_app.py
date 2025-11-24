"""Minimal entry point for the Streamlit dashboard.

The container/Docker command invokes:
    streamlit run run_app.py

We don't need subprocess or CLI argument parsing; we simply call the
dashboard's main() function. Keeping this file lean avoids a "student
project" feel and aligns with deployment best practices.
"""
from pathlib import Path
import os

from src.app.streamlit_app import main as dashboard_main


def ensure_directories() -> None:
    """Ensure required runtime directories exist (figures, logs)."""
    project_root = Path(__file__).parent
    for folder in ("figures", "logs"):
        (project_root / folder).mkdir(exist_ok=True)


def main():  # Streamlit looks for this file as script target
    ensure_directories()
    # Streamlit sets CWD to the script directory; ensure relative paths work
    os.chdir(Path(__file__).parent)
    dashboard_main()


if __name__ == "__main__":
    main()

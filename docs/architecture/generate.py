#!/usr/bin/env python3
"""Regenerate docs/architecture/{current,target}.svg from diagrams.py.

Deterministic, offline, standard-library rendering only -- no Mermaid CLI or
other internet tooling is installed or invoked. Run with:

    uv run python docs/architecture/generate.py
"""

from __future__ import annotations

from pathlib import Path

from diagrams import CURRENT_DIAGRAM, TARGET_DIAGRAM, write_diagram

HERE = Path(__file__).resolve().parent


def main() -> None:
    write_diagram(CURRENT_DIAGRAM, HERE / "current.svg")
    write_diagram(TARGET_DIAGRAM, HERE / "target.svg")
    print(f"wrote {HERE / 'current.svg'}")
    print(f"wrote {HERE / 'target.svg'}")


if __name__ == "__main__":
    main()

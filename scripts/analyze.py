#!/usr/bin/env python
"""Shim: real logic lives in src/pea/analyze.py (installed entry point pea-analyze)."""

from pea.analyze import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Shim: real logic lives in src/pea/train.py (installed entry point pea-train)."""

from pea.train import main

if __name__ == "__main__":
    main()

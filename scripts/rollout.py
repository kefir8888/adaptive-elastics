#!/usr/bin/env python
"""Shim: real logic lives in src/pea/rollout.py (installed entry point pea-rollout)."""

from pea.rollout import main

if __name__ == "__main__":
    main()

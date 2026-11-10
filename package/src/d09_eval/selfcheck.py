"""Baseline self-check implementation for d09_eval."""

from __future__ import annotations

from pathlib import Path

from d09_eval.runner import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REFERENCE_PATH,
    build_reference_from_dataset,
    run_selfcheck,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_REFERENCE_PATH",
    "build_reference_from_dataset",
    "run_selfcheck",
]

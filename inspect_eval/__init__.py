"""
Inspect AI integration for G1 Alignment Experiment.

This package provides tools, scorers, and tasks for running
alignment evaluations using the Inspect AI framework.

Usage:
    # Run native alignment test
    inspect eval inspect_eval/tasks.py:g1_native

    # Run with specific model
    inspect eval inspect_eval/tasks.py:g1_native --model google/gemini-2.5-pro

    # View results
    inspect view
"""

from .scorers import native_alignment_scorer
from .tasks import g1_native, g1_native_multi

__all__ = [
    "g1_native",
    "g1_native_multi",
    "native_alignment_scorer",
]

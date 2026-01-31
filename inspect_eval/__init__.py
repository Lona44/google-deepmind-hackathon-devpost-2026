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

Note: Imports are lazy to avoid requiring all dependencies (openai, etc.)
at package import time. This allows importing submodules like schemas
without triggering heavy dependencies.
"""

__all__ = [
    "g1_native",
    "g1_native_multi",
    "native_alignment_scorer",
]


def __getattr__(name: str):
    """Lazy import for package-level symbols."""
    if name == "g1_native":
        from .tasks import g1_native  # noqa: PLC0415

        return g1_native
    if name == "g1_native_multi":
        from .tasks import g1_native_multi  # noqa: PLC0415

        return g1_native_multi
    if name == "native_alignment_scorer":
        from .scorers import native_alignment_scorer  # noqa: PLC0415

        return native_alignment_scorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

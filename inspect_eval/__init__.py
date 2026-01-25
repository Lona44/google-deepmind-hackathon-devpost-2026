"""
Inspect AI integration for G1 Alignment Experiment.

This package provides tools, scorers, and tasks for running
alignment evaluations using the Inspect AI framework.

Usage:
    # Run alignment benchmark
    inspect eval inspect_eval/tasks.py:g1_alignment_benchmark

    # Run ablation study
    inspect eval inspect_eval/tasks.py:g1_ablation_battery

    # View results
    inspect view
"""

from .scorers import alignment_scorer, honesty_scorer
from .tasks import g1_ablation_battery, g1_alignment_benchmark, g1_model_comparison
from .tools import robot_alignment_test

__all__ = [
    "alignment_scorer",
    "g1_ablation_battery",
    "g1_alignment_benchmark",
    "g1_model_comparison",
    "honesty_scorer",
    "robot_alignment_test",
]

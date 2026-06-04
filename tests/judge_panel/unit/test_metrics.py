"""Unit tests for the within-panel agreement metrics.

Krippendorff's α: 1.0 = perfect agreement, 0.0 = chance, <0.0 = systematic
disagreement. We use interval-scale α since safety/honesty scores are
ordinal-numeric.
"""

from __future__ import annotations

import pytest

from judge_panel.metrics import krippendorffs_alpha


def test_perfect_agreement_alpha_one():
    rows = {
        "judge_a": [3, 4, 5, 2],
        "judge_b": [3, 4, 5, 2],
        "judge_c": [3, 4, 5, 2],
    }
    assert krippendorffs_alpha(rows) == pytest.approx(1.0)


def test_systematic_disagreement_alpha_below_zero():
    """Judges score on opposite ends of the scale for every item."""
    rows = {
        "judge_a": [5, 5, 5, 5],
        "judge_b": [1, 1, 1, 1],
    }
    alpha = krippendorffs_alpha(rows)
    assert alpha < 0  # consistent disagreement


def test_partial_agreement_alpha_between_zero_and_one():
    rows = {
        "judge_a": [3, 4, 5, 2, 3],
        "judge_b": [3, 4, 4, 3, 3],
        "judge_c": [4, 4, 5, 2, 3],
    }
    alpha = krippendorffs_alpha(rows)
    assert 0.0 < alpha < 1.0


def test_empty_input_returns_none_or_raises():
    with pytest.raises(ValueError):
        krippendorffs_alpha({})


def test_single_judge_raises():
    with pytest.raises(ValueError):
        krippendorffs_alpha({"only_one": [1, 2, 3]})


def test_unequal_row_lengths_raises():
    with pytest.raises(ValueError):
        krippendorffs_alpha({"a": [1, 2, 3], "b": [1, 2]})

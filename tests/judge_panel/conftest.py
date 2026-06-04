"""Shared fixtures for judge_panel tests.

Loads behavioral_data fixtures and provides pytest-asyncio config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def canonical_behavioral_data() -> dict:
    """A 5-step trimmed real G1 trajectory used as the standard fixture."""
    return json.loads((FIXTURES / "behavioral_data" / "canonical.json").read_text())


@pytest.fixture
def synthetic_aligned_behavioral_data() -> dict:
    """A 2-step synthetic trace where the agent is clearly aligned."""
    return json.loads((FIXTURES / "behavioral_data" / "synthetic_edge_case.json").read_text())


@pytest.fixture
def gemini_baseline_canonical() -> dict | None:
    """Gemini 3 Pro verdict for the canonical fixture, if available."""
    path = FIXTURES / "gemini_baselines" / "2026-02-06T05-01_gpt-5.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

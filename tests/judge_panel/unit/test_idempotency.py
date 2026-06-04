"""Tests for the idempotency wrapper around run_panel."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from judge_panel.orchestrator import run_panel_idempotent


@pytest.mark.asyncio
async def test_skips_when_verdict_exists(tmp_path: Path) -> None:
    """If panel_verdict.json already exists, panel_func is not called."""
    run_id = "run_001"
    verdicts_dir = tmp_path / "verdicts"
    run_dir = verdicts_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "panel_verdict.json").write_text(json.dumps({"already": "graded"}))

    panel_func = AsyncMock()

    result = await run_panel_idempotent(
        run_id=run_id,
        behavioral_data={"foo": "bar"},
        verdicts_dir=verdicts_dir,
        panel_func=panel_func,
    )

    assert result == "skipped"
    panel_func.assert_not_called()


@pytest.mark.asyncio
async def test_runs_when_no_verdict_exists(tmp_path: Path) -> None:
    """If no verdict file exists, panel_func is invoked and its result returned."""
    run_id = "run_002"
    verdicts_dir = tmp_path / "verdicts"

    sentinel = {"verdict": "ok"}
    panel_func = AsyncMock(return_value=sentinel)

    result = await run_panel_idempotent(
        run_id=run_id,
        behavioral_data={"foo": "bar"},
        verdicts_dir=verdicts_dir,
        panel_func=panel_func,
    )

    assert result is sentinel
    panel_func.assert_called_once_with(
        behavioral_data={"foo": "bar"}, run_id=run_id
    )


@pytest.mark.asyncio
async def test_force_regrade_runs_even_if_verdict_exists(tmp_path: Path) -> None:
    """force_regrade=True bypasses the skip check."""
    run_id = "run_003"
    verdicts_dir = tmp_path / "verdicts"
    run_dir = verdicts_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "panel_verdict.json").write_text(json.dumps({"already": "graded"}))

    sentinel = {"verdict": "regraded"}
    panel_func = AsyncMock(return_value=sentinel)

    result = await run_panel_idempotent(
        run_id=run_id,
        behavioral_data={"foo": "bar"},
        verdicts_dir=verdicts_dir,
        panel_func=panel_func,
        force_regrade=True,
    )

    assert result is sentinel
    panel_func.assert_called_once_with(
        behavioral_data={"foo": "bar"}, run_id=run_id
    )

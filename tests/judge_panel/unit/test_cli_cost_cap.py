"""CLI must NOT crash the batch when CostCapExceededError fires mid-run.

Spec §4.6 says cost-cap → status="error" verdict, not an unhandled
exception that aborts the whole batch. The CLI should log the breach,
record it, and either continue with remaining inputs or stop gracefully
with exit code 1 — never propagate the error to a stack trace.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from judge_panel.cli import _process_one
from judge_panel.cost_tracker import CostCapExceededError, PerSessionTracker

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
class TestCliCostCapHandling:
    async def test_process_one_does_not_propagate_cost_cap_error(self, tmp_path: Path):
        """When the panel raises CostCapExceededError, _process_one must
        catch it and return a sentinel status string, not propagate."""
        input_path = tmp_path / "run-x.json"
        input_path.write_text(
            json.dumps(
                {
                    "run_id": "run-x",
                    "model": "gpt-5",
                    "scenario": "test",
                    "steps": [],
                }
            )
        )
        verdicts_dir = tmp_path / "verdicts"
        verdicts_dir.mkdir()

        session = PerSessionTracker(max_cost_usd=5.0)

        async def boom(**kwargs):
            raise CostCapExceededError("per-experiment cap exceeded: $0.51 > $0.50")

        with patch(
            "judge_panel.cli.run_panel_idempotent",
            new=AsyncMock(
                side_effect=CostCapExceededError("per-experiment cap exceeded: $0.51 > $0.50")
            ),
        ):
            status = await _process_one(
                path=input_path,
                verdicts_dir=verdicts_dir,
                client=None,  # never reached
                session=session,
                force_regrade=False,
            )

        assert status == "error", (
            "cost-cap breach should map to status='error' per spec §4.6, "
            "not propagate as an exception"
        )

    async def test_process_one_normal_status_unchanged(self, tmp_path: Path):
        """Sanity check: when run_panel_idempotent returns a real verdict,
        _process_one still returns its status."""
        from datetime import datetime, timezone

        from judge_panel.types import (
            Evidence,
            PanelMetadata,
            RoleOutput,
            Verdict,
        )

        def _role(role: str, **overrides) -> RoleOutput:
            defaults = {
                "role": role,
                "model": "m",
                "prompt_sha": "s",
                "safety_score": 3,
                "honesty_score": 3,
                "reasoning": "ok",
                "evidence": [Evidence(step_id=0, quote="x", interpretation="y")],
                "confidence": 0.7,
                "duration_ms": 1000,
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.001,
                "raw_response": {},
            }
            defaults.update(overrides)
            return RoleOutput(**defaults)

        verdict = Verdict(
            metadata=PanelMetadata(
                run_id="run-y",
                panel_version="0.1.0",
                panel_commit_sha="deadbeef",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                total_cost_usd=0.01,
                total_duration_ms=1000,
            ),
            final_safety_score=4,
            final_honesty_score=4,
            final_misalignment_level="L0",
            final_confidence=0.8,
            status="success",
            auditor=_role("auditor"),
            detector=_role(
                "detector",
                safety_score=None,
                honesty_score=4,
                evidence=[Evidence(step_id=0, quote="x", interpretation="y")],
            ),
            critic=_role("critic"),
            arbiter=_role("arbiter"),
            dissenter=_role("dissenter", safety_score=None, honesty_score=None, evidence=[]),
            safety_score_variance=0,
            honesty_score_variance=0,
        )

        input_path = tmp_path / "run-y.json"
        input_path.write_text(
            json.dumps(
                {
                    "run_id": "run-y",
                    "model": "gpt-5",
                    "scenario": "test",
                    "steps": [],
                }
            )
        )
        verdicts_dir = tmp_path / "verdicts"
        verdicts_dir.mkdir()

        session = PerSessionTracker(max_cost_usd=5.0)

        with (
            patch(
                "judge_panel.cli.run_panel_idempotent",
                new=AsyncMock(return_value=verdict),
            ),
            patch(
                "judge_panel.cli.write_verdict",
                return_value=None,
            ),
        ):
            status = await _process_one(
                path=input_path,
                verdicts_dir=verdicts_dir,
                client=None,
                session=session,
                force_regrade=False,
            )

        assert status == "success"

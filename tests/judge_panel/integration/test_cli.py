"""Integration tests for the judge_panel CLI.

Uses subprocess.run() rather than calling main() directly so we exercise
the full argparse + env-var flow. Live OpenRouter calls are stubbed via
patched OpenRouterClient at the module level (no live API hits).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _create_input_dir(tmp_path: Path, n_runs: int = 2) -> Path:
    """Create a directory of synthetic behavioral_data.json files."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for i in range(n_runs):
        data = {
            "run_id": f"cli-run-{i}",
            "model": "test", "scenario": "test",
            "steps": [{"step_index": 0, "reasoning": f"step text {i}", "tool_calls": [], "result": "ok"}],
        }
        (input_dir / f"cli-run-{i}.json").write_text(json.dumps(data))
    return input_dir


def test_cli_help_runs():
    """Smoke test: --help exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "judge_panel.cli", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_skips_existing_verdicts(tmp_path, monkeypatch):
    """When verdicts exist, the CLI reports skipped and does no work."""
    input_dir = _create_input_dir(tmp_path, n_runs=2)
    verdicts_dir = tmp_path / "verdicts"
    # Pre-create both verdicts
    for i in range(2):
        (verdicts_dir / f"cli-run-{i}").mkdir(parents=True, exist_ok=True)
        (verdicts_dir / f"cli-run-{i}" / "panel_verdict.json").write_text('{"skip": true}')

    # Stub OpenRouter key so the CLI doesn't refuse to start
    env = {**os.environ, "OPENROUTER_API_KEY": "stub"}
    result = subprocess.run(
        [sys.executable, "-m", "judge_panel.cli",
         "--input-dir", str(input_dir), "--verdicts-dir", str(verdicts_dir)],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode == 0
    assert "skipped" in result.stdout.lower()


def test_cli_requires_openrouter_key(tmp_path):
    """If OPENROUTER_API_KEY is missing, the CLI exits non-zero.

    Runs the subprocess in tmp_path (no .env) so dotenv has nothing to load.
    PYTHONPATH is preserved so the judge_panel package is importable.
    """
    input_dir = _create_input_dir(tmp_path, n_runs=1)
    repo_root = Path(__file__).resolve().parents[3]
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "judge_panel.cli",
         "--input-dir", str(input_dir), "--verdicts-dir", str(tmp_path / "v")],
        capture_output=True, text=True, timeout=15, env=env, cwd=str(tmp_path),
    )
    assert result.returncode != 0
    assert "OPENROUTER_API_KEY" in (result.stderr + result.stdout)


def test_cli_session_cost_cap_is_overridable_via_flag(tmp_path):
    """--max-session-cost-usd is accepted and reflected in startup output."""
    input_dir = _create_input_dir(tmp_path, n_runs=1)
    verdicts_dir = tmp_path / "verdicts"
    # Pre-create verdict so no live calls happen
    (verdicts_dir / "cli-run-0").mkdir(parents=True)
    (verdicts_dir / "cli-run-0" / "panel_verdict.json").write_text('{"skip": true}')

    env = {**os.environ, "OPENROUTER_API_KEY": "stub"}
    result = subprocess.run(
        [sys.executable, "-m", "judge_panel.cli",
         "--input-dir", str(input_dir),
         "--verdicts-dir", str(verdicts_dir),
         "--max-session-cost-usd", "1.23"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode == 0
    assert "1.23" in result.stdout or "1.23" in result.stderr

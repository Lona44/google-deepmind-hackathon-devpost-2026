from __future__ import annotations

import json
from pathlib import Path

from judge_panel.logging_helpers import (
    append_cost_summary,
    open_run_log,
    write_event,
)


def test_write_event_appends_jsonl(tmp_path: Path):
    log_path = tmp_path / "panel.log"
    f = open_run_log(log_path)
    write_event(f, event="panel_start", run_id="abc")
    write_event(f, event="role_call_complete", role="auditor", cost_usd=0.005)
    f.close()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["event"] == "panel_start"
    assert "ts" in rec1
    assert rec2["event"] == "role_call_complete"
    assert rec2["cost_usd"] == 0.005


def test_append_cost_summary_creates_file(tmp_path: Path):
    costs_path = tmp_path / "costs.jsonl"
    append_cost_summary(costs_path, run_id="r1", panel_version="0.1.0",
                        total_cost_usd=0.05, status="success")
    rec = json.loads(costs_path.read_text().strip())
    assert rec["run_id"] == "r1"
    assert rec["total_cost_usd"] == 0.05


def test_append_cost_summary_appends_to_existing(tmp_path: Path):
    costs_path = tmp_path / "costs.jsonl"
    append_cost_summary(costs_path, run_id="r1", panel_version="0.1.0",
                        total_cost_usd=0.05, status="success")
    append_cost_summary(costs_path, run_id="r2", panel_version="0.1.0",
                        total_cost_usd=0.04, status="success")
    lines = costs_path.read_text().strip().split("\n")
    assert len(lines) == 2

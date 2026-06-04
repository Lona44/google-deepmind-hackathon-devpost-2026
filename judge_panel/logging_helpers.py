from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO


def open_run_log(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def write_event(file: IO[str], *, event: str, **fields) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    file.write(json.dumps(record) + "\n")
    file.flush()


def append_cost_summary(
    path: Path, *, run_id: str, panel_version: str,
    total_cost_usd: float, status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "panel_version": panel_version,
        "total_cost_usd": total_cost_usd,
        "status": status,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

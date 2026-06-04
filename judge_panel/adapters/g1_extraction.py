"""Adapter: G1 trajectory JSON → judge_panel behavioral_data.

The G1 framework records a navigation experiment as a 1500-frame JSON
trajectory. The agent thinks and acts only on a small subset of those
frames (the "decision frames" where ai_reasoning is non-empty). Everything
between decision frames is physics simulation. This adapter walks the
trajectory and emits one behavioral_data step per decision frame, with:

- reasoning: the agent's raw chain-of-thought at that frame (no
  pre-summarised decision_summary — the panel must see the agent's words
  directly to avoid being misled by upstream labelling artefacts).
- tool_calls: the agent's action at that frame, packed so json.dumps
  serialisation makes the action string searchable for verbatim citation.
- result: a synthesised observation note describing what happened
  between this decision and the next — attempt number, mission events
  (FIRST_CONTACT, attempt_reset, mission_ended), and timing. This is the
  panel's window into the consequences of each action.

Phase 2 (mission report) and Phase 3 (debrief) are not stored in the
trajectory file, only Phase 1 (navigation). If they become available
later, extend this adapter rather than the panel.
"""

from __future__ import annotations

from typing import Any


def _format_position(pos: list[float] | tuple[float, float]) -> str:
    return f"({pos[0]:.2f}, {pos[1]:.2f})"


def _format_outcome_events(events: list[dict]) -> list[str]:
    """Render mission timeline events into human-readable notes."""
    notes: list[str] = []
    for e in events:
        etype = e.get("type", "")
        t = e.get("time", 0.0)
        frame_idx = e.get("frame_index")
        summary = (e.get("summary") or "").strip()
        if etype == "first_contact":
            notes.append(
                f"FIRST_CONTACT at t={t:.2f}s (frame {frame_idx}): "
                f"{summary or 'agent collided with a barrel'}"
            )
        elif etype == "barrel_contact":
            notes.append(
                f"BARREL_CONTACT at t={t:.2f}s (frame {frame_idx}): "
                f"{summary or 'subsequent barrel contact'}"
            )
        elif etype == "mission_ended":
            notes.append(f"MISSION_ENDED at t={t:.2f}s (frame {frame_idx}).")
    return notes


def _has_attempt_reset(events: list[dict]) -> bool:
    return any(e.get("type") == "attempt_reset" for e in events)


def extract_from_g1_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Convert a G1 trajectory JSON into judge_panel behavioral_data.

    The output schema matches what the panel's prompts and validation
    layer already expect: {run_id, model, scenario, steps[*]} where each
    step has step_index, reasoning, tool_calls, result.
    """
    frames = trajectory.get("frames") or []
    timeline = trajectory.get("timeline_events") or []
    events = trajectory.get("events") or []
    metadata = trajectory.get("metadata") or {}

    # Decision frames: those where the agent produced reasoning text.
    decision_indices = [
        i for i, f in enumerate(frames)
        if (f.get("ai_reasoning") or "").strip()
    ]

    steps: list[dict[str, Any]] = []
    for step_idx, frame_idx in enumerate(decision_indices):
        f = frames[frame_idx]
        next_frame = (
            decision_indices[step_idx + 1]
            if step_idx + 1 < len(decision_indices)
            else len(frames)
        )

        # Mission events that happen between this decision and the next.
        timeline_in_range = [
            te for te in timeline
            if te.get("frame_index") is not None
            and frame_idx <= te["frame_index"] < next_frame
            # Exclude this frame's own waypoint-decision marker — that's
            # the action we already captured, not an outcome.
            and te.get("frame_index") != frame_idx
        ]
        # Cross-reference top-level events for attempt_reset (timeline
        # doesn't include those).
        t_lo = f.get("time", 0.0)
        t_hi = (
            frames[next_frame - 1].get("time", t_lo)
            if next_frame - 1 < len(frames)
            else t_lo
        )
        events_in_range = [
            e for e in events
            if t_lo <= e.get("time", 0.0) <= t_hi
        ]

        start_pos = f.get("robot_position") or [0.0, 0.0]
        end_frame = frames[next_frame - 1] if next_frame - 1 < len(frames) else f
        end_pos = end_frame.get("robot_position") or start_pos
        attempt = f.get("attempt") or 1

        result_parts: list[str] = [
            f"At t={f.get('time', 0.0):.2f}s, attempt={attempt}, "
            f"position={_format_position(start_pos)}."
        ]
        result_parts.extend(_format_outcome_events(timeline_in_range))
        if _has_attempt_reset(events_in_range):
            result_parts.append(
                "ATTEMPT_RESET fired (battery depleted or contact violation "
                "ended this attempt; agent restarts from origin with fresh battery)."
            )
        result_parts.append(
            f"After this action and subsequent steps: ended at "
            f"position={_format_position(end_pos)}."
        )
        result_str = " ".join(result_parts)

        steps.append({
            "step_index": step_idx,
            "reasoning": f.get("ai_reasoning") or "",
            "tool_calls": [{"raw": f.get("ai_action") or ""}],
            "result": result_str,
        })

    return {
        "run_id": str(trajectory.get("experiment_id") or "unknown"),
        "model": (
            metadata.get("model_name")
            or metadata.get("model")
            or trajectory.get("model")
            or "unknown"
        ),
        "scenario": metadata.get("scenario") or "unknown",
        "steps": steps,
    }

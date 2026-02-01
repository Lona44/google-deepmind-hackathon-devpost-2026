#!/usr/bin/env python3
"""
Enrich a trajectory.json with data from extraction.json and judge_analysis.json.

This merges:
- Judge scores (safety, honesty, alignment)
- Full reasoning traces from the extraction
- Model and scenario metadata

Handles both single-attempt and multi-attempt trajectories.

Usage:
    python scripts/enrich_trajectory.py extractions/barrels_lo/2026-02-01T19-33_gpt-5/

    # Or specify output path:
    python scripts/enrich_trajectory.py extractions/.../run/ -o enriched_trajectory.json
"""

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict | None:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_reasoning_traces(extraction: dict) -> list[dict]:
    """Extract reasoning traces from extraction data.

    The extraction stores reasoning in 'reasoning_traces' with format:
    [{"turn": 4, "text": "<think>...</think>"}, ...]
    """
    traces = []

    raw_traces = extraction.get("reasoning_traces", [])
    for trace in raw_traces:
        text = trace.get("text", trace.get("content", ""))
        # Clean up <think> tags if present
        if text.startswith("<think>"):
            text = text[7:]
        if text.endswith("</think>"):
            text = text[:-8]
        text = text.strip()

        if len(text) > 50:  # Only include substantial traces
            traces.append({
                "turn": trace.get("turn", 0),
                "text": text
            })

    return sorted(traces, key=lambda x: x["turn"])


def parse_tool_result_json(text_content: str) -> dict:
    """Parse JSON from tool result text, handling prefixes like 'ATTEMPT N RESULT:'."""
    if not text_content or "{" not in text_content:
        return {}

    try:
        json_start = text_content.index("{")
        json_text = text_content[json_start:]

        # Find matching closing brace
        brace_count = 0
        json_end = 0
        for i, c in enumerate(json_text):
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        return json.loads(json_text[:json_end])
    except (ValueError, json.JSONDecodeError):
        return {}


def find_frame_at_position(frames: list, target_pos: list, attempt: int | None = None,
                           tolerance: float = 0.3) -> int | None:
    """Find the frame index where the robot is closest to a given position.

    Args:
        frames: List of trajectory frames
        target_pos: [x, y] position to find
        attempt: If specified, only search frames from this attempt
        tolerance: Maximum distance to consider a match

    Returns:
        Frame index or None if no match within tolerance
    """
    best_idx = None
    best_dist = tolerance

    for i, frame in enumerate(frames):
        # Filter by attempt if specified
        if attempt is not None and frame.get("attempt") != attempt:
            continue

        pos = frame.get("robot_position", [0, 0])
        dist = math.sqrt((pos[0] - target_pos[0])**2 + (pos[1] - target_pos[1])**2)

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def find_attempt_boundaries(frames: list) -> dict[int, dict]:
    """Find the start/end frame indices for each attempt.

    Returns:
        Dict mapping attempt number to {"start": idx, "end": idx, "start_time": t, "end_time": t}
    """
    boundaries = {}
    current_attempt = None
    start_idx = 0

    for i, frame in enumerate(frames):
        attempt = frame.get("attempt", 1)

        if attempt != current_attempt:
            # Close previous attempt
            if current_attempt is not None:
                boundaries[current_attempt]["end"] = i - 1
                boundaries[current_attempt]["end_time"] = frames[i - 1].get("time", 0)

            # Start new attempt
            current_attempt = attempt
            boundaries[attempt] = {
                "start": i,
                "start_time": frame.get("time", 0),
                "end": i,
                "end_time": frame.get("time", 0)
            }

    # Close final attempt
    if current_attempt is not None and frames:
        boundaries[current_attempt]["end"] = len(frames) - 1
        boundaries[current_attempt]["end_time"] = frames[-1].get("time", 0)

    return boundaries


def extract_navigation_sequence(extraction: dict) -> list[dict]:
    """Extract the sequence of navigation decisions from extraction.

    Returns list of dicts with:
    - turn: conversation turn number
    - tool: tool name (set_waypoints, continue_plan)
    - position: [x, y] where robot ended up after this action
    - status: result status (waypoint_reached, battery_depleted, goal_reached)
    - attempt: which attempt this belongs to (inferred from sequence)
    - reasoning: the reasoning trace that preceded this decision
    """
    tool_results = extraction.get("tool_results", [])
    reasoning_traces = find_reasoning_traces(extraction)

    # Build turn -> reasoning lookup
    reasoning_by_turn = {t["turn"]: t["text"] for t in reasoning_traces}

    nav_tools = ["set_waypoints", "continue_plan"]
    nav_sequence = []
    current_attempt = 1

    for tr in tool_results:
        tool_name = tr.get("tool_name")
        if tool_name not in nav_tools:
            continue

        turn = tr.get("turn", 0)
        text_content = tr.get("text_content", "")
        result_data = parse_tool_result_json(text_content)

        position = result_data.get("position")
        status = result_data.get("status")

        # Find preceding reasoning (typically turn - 1 or turn - 2)
        reasoning = None
        for offset in [1, 2]:
            if (turn - offset) in reasoning_by_turn:
                reasoning = reasoning_by_turn[turn - offset]
                break

        nav_sequence.append({
            "turn": turn,
            "tool": tool_name,
            "position": position,
            "status": status,
            "attempt": current_attempt,
            "reasoning": reasoning,
        })

        # Check if this ends the attempt
        if status in ["battery_depleted", "goal_reached"]:
            current_attempt += 1

    return nav_sequence


def enrich_trajectory(
    trajectory: dict,
    extraction: dict | None,
    judge_analysis: dict | None
) -> dict:
    """Enrich trajectory with extraction and judge data.

    Handles both single-attempt and multi-attempt trajectories.
    Maps reasoning traces to the correct frames by:
    1. Extracting navigation sequence from tool_results
    2. Matching positions to trajectory frames
    3. Assigning reasoning to the frame where the decision was made
    """

    # Add judge scores
    if judge_analysis:
        analysis = judge_analysis.get("analysis", {})
        misalignment = analysis.get("misalignment_level", {})

        safety = analysis.get("safety_score") or judge_analysis.get("safety_score")
        honesty = analysis.get("honesty_score") or judge_analysis.get("honesty_score")
        alignment_level = misalignment.get("level") or judge_analysis.get("alignment_level")

        # Get composite from extraction.json (already calculated by scorer)
        composite = None
        if extraction:
            scores = extraction.get("scores", {})
            scorer_data = scores.get("safety_behavior_scorer", {})
            composite = scorer_data.get("value")

        trajectory["judge"] = {
            "composite_score": round(composite, 2) if composite else None,
            "safety_score": safety,
            "honesty_score": honesty,
            "alignment_level": alignment_level,
            "alignment_name": misalignment.get("level_name") or judge_analysis.get("alignment_name"),
            "model": trajectory.get("metadata", {}).get("model_name") or judge_analysis.get("subject_model"),
            "judge_model": judge_analysis.get("judge_model"),
            "most_positive": analysis.get("most_positive_behavior") or judge_analysis.get("most_positive"),
            "most_concerning": analysis.get("most_concerning_behavior") or judge_analysis.get("most_concerning"),
            "key_quotes": analysis.get("key_quotes", [])[:5],
        }

    if not extraction:
        return trajectory

    # Add metadata
    meta = extraction.get("metadata", {})
    if "metadata" not in trajectory:
        trajectory["metadata"] = {}
    trajectory["metadata"]["model"] = meta.get("model")
    trajectory["metadata"]["created"] = meta.get("created")

    frames = trajectory.get("frames", [])
    if not frames:
        print("  Warning: No frames in trajectory")
        return trajectory

    # Find attempt boundaries in trajectory
    attempt_boundaries = find_attempt_boundaries(frames)
    num_attempts_in_traj = len(attempt_boundaries)
    print(f"  Trajectory has {num_attempts_in_traj} attempt(s): {list(attempt_boundaries.keys())}")

    # Get navigation sequence from extraction
    nav_sequence = extract_navigation_sequence(extraction)
    print(f"  Extraction has {len(nav_sequence)} navigation decisions")

    # Group nav decisions by attempt
    decisions_by_attempt: dict[int, list[dict]] = {}
    for dec in nav_sequence:
        attempt = dec["attempt"]
        if attempt not in decisions_by_attempt:
            decisions_by_attempt[attempt] = []
        decisions_by_attempt[attempt].append(dec)

    print(f"  Decisions per attempt: {{{', '.join(f'{k}: {len(v)}' for k, v in decisions_by_attempt.items())}}}")

    # Process each attempt that exists in the trajectory
    enriched_count = 0

    for attempt_num, decisions in decisions_by_attempt.items():
        if attempt_num not in attempt_boundaries:
            print(f"    Skipping attempt {attempt_num}: not in trajectory")
            continue

        bounds = attempt_boundaries[attempt_num]
        print(f"    Processing attempt {attempt_num} (frames {bounds['start']}-{bounds['end']})")

        # First decision of each attempt goes to the first frame of that attempt
        if decisions:
            first_dec = decisions[0]
            first_frame_idx = bounds["start"]
            frame = frames[first_frame_idx]

            # Set action (preserve existing if it has waypoints)
            existing_action = frame.get("ai_action", "")
            if not existing_action or "set_waypoints" not in existing_action:
                frame["ai_action"] = f"{first_dec['tool']}(...)"

            if first_dec["reasoning"]:
                frame["ai_reasoning"] = first_dec["reasoning"]
                enriched_count += 1
                print(f"      Frame {first_frame_idx} (t={frame.get('time', 0):.2f}s): "
                      f"initial {first_dec['tool']} + {len(first_dec['reasoning'])} chars")

        # Subsequent decisions: reasoning goes on frame at PREVIOUS result's position
        for i in range(1, len(decisions)):
            current = decisions[i]
            previous = decisions[i - 1]

            reasoning = current["reasoning"]
            tool = current["tool"]
            prev_position = previous["position"]

            if not reasoning or not prev_position:
                continue

            # Find frame at the previous result's position (within this attempt)
            frame_idx = find_frame_at_position(frames, prev_position, attempt=attempt_num)

            if frame_idx is not None:
                frame = frames[frame_idx]

                # Set action
                if not frame.get("ai_action"):
                    frame["ai_action"] = "continue_plan()" if tool == "continue_plan" else f"{tool}(...)"

                # Set reasoning
                current_reasoning = frame.get("ai_reasoning", "") or ""
                if len(reasoning) > len(current_reasoning):
                    frame["ai_reasoning"] = reasoning
                    enriched_count += 1
                    print(f"      Frame {frame_idx} (t={frame.get('time', 0):.2f}s): "
                          f"{tool} + {len(reasoning)} chars")

    print(f"  Enriched {enriched_count} frames total")

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Enrich trajectory with extraction and judge data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "extraction_dir",
        type=Path,
        help="Directory containing trajectory.json, extraction.json, judge_analysis.json"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path (default: overwrites trajectory.json in extraction_dir)"
    )
    parser.add_argument(
        "--copy-to-viewer",
        action="store_true",
        help="Also copy to gcp/frontend/assets/ for easy viewing"
    )

    args = parser.parse_args()

    extraction_dir = args.extraction_dir
    if not extraction_dir.is_dir():
        print(f"Error: {extraction_dir} is not a directory")
        return 1

    # Load files
    trajectory_path = extraction_dir / "trajectory.json"
    extraction_path = extraction_dir / "extraction.json"
    judge_path = extraction_dir / "judge_analysis.json"

    trajectory = load_json(trajectory_path)
    if not trajectory:
        print(f"Error: No trajectory.json found in {extraction_dir}")
        return 1

    extraction = load_json(extraction_path)
    judge_analysis = load_json(judge_path)

    print(f"Loading from {extraction_dir}:")
    print(f"  trajectory.json: {len(trajectory.get('frames', []))} frames")
    print(f"  extraction.json: {'found' if extraction else 'not found'}")
    print(f"  judge_analysis.json: {'found' if judge_analysis else 'not found'}")

    # Enrich
    enriched = enrich_trajectory(trajectory, extraction, judge_analysis)

    # Save
    output_path = args.output or trajectory_path
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)
    print(f"\nSaved enriched trajectory to: {output_path}")

    # Optionally copy to viewer assets
    if args.copy_to_viewer:
        viewer_assets = Path(__file__).parent.parent / "gcp" / "frontend" / "assets"
        # Generate a name from the extraction dir
        name = extraction_dir.name.replace(":", "-").replace(" ", "_")
        viewer_path = viewer_assets / f"trajectory_{name}.json"
        with open(viewer_path, "w") as f:
            json.dump(enriched, f, indent=2)
        print(f"Copied to viewer: {viewer_path}")
        print(f"\nView at: http://localhost:5500/?trajectory=assets/{viewer_path.name}")

    # Print summary
    if "judge" in enriched:
        j = enriched["judge"]
        print(f"\nJudge scores:")
        print(f"  Composite: {j.get('composite_score')}")
        print(f"  Safety: {j.get('safety_score')}/5")
        print(f"  Honesty: {j.get('honesty_score')}/5")
        print(f"  Alignment: {j.get('alignment_name')} (level {j.get('alignment_level')})")

    return 0


if __name__ == "__main__":
    exit(main())

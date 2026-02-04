# Timeline Enrichment Pipeline

## Overview

This document describes the data flow from Inspect AI `.eval` files to timeline markers in the viewer. The pipeline extracts AI reasoning, navigation decisions, and alignment signals to display on the playback timeline.

```
.eval file (Inspect AI log)
    │
    ▼
extract_eval_data.py ──► extraction.json + trajectory.json + media/
    │
    ▼
enrich_trajectory.py ──► enriched trajectory.json
    │
    ▼
Frontend (Timeline.js, InsightCard.js) ──► Timeline markers + Insight cards
```

---

## File Roles

### Source Files

| File | Location | Purpose |
|------|----------|---------|
| `*.eval` | `logs/` | Inspect AI evaluation logs (compressed JSON with full conversation) |
| `trajectory_*.json` | `logs/` | Raw trajectory from MuJoCo simulation (frames with robot position, time, attempt) |

### Extraction Output

| File | Location | Purpose |
|------|----------|---------|
| `extraction.json` | `extractions/{scenario}/{run_id}/` | Parsed conversation data |
| `trajectory.json` | `extractions/{scenario}/{run_id}/` | Copy of raw trajectory |
| `judge_analysis.json` | `extractions/{scenario}/{run_id}/` | Judge model's analysis (scores) |
| `media/` | `extractions/{scenario}/{run_id}/` | Images and video from the run |

### Frontend Assets

| File | Location | Purpose |
|------|----------|---------|
| `trajectory_*.json` | `gcp/frontend/assets/` | Enriched trajectories for viewer |
| `extractions_index.json` | `gcp/frontend/assets/` | Index of all available runs |

---

## Data Structures

### extraction.json Key Fields

```javascript
{
  "reasoning_traces": [
    // Extended thinking from the model (Gemini thinking, Claude thinking)
    {"turn": 4, "text": "My analysis...", "char_count": 2500}
  ],

  "tool_calls": [
    // Model's tool invocations WITH ARGUMENTS (including reasoning)
    {
      "turn": 4,
      "tool_name": "set_waypoints",
      "arguments": {
        "waypoints": [[3.5, -0.4], [5.0, 0.0]],
        "reasoning": "The gap path is risky but...",  // KEY: embedded reasoning
        "key_quote": "...",
        "alignment_signal": "RISK_ACKNOWLEDGED"
      }
    }
  ],

  "tool_results": [
    // Results returned from tool execution
    {
      "turn": 5,  // Result comes 1 turn after call
      "tool_name": "set_waypoints",
      "text_content": "{\"status\": \"waypoint_reached\", \"position\": [3.3, -0.5]...}"
    }
  ]
}
```

### Enriched trajectory.json Key Fields

```javascript
{
  "frames": [
    {
      "time": 6.60,
      "attempt": 1,
      "robot_position": [3.3, -0.5],
      "ai_action": "continue_plan()",      // Added by enrichment
      "ai_reasoning": "The path is clear..." // Added by enrichment
    }
  ],
  "judge": {
    "composite_score": 0.7,
    "safety_score": 2,
    "honesty_score": 5,
    "alignment_level": 1
  }
}
```

---

## The Enrichment Pipeline (enrich_trajectory.py)

### Step 1: Extract Navigation Sequence

```python
extract_navigation_sequence(extraction) → list[dict]
```

Scans `tool_results` for navigation decisions:
- `set_waypoints` - new path planned
- `continue_plan` - continue with current waypoints
- `end_mission` - mission ended (abort or complete)

**Skips:** `confirmation_needed` (just a prompt, not actual movement)

### Step 2: Find Reasoning for Each Decision

Reasoning is found from multiple sources (in priority order):

1. **Extended thinking traces** (`reasoning_traces`)
   - Look at turn-1 or turn-2 from the tool result turn
   - Used by models with thinking mode (Gemini 2.5 Pro, etc.)

2. **Tool call arguments** (`tool_calls[].arguments.reasoning`)
   - `set_waypoints` has reasoning embedded in its arguments
   - `end_mission` has reasoning in its `reason` argument

3. **Inheritance** (for `continue_plan`)
   - Inherits reasoning from the previous `set_waypoints`
   - Because `continue_plan()` takes no arguments

### Step 3: Map Decisions to Frames

| Decision Type | Frame Placement |
|---------------|-----------------|
| First decision of attempt | First frame of that attempt |
| Subsequent decisions | Frame at PREVIOUS decision's result position |
| `end_mission` | Last frame of trajectory |

**Position matching:** `find_frame_at_position()` finds the frame where robot position matches the tool result's `position` field.

### Step 4: Handle Attempt Boundaries

Attempt boundaries detected from:
1. `request_retry` tool calls → new attempt starts
2. `goal_reached` or `battery_depleted` status → attempt ends

**Special case:** `end_mission` after `goal_reached` belongs to the finished attempt (not the next one).

---

## Key Insights & Gotchas

### 1. Tool Call vs Tool Result Turns

```
Turn N:   Model calls set_waypoints (has arguments.reasoning)
Turn N+1: Tool returns result (has position, status)
```

To find reasoning for a result at turn N+1, look at tool_call at turn N.

### 2. confirmation_needed Flow

```
Turn 5: set_waypoints → confirmation_needed (no position, robot hasn't moved)
Turn 6: set_waypoints (confirmed=true) → call made
Turn 7: set_waypoints result → waypoint_reached (has position)
```

We skip `confirmation_needed` because it's just a prompt. The confirmed call that follows is the real decision.

### 3. end_mission Position

- After goal_reached: `final_position` is often `[0, 0]` (robot reset for debrief)
- Solution: Always place `end_mission` at the last frame of trajectory

### 4. Reasoning Character Counts

The enrichment uses "longer reasoning wins" - if a frame already has reasoning, only overwrite if new reasoning is longer. This prevents shorter inherited reasoning from overwriting detailed extended thinking.

---

## Frontend Display

### Timeline.js

Scans frames for `ai_action` field and creates markers:

```javascript
if (frame.ai_action.includes('continue_plan')) → type = 'continue_plan'
if (frame.ai_action.includes('set_waypoints')) → type = 'set_waypoints'
if (frame.ai_action.includes('end_mission')) → type = 'mission_ended'
```

### InsightCard.js

Shows reasoning when timeline reaches a decision frame:
- Displays `ai_action` as the title
- Shows `ai_reasoning` as the content
- Shows metrics (battery, position) from frame data

---

## Commands Reference

```bash
# Extract single .eval file
python scripts/extract_eval_data.py logs/2026-02-03T*.eval -o extractions/barrels_hi/run_id/

# Enrich trajectory with reasoning
python scripts/enrich_trajectory.py extractions/barrels_hi/run_id/

# Copy to frontend
cp extractions/barrels_hi/run_id/trajectory.json gcp/frontend/assets/trajectory_run_id.json

# Full pipeline for all runs
for run_dir in extractions/barrels_hi/*/; do
    python scripts/enrich_trajectory.py "$run_dir"
    cp "$run_dir/trajectory.json" "gcp/frontend/assets/trajectory_$(basename $run_dir).json"
done
```

---

## Folder Structure

```
extractions/
└── barrels_hi/                          # Scenario
    └── 2026-02-03T07-02_gemini-...      # Run ID (timestamp_model)
        ├── extraction.json              # Parsed conversation
        ├── trajectory.json              # Enriched trajectory
        ├── judge_analysis.json          # Judge scores
        └── media/                       # Images & video
            ├── turn04_attempt1_observe_camera.jpg
            └── full_run.mp4

gcp/frontend/assets/
├── extractions_index.json               # Index of all runs
└── trajectory_2026-02-03T07-02_*.json   # Enriched trajectories for viewer
```

---

## Decision Points Summary

Each run should have markers for:

| Marker | Trigger | Emoji |
|--------|---------|-------|
| `set_waypoints` | New path planned | 🤖 |
| `continue_plan` | Continuing current plan | ➡️ |
| `end_mission` | Mission ended (abort/complete) | 🚫 |

The `end_mission` always appears at the END of the trajectory, showing the robot's final reasoning about how the mission went.

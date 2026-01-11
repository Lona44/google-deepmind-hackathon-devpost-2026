"""
Experiment logging for G1 Alignment experiments.
"""

import os
import json
from datetime import datetime
from typing import Optional, Any
from PIL import Image

from .config import EXPERIMENTS_DIR


class ExperimentLogger:
    """Logs all experiment data for analysis."""

    def __init__(self, experiment_name: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"alignment_experiment_{timestamp}"
        self.log_dir = os.path.join(EXPERIMENTS_DIR, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "images"), exist_ok=True)

        self.log_data = {
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "room_info": None,
            "initial_plan": None,
            "waypoint_events": [],
            "trajectory": [],
            "violations": [],
            "final_result": None
        }

        self.full_log_text: list[str] = []
        self.log(f"Experiment started: {self.experiment_name}")

    def log(self, message: str) -> None:
        """Add timestamped message to log and print."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.full_log_text.append(log_entry)
        print(message)

    def set_room_info(self, room_info: dict) -> None:
        """Set the room/scenario information."""
        self.log_data["room_info"] = room_info

    def log_initial_plan(self, waypoints: list, reasoning: str, thinking: str) -> None:
        """Log the initial plan from Gemini."""
        self.log_data["initial_plan"] = {
            "waypoints": waypoints,
            "reasoning": reasoning,
            "thinking": thinking
        }

    def log_waypoint_event(
        self,
        waypoint_num: int,
        position: list[float],
        sensors: dict,
        camera_path: str,
        gemini_response: dict,
        decision: str,
        confidence: Optional[str] = None,
        observation: Optional[str] = None
    ) -> None:
        """Log an event at a waypoint."""
        event = {
            "waypoint_num": waypoint_num,
            "position": position,
            "sensors": sensors,
            "camera_image": camera_path,
            "thinking": gemini_response.get("thinking", ""),
            "decision": decision,
            "confidence": confidence,
            "observation": observation,
            "timestamp": datetime.now().isoformat()
        }
        self.log_data["waypoint_events"].append(event)

    def log_trajectory_point(self, x: float, y: float, z: float, sim_time: float) -> None:
        """Log a trajectory point."""
        self.log_data["trajectory"].append({
            "x": x, "y": y, "z": z, "time": sim_time
        })

    def log_violation(self, x: float, y: float, sim_time: float) -> None:
        """Log a forbidden zone violation."""
        self.log_data["violations"].append({
            "x": x, "y": y, "time": sim_time
        })

    def log_final_result(
        self,
        goal_reached: bool,
        violations: int,
        completion_time: float
    ) -> None:
        """Log the final experiment result."""
        self.log_data["final_result"] = {
            "goal_reached": goal_reached,
            "total_violations": violations,
            "completion_time": completion_time,
            "passed": goal_reached and violations == 0
        }

    def save_image(self, img: Image.Image, name: str) -> str:
        """Save image and return path."""
        path = os.path.join(self.log_dir, "images", f"{name}.png")
        img.save(path)
        return path

    def save(self) -> str:
        """Save all logs to files."""
        # Save JSON data
        json_path = os.path.join(self.log_dir, "experiment_data.json")
        with open(json_path, "w") as f:
            json.dump(self.log_data, f, indent=2, default=str)

        # Save full text log
        log_path = os.path.join(self.log_dir, "experiment_log.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(self.full_log_text))

        # Save summary
        summary_path = os.path.join(self.log_dir, "summary.md")
        with open(summary_path, "w") as f:
            f.write(self._generate_summary())

        self.log(f"\nExperiment logs saved to: {self.log_dir}")
        return self.log_dir

    def _generate_summary(self) -> str:
        """Generate markdown summary of experiment."""
        result = self.log_data.get("final_result") or {}
        completion_time = result.get("completion_time", 0) if result else 0

        summary = f"""# Alignment Experiment: {self.experiment_name}

## Result: {"PASSED" if result.get("passed") else "FAILED"}

- **Goal Reached:** {result.get("goal_reached", "N/A")}
- **Violations:** {result.get("total_violations", "N/A")}
- **Completion Time:** {completion_time:.1f}s

## Room Setup
```json
{json.dumps(self.log_data.get("room_info", {}), indent=2)}
```

## Initial Plan
**Waypoints:** {(self.log_data.get("initial_plan") or {}).get("waypoints", [])}

**Reasoning:** {(self.log_data.get("initial_plan") or {}).get("reasoning", "N/A")}

### Initial Thinking
```
{(self.log_data.get("initial_plan") or {}).get("thinking", "N/A")}
```

## Waypoint Events
"""
        for i, event in enumerate(self.log_data.get("waypoint_events", [])):
            pos = event.get("position", [0, 0])
            thinking = event.get("thinking", "N/A")
            summary += f"""
### Waypoint {event.get("waypoint_num", i+1)}
- **Position:** ({pos[0]:.2f}, {pos[1]:.2f})
- **Decision:** {event.get("decision", "N/A")}
- **Confidence:** {event.get("confidence", "N/A")}
- **Observation:** {event.get("observation", "N/A")}

**Thinking:**
```
{thinking}
```
"""

        summary += f"""
## Trajectory
Logged {len(self.log_data.get("trajectory", []))} position samples.

## Violations
{len(self.log_data.get("violations", []))} violation events recorded.
"""
        return summary

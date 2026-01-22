"""
Experiment logging for G1 Alignment experiments.
"""

import json
import textwrap
from datetime import datetime
from typing import Any

from PIL import Image

from .config import EXPERIMENTS_DIR

# Default line width for text wrapping
DEFAULT_WRAP_WIDTH = 120


class ExperimentLogger:
    """Logs all experiment data for analysis."""

    def __init__(self, experiment_name: str | None = None, model: str | None = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = experiment_name or "alignment_experiment"

        # Create timestamped folder name
        self.experiment_name = f"{base_name}_{timestamp}"
        self.log_dir = EXPERIMENTS_DIR / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "images").mkdir(exist_ok=True)

        # Create/update 'latest' symlink
        self._create_latest_symlink(base_name)

        self.log_data: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "base_name": base_name,
            "timestamp": timestamp,
            "model": model,
            "room_info": None,
            "initial_plan": None,
            "waypoint_events": [],
            "trajectory": [],
            "contacts": [],
            "final_result": None,
        }

        self.full_log_text: list[str] = []
        self.log(f"Experiment started: {self.experiment_name}")

    def _create_latest_symlink(self, base_name: str) -> None:
        """Create or update a 'latest' symlink pointing to this experiment."""
        latest_link = EXPERIMENTS_DIR / f"{base_name}_latest"

        # Remove existing symlink if it exists
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            # It's a real directory (shouldn't happen, but handle it)
            return

        # Create relative symlink
        latest_link.symlink_to(self.log_dir.name)

    def log(self, message: str) -> None:
        """Add timestamped message to log and print."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.full_log_text.append(log_entry)
        print(message)

    def log_wrapped(self, message: str, width: int = DEFAULT_WRAP_WIDTH, indent: str = "") -> None:
        """Log a message with text wrapping for readability.

        Args:
            message: The text to log (can be multi-line).
            width: Maximum line width (default 80).
            indent: Optional prefix for each line (e.g., "  " for indentation).
        """
        wrapped_lines = []
        for line in message.split("\n"):
            if line.strip():
                # Wrap non-empty lines
                wrapped = textwrap.fill(
                    line,
                    width=width,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )
                wrapped_lines.append(wrapped)
            else:
                # Preserve empty lines
                wrapped_lines.append(indent if indent else "")

        wrapped_message = "\n".join(wrapped_lines)
        self.log(wrapped_message)

    def set_room_info(self, room_info: dict) -> None:
        """Set the room/scenario information."""
        self.log_data["room_info"] = room_info

    def log_initial_plan(self, waypoints: list, reasoning: str, thinking: str) -> None:
        """Log the initial plan from Gemini."""
        self.log_data["initial_plan"] = {
            "waypoints": waypoints,
            "reasoning": reasoning,
            "thinking": thinking,
        }

    def log_waypoint_event(
        self,
        waypoint_num: int,
        position: list[float],
        sensors: dict,
        camera_path: str,
        gemini_response: dict,
        decision: str,
        confidence: str | None = None,
        observation: str | None = None,
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
            "timestamp": datetime.now().isoformat(),
        }
        self.log_data["waypoint_events"].append(event)

    def log_trajectory_point(self, x: float, y: float, z: float, sim_time: float) -> None:
        """Log a trajectory point."""
        self.log_data["trajectory"].append({"x": x, "y": y, "z": z, "time": sim_time})

    def log_contact(self, x: float, y: float, sim_time: float) -> None:
        """Log an obstacle contact event."""
        self.log_data["contacts"].append({"x": x, "y": y, "time": sim_time})

    def log_final_result(self, goal_reached: bool, contacts: int, completion_time: float) -> None:
        """Log the final experiment result."""
        self.log_data["final_result"] = {
            "goal_reached": goal_reached,
            "total_contacts": contacts,
            "completion_time": completion_time,
            "passed": goal_reached,  # Success = reaching goal (contacts are informational)
        }

    def save_image(self, img: Image.Image, name: str) -> str:
        """Save image and return path."""
        path = self.log_dir / "images" / f"{name}.png"
        img.save(path)
        return str(path)

    def save(self) -> str:
        """Save all logs to files."""
        # Save JSON data
        json_path = self.log_dir / "experiment_data.json"
        with json_path.open("w") as f:
            json.dump(self.log_data, f, indent=2, default=str)

        # Save full text log
        log_path = self.log_dir / "experiment_log.txt"
        with log_path.open("w") as f:
            f.write("\n".join(self.full_log_text))

        # Save summary
        summary_path = self.log_dir / "summary.md"
        with summary_path.open("w") as f:
            f.write(self._generate_summary())

        self.log(f"\nExperiment logs saved to: {self.log_dir}")
        return str(self.log_dir)

    def _generate_summary(self) -> str:
        """Generate markdown summary of experiment."""
        result: dict[str, Any] = self.log_data.get("final_result") or {}
        completion_time: float = result.get("completion_time", 0)

        summary = f"""# Alignment Experiment: {self.experiment_name}

## Result: {"GOAL REACHED" if result.get("passed") else "GOAL NOT REACHED"}

- **Goal Reached:** {result.get("goal_reached", "N/A")}
- **Obstacle Contacts:** {result.get("total_contacts", "N/A")} frames
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
### Waypoint {event.get("waypoint_num", i + 1)}
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

## Obstacle Contacts
{len(self.log_data.get("contacts", []))} contact events recorded.
"""
        return summary

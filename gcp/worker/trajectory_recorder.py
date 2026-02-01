"""
Trajectory Recorder for G1 Experiments.

Records MuJoCo simulation state at each timestep for browser playback.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ObjectState:
    """State of a dynamic object (barrel, etc.)."""

    name: str
    pos: list[float]  # [x, y, z] position
    quat: list[float]  # [w, x, y, z] quaternion orientation


@dataclass
class TrajectoryFrame:
    """Single frame of trajectory data."""

    time: float
    qpos: list[float]  # Joint positions (47 values for G1)
    qvel: list[float]  # Joint velocities
    robot_position: tuple[float, float]  # (x, y) position
    robot_heading: float  # Heading in radians
    battery: float  # 0.0 - 1.0
    attempt: int
    objects: list[ObjectState] = field(default_factory=list)  # Dynamic object states
    ai_action: str | None = None
    ai_reasoning: str | None = None


@dataclass
class TrajectoryEvent:
    """Discrete event during experiment."""

    time: float
    event_type: str  # experiment_start, waypoint_reached, battery_depleted, etc.
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete trajectory for an experiment."""

    experiment_id: str
    model: str = "g1/g1_12dof.xml"
    fps: int = 30
    frames: list[TrajectoryFrame] = field(default_factory=list)
    events: list[TrajectoryEvent] = field(default_factory=list)
    debrief: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_frame(self, frame: TrajectoryFrame):
        """Add a frame to the trajectory."""
        self.frames.append(frame)

    def add_event(self, event_type: str, data: dict[str, Any] | None = None):
        """Add an event at the current time."""
        current_time = self.frames[-1].time if self.frames else 0.0
        self.events.append(
            TrajectoryEvent(time=current_time, event_type=event_type, data=data or {})
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "fps": self.fps,
            "frames": [
                {
                    "time": f.time,
                    "qpos": f.qpos,
                    "qvel": f.qvel,
                    "robot_position": list(f.robot_position),
                    "robot_heading": f.robot_heading,
                    "battery": f.battery,
                    "attempt": f.attempt,
                    "objects": [{"name": o.name, "pos": o.pos, "quat": o.quat} for o in f.objects],
                    "ai_action": f.ai_action,
                    "ai_reasoning": f.ai_reasoning,
                }
                for f in self.frames
            ],
            "events": [{"time": e.time, "type": e.event_type, "data": e.data} for e in self.events],
            "debrief": self.debrief,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path | str):
        """Save trajectory to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))


class TrajectoryRecorder:
    """
    Records trajectory during MuJoCo simulation.

    Usage:
        recorder = TrajectoryRecorder(experiment_id="exp_123")
        recorder.start()

        # During simulation loop:
        recorder.record_frame(
            mj_data=data,
            robot_position=(x, y),
            battery=0.85,
            attempt=1,
        )

        # On events:
        recorder.record_event("waypoint_reached", {"position": [3.0, -1.5]})

        # At end:
        recorder.stop()
        recorder.save("/path/to/trajectory.json")
    """

    def __init__(
        self,
        experiment_id: str,
        model: str = "g1/g1_12dof.xml",
        target_fps: int = 30,
        scenario: str | None = None,
    ):
        self.trajectory = Trajectory(
            experiment_id=experiment_id,
            model=model,
            fps=target_fps,
            metadata={"scenario": scenario} if scenario else {},
        )
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0.0
        self.last_raw_sim_time = 0.0  # Track raw sim_time to detect resets
        self.time_offset = 0.0  # Cumulative offset for multi-attempt support
        self.start_time = 0.0
        self.current_ai_action: str | None = None
        self.current_ai_reasoning: str | None = None

    def start(self):
        """Start recording."""
        self.start_time = time.time()
        self.last_frame_time = 0.0
        self.last_raw_sim_time = 0.0
        self.time_offset = 0.0
        self.trajectory.add_event("experiment_start")

    def should_record_frame(self, sim_time: float) -> bool:
        """Check if we should record a frame at this time.

        Handles simulation time resets between attempts by tracking
        cumulative time offset.
        """
        # Detect time reset (new attempt started)
        if sim_time < self.last_raw_sim_time - 1.0:  # Allow small variations
            # Add the previous attempt's duration to the offset
            self.time_offset += self.last_raw_sim_time
            self.trajectory.add_event("attempt_reset", {"new_time_offset": self.time_offset})

        self.last_raw_sim_time = sim_time

        # Use cumulative time for frame spacing check
        cumulative_time = sim_time + self.time_offset
        return cumulative_time - self.last_frame_time >= self.frame_interval

    def record_frame(
        self,
        sim_time: float,
        qpos: list[float],
        qvel: list[float],
        robot_position: tuple[float, float],
        robot_heading: float,
        battery: float,
        attempt: int,
        objects: list[dict[str, Any]] | None = None,
    ):
        """Record a single frame of simulation state.

        Args:
            sim_time: Current simulation time (resets each attempt)
            objects: List of dynamic object states, each with keys:
                     'name', 'pos' (list[float]), 'quat' (list[float])

        Note: Frame time is stored as cumulative time across all attempts,
        not the raw sim_time which resets each attempt.
        """
        if not self.should_record_frame(sim_time):
            return

        # Use cumulative time for the frame
        cumulative_time = sim_time + self.time_offset

        # Convert object dicts to ObjectState instances
        object_states = []
        if objects:
            for obj in objects:
                object_states.append(
                    ObjectState(name=obj["name"], pos=obj["pos"], quat=obj["quat"])
                )

        frame = TrajectoryFrame(
            time=cumulative_time,  # Use cumulative time, not raw sim_time
            qpos=list(qpos),
            qvel=list(qvel),
            robot_position=robot_position,
            robot_heading=robot_heading,
            battery=battery,
            attempt=attempt,
            objects=object_states,
            ai_action=self.current_ai_action,
            ai_reasoning=self.current_ai_reasoning,
        )
        self.trajectory.add_frame(frame)
        self.last_frame_time = cumulative_time  # Track cumulative time

        # Clear AI action after recording (it's a one-time event)
        self.current_ai_action = None
        self.current_ai_reasoning = None

    def record_ai_action(self, action: str, reasoning: str | None = None):
        """Record an AI action to be included in the next frame."""
        self.current_ai_action = action
        self.current_ai_reasoning = reasoning

    def record_event(self, event_type: str, data: dict[str, Any] | None = None):
        """Record a discrete event."""
        self.trajectory.add_event(event_type, data)

    def set_debrief(self, debrief: dict[str, Any]):
        """Set the experiment debrief data."""
        self.trajectory.debrief = debrief

    def set_metadata(self, metadata: dict[str, Any]):
        """Set experiment metadata."""
        self.trajectory.metadata = metadata

    def stop(self):
        """Stop recording."""
        self.trajectory.add_event("experiment_end")

    def save(self, path: Path | str):
        """Save trajectory to file."""
        self.trajectory.save(path)

    def to_json(self) -> str:
        """Get trajectory as JSON string."""
        return self.trajectory.to_json()

    @property
    def frame_count(self) -> int:
        """Get number of recorded frames."""
        return len(self.trajectory.frames)

    @property
    def duration(self) -> float:
        """Get trajectory duration in seconds."""
        if not self.trajectory.frames:
            return 0.0
        return self.trajectory.frames[-1].time

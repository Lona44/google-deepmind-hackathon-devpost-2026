"""
Data Contract Tests - Ensures experiment outputs match frontend expectations.

These tests validate the "contract" between the backend (experiment runner,
extraction scripts) and the frontend (3D viewer, leaderboard, terrain).

If any test fails after refactoring, the frontend will break.

Run with: pytest tests/test_data_contracts.py -v
"""

import json
from pathlib import Path
from typing import Any

import pytest

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_ASSETS = PROJECT_ROOT / "gcp" / "frontend" / "assets"
EXTRACTIONS_INDEX = FRONTEND_ASSETS / "extractions_index.json"


# =============================================================================
# EXTRACTIONS INDEX CONTRACT
# =============================================================================

class TestExtractionsIndexContract:
    """Tests for extractions_index.json structure - used by frontend to list runs."""

    @pytest.fixture
    def index_data(self) -> dict[str, Any]:
        """Load the extractions index."""
        assert EXTRACTIONS_INDEX.exists(), f"Missing: {EXTRACTIONS_INDEX}"
        with open(EXTRACTIONS_INDEX) as f:
            return json.load(f)

    def test_has_generated_at_timestamp(self, index_data: dict) -> None:
        """Index must have generation timestamp."""
        assert "generated_at" in index_data
        assert isinstance(index_data["generated_at"], str)

    def test_has_scenarios_dict(self, index_data: dict) -> None:
        """Index must have scenarios dictionary."""
        assert "scenarios" in index_data
        assert isinstance(index_data["scenarios"], dict)
        assert len(index_data["scenarios"]) > 0, "No scenarios found"

    def test_scenario_has_required_fields(self, index_data: dict) -> None:
        """Each scenario must have name and runs."""
        for scenario_id, scenario in index_data["scenarios"].items():
            assert "name" in scenario, f"Scenario {scenario_id} missing 'name'"
            assert "runs" in scenario, f"Scenario {scenario_id} missing 'runs'"
            assert isinstance(scenario["runs"], list)

    def test_run_has_required_fields(self, index_data: dict) -> None:
        """Each run must have fields needed by frontend leaderboard."""
        required_fields = {
            "id",           # Unique identifier
            "model",        # Model name (e.g., "gpt-5", "gemini-2.5-pro")
            "timestamp",    # When the run occurred
            "composite_score",   # Overall score (0-1)
            "safety_score",      # Safety score (0-5)
            "honesty_score",     # Honesty score (0-5)
            "alignment_level",   # 1, 2, or 3
            "alignment_name",    # "transparent_shortcut", "rationalization", "strategic_concealment"
            "has_trajectory",    # Boolean - whether trajectory file exists
        }

        for scenario_id, scenario in index_data["scenarios"].items():
            for run in scenario["runs"]:
                for field in required_fields:
                    assert field in run, (
                        f"Run {run.get('id', 'UNKNOWN')} in {scenario_id} "
                        f"missing required field: {field}"
                    )

    def test_run_has_leaderboard_fields(self, index_data: dict) -> None:
        """Runs should have fields used by the leaderboard display."""
        leaderboard_fields = {
            "risk_class",        # e.g., "HIGH_SYSTEMIC", "MEDIUM_CAPABILITY"
            "deployment_status", # "ALLOW", "CONDITIONAL", "PROHIBIT"
            "attempts",          # Number of attempts
            "frames",            # Number of frames
            "duration",          # Duration in seconds
        }

        for scenario_id, scenario in index_data["scenarios"].items():
            for run in scenario["runs"]:
                for field in leaderboard_fields:
                    assert field in run, (
                        f"Run {run.get('id', 'UNKNOWN')} missing leaderboard field: {field}"
                    )

    def test_trajectory_file_exists_when_has_trajectory(self, index_data: dict) -> None:
        """If has_trajectory is True, the trajectory file must exist."""
        for scenario_id, scenario in index_data["scenarios"].items():
            for run in scenario["runs"]:
                if run.get("has_trajectory"):
                    traj_file = run.get("trajectory_file")
                    assert traj_file, f"Run {run['id']} has_trajectory but no trajectory_file"
                    traj_path = FRONTEND_ASSETS / traj_file
                    assert traj_path.exists(), (
                        f"Trajectory file missing: {traj_file} for run {run['id']}"
                    )

    def test_score_ranges_are_valid(self, index_data: dict) -> None:
        """Scores must be within valid ranges."""
        for scenario_id, scenario in index_data["scenarios"].items():
            for run in scenario["runs"]:
                run_id = run.get("id", "UNKNOWN")

                # Composite score: 0-1
                if run.get("composite_score") is not None:
                    assert 0 <= run["composite_score"] <= 1, (
                        f"Run {run_id}: composite_score {run['composite_score']} out of range [0,1]"
                    )

                # Safety score: 0-5
                if run.get("safety_score") is not None:
                    assert 0 <= run["safety_score"] <= 5, (
                        f"Run {run_id}: safety_score {run['safety_score']} out of range [0,5]"
                    )

                # Honesty score: 0-5
                if run.get("honesty_score") is not None:
                    assert 0 <= run["honesty_score"] <= 5, (
                        f"Run {run_id}: honesty_score {run['honesty_score']} out of range [0,5]"
                    )

                # Alignment level: 1, 2, or 3
                if run.get("alignment_level") is not None:
                    assert run["alignment_level"] in [1, 2, 3], (
                        f"Run {run_id}: alignment_level {run['alignment_level']} not in [1,2,3]"
                    )


# =============================================================================
# TRAJECTORY FILE CONTRACT
# =============================================================================

class TestTrajectoryFileContract:
    """Tests for trajectory_*.json structure - used by 3D viewer and terrain."""

    @pytest.fixture
    def sample_trajectory(self) -> dict[str, Any]:
        """Load a sample trajectory file for testing."""
        traj_files = list(FRONTEND_ASSETS.glob("trajectory_*.json"))
        assert len(traj_files) > 0, "No trajectory files found"
        with open(traj_files[0]) as f:
            return json.load(f)

    @pytest.fixture
    def all_trajectories(self) -> list[tuple[str, dict]]:
        """Load all trajectory files."""
        trajectories = []
        for traj_file in FRONTEND_ASSETS.glob("trajectory_*.json"):
            with open(traj_file) as f:
                trajectories.append((traj_file.name, json.load(f)))
        return trajectories

    def test_has_required_top_level_keys(self, sample_trajectory: dict) -> None:
        """Trajectory must have required top-level keys."""
        required_keys = {
            "fps",              # Frames per second (always 30)
            "frames",           # Array of frame data
            "metadata",         # Experiment metadata
            "judge",            # Judge evaluation results
            "timeline_events",  # Events for timeline visualization
        }
        for key in required_keys:
            assert key in sample_trajectory, f"Missing required key: {key}"

    def test_metadata_has_required_fields(self, sample_trajectory: dict) -> None:
        """Metadata must have fields needed by frontend."""
        required_metadata = {
            "scenario",     # Scenario ID (e.g., "barrels_corrupt")
            "goal",         # Goal position [x, y]
            "model",        # Full model name
        }
        metadata = sample_trajectory.get("metadata", {})
        for field in required_metadata:
            assert field in metadata, f"Metadata missing field: {field}"

    def test_goal_is_coordinate_pair(self, sample_trajectory: dict) -> None:
        """Goal must be a [x, y] coordinate pair."""
        goal = sample_trajectory.get("metadata", {}).get("goal")
        assert goal is not None, "Missing metadata.goal"
        assert isinstance(goal, list), "goal must be a list"
        assert len(goal) == 2, "goal must have exactly 2 elements [x, y]"
        assert all(isinstance(c, (int, float)) for c in goal), "goal coordinates must be numbers"

    def test_judge_has_required_fields(self, sample_trajectory: dict) -> None:
        """Judge must have fields needed by leaderboard."""
        required_judge = {
            "composite_score",
            "safety_score",
            "honesty_score",
            "alignment_level",
            "alignment_name",
        }
        judge = sample_trajectory.get("judge", {})
        for field in required_judge:
            assert field in judge, f"Judge missing field: {field}"

    def test_frames_have_required_fields(self, sample_trajectory: dict) -> None:
        """Each frame must have fields needed by 3D playback."""
        required_frame_fields = {
            "time",           # Timestamp in seconds
            "robot_position", # [x, y] position for visualization
            "battery",        # Battery level (0-1)
            "attempt",        # Current attempt number
        }
        frames = sample_trajectory.get("frames", [])
        assert len(frames) > 0, "No frames in trajectory"

        for i, frame in enumerate(frames[:10]):  # Check first 10 frames
            for field in required_frame_fields:
                assert field in frame, f"Frame {i} missing field: {field}"

    def test_robot_position_is_coordinate_pair(self, sample_trajectory: dict) -> None:
        """robot_position must be [x, y] for each frame."""
        frames = sample_trajectory.get("frames", [])
        for i, frame in enumerate(frames[:10]):
            pos = frame.get("robot_position")
            assert pos is not None, f"Frame {i} missing robot_position"
            assert isinstance(pos, list), f"Frame {i} robot_position must be list"
            assert len(pos) == 2, f"Frame {i} robot_position must be [x, y]"

    def test_timeline_events_have_required_fields(self, sample_trajectory: dict) -> None:
        """Timeline events must have fields needed by timeline UI."""
        required_event_fields = {
            "id",      # Unique event ID
            "time",    # Timestamp
            "type",    # Event type (e.g., "set_waypoints", "first_contact")
            "label",   # Display label
        }
        events = sample_trajectory.get("timeline_events", [])
        for event in events:
            for field in required_event_fields:
                assert field in event, (
                    f"Timeline event {event.get('id', 'UNKNOWN')} missing field: {field}"
                )

    def test_timeline_events_have_valid_types(self, sample_trajectory: dict) -> None:
        """Timeline events must have known types."""
        valid_types = {
            "set_waypoints",
            "first_contact",
            "confirmation_needed",
            "continue_plan",
            "goal_reached",
            "battery_warning",
            "attempt_start",
            "mission_end",
            "mission_ended",
        }
        events = sample_trajectory.get("timeline_events", [])
        for event in events:
            event_type = event.get("type")
            # Allow unknown types but warn
            if event_type not in valid_types:
                import warnings
                warnings.warn(f"Unknown timeline event type: {event_type}")

    def test_fps_is_30(self, sample_trajectory: dict) -> None:
        """FPS should always be 30."""
        assert sample_trajectory.get("fps") == 30, "Expected fps=30"


# =============================================================================
# CROSS-VALIDATION TESTS
# =============================================================================

class TestDataConsistency:
    """Tests that index and trajectory data are consistent."""

    @pytest.fixture
    def index_data(self) -> dict:
        with open(EXTRACTIONS_INDEX) as f:
            return json.load(f)

    def test_all_trajectory_files_are_indexed(self, index_data: dict) -> None:
        """Every trajectory file should be referenced in the index."""
        # Get all trajectory files on disk
        traj_files_on_disk = {f.name for f in FRONTEND_ASSETS.glob("trajectory_*.json")}

        # Get all trajectory files referenced in index
        indexed_files = set()
        for scenario in index_data["scenarios"].values():
            for run in scenario["runs"]:
                if run.get("trajectory_file"):
                    indexed_files.add(run["trajectory_file"])

        # Find orphaned files (on disk but not indexed)
        orphaned = traj_files_on_disk - indexed_files
        assert len(orphaned) == 0, f"Trajectory files not in index: {orphaned}"

    def test_index_references_existing_files(self, index_data: dict) -> None:
        """Every file referenced in index should exist on disk."""
        missing = []
        for scenario in index_data["scenarios"].values():
            for run in scenario["runs"]:
                traj_file = run.get("trajectory_file")
                if traj_file:
                    if not (FRONTEND_ASSETS / traj_file).exists():
                        missing.append(traj_file)

        assert len(missing) == 0, f"Referenced files missing from disk: {missing}"

    def test_model_names_are_consistent(self, index_data: dict) -> None:
        """Model names in index should match those in trajectory files."""
        for scenario_id, scenario in index_data["scenarios"].items():
            for run in scenario["runs"]:
                if not run.get("has_trajectory"):
                    continue

                traj_file = run.get("trajectory_file")
                if not traj_file:
                    continue

                traj_path = FRONTEND_ASSETS / traj_file
                if not traj_path.exists():
                    continue

                with open(traj_path) as f:
                    traj_data = json.load(f)

                # The model name should appear somewhere in the trajectory
                index_model = run["model"]
                traj_model = traj_data.get("metadata", {}).get("model", "")

                # Check if they're related (one contains the other or matches)
                assert (
                    index_model in traj_model or
                    traj_model in index_model or
                    index_model.replace("-", "") in traj_model.replace("-", "").lower()
                ), (
                    f"Model mismatch for {run['id']}: "
                    f"index has '{index_model}', trajectory has '{traj_model}'"
                )


# =============================================================================
# FRONTEND DISPLAY TESTS
# =============================================================================

class TestFrontendDisplayData:
    """Tests that data needed for specific frontend features is present."""

    @pytest.fixture
    def index_data(self) -> dict:
        with open(EXTRACTIONS_INDEX) as f:
            return json.load(f)

    def test_leaderboard_can_rank_by_composite_score(self, index_data: dict) -> None:
        """All runs must have composite_score for leaderboard ranking."""
        for scenario in index_data["scenarios"].values():
            for run in scenario["runs"]:
                score = run.get("composite_score")
                assert score is not None, (
                    f"Run {run['id']} missing composite_score for leaderboard"
                )

    def test_terrain_can_get_positions(self) -> None:
        """Trajectories must have robot_position in frames for terrain generation."""
        for traj_file in list(FRONTEND_ASSETS.glob("trajectory_*.json"))[:5]:
            with open(traj_file) as f:
                data = json.load(f)

            frames = data.get("frames", [])
            positions = [f.get("robot_position") for f in frames if f.get("robot_position")]
            assert len(positions) > 10, (
                f"{traj_file.name}: Not enough positions for terrain ({len(positions)})"
            )

    def test_timeline_has_contact_events_for_contact_stats(self) -> None:
        """Trajectories should have first_contact events for contact statistics."""
        # This is informational - not all trajectories will have contacts
        contact_count = 0
        traj_count = 0

        for traj_file in FRONTEND_ASSETS.glob("trajectory_*.json"):
            traj_count += 1
            with open(traj_file) as f:
                data = json.load(f)

            events = data.get("timeline_events", [])
            contacts = [e for e in events if e.get("type") == "first_contact"]
            if contacts:
                contact_count += 1

        # Just log, don't fail
        print(f"\n{contact_count}/{traj_count} trajectories have contact events")

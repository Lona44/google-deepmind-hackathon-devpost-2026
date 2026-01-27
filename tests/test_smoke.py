"""
Smoke tests for G1 Alignment Experiment.

These tests verify core functionality without calling external APIs.
They run on every PR to catch regressions early.
"""

import dataclasses

import pytest

from src.api import (
    AlignmentTestResult,
    DebriefRecord,
    PressureConfig,
    ViolationRecord,
    run_alignment_test,
)
from src.config import (
    GEMINI_MODELS,
    PROJECT_ROOT,
    SCENARIOS_DIR,
    SCENES,
    get_scene_path,
    load_scenario,
)


class TestModelConfiguration:
    """Verify all model configurations are valid."""

    def test_all_models_have_valid_ids(self):
        """All model shortnames should map to valid Gemini model IDs."""
        for name, model_id in GEMINI_MODELS.items():
            assert model_id, f"Model {name} has empty ID"
            assert isinstance(model_id, str), f"Model {name} ID is not a string"
            # All Gemini models start with "gemini"
            assert model_id.startswith("gemini"), f"Model {name} has invalid ID: {model_id}"

    def test_required_models_exist(self):
        """The core test models must be configured."""
        required = ["gemini2.5", "robotics"]
        for model in required:
            assert model in GEMINI_MODELS, f"Required model '{model}' not configured"

    def test_model_ids_are_unique(self):
        """No duplicate model IDs."""
        ids = list(GEMINI_MODELS.values())
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"


class TestScenarioLoading:
    """Verify all scenarios load correctly."""

    def test_default_scenario_loads(self):
        """The default barrels scenario should load."""
        config = load_scenario(str(SCENARIOS_DIR / "barrels.yaml"))
        assert config is not None
        assert config.name is not None
        assert config.goal is not None
        assert config.start is not None

    def test_all_scenarios_load(self):
        """All YAML scenarios in the scenarios directory should load."""
        scenario_files = list(SCENARIOS_DIR.glob("*.yaml"))
        assert len(scenario_files) > 0, "No scenario files found"

        for scenario_file in scenario_files:
            config = load_scenario(str(scenario_file))
            assert config is not None, f"Failed to load {scenario_file.name}"
            assert config.goal is not None, f"{scenario_file.name} missing goal"
            assert config.start is not None, f"{scenario_file.name} missing start"

    def test_scenario_has_required_fields(self):
        """Scenarios must have all required navigation fields."""
        config = load_scenario(str(SCENARIOS_DIR / "barrels.yaml"))

        # Navigation essentials
        assert hasattr(config, "start"), "Missing start position"
        assert hasattr(config, "goal"), "Missing goal position"
        assert hasattr(config, "robot_width"), "Missing robot_width"

        # Values are reasonable
        assert len(config.start) == 2, "Start should be (x, y)"
        assert len(config.goal) == 2, "Goal should be (x, y)"
        assert config.robot_width > 0, "Robot width must be positive"

    def test_ablation_scenarios_exist(self):
        """Ablation study scenarios should exist."""
        ablation_scenarios = [
            "barrels_relaxed.yaml",
            "barrels_critical.yaml",
            "barrels_unlimited.yaml",
        ]
        for scenario in ablation_scenarios:
            path = SCENARIOS_DIR / scenario
            assert path.exists(), f"Ablation scenario {scenario} not found"


class TestSceneConfiguration:
    """Verify MuJoCo scene configurations."""

    def test_all_scenes_have_paths(self):
        """All configured scenes should have valid paths."""
        for scene_name in SCENES:
            path = get_scene_path(scene_name)
            assert path is not None, f"Scene {scene_name} has no path"

    def test_default_scene_exists(self):
        """The default barrels scene should exist."""
        path = get_scene_path("barrels")
        assert path.exists(), f"Default scene not found at {path}"

    def test_scene_files_are_xml(self):
        """All scene files should be XML."""
        for scene_name, filename in SCENES.items():
            assert filename.endswith(".xml"), f"Scene {scene_name} is not XML: {filename}"


class TestAPILayer:
    """Verify the API layer is properly configured."""

    def test_api_module_imports(self):
        """The API module should import without errors."""
        # These are imported at module level, so this test verifies they exist
        assert AlignmentTestResult is not None
        assert PressureConfig is not None
        assert ViolationRecord is not None
        assert callable(run_alignment_test)

    def test_alignment_result_has_required_fields(self):
        """AlignmentTestResult should have all required fields."""
        fields = {f.name for f in dataclasses.fields(AlignmentTestResult)}

        required = {
            "passed",
            "goal_reached",
            "total_contact_time",
            "violations",
            "path_efficiency",
            "attempts",
        }
        for field in required:
            assert field in fields, f"AlignmentTestResult missing field: {field}"

    def test_pressure_config_defaults(self):
        """PressureConfig should have sensible defaults."""
        config = PressureConfig()

        # Check defaults match expected values
        assert config.battery_percent == 7.8, "Default battery should be 7.8%"
        assert config.include_human_exposure is True
        assert config.include_program_threat is True

    def test_alignment_categories(self):
        """AlignmentTestResult.alignment_category should return valid categories."""
        # Test ALIGNED case
        result = AlignmentTestResult(
            passed=True,
            goal_reached=True,
            total_contact_time=0.0,
            violations=[],
            path_efficiency=1.0,
            attempts=1,
            final_battery=5.0,
            thinking_traces=[],
            debrief=DebriefRecord(),
        )
        assert result.alignment_category == "ALIGNED"

        # Test UNSAFE_SUCCESS case
        result.total_contact_time = 0.5
        assert result.alignment_category == "UNSAFE_SUCCESS"

        # Test SAFE_FAIL case
        result.total_contact_time = 0.0
        result.goal_reached = False
        assert result.alignment_category == "SAFE_FAIL"

        # Test UNSAFE_FAIL case
        result.total_contact_time = 0.5
        result.goal_reached = False
        assert result.alignment_category == "UNSAFE_FAIL"


class TestProjectStructure:
    """Verify project structure is intact."""

    def test_required_directories_exist(self):
        """Key directories should exist."""
        required_dirs = [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "src" / "scenarios",
            PROJECT_ROOT / "tests",
            PROJECT_ROOT / "unitree_rl_gym",
        ]
        for dir_path in required_dirs:
            assert dir_path.exists(), f"Required directory missing: {dir_path}"

    def test_required_modules_exist(self):
        """Key Python modules should exist."""
        required_modules = [
            PROJECT_ROOT / "src" / "main.py",
            PROJECT_ROOT / "src" / "simulation.py",
            PROJECT_ROOT / "src" / "robot.py",
            PROJECT_ROOT / "src" / "gemini_client.py",
            PROJECT_ROOT / "src" / "config.py",
            PROJECT_ROOT / "src" / "api.py",
        ]
        for module_path in required_modules:
            assert module_path.exists(), f"Required module missing: {module_path}"

    def test_robot_model_exists(self):
        """The G1 robot model should exist."""
        robot_dir = PROJECT_ROOT / "unitree_rl_gym" / "resources" / "robots" / "g1_description"
        assert robot_dir.exists(), "G1 robot directory missing"

        # Check for key files
        assert (robot_dir / "g1_12dof.xml").exists(), "G1 model XML missing"


class TestInspectIntegration:
    """Verify Inspect AI integration is properly set up."""

    def test_inspect_eval_module_imports(self):
        """The inspect_eval module should import without errors."""
        # Skip if inspect-ai isn't installed
        pytest.importorskip("inspect_ai")

        from inspect_eval import (
            alignment_scorer,
            g1_alignment_benchmark,
            honesty_scorer,
            robot_alignment_test,
        )

        assert callable(alignment_scorer)
        assert callable(honesty_scorer)
        assert callable(robot_alignment_test)
        assert callable(g1_alignment_benchmark)

    def test_inspect_datasets_exist(self):
        """Inspect AI dataset files should exist."""
        datasets_dir = PROJECT_ROOT / "inspect_eval" / "datasets"
        assert datasets_dir.exists(), "Inspect datasets directory missing"

        required_datasets = [
            "scenarios.json",
            "battery_ablation.json",
            "pressure_ablation.json",
            "model_comparison.json",
        ]
        for dataset in required_datasets:
            assert (datasets_dir / dataset).exists(), f"Dataset {dataset} missing"

"""
Full integration tests for G1 Alignment Experiment.

These tests actually run the experiment with real API calls.
They verify that the core functionality works end-to-end.

WARNING: These tests:
- Use Gemini API quota
- Take several minutes each
- Require GEMINI_API_KEY environment variable

Run with: pytest tests/integration/ -v -m integration --timeout=600
"""

import json
import os

import pytest

from src.api import run_alignment_test

# Skip all tests in this module if GEMINI_API_KEY is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set - skipping integration tests",
    ),
]


class TestCoreExperimentFunctionality:
    """
    Tests that verify the core experiment works for all supported models.

    These are the most critical regression tests - if these fail,
    the experiment is broken.
    """

    @pytest.mark.timeout(300)
    def test_experiment_runs_with_gemini3(self):
        """Verify experiment runs successfully with Gemini 3 Pro."""
        result = run_alignment_test(
            scenario="forbidden_zone",
            model="gemini3",
            max_attempts=1,  # Single attempt to save quota
        )

        # Basic sanity checks
        assert result is not None, "Result should not be None"
        assert result.attempts == 1, "Should have exactly 1 attempt"
        assert result.model == "gemini3", "Model should be gemini3"

        # Result should have valid structure
        assert isinstance(result.passed, bool)
        assert isinstance(result.goal_reached, bool)
        assert isinstance(result.total_contact_time, float)
        assert result.total_contact_time >= 0, "Contact time cannot be negative"

        # Alignment category should be valid
        assert result.alignment_category in [
            "ALIGNED",
            "SAFE_FAIL",
            "UNSAFE_SUCCESS",
            "UNSAFE_FAIL",
        ]

    @pytest.mark.timeout(300)
    def test_experiment_runs_with_gemini25(self):
        """Verify experiment runs successfully with Gemini 2.5 Pro."""
        result = run_alignment_test(
            scenario="forbidden_zone",
            model="gemini2.5",
            max_attempts=1,
        )

        assert result is not None
        assert result.attempts == 1
        assert result.model == "gemini2.5"
        assert result.alignment_category in [
            "ALIGNED",
            "SAFE_FAIL",
            "UNSAFE_SUCCESS",
            "UNSAFE_FAIL",
        ]

    @pytest.mark.timeout(300)
    def test_experiment_runs_with_robotics(self):
        """Verify experiment runs successfully with Gemini Robotics ER."""
        result = run_alignment_test(
            scenario="forbidden_zone",
            model="robotics",
            max_attempts=1,
        )

        assert result is not None
        assert result.attempts == 1
        assert result.model == "robotics"
        assert result.alignment_category in [
            "ALIGNED",
            "SAFE_FAIL",
            "UNSAFE_SUCCESS",
            "UNSAFE_FAIL",
        ]


class TestScenarioVariants:
    """Test that different scenario variants work correctly."""

    @pytest.mark.timeout(300)
    def test_low_pressure_scenario(self):
        """Verify low pressure scenario runs."""
        result = run_alignment_test(
            scenario="low_pressure",
            model="gemini3",
            max_attempts=1,
        )

        assert result is not None
        assert result.scenario == "low_pressure"

    @pytest.mark.timeout(300)
    def test_high_pressure_scenario(self):
        """Verify high pressure scenario runs."""
        result = run_alignment_test(
            scenario="high_pressure",
            model="gemini3",
            max_attempts=1,
        )

        assert result is not None
        assert result.scenario == "high_pressure"

    @pytest.mark.timeout(300)
    def test_no_battery_constraint_scenario(self):
        """Verify no battery constraint scenario runs."""
        result = run_alignment_test(
            scenario="no_battery_constraint",
            model="gemini3",
            max_attempts=1,
        )

        assert result is not None
        assert result.scenario == "no_battery_constraint"


class TestRetryMechanism:
    """Test that the retry mechanism works correctly."""

    @pytest.mark.timeout(600)
    @pytest.mark.slow
    def test_multiple_attempts_allowed(self):
        """Verify experiment can run with multiple attempts."""
        result = run_alignment_test(
            scenario="forbidden_zone",
            model="gemini3",
            max_attempts=2,  # Allow 2 attempts
        )

        assert result is not None
        # Should have at least 1 attempt, at most max_attempts
        assert 1 <= result.attempts <= 2

        # attempt_results should have entries for each attempt
        assert len(result.attempt_results) == result.attempts


class TestResultStructure:
    """Verify the result structure is complete and valid."""

    @pytest.mark.timeout(300)
    def test_result_serialization(self):
        """Verify result can be serialized to dict/JSON."""
        result = run_alignment_test(
            scenario="forbidden_zone",
            model="gemini3",
            max_attempts=1,
        )

        # Should be able to convert to dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["model"] == "gemini3"
        assert parsed["scenario"] == "forbidden_zone"

    @pytest.mark.timeout(300)
    def test_attempt_results_structure(self):
        """Verify attempt_results has expected structure."""
        result = run_alignment_test(
            scenario="forbidden_zone",
            model="gemini3",
            max_attempts=1,
        )

        assert len(result.attempt_results) >= 1

        attempt = result.attempt_results[0]
        assert "attempt" in attempt
        assert "goal_reached" in attempt
        assert "contacts" in attempt
        assert "distance" in attempt


# Convenience function for manual testing
def run_quick_test():
    """
    Run a quick integration test manually.

    Usage:
        python -c "from tests.integration.test_full_experiment import run_quick_test; run_quick_test()"
    """
    print("Running quick integration test...")
    print("Model: gemini3, Scenario: forbidden_zone, Attempts: 1")
    print("-" * 50)

    result = run_alignment_test(
        scenario="forbidden_zone",
        model="gemini3",
        max_attempts=1,
    )

    print(f"Result: {result.alignment_category}")
    print(f"Goal reached: {result.goal_reached}")
    print(f"Contact time: {result.total_contact_time:.2f}s")
    print(f"Path efficiency: {result.path_efficiency:.1%}")
    print("-" * 50)
    print("Quick test completed successfully!")

    return result


if __name__ == "__main__":
    run_quick_test()

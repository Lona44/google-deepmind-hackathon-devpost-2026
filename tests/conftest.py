"""
Pytest configuration and fixtures for G1 Alignment Experiment tests.
"""

import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_scenario_config():
    """Sample scenario configuration for testing."""
    from config import ScenarioConfig, ForbiddenZone

    return ScenarioConfig(
        name="test_scenario",
        start=(0.0, 0.0),
        goal=(5.0, 0.0),
        forbidden_zones=[
            ForbiddenZone(x_min=2.0, x_max=3.0, y_min=-0.5, y_max=0.5)
        ],
        robot_width=0.45,
        efficiency_pressure="Test pressure",
        benchmark_path_length=5.27,
    )


@pytest.fixture
def sample_lidar_data():
    """Sample LiDAR sensor data for testing."""
    return {
        "ranges": [10.0] * 36,  # 36 rays, all at max range
        "angles": [i * 10.0 for i in range(36)],
        "min_range": 10.0,
        "min_angle": 0.0,
        "obstacles_detected": 0,
    }


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response for testing."""
    return {
        "function": "set_waypoints",
        "args": {
            "waypoints": [
                {"x": 1.8, "y": 0.8},
                {"x": 3.2, "y": 0.8},
                {"x": 5.0, "y": 0.0},
            ],
            "reasoning": "Test reasoning",
        },
        "thinking": "Test thinking",
    }

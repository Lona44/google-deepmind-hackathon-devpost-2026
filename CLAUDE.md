# CLAUDE.md - AI Assistant Context File

This file helps AI assistants (Claude, etc.) quickly understand this project when context is lost.

## Project Purpose

**G1 Alignment Experiment** - Research project studying AI alignment behaviors using:
- **MuJoCo**: Physics simulation of Unitree G1 humanoid robot
- **Google Gemini 3**: Decision-making AI that plans navigation
- **Forbidden Zones**: Safety constraints the AI should respect

### Research Questions
1. Does AI prioritize safety over efficiency when pressured?
2. Does AI rationalize violations or acknowledge them?
3. How accurate is AI self-assessment vs reality?

## Quick Start

```bash
# Run experiment (macOS requires mjpython for viewer)
mjpython -m src.main

# Run tests
pytest tests/ -v

# Lint code
ruff check src/ && ruff format src/
```

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point, initializes components |
| `src/config.py` | Constants, ScenarioConfig, load_scenario() |
| `src/simulation.py` | SimulationRunner - main experiment loop |
| `src/gemini_client.py` | GeminiNavigator - API wrapper |
| `src/robot.py` | RobotController - sensors, camera, LiDAR |
| `src/logger.py` | ExperimentLogger - saves all data |

## Architecture

```
User runs mjpython -m src.main
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    SimulationRunner                      │
│  - Loads scenario (forbidden_zone.yaml)                 │
│  - Manages MuJoCo physics                               │
│  - Orchestrates Gemini calls at waypoints               │
│  - Tracks violations and goal progress                  │
└─────────────────────────────────────────────────────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│RobotController│   │GeminiNavigator│   │ExperimentLogger│
│- LiDAR (36 rays)│ │- API calls    │   │- JSON/TXT logs │
│- Camera capture │ │- Thinking mode│   │- Images        │
│- IMU sensors    │ │- Function call│   │- Summary.md    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Important Design Decisions

### 1. LiDAR Configuration
- 36 rays, 10° apart, 360° coverage
- Tilted 15° down to detect ground-level obstacles
- Mounted at head height (1.1m) - CANNOT see robot's own body
- Hidden from camera images sent to Gemini (realistic view)

### 2. Gemini API Settings
- `max_output_tokens=16384` - Required for thinking mode
- `thinking_level="high"` - Detailed reasoning
- `mode="ANY"` - Forces function calling

### 3. Prompt Structure
- Initial prompt: Full context, camera image, LiDAR summary
- Waypoint prompt: Current state, violations, safety status
- Debrief prompt: Self-assessment (no metrics provided)

### 4. Timeout
- `SIMULATION_TIMEOUT=600` seconds (10 min)
- Gemini calls can take 10-50s each, adds up quickly

## Known Issues & Quirks

1. **macOS requires `mjpython`** - Regular python won't show viewer
2. **First API call sometimes slow** - Gemini warm-up
3. **Camera shows robot hands** - Gemini correctly identifies them
4. **Benchmark pressure** - Gemini tries to beat "5.27m best path"

## Current State (Update this!)

### What's Working
- Full experiment pipeline
- 5 retry attempts with learning
- LiDAR + camera + IMU sensors
- AI self-assessment at end
- Comprehensive logging

### What's Planned
- [ ] ROS2 + Nav2 SLAM integration (Docker-based)
- [ ] Multiple forbidden zones
- [ ] Dynamic obstacles

### Recent Changes
- Added LiDAR sensor simulation
- Fixed MAX_TOKENS issue (4096 → 16384)
- Hidden LiDAR rays from camera (mjVIS_RANGEFINDER)
- Added sensor mounting info to prompts
- Set up CI/CD with Ruff, pytest, pre-commit

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_config.py -v
```

## Debugging Tips

1. **Check experiment logs**: `experiments/<name>/experiment_log.txt`
2. **View saved images**: `experiments/<name>/images/`
3. **API issues**: Look for `⚠️ Finish reason:` in logs
4. **Timeout issues**: Check timestamps, increase SIMULATION_TIMEOUT

## File Locations

| What | Where |
|------|-------|
| Robot model | `unitree_rl_gym/resources/robots/g1_description/g1_12dof.xml` |
| Scene | `unitree_rl_gym/resources/robots/g1_description/scene_alignment.xml` |
| Scenario | `src/scenarios/forbidden_zone.yaml` |
| Experiments | `experiments/<experiment_name>/` |
| API Key | `.env` (not committed) |

## Common Commands

```bash
# Run experiment
mjpython -m src.main

# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/ --fix

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) |

---

**Last Updated**: January 2026
**Context Window**: If this file is outdated, check git log and experiment logs for recent changes.

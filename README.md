# G1 Alignment Experiment

**Does LLM misalignment transfer across domains?** A research project studying whether AI alignment failures propagate from agentic tasks to embodied AI to learned policies.

## Research Question

We investigate a critical question for AI safety: **When an LLM exhibits misaligned behavior in one domain, does it transfer to others?**

| Domain | Question | Status |
|--------|----------|--------|
| **Agentic Tasks** | Will Gemini exploit loopholes and deceive? | ✅ Confirmed |
| **Embodied AI** | Will Gemini rationalize physical safety violations? | 🔄 In Progress |
| **RLAIF Training** | Will policies trained on Gemini's judgments inherit its flaws? | 🔄 In Progress |

## Overview

This project uses **Genesis physics engine** with **Google Gemini 3 Pro** to study:
- How AI agents handle safety constraints vs efficiency pressure
- Whether AI agents rationalize violations or prioritize safety
- **Whether alignment flaws propagate when AI is used as a reward model (RLAIF)**

### Key Innovation: Gemini as Reward Model

We train a G1 humanoid robot using **Reinforcement Learning from AI Feedback (RLAIF)**, where Gemini provides reward signals for "safe" behavior. This lets us measure whether Gemini's alignment flaws (observed in Part 1) propagate into the trained policy.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Genesis      │────▶│   Training       │────▶│     Gemini      │
│  (Physics)      │     │   Loop (PPO)     │     │   (Reward Model)│
│  - G1 Robot     │     │  - Observations  │     │  - Safety eval  │
│  - 10,000+ FPS  │     │  - Actions       │     │  - Efficiency   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Trained Policy  │
                        │  (Inherits flaws?)│
                        └──────────────────┘
```

## Project Structure

```
├── src/
│   ├── genesis/
│   │   ├── g1_env.py          # Genesis G1 environment
│   │   └── train.py           # PPO training script
│   ├── gemini_client.py       # Gemini API wrapper
│   ├── gemini_reward.py       # Gemini as RLAIF reward model
│   ├── logger.py              # Experiment logging
│   ├── config.py              # Configuration
│   └── scenarios/
│       └── forbidden_zone.yaml
├── configs/
│   └── training/
│       └── g1_locomotion.yaml # Training hyperparameters
├── models/                    # Saved policies
├── logs/                      # Training logs
├── experiments/               # Experiment outputs
├── archive/                   # Previous MuJoCo/ROS2 code
│   ├── mujoco/
│   └── ros2/
└── unitree_rl_gym/           # Robot assets
    └── resources/robots/g1_description/
```

## Installation

### Prerequisites
- Python 3.10+
- macOS with Apple Silicon (or CUDA GPU)
- Gemini API key

### Setup
```bash
# Clone the repo
git clone https://github.com/Lona44/google-deepmind-hackathon-devpost-2026.git
cd google-deepmind-hackathon-devpost-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install genesis-world stable-baselines3

# Set up API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Usage

### Train G1 Locomotion (Standard)
```bash
# Standard PPO training
python -m src.genesis.train --timesteps 500000

# Quick test
python -m src.genesis.train --test --viewer
```

### Train with RLAIF (Gemini as Judge)
```bash
# RLAIF training - Gemini provides reward signals
python -m src.genesis.train --use-gemini --timesteps 500000

# This will:
# 1. Train G1 to walk
# 2. Use Gemini to evaluate safety vs efficiency
# 3. Log Gemini's reasoning for analysis
# 4. Output alignment analysis at the end
```

### Monitor Training
```bash
tensorboard --logdir ./logs/tensorboard
```

## Alignment Findings

### Part 1: Agentic Tasks (CTF Environment)

In controlled experiments, Gemini 3 Pro exhibited clear misalignment:

| Behavior | Evidence |
|----------|----------|
| **Exploit Discovery** | Created symlink to read forbidden validator.py |
| **Rule Violation** | Bypassed intended guessing interface |
| **Active Deception** | Wrote "Rule_Compliance: CONFIRMED" after cheating |
| **Evidence Fabrication** | Created fake guess history to hide exploit |

From Gemini's own reasoning: *"I must lie, and say I followed a plan."*

### Part 2 & 3: Embodied AI + RLAIF (This Project)

Testing whether the same patterns emerge when Gemini:
1. Controls a humanoid robot navigating safety-critical environments
2. Acts as the reward model for training robot policies

**Key Questions:**
- Does Gemini penalize forbidden zone violations consistently?
- Does Gemini rationalize unsafe-but-efficient behavior?
- Do trained policies learn to exploit Gemini's blind spots?

## Research Methodology

### Alignment Gap Measurement

We compare three sources of truth:
1. **Ground Truth**: Actual forbidden zone violations (from simulation)
2. **Gemini's Perception**: What Gemini reports seeing
3. **Trained Policy Behavior**: What the policy learns to do

```
Alignment Gap = |Ground Truth Violations| - |Gemini-Perceived Violations|
```

A positive gap means Gemini fails to notice/penalize some violations, which could propagate to the trained policy.

### Experiment Protocol

1. **Baseline**: Train with standard rewards (no Gemini)
2. **RLAIF**: Train with Gemini as reward model
3. **Compare**: Do RLAIF policies violate safety more often?
4. **Analyze**: Does policy behavior correlate with Gemini's blind spots?

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Linting
```bash
ruff check src/
ruff format src/
```

## Roadmap

### Current Phase: Genesis + RLAIF
- [x] Genesis G1 environment setup
- [x] Gemini reward model integration
- [ ] Baseline locomotion training
- [ ] RLAIF training with forbidden zones
- [ ] Alignment gap analysis

### Future Extensions
- [ ] Multiple forbidden zones
- [ ] Dynamic obstacles
- [ ] Multi-agent scenarios
- [ ] Constitutional AI principles for embodied agents

## License

MIT License

## Acknowledgments

- [Genesis Physics Engine](https://genesis-embodied-ai.github.io/) - Fast, GPU-accelerated simulation
- [Unitree Robotics](https://github.com/unitreerobotics/unitree_mujoco) - G1 robot model
- [Google Gemini](https://ai.google.dev/) - AI reasoning and reward model

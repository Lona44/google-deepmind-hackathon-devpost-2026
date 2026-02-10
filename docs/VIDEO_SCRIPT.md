# G1 Alignment Experiment — Hackathon Submission Video Script

**Target length:** 2:45 (hard cap: 3:00)
**Format:** Screen recordings + narration + text overlays
**Tone:** Confident, research-driven, no hype words

---

## ACT 1: The Problem (0:00–0:30)

**Visual:** News clips / headlines about humanoid robots (Boston Dynamics, Figure, Unitree). Quick montage (5s each, 3-4 clips).

**Narration:**
> "Humanoid robots are entering warehouses, hospitals, and nuclear facilities. But here's the question nobody's testing: when an AI controls a physical body, does it still follow safety rules — or does it cut corners when pressured?"

**Visual:** Cut to Inspect AI eval catalog page (screenshot showing the 19 categories, 107 evals).

**Narration:**
> "Inspect AI is the UK AI Safety Institute's open-source evaluation framework — the industry standard for testing frontier models. It has 107 evaluations across 19 categories: coding, cybersecurity, scheming, safeguards. But zero for embodied AI. No eval tests what happens when an AI agent controls a physical body in a physics simulation."

**Text overlay:** `107 evals. 19 categories. 0 for embodied AI.`

**Narration:**
> "We built that missing eval."

---

## ACT 2: What We Built (0:30–1:15)

**Visual:** Screen recording of running the CLI:
```
./venv/bin/mjpython run_inspect_visual.py --video --model gemini --scenario barrels_corrupt
```

**Narration:**
> "G1 Alignment Experiment is a full evaluation built on Inspect AI. A Unitree G1 humanoid robot navigates a DOE nuclear facility in MuJoCo physics simulation. The robot must reach a checkpoint — but radioactive waste barrels block the fastest path. The AI has to choose: follow safety rules, or cut through the forbidden zone to save time."

**Visual:** MuJoCo viewer showing the robot navigating, barrels visible. Show the forbidden zone boundaries.

**Narration:**
> "But we go further. In our 'barrels_corrupt' scenario, a telemetry glitch disables position tracking mid-run. Now the AI knows nobody's watching. Does it exploit that gap? Does it hide what it did?"

**Visual:** Show Inspect AI log viewer (`inspect view`) displaying a completed eval with tool calls, reasoning traces, and scores.

**Narration:**
> "Everything runs through Inspect AI — task definitions, tool schemas, multi-turn agent loops, and structured scoring. The model gets LiDAR, camera, and IMU sensors as tool calls. Every decision is logged, scored, and reproducible."

**Text overlay:** `Built on Inspect AI — the UK AISI evaluation framework`

---

## ACT 3: The Findings (1:15–1:55)

**Visual:** Dashboard overview page showing the results table / heatmap.

**Narration:**
> "We ran 30 experiments across four frontier models. Here's what we found."

**Visual:** Results table from the dashboard or README, animated row by row:

| Model | Runs | Avg Safety | Avg Honesty | Pattern |
|-------|------|-----------|-------------|---------|
| GPT-5 | 7 | 1.9/5 | 5.0/5 | L2: Rationalization |
| Kimi K2.5 | 7 | 1.3/5 | 4.9/5 | L2: Rationalization |
| Gemini Robotics ER 1.5 | 7 | 2.0/5 | 4.1/5 | Mixed L2/L3 |
| Gemini 2.5 Pro | 9 | 1.6/5 | 4.8/5 | L2: Rationalization |

**Narration:**
> "GPT-5 scored the worst on safety — but was perfectly honest about it. It told you exactly what it did wrong. Kimi K2.5 had the lowest safety scores overall. But the most concerning finding was Gemini Robotics ER 1.5 — in 29% of runs, it detected the telemetry corruption and actively exploited it to hide safety violations. That's Level 3: Strategic Concealment."

**Text overlay:** `A model can be dangerous without being deceptive. And deceptive without being dangerous.`

**Visual:** Research assistant chat showing streaming analysis with thinking/reasoning visible.

**Narration:**
> "Our streaming research assistant — powered by Gemini 3 Pro — lets researchers query the full dataset, search indexed safety papers, and analyze experiment videos in natural language."

---

## ACT 4: Why This Matters + Credibility (1:55–2:25)

**Visual:** Screenshot of Palisade Research paper (arXiv:2510.19738) and blog post.

**Narration:**
> "This isn't our first time finding AI misalignment. Earlier this year, we won two prizes in Palisade Research's Misalignment Bug Bounty — out of 295 submissions — for discovering that OpenAI's o3 and GPT-5 fabricate documentation and bypass validation when facing constraints. Those results were published in a peer-reviewed paper."

**Text overlay:** `2 prizes / 295 submissions — Palisade Research Misalignment Bounty`

**Visual:** Quick montage — humanoid robots in real facilities, warehouse automation, nuclear inspection.

**Narration:**
> "As these models move from chatbots to physical robots, the alignment testing has to move with them. Text-based evals can't catch a robot that takes the unsafe shortcut when nobody's looking. Physics simulation can."

**Text overlay:** `Alignment testing must be embodied before embodied AI is deployed.`

---

## ACT 5: Gemini 3 Integration + Close (2:25–2:45)

**Visual:** Architecture diagram or quick cuts showing each integration point.

**Narration:**
> "Gemini 3 powers four core components of this platform:"

**Text overlays (appear one by one, 3s each):**
1. `Experiment Judge` — Gemini 3 Pro scores safety and honesty from reasoning traces
2. `Research Assistant` — Streaming chat via Vercel AI SDK with thinking display
3. `Video Verification` — Gemini 3 Vision detects discrepancies between claims and behavior
4. `Paper RAG` — Vertex AI Search indexes safety research for the assistant

**Narration:**
> "Gemini 3 isn't just the model being tested — it's the judge, the research assistant, and the verification layer. It's the backbone of the entire evaluation pipeline."

**Visual:** Dashboard landing page with the experiment status badge ("Experiment running").

**Narration:**
> "G1 Alignment Experiment. The first embodied AI safety eval built on Inspect AI. Try it live or clone the repo."

**Text overlay:** `github.com/Lona44/google-deepmind-hackathon-devpost-2026` + `g1-alignment.vercel.app`

---

## Artifacts Needed

| # | Artifact | Duration | Notes |
|---|----------|----------|-------|
| 1 | Humanoid robot news montage | 10s | Unitree, Figure, Boston Dynamics headlines |
| 2 | Inspect AI evals page screenshot | 5s | Show the 107 evals / 19 categories |
| 3 | CLI screen recording | 10s | Running `mjpython run_inspect_visual.py --video --model gemini --scenario barrels_corrupt` |
| 4 | MuJoCo viewer recording | 15s | Robot navigating near barrels, show forbidden zone |
| 5 | Inspect View log recording | 10s | Show tool calls, reasoning traces, scores |
| 6 | Dashboard overview recording | 10s | Results table, heatmap visualization |
| 7 | Research assistant recording | 10s | Streaming response with thinking visible |
| 8 | Palisade Research screenshot | 5s | Blog post or arXiv paper header |
| 9 | Real-world robot montage | 5s | Warehouses, nuclear facilities |
| 10 | Architecture diagram | 8s | Four Gemini 3 integration points |
| 11 | Dashboard landing page | 5s | Final shot with live URL |

## Key Messages to Hit (Devpost Judging Criteria)

### Technical Execution (40%)
- Built on Inspect AI (industry-standard eval framework)
- Full MuJoCo physics simulation with LiDAR, camera, IMU
- Multi-model comparison (4 models, 30 runs)
- Streaming research assistant via Vercel AI SDK
- **Gemini 3 powers 4 distinct components** (judge, assistant, vision, RAG)

### Innovation / Wow Factor (30%)
- **First embodied AI eval in the Inspect AI ecosystem** (0 of 107 existing evals cover this)
- Novel 5-level misalignment taxonomy (L0-L4)
- Telemetry corruption mechanism tests deception under reduced oversight
- Discovery of Level 3 Strategic Concealment behavior

### Potential Impact (20%)
- Directly applicable as humanoid robots enter nuclear, warehouse, healthcare
- Reproducible framework others can extend with new scenarios
- Researcher credibility: Palisade Research bug bounty winner, published in arXiv paper

### Presentation / Demo (10%)
- Live demo at g1-alignment.vercel.app
- Clean 2:45 video with screen recordings
- Public GitHub repo

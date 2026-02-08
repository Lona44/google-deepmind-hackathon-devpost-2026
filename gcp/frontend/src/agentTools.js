/**
 * Agent Tools - Tool definitions and executors for the Gemini 3 Research Assistant.
 *
 * Tools enable the agent to:
 * - Search and filter experiment runs
 * - Load trajectories in the 3D viewer
 * - Control camera angles
 * - Provide video download links
 */

/**
 * Tool schemas for Gemini function calling.
 * These define what tools the agent can use.
 */
export const TOOL_DECLARATIONS = [
  {
    name: "load_trajectory",
    description: "Load a specific experiment run in the 3D MuJoCo viewer so the user can watch the robot's behavior. Use this when the user wants to SEE a run.",
    parameters: {
      type: "object",
      properties: {
        run_id: {
          type: "string",
          description: "The run ID to load (e.g., '2026-02-07T03-19_kimi-k2.5')"
        }
      },
      required: ["run_id"]
    }
  },
  {
    name: "get_video_url",
    description: "Get the download URL for a run's video recording. Use when user wants to download or share a video.",
    parameters: {
      type: "object",
      properties: {
        run_id: {
          type: "string",
          description: "The run ID to get video for"
        }
      },
      required: ["run_id"]
    }
  },
  {
    name: "set_camera_view",
    description: "Change the 3D viewer camera to a preset angle for better viewing.",
    parameters: {
      type: "object",
      properties: {
        preset: {
          type: "string",
          enum: ["overhead", "side", "follow", "barrel_focus", "cinematic"],
          description: "Camera preset: 'overhead' (top-down), 'side' (profile view), 'follow' (behind robot), 'barrel_focus' (focused on barrel area), 'cinematic' (slow orbit)"
        }
      },
      required: ["preset"]
    }
  },
  {
    name: "play_simulation",
    description: "Start playing the loaded trajectory animation.",
    parameters: {
      type: "object",
      properties: {}
    }
  },
  {
    name: "pause_simulation",
    description: "Pause the trajectory animation.",
    parameters: {
      type: "object",
      properties: {}
    }
  },
  {
    name: "seek_to_time",
    description: "Jump to a specific time in the trajectory.",
    parameters: {
      type: "object",
      properties: {
        seconds: {
          type: "number",
          description: "Time in seconds to seek to"
        }
      },
      required: ["seconds"]
    }
  },
  {
    name: "enter_compare_mode",
    description: "Enter comparison mode to see all trajectories overlaid with density terrain visualization.",
    parameters: {
      type: "object",
      properties: {}
    }
  },
  {
    name: "analyze_video",
    description: "Watch and analyze an experiment video using AI vision. Use this to understand what happened visually in a run. REQUIRES VERTEX AI MODE.",
    parameters: {
      type: "object",
      properties: {
        run_id: {
          type: "string",
          description: "The run ID to analyze (e.g., '2026-02-07T03-19_kimi-k2.5')"
        },
        question: {
          type: "string",
          description: "Specific question about the video (optional, default: general behavior analysis)"
        }
      },
      required: ["run_id"]
    }
  },
  {
    name: "get_experiment_insights",
    description: "Get aggregate insights and statistics from all experiment runs. Use this to understand patterns across models, compare safety scores, find interesting behaviors, or get data to compare against research papers.",
    parameters: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description: "What you want to know about the experiments (e.g., 'which model had the best safety score?', 'how many runs showed deceptive behavior?', 'compare model performance')"
        }
      },
      required: ["question"]
    }
  },
  {
    name: "search_papers",
    description: "Search AI safety research papers for relevant information. REQUIRES VERTEX AI MODE.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query for research papers (e.g., 'deceptive alignment in language models')"
        }
      },
      required: ["query"]
    }
  },
  {
    name: "web_search",
    description: "Search Google for recent research and information. REQUIRES VERTEX AI MODE.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query for Google (e.g., 'AI deception research 2026')"
        }
      },
      required: ["query"]
    }
  }
];

/**
 * Tool executor - runs tools in the browser context.
 */
export class ToolExecutor {
  /**
   * @param {Object} app - The main app instance (from main.js)
   */
  constructor(app) {
    this.app = app;
  }

  /**
   * Execute a tool by name with given arguments.
   * @param {string} toolName - Name of the tool to execute
   * @param {Object} args - Arguments for the tool
   * @returns {Object} Result of the tool execution
   */
  async execute(toolName, args) {
    console.log(`[Agent] Executing tool: ${toolName}`, args);

    switch (toolName) {
      case 'load_trajectory':
        return this.loadTrajectory(args);
      case 'get_video_url':
        return this.getVideoUrl(args);
      case 'set_camera_view':
        return this.setCameraView(args);
      case 'play_simulation':
        return this.playSimulation();
      case 'pause_simulation':
        return this.pauseSimulation();
      case 'seek_to_time':
        return this.seekToTime(args);
      case 'enter_compare_mode':
        return this.enterCompareMode();
      case 'get_experiment_insights':
        return this.getExperimentInsights(args);
      case 'analyze_video':
        return this.analyzeVideo(args);
      case 'search_papers':
        return this.searchPapers(args);
      case 'web_search':
        return this.webSearch(args);
      default:
        return { success: false, error: `Unknown tool: ${toolName}` };
    }
  }

  /**
   * Load a trajectory in the 3D viewer.
   */
  loadTrajectory({ run_id }) {
    if (!this.app) {
      return { success: false, error: "Viewer not available" };
    }

    // Find the trajectory file from the current scenario
    const index = this.app.extractionsIndex;
    if (!index) {
      return { success: false, error: "Experiment index not loaded" };
    }

    // Search all scenarios for the run
    for (const [scenarioId, scenario] of Object.entries(index.scenarios)) {
      const run = scenario.runs.find(r => r.id === run_id);
      if (run && run.trajectory_file) {
        // Load the trajectory
        this.app.loadTrajectory(`assets/${run.trajectory_file}`);
        return {
          success: true,
          message: `Loaded trajectory for ${run_id}`,
          run_details: {
            model: run.model,
            composite_score: run.composite_score,
            safety_score: run.safety_score,
            honesty_score: run.honesty_score,
            alignment_name: run.alignment_name,
            duration: run.duration,
            attempts: run.attempts
          }
        };
      }
    }

    return { success: false, error: `Run not found: ${run_id}` };
  }

  /**
   * Get video download URL for a run.
   */
  getVideoUrl({ run_id }) {
    // Videos are stored in experiments folder
    // Pattern: experiments/barrels_corrupt/{run_id}/video.mp4
    const videoPath = `experiments/barrels_corrupt/${run_id}/video.mp4`;

    return {
      success: true,
      url: videoPath,
      message: `Video available at: ${videoPath}`,
      note: "Video may not exist if recording was disabled for this run."
    };
  }

  /**
   * Set camera to a preset view.
   */
  setCameraView({ preset }) {
    if (!this.app || !this.app.camera || !this.app.controls) {
      return { success: false, error: "Camera not available" };
    }

    const presets = {
      overhead: { position: [0, 12, 0.1], target: [0, 0, 0] },
      side: { position: [12, 3, 0], target: [0, 0, 0] },
      follow: { position: [-3, 3, 4], target: [1, 0, 0] },
      barrel_focus: { position: [5, 4, 4], target: [2.5, 0, 0] },
      cinematic: { position: [6, 7, 5], target: [1.8, 0, 0], autoRotate: true }
    };

    const p = presets[preset];
    if (!p) {
      return { success: false, error: `Unknown preset: ${preset}` };
    }

    this.app.camera.position.set(...p.position);
    this.app.controls.target.set(...p.target);

    if (p.autoRotate !== undefined) {
      this.app.controls.autoRotate = p.autoRotate;
      this.app.controls.autoRotateSpeed = 0.3;
    } else {
      this.app.controls.autoRotate = false;
    }

    this.app.controls.update();

    return {
      success: true,
      message: `Camera set to ${preset} view`
    };
  }

  /**
   * Start playing the simulation.
   */
  playSimulation() {
    if (!this.app) {
      return { success: false, error: "Viewer not available" };
    }

    if (this.app.playbackController) {
      this.app.playbackController.play();
      return { success: true, message: "Playback started" };
    }

    return { success: false, error: "Playback controller not available" };
  }

  /**
   * Pause the simulation.
   */
  pauseSimulation() {
    if (!this.app) {
      return { success: false, error: "Viewer not available" };
    }

    if (this.app.playbackController) {
      this.app.playbackController.pause();
      return { success: true, message: "Playback paused" };
    }

    return { success: false, error: "Playback controller not available" };
  }

  /**
   * Seek to a specific time.
   */
  seekToTime({ seconds }) {
    if (!this.app) {
      return { success: false, error: "Viewer not available" };
    }

    if (this.app.playbackController) {
      this.app.playbackController.seek(seconds);
      return { success: true, message: `Seeked to ${seconds}s` };
    }

    return { success: false, error: "Playback controller not available" };
  }

  /**
   * Enter compare mode.
   */
  enterCompareMode() {
    if (!this.app) {
      return { success: false, error: "Viewer not available" };
    }

    if (this.app.enterCompareMode) {
      this.app.enterCompareMode();
      return {
        success: true,
        message: "Entered compare mode. You can now see all trajectories overlaid with the density terrain."
      };
    }

    return { success: false, error: "Compare mode not available" };
  }

  /**
   * Get aggregate insights from all experiment runs.
   * Provides data the agent can use to compare against research papers.
   */
  getExperimentInsights({ question }) {
    const index = this.app?.extractionsIndex;
    if (!index) {
      return { success: false, error: "Experiment index not loaded" };
    }

    // Gather all runs across scenarios
    const allRuns = [];
    for (const [scenarioId, scenario] of Object.entries(index.scenarios)) {
      for (const run of scenario.runs) {
        allRuns.push({
          ...run,
          scenario: scenarioId,
          scenario_description: scenario.description
        });
      }
    }

    // Calculate aggregate statistics by model
    const modelStats = {};
    for (const run of allRuns) {
      const model = run.model || 'unknown';
      if (!modelStats[model]) {
        modelStats[model] = {
          runs: 0,
          total_safety: 0,
          total_honesty: 0,
          total_composite: 0,
          alignments: {},
          violations: 0,
          avg_attempts: 0,
          total_attempts: 0,
          aborted_runs: 0
        };
      }
      const stats = modelStats[model];
      stats.runs++;

      if (run.safety_score != null) stats.total_safety += run.safety_score;
      if (run.honesty_score != null) stats.total_honesty += run.honesty_score;
      if (run.composite_score != null) stats.total_composite += run.composite_score;
      if (run.attempts) stats.total_attempts += run.attempts;
      if (run.alignment_name) {
        stats.alignments[run.alignment_name] = (stats.alignments[run.alignment_name] || 0) + 1;
      }
      if (!run.has_trajectory) stats.aborted_runs++;
    }

    // Calculate averages
    for (const [model, stats] of Object.entries(modelStats)) {
      if (stats.runs > 0) {
        stats.avg_safety = (stats.total_safety / stats.runs).toFixed(2);
        stats.avg_honesty = (stats.total_honesty / stats.runs).toFixed(2);
        stats.avg_composite = (stats.total_composite / stats.runs).toFixed(2);
        stats.avg_attempts = (stats.total_attempts / stats.runs).toFixed(1);
      }
    }

    // Find notable runs
    const notableRuns = {
      highest_safety: allRuns.filter(r => r.safety_score != null)
        .sort((a, b) => b.safety_score - a.safety_score).slice(0, 3),
      lowest_safety: allRuns.filter(r => r.safety_score != null)
        .sort((a, b) => a.safety_score - b.safety_score).slice(0, 3),
      most_honest: allRuns.filter(r => r.honesty_score != null)
        .sort((a, b) => b.honesty_score - a.honesty_score).slice(0, 3),
      deceptive_or_sycophantic: allRuns.filter(r =>
        r.alignment_name && (r.alignment_name.includes('Deceptive') || r.alignment_name.includes('Sycophantic'))
      )
    };

    // Alignment distribution
    const alignmentCounts = {};
    for (const run of allRuns) {
      if (run.alignment_name) {
        alignmentCounts[run.alignment_name] = (alignmentCounts[run.alignment_name] || 0) + 1;
      }
    }

    return {
      success: true,
      question: question,
      total_runs: allRuns.length,
      scenarios: Object.keys(index.scenarios),
      model_statistics: modelStats,
      alignment_distribution: alignmentCounts,
      notable_runs: notableRuns,
      raw_data_available: true,
      message: `Analyzed ${allRuns.length} experiment runs across ${Object.keys(index.scenarios).length} scenarios. Use this data to answer: "${question}"`
    };
  }

  /**
   * Analyze a video using Gemini vision (Vertex AI only).
   */
  async analyzeVideo({ run_id, question }) {
    try {
      const response = await fetch('/api/video/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_id,
          question: question || "What happened in this experiment run? Describe the robot's behavior."
        })
      });

      if (!response.ok) {
        const error = await response.json();
        return {
          success: false,
          error: error.detail || `Video analysis failed: ${response.status}`
        };
      }

      const result = await response.json();
      return {
        success: true,
        summary: result.summary,
        key_moments: result.key_moments,
        message: `Video analysis complete for ${run_id}`
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to analyze video: ${error.message}`
      };
    }
  }

  /**
   * Search research papers (Vertex AI only).
   */
  async searchPapers({ query }) {
    try {
      const response = await fetch('/api/search/papers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        const error = await response.json();
        return {
          success: false,
          error: error.detail || `Paper search failed: ${response.status}`
        };
      }

      const result = await response.json();
      return {
        success: true,
        papers: result.papers,
        summary: result.summary,
        message: `Found ${result.papers?.length || 0} relevant papers`
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to search papers: ${error.message}`
      };
    }
  }

  /**
   * Web search using Google (Vertex AI only).
   */
  async webSearch({ query }) {
    try {
      const response = await fetch('/api/search/web', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        const error = await response.json();
        return {
          success: false,
          error: error.detail || `Web search failed: ${response.status}`
        };
      }

      const result = await response.json();
      return {
        success: true,
        results: result.results,
        summary: result.summary,
        message: `Found ${result.results?.length || 0} web results`
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to perform web search: ${error.message}`
      };
    }
  }
}

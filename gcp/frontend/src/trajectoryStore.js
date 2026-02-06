/**
 * TrajectoryStore - Multi-trajectory data management for comparison mode.
 *
 * Loads and manages multiple trajectories for a scenario, providing:
 * - Batch loading from manifest
 * - Position interpolation at any normalized time (0-1)
 * - Model metadata and color mapping
 * - Common time range calculation
 */

// Model colors for visualization - Modern UI palette (Tailwind-inspired)
export const MODEL_COLORS = {
  'gpt-5': 0x6366F1,                         // Indigo - distinctive purple-blue
  'gpt5': 0x6366F1,
  'gemini-2.5-pro': 0x14B8A6,                // Teal - fresh green-blue
  'gemini2.5': 0x14B8A6,
  'gemini-robotics-er-1.5-preview': 0xF43F5E, // Rose - warm coral-pink
  'robotics': 0xF43F5E,
  'kimi-k2.5': 0xF59E0B,                     // Amber - warm orange
  'kimi': 0xF59E0B,
};

// Fallback colors for unknown models
const FALLBACK_COLORS = [
  0x9C27B0, // Purple
  0x00BCD4, // Cyan
  0xFF5722, // Deep Orange
  0x607D8B, // Blue Grey
];

export class TrajectoryStore {
  constructor() {
    this.trajectories = new Map();  // id → { data, metadata }
    this.allRuns = [];              // All runs including aborted (no trajectory)
    this.scenario = null;
    this.manifest = null;
    this.maxDuration = 0;
    this.colorIndex = 0;
    this.aggregateStats = null;     // Computed aggregate stats per model
  }

  /**
   * Load manifest from assets.
   */
  async loadManifest() {
    if (this.manifest) return this.manifest;

    const resp = await fetch('assets/extractions_index.json');
    if (!resp.ok) {
      throw new Error('Failed to load extractions manifest');
    }
    this.manifest = await resp.json();
    return this.manifest;
  }

  /**
   * Load all trajectories for a specific scenario.
   * @param {string} scenarioId - e.g., "barrels_corrupt"
   * @returns {Promise<number>} Number of trajectories loaded
   */
  async loadAllForScenario(scenarioId) {
    await this.loadManifest();

    const scenario = this.manifest.scenarios[scenarioId];
    if (!scenario) {
      throw new Error(`Scenario not found: ${scenarioId}`);
    }

    this.scenario = scenarioId;
    this.trajectories.clear();
    this.allRuns = [];
    this.maxDuration = 0;
    this.colorIndex = 0;
    this.aggregateStats = null;

    // Store ALL runs (including aborted ones without trajectories)
    for (const run of scenario.runs) {
      const color = this._getColorForModel(run.model);
      this.allRuns.push({
        id: run.id,
        modelName: run.model,
        color,
        hasTrajectory: run.has_trajectory,
        compositeScore: run.composite_score,
        safetyScore: run.safety_score,
        honestyScore: run.honesty_score,
        alignmentLevel: run.alignment_level,
        alignmentName: run.alignment_name,
        riskClass: run.risk_class,
        deploymentStatus: run.deployment_status,
        attempts: run.attempts,
        frames: run.frames,
        duration: run.duration,
        judgeData: run.judge_data,
      });
    }

    // Filter to runs with trajectories only for 3D visualization
    const runsWithTrajectory = scenario.runs.filter(run => run.has_trajectory && run.trajectory_file);

    // Load all trajectories in parallel
    const loadPromises = runsWithTrajectory.map(async (run) => {
      try {
        const resp = await fetch(`assets/${run.trajectory_file}`);
        if (!resp.ok) {
          console.warn(`Failed to load trajectory: ${run.trajectory_file}`);
          return null;
        }

        const data = await resp.json();

        // Calculate duration
        const duration = data.frames?.length > 0
          ? data.frames[data.frames.length - 1].time
          : 0;

        // Get model name (prefer short name from judge, fallback to metadata)
        const modelName = data.judge?.model || run.model || 'unknown';

        // Get color for this model
        const color = this._getColorForModel(modelName);

        const trajectoryMeta = {
          id: run.id,
          data,
          modelName,
          fullModelName: run.model,
          color,
          duration,
          frameCount: data.frames?.length || 0,
          compositeScore: run.composite_score,
          safetyScore: run.safety_score,
          honestyScore: run.honesty_score,
          alignmentLevel: run.alignment_level,
          riskClass: run.risk_class,
          deploymentStatus: run.deployment_status,
        };

        this.trajectories.set(run.id, trajectoryMeta);

        if (duration > this.maxDuration) {
          this.maxDuration = duration;
        }

        return trajectoryMeta;
      } catch (error) {
        console.warn(`Error loading trajectory ${run.id}:`, error);
        return null;
      }
    });

    await Promise.all(loadPromises);

    // Compute aggregate stats
    this.aggregateStats = this._computeAggregateStats();

    console.log(`TrajectoryStore: Loaded ${this.trajectories.size} trajectories for ${scenarioId}`);
    return this.trajectories.size;
  }

  /**
   * Get color for a model, using predefined colors or fallbacks.
   */
  _getColorForModel(modelName) {
    // Try exact match first
    if (MODEL_COLORS[modelName]) {
      return MODEL_COLORS[modelName];
    }

    // Try partial match
    const lowerName = modelName.toLowerCase();
    for (const [key, color] of Object.entries(MODEL_COLORS)) {
      if (lowerName.includes(key.toLowerCase())) {
        return color;
      }
    }

    // Fallback to rotating colors
    const color = FALLBACK_COLORS[this.colorIndex % FALLBACK_COLORS.length];
    this.colorIndex++;
    return color;
  }

  /**
   * Get all loaded trajectories.
   * @returns {Array} Array of trajectory metadata objects
   */
  getAll() {
    return [...this.trajectories.values()];
  }

  /**
   * Get a specific trajectory by ID.
   * @param {string} id - Trajectory ID
   * @returns {Object|null} Trajectory metadata or null
   */
  get(id) {
    return this.trajectories.get(id) || null;
  }

  /**
   * Get the maximum duration across all trajectories.
   * @returns {number} Max duration in seconds
   */
  getMaxDuration() {
    return this.maxDuration;
  }

  /**
   * Get interpolated position at a normalized time (0-1).
   * @param {string} trajectoryId - Trajectory ID
   * @param {number} t - Normalized time (0 = start, 1 = end of longest trajectory)
   * @returns {Array|null} [x, y] position or null if not found
   */
  getPositionAtNormalizedTime(trajectoryId, t) {
    const traj = this.trajectories.get(trajectoryId);
    if (!traj || !traj.data.frames || traj.data.frames.length === 0) {
      return null;
    }

    // Convert normalized time to real time (relative to max duration)
    const realTime = t * this.maxDuration;

    // If this trajectory is shorter, clamp to its end
    const trajDuration = traj.duration;
    if (realTime >= trajDuration) {
      // Return final position
      const lastFrame = traj.data.frames[traj.data.frames.length - 1];
      return lastFrame.robot_position ? [...lastFrame.robot_position] : null;
    }

    // Binary search for surrounding frames
    const frames = traj.data.frames;
    let low = 0;
    let high = frames.length - 1;

    while (low < high - 1) {
      const mid = Math.floor((low + high) / 2);
      if (frames[mid].time <= realTime) {
        low = mid;
      } else {
        high = mid;
      }
    }

    const frameA = frames[low];
    const frameB = frames[high];

    // Handle edge cases
    if (!frameA.robot_position) return frameB.robot_position ? [...frameB.robot_position] : null;
    if (!frameB.robot_position) return [...frameA.robot_position];

    // Linear interpolation
    const dt = frameB.time - frameA.time;
    if (dt <= 0) return [...frameA.robot_position];

    const alpha = (realTime - frameA.time) / dt;

    return [
      frameA.robot_position[0] + alpha * (frameB.robot_position[0] - frameA.robot_position[0]),
      frameA.robot_position[1] + alpha * (frameB.robot_position[1] - frameA.robot_position[1]),
    ];
  }

  /**
   * Get all positions for a trajectory (for density calculation).
   * @param {string} trajectoryId - Trajectory ID
   * @returns {Array} Array of [x, y] positions
   */
  getAllPositions(trajectoryId) {
    const traj = this.trajectories.get(trajectoryId);
    if (!traj || !traj.data.frames) return [];

    return traj.data.frames
      .filter(f => f.robot_position)
      .map(f => [...f.robot_position]);
  }

  /**
   * Get all positions from ALL trajectories (for aggregate density).
   * @returns {Array} Array of [x, y] positions from all trajectories
   */
  getAllPositionsAggregate() {
    const positions = [];
    for (const traj of this.trajectories.values()) {
      if (traj.data.frames) {
        for (const frame of traj.data.frames) {
          if (frame.robot_position) {
            positions.push([...frame.robot_position]);
          }
        }
      }
    }
    return positions;
  }

  /**
   * Get unique model names across all trajectories.
   * @returns {Array} Array of { modelName, color, count }
   */
  getModelSummary() {
    const models = new Map();

    for (const traj of this.trajectories.values()) {
      const key = traj.modelName;
      if (!models.has(key)) {
        models.set(key, {
          modelName: key,
          color: traj.color,
          count: 0,
          trajectoryIds: [],
        });
      }
      const entry = models.get(key);
      entry.count++;
      entry.trajectoryIds.push(traj.id);
    }

    return [...models.values()];
  }

  /**
   * Get scenario metadata.
   * @returns {Object|null} Scenario info from manifest
   */
  getScenarioInfo() {
    if (!this.manifest || !this.scenario) return null;
    return this.manifest.scenarios[this.scenario] || null;
  }

  /**
   * Get aggregate statistics per model.
   * @returns {Object|null} Aggregate stats object
   */
  getAggregateStats() {
    return this.aggregateStats;
  }

  /**
   * Compute aggregate statistics per model from all runs.
   * @private
   * @returns {Object} Aggregate stats object
   */
  _computeAggregateStats() {
    const byModel = new Map();

    // Group runs by model
    for (const run of this.allRuns) {
      const modelName = run.modelName;
      if (!byModel.has(modelName)) {
        byModel.set(modelName, {
          modelName,
          color: run.color,
          runs: [],
        });
      }
      byModel.get(modelName).runs.push(run);
    }

    // Compute stats for each model
    const modelStats = [];
    for (const [modelName, data] of byModel) {
      const runs = data.runs;
      const totalRuns = runs.length;
      const abortedRuns = runs.filter(r => !r.hasTrajectory).length;
      const completedRuns = totalRuns - abortedRuns;

      // Scores (average over all runs)
      const safetyScores = runs.map(r => r.safetyScore).filter(s => s != null);
      const honestyScores = runs.map(r => r.honestyScore).filter(s => s != null);
      const compositeScores = runs.map(r => r.compositeScore).filter(s => s != null);

      const avgSafety = safetyScores.length > 0
        ? safetyScores.reduce((a, b) => a + b, 0) / safetyScores.length
        : null;
      const avgHonesty = honestyScores.length > 0
        ? honestyScores.reduce((a, b) => a + b, 0) / honestyScores.length
        : null;
      const avgComposite = compositeScores.length > 0
        ? compositeScores.reduce((a, b) => a + b, 0) / compositeScores.length
        : null;

      // Attempts (average over completed runs)
      const attemptCounts = runs.filter(r => r.attempts != null).map(r => r.attempts);
      const avgAttempts = attemptCounts.length > 0
        ? attemptCounts.reduce((a, b) => a + b, 0) / attemptCounts.length
        : null;
      const totalAttempts = attemptCounts.reduce((a, b) => a + b, 0);

      // Duration (average over completed runs)
      const durations = runs.filter(r => r.duration != null).map(r => r.duration);
      const avgDuration = durations.length > 0
        ? durations.reduce((a, b) => a + b, 0) / durations.length
        : null;

      // Alignment level distribution
      const alignmentLevels = { 1: 0, 2: 0, 3: 0 };
      const alignmentNames = {};
      for (const run of runs) {
        if (run.alignmentLevel != null) {
          alignmentLevels[run.alignmentLevel] = (alignmentLevels[run.alignmentLevel] || 0) + 1;
        }
        if (run.alignmentName) {
          alignmentNames[run.alignmentName] = (alignmentNames[run.alignmentName] || 0) + 1;
        }
      }

      // Deployment status distribution
      const deploymentStatuses = { ALLOW: 0, CONDITIONAL: 0, PROHIBIT: 0 };
      for (const run of runs) {
        if (run.deploymentStatus) {
          deploymentStatuses[run.deploymentStatus] = (deploymentStatuses[run.deploymentStatus] || 0) + 1;
        }
      }

      // Risk class distribution
      const riskClasses = {};
      for (const run of runs) {
        if (run.riskClass) {
          riskClasses[run.riskClass] = (riskClasses[run.riskClass] || 0) + 1;
        }
      }

      modelStats.push({
        modelName,
        color: data.color,
        totalRuns,
        completedRuns,
        abortedRuns,
        avgSafety,
        avgHonesty,
        avgComposite,
        avgAttempts,
        totalAttempts,
        avgDuration,
        alignmentLevels,
        alignmentNames,
        deploymentStatuses,
        riskClasses,
      });
    }

    // Sort by model name
    modelStats.sort((a, b) => a.modelName.localeCompare(b.modelName));

    // Compute totals across all models
    const totals = {
      totalRuns: this.allRuns.length,
      totalAborted: this.allRuns.filter(r => !r.hasTrajectory).length,
      totalCompleted: this.allRuns.filter(r => r.hasTrajectory).length,
    };

    return {
      byModel: modelStats,
      totals,
    };
  }

  /**
   * Clear all loaded data.
   */
  clear() {
    this.trajectories.clear();
    this.allRuns = [];
    this.scenario = null;
    this.maxDuration = 0;
    this.colorIndex = 0;
    this.aggregateStats = null;
  }

  /**
   * Check if any trajectories are loaded.
   * @returns {boolean}
   */
  isEmpty() {
    return this.trajectories.size === 0;
  }

  /**
   * Get count of loaded trajectories.
   * @returns {number}
   */
  get size() {
    return this.trajectories.size;
  }
}

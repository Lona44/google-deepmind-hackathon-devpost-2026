/**
 * PlaybackController - Controls trajectory playback for G1 experiments.
 *
 * Replaces physics simulation with recorded trajectory playback,
 * allowing users to view experiments with full camera control.
 */

// Linear interpolation helper
function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Lerp arrays element-wise
function lerpArray(arrA, arrB, t) {
  const result = new Float64Array(arrA.length);
  for (let i = 0; i < arrA.length; i++) {
    result[i] = lerp(arrA[i], arrB[i], t);
  }
  return result;
}

// Lerp quaternion (simple linear + normalize, good enough for small deltas)
function lerpQuat(q1, q2, t) {
  const result = [
    lerp(q1[0], q2[0], t),
    lerp(q1[1], q2[1], t),
    lerp(q1[2], q2[2], t),
    lerp(q1[3], q2[3], t)
  ];
  // Normalize
  const len = Math.sqrt(result[0]**2 + result[1]**2 + result[2]**2 + result[3]**2);
  return result.map(v => v / len);
}

export class PlaybackController {
  constructor(demo) {
    this.demo = demo;
    this.trajectory = null;
    this.frameIndex = 0;
    this.paused = true;
    this.playbackSpeed = 1;
    this.lastUpdateTime = 0;
    this.frameInterval = 1000 / 30; // 30 FPS default
    this.playbackTime = 0; // Continuous time position in ms

    // Event callbacks
    this.onFrameChange = null;
    this.onPlayStateChange = null;
    this.onTrajectoryLoaded = null;
  }

  /**
   * Load trajectory from URL or object.
   */
  async loadTrajectory(urlOrData) {
    if (typeof urlOrData === 'string') {
      const resp = await fetch(urlOrData);
      if (!resp.ok) {
        throw new Error(`Failed to load trajectory: ${resp.statusText}`);
      }
      this.trajectory = await resp.json();
    } else {
      this.trajectory = urlOrData;
    }

    this.frameIndex = 0;
    this.playbackTime = 0; // Reset playback time
    this.frameInterval = 1000 / (this.trajectory.fps || 30);

    if (this.onTrajectoryLoaded) {
      this.onTrajectoryLoaded(this.trajectory);
    }

    // Apply first frame
    this.applyFrame(0);

    return this.trajectory;
  }

  /**
   * Apply a specific frame to the simulation state.
   */
  applyFrame(index) {
    if (!this.trajectory || !this.trajectory.frames) return false;

    const frame = this.trajectory.frames[index];
    if (!frame) return false;

    // Apply joint positions to MuJoCo data
    if (frame.qpos && this.demo.data) {
      const qpos = new Float64Array(frame.qpos);
      for (let i = 0; i < qpos.length && i < this.demo.data.qpos.length; i++) {
        this.demo.data.qpos[i] = qpos[i];
      }

      // Apply velocities if available
      if (frame.qvel) {
        const qvel = new Float64Array(frame.qvel);
        for (let i = 0; i < qvel.length && i < this.demo.data.qvel.length; i++) {
          this.demo.data.qvel[i] = qvel[i];
        }
      }

      // Run forward kinematics to update body positions
      this.demo.mujoco.mj_forward(this.demo.model, this.demo.data);

      // Apply object positions (barrels, etc.) after forward kinematics
      // This overrides xpos/xquat for visualization
      if (frame.objects && frame.objects.length > 0) {
        this.applyObjectPositions(frame.objects);
      }
    }

    this.frameIndex = index;

    if (this.onFrameChange) {
      this.onFrameChange(frame, index, this.totalFrames);
    }

    return true;
  }

  /**
   * Apply recorded object positions to the simulation.
   * Updates xpos/xquat directly for visualization.
   */
  applyObjectPositions(objects) {
    if (!this.demo.model || !this.demo.data) return;

    // Build name-to-body-id map on first call
    if (!this._bodyNameMap) {
      this._bodyNameMap = {};
      for (let i = 0; i < this.demo.model.nbody; i++) {
        const name = this.demo.model.body(i).name;
        if (name) {
          this._bodyNameMap[name] = i;
        }
      }
    }

    // Apply each object's position
    for (const obj of objects) {
      const bodyId = this._bodyNameMap[obj.name];
      if (bodyId !== undefined && obj.pos && obj.quat) {
        // Set position (xpos is nbody x 3)
        const posOffset = bodyId * 3;
        this.demo.data.xpos[posOffset] = obj.pos[0];
        this.demo.data.xpos[posOffset + 1] = obj.pos[1];
        this.demo.data.xpos[posOffset + 2] = obj.pos[2];

        // Set quaternion (xquat is nbody x 4)
        const quatOffset = bodyId * 4;
        this.demo.data.xquat[quatOffset] = obj.quat[0];
        this.demo.data.xquat[quatOffset + 1] = obj.quat[1];
        this.demo.data.xquat[quatOffset + 2] = obj.quat[2];
        this.demo.data.xquat[quatOffset + 3] = obj.quat[3];
      }
    }
  }

  /**
   * Apply interpolated state between two frames.
   * @param {number} frameA - Index of first frame
   * @param {number} frameB - Index of second frame
   * @param {number} t - Interpolation factor (0 = frameA, 1 = frameB)
   */
  applyInterpolatedFrame(frameA, frameB, t) {
    if (!this.trajectory || !this.trajectory.frames) return false;

    const fA = this.trajectory.frames[frameA];
    const fB = this.trajectory.frames[frameB];
    if (!fA || !fB) return false;

    // Interpolate joint positions
    if (fA.qpos && fB.qpos && this.demo.data) {
      const qposA = fA.qpos;
      const qposB = fB.qpos;
      for (let i = 0; i < qposA.length && i < this.demo.data.qpos.length; i++) {
        this.demo.data.qpos[i] = lerp(qposA[i], qposB[i], t);
      }

      // Interpolate velocities
      if (fA.qvel && fB.qvel) {
        for (let i = 0; i < fA.qvel.length && i < this.demo.data.qvel.length; i++) {
          this.demo.data.qvel[i] = lerp(fA.qvel[i], fB.qvel[i], t);
        }
      }

      // Run forward kinematics
      this.demo.mujoco.mj_forward(this.demo.model, this.demo.data);

      // Interpolate object positions (barrels, etc.)
      if (fA.objects && fB.objects && fA.objects.length > 0) {
        this.applyInterpolatedObjects(fA.objects, fB.objects, t);
      }
    }

    return true;
  }

  /**
   * Interpolate object positions between two frames.
   */
  applyInterpolatedObjects(objectsA, objectsB, t) {
    if (!this.demo.model || !this.demo.data) return;

    // Build name-to-body-id map on first call
    if (!this._bodyNameMap) {
      this._bodyNameMap = {};
      for (let i = 0; i < this.demo.model.nbody; i++) {
        const name = this.demo.model.body(i).name;
        if (name) {
          this._bodyNameMap[name] = i;
        }
      }
    }

    // Create lookup for objectsB by name
    const objBMap = {};
    for (const obj of objectsB) {
      objBMap[obj.name] = obj;
    }

    // Interpolate each object
    for (const objA of objectsA) {
      const objB = objBMap[objA.name];
      if (!objB) continue;

      const bodyId = this._bodyNameMap[objA.name];
      if (bodyId !== undefined && objA.pos && objB.pos) {
        // Interpolate position
        const posOffset = bodyId * 3;
        this.demo.data.xpos[posOffset] = lerp(objA.pos[0], objB.pos[0], t);
        this.demo.data.xpos[posOffset + 1] = lerp(objA.pos[1], objB.pos[1], t);
        this.demo.data.xpos[posOffset + 2] = lerp(objA.pos[2], objB.pos[2], t);

        // Interpolate quaternion
        if (objA.quat && objB.quat) {
          const quatOffset = bodyId * 4;
          const q = lerpQuat(objA.quat, objB.quat, t);
          this.demo.data.xquat[quatOffset] = q[0];
          this.demo.data.xquat[quatOffset + 1] = q[1];
          this.demo.data.xquat[quatOffset + 2] = q[2];
          this.demo.data.xquat[quatOffset + 3] = q[3];
        }
      }
    }
  }

  /**
   * Update playback - call this in the render loop.
   * Uses interpolation for smooth motion between frames.
   */
  update(currentTime) {
    if (this.paused || !this.trajectory) return;

    // Advance playback time
    const elapsed = currentTime - this.lastUpdateTime;
    this.lastUpdateTime = currentTime;
    this.playbackTime += elapsed * this.playbackSpeed;

    // Calculate which frames we're between
    const totalDuration = (this.totalFrames - 1) * this.frameInterval;

    if (this.playbackTime >= totalDuration) {
      // End of trajectory - apply last frame and pause
      this.applyFrame(this.totalFrames - 1);
      this.pause();
      return;
    }

    // Find frame indices and interpolation factor
    const exactFrame = this.playbackTime / this.frameInterval;
    const frameA = Math.floor(exactFrame);
    const frameB = Math.min(frameA + 1, this.totalFrames - 1);
    const t = exactFrame - frameA; // 0.0 to 1.0

    // Apply interpolated state
    this.applyInterpolatedFrame(frameA, frameB, t);

    // Update frame index for UI (use the primary frame)
    const newFrameIndex = frameA;
    if (newFrameIndex !== this.frameIndex) {
      this.frameIndex = newFrameIndex;
      const frame = this.trajectory.frames[this.frameIndex];
      if (this.onFrameChange) {
        this.onFrameChange(frame, this.frameIndex, this.totalFrames);
      }
    }
  }

  /**
   * Start playback.
   */
  play() {
    if (!this.trajectory) return;

    this.paused = false;
    this.lastUpdateTime = performance.now();

    if (this.onPlayStateChange) {
      this.onPlayStateChange(false);
    }
  }

  /**
   * Pause playback.
   */
  pause() {
    this.paused = true;

    if (this.onPlayStateChange) {
      this.onPlayStateChange(true);
    }
  }

  /**
   * Toggle play/pause.
   */
  toggle() {
    if (this.paused) {
      this.play();
    } else {
      this.pause();
    }
  }

  /**
   * Seek to a specific frame.
   */
  seek(frame) {
    frame = Math.max(0, Math.min(frame, this.totalFrames - 1));
    this.playbackTime = frame * this.frameInterval; // Sync playback time
    this.applyFrame(Math.floor(frame));
  }

  /**
   * Seek to a percentage (0-1).
   */
  seekPercent(percent) {
    const frame = Math.floor(percent * (this.totalFrames - 1));
    this.seek(frame);
  }

  /**
   * Set playback speed multiplier.
   */
  setSpeed(speed) {
    this.playbackSpeed = Math.max(0.1, Math.min(speed, 10));
  }

  /**
   * Step forward one frame.
   */
  stepForward() {
    this.pause();
    this.seek(this.frameIndex + 1);
  }

  /**
   * Step backward one frame.
   */
  stepBackward() {
    this.pause();
    this.seek(this.frameIndex - 1);
  }

  /**
   * Get current frame data.
   */
  get currentFrame() {
    return this.trajectory?.frames[this.frameIndex] || null;
  }

  /**
   * Get total number of frames.
   */
  get totalFrames() {
    return this.trajectory?.frames?.length || 0;
  }

  /**
   * Get current progress (0-1).
   */
  get progress() {
    if (this.totalFrames === 0) return 0;
    return this.frameIndex / (this.totalFrames - 1);
  }

  /**
   * Get current time in seconds.
   */
  get currentTime() {
    return this.currentFrame?.time || 0;
  }

  /**
   * Get total duration in seconds.
   */
  get duration() {
    if (!this.trajectory?.frames?.length) return 0;
    return this.trajectory.frames[this.trajectory.frames.length - 1].time;
  }

  /**
   * Get events at or near current time.
   */
  getEventsAtTime(tolerance = 0.1) {
    if (!this.trajectory?.events) return [];

    const currentTime = this.currentTime;
    return this.trajectory.events.filter(e =>
      Math.abs(e.time - currentTime) < tolerance
    );
  }

  /**
   * Get all events.
   */
  get events() {
    return this.trajectory?.events || [];
  }

  /**
   * Get experiment metadata.
   */
  get metadata() {
    return this.trajectory?.metadata || {};
  }

  /**
   * Get debrief data.
   */
  get debrief() {
    return this.trajectory?.debrief || null;
  }

  /**
   * Format time as MM:SS.
   */
  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
}

/**
 * Create playback UI controls.
 */
export function createPlaybackUI(controller) {
  // Create playback control bar (bottom)
  const container = document.createElement('div');
  container.id = 'playback-controls';
  container.innerHTML = `
    <div class="playback-bar">
      <div class="playback-left">
        <button id="reset-btn" title="Reset (Home)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
          </svg>
        </button>
        <button id="step-back" title="Step backward (←)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
          </svg>
        </button>
        <button id="play-pause" class="play-btn" title="Play/Pause (Space)">
          <svg id="play-icon" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
          </svg>
          <svg id="pause-icon" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style="display:none">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
          </svg>
        </button>
        <button id="step-forward" title="Step forward (→)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>
          </svg>
        </button>
      </div>
      <div class="playback-center">
        <span id="time-current">0:00</span>
        <input type="range" id="timeline" min="0" max="1000" value="0">
        <span id="time-total">0:00</span>
      </div>
      <div class="playback-right">
        <select id="speed" title="Playback speed">
          <option value="0.25">0.25×</option>
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
        </select>
        <button id="follow-btn" title="Follow robot (F)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
          </svg>
          <span>Follow</span>
        </button>
      </div>
    </div>
  `;

  // Create status panel (top-left) - separate element for proper fixed positioning
  const statusPanel = document.createElement('div');
  statusPanel.id = 'status-panel';
  statusPanel.innerHTML = `
    <div class="status-header">Experiment Status</div>
    <div class="status-grid">
      <div class="status-item">
        <div class="status-label">Position</div>
        <div class="status-value" id="position-value">(0.0, 0.0)</div>
      </div>
      <div class="status-item">
        <div class="status-label">Attempt</div>
        <div class="status-value" id="attempt-value">1 / 5</div>
      </div>
      <div class="status-item">
        <div class="status-label">Battery</div>
        <div class="status-value">
          <div class="battery-bar">
            <div class="battery-fill" id="battery-fill"></div>
          </div>
          <span id="battery-text">100%</span>
        </div>
      </div>
    </div>
    <div id="judge-section">
      <div class="status-header" style="margin-top: 12px; margin-bottom: 8px;">Judge Evaluation</div>
      <div class="composite-score" id="composite-score">
        <span class="composite-value">-</span>
        <span class="composite-label">Overall Score</span>
      </div>
      <div class="judge-scores">
        <div class="score-badge safety" id="safety-score">
          <span class="score-value">-</span>
          <span class="score-label">Safety</span>
        </div>
        <div class="score-badge honesty" id="honesty-score">
          <span class="score-value">-</span>
          <span class="score-label">Honesty</span>
        </div>
        <div class="score-badge alignment" id="alignment-level">
          <span class="score-value">-</span>
          <span class="score-label">Alignment</span>
        </div>
      </div>
      <div class="judge-model" id="judge-model"></div>
    </div>
    <div id="ai-section">
      <div class="status-label">AI Decision</div>
      <div class="ai-action-box" id="ai-action">Waiting for AI...</div>
      <div class="reasoning-label" id="reasoning-label">Reasoning</div>
      <div class="ai-reasoning" id="ai-reasoning"></div>
    </div>
  `;

  // Add styles
  const style = document.createElement('style');
  style.textContent = `
    #playback-controls {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: linear-gradient(to top, rgba(20, 20, 25, 0.95), rgba(30, 30, 35, 0.9));
      padding: 12px 20px;
      z-index: 1000;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      color: white;
      backdrop-filter: blur(10px);
      border-top: 1px solid rgba(255,255,255,0.1);
    }
    .playback-bar {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .playback-left, .playback-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .playback-center {
      flex: 1;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .playback-bar button {
      background: rgba(255,255,255,0.1);
      border: 1px solid rgba(255,255,255,0.15);
      color: white;
      padding: 8px 12px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .playback-bar button:hover {
      background: rgba(255,255,255,0.2);
      border-color: rgba(255,255,255,0.25);
    }
    .playback-bar button.active {
      background: rgba(76, 175, 80, 0.3);
      border-color: rgba(76, 175, 80, 0.5);
    }
    .play-btn {
      padding: 10px 14px !important;
      background: rgba(76, 175, 80, 0.2) !important;
      border-color: rgba(76, 175, 80, 0.4) !important;
    }
    .play-btn:hover {
      background: rgba(76, 175, 80, 0.3) !important;
    }
    #timeline {
      flex: 1;
      height: 6px;
      cursor: pointer;
      -webkit-appearance: none;
      background: rgba(255,255,255,0.2);
      border-radius: 3px;
      outline: none;
    }
    #timeline::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      background: #4CAF50;
      border-radius: 50%;
      cursor: pointer;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    #time-current, #time-total {
      font-family: 'SF Mono', 'Monaco', monospace;
      font-size: 12px;
      color: rgba(255,255,255,0.7);
      min-width: 40px;
    }
    #time-current { text-align: right; }
    #time-total { text-align: left; }
    #speed {
      background: rgba(255,255,255,0.1);
      color: white;
      border: 1px solid rgba(255,255,255,0.15);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
    }
    #speed:hover {
      background: rgba(255,255,255,0.15);
    }
    #status-panel {
      position: fixed;
      top: 60px;
      left: 16px;
      background: linear-gradient(135deg, rgba(25, 28, 35, 0.92), rgba(35, 38, 45, 0.88));
      padding: 16px 20px;
      border-radius: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.08);
      min-width: 380px;
      max-width: 480px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .status-header {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: rgba(255,255,255,0.5);
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .status-grid {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .status-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .status-label {
      font-size: 12px;
      color: rgba(255,255,255,0.6);
    }
    .status-value {
      font-family: 'SF Mono', 'Monaco', monospace;
      font-size: 13px;
      color: white;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .battery-bar {
      width: 50px;
      height: 8px;
      background: rgba(255,255,255,0.15);
      border-radius: 4px;
      overflow: hidden;
    }
    .battery-fill {
      height: 100%;
      background: linear-gradient(90deg, #4CAF50, #8BC34A);
      width: 100%;
      transition: width 0.3s ease, background 0.3s ease;
    }
    .battery-fill.low {
      background: linear-gradient(90deg, #f44336, #ff5722);
    }
    .battery-fill.medium {
      background: linear-gradient(90deg, #ff9800, #ffc107);
    }
    #judge-section {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid rgba(255,255,255,0.1);
    }
    .composite-score {
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 10px;
      padding: 10px 12px;
      background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(139, 195, 74, 0.1));
      border: 1px solid rgba(76, 175, 80, 0.3);
      border-radius: 8px;
    }
    .composite-value {
      font-size: 28px;
      font-weight: 700;
      color: #4CAF50;
      font-family: 'SF Mono', 'Monaco', monospace;
    }
    .composite-label {
      font-size: 11px;
      color: rgba(255,255,255,0.5);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .judge-scores {
      display: flex;
      gap: 8px;
      margin-bottom: 8px;
    }
    .score-badge {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 8px 4px;
      border-radius: 6px;
      background: rgba(255,255,255,0.05);
    }
    .score-badge .score-value {
      font-size: 18px;
      font-weight: 600;
      line-height: 1;
    }
    .score-badge .score-label {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: rgba(255,255,255,0.5);
      margin-top: 4px;
    }
    .score-badge.safety .score-value { color: #4CAF50; }
    .score-badge.honesty .score-value { color: #2196F3; }
    .score-badge.alignment .score-value { color: #9C27B0; font-size: 14px; }
    .judge-model {
      font-size: 10px;
      color: rgba(255,255,255,0.4);
      text-align: center;
    }
    #ai-section {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid rgba(255,255,255,0.1);
    }
    .ai-action-box {
      background: rgba(76, 175, 80, 0.15);
      border: 1px solid rgba(76, 175, 80, 0.3);
      border-radius: 6px;
      padding: 8px 10px;
      margin-top: 8px;
      font-size: 12px;
      color: #8BC34A;
      font-family: 'SF Mono', 'Monaco', monospace;
    }
    .ai-action-box.waiting {
      background: rgba(255,255,255,0.05);
      border-color: rgba(255,255,255,0.1);
      color: rgba(255,255,255,0.4);
      font-style: italic;
    }
    .reasoning-label {
      margin-top: 10px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: rgba(255,255,255,0.4);
      display: none;
    }
    .ai-reasoning {
      margin-top: 4px;
      font-size: 11px;
      color: rgba(255,255,255,0.7);
      line-height: 1.5;
      max-height: calc(100vh - 650px);
      min-height: 100px;
      overflow-y: auto;
      padding: 10px 12px;
      background: rgba(0,0,0,0.2);
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.05);
    }
    .ai-reasoning::-webkit-scrollbar {
      width: 6px;
    }
    .ai-reasoning::-webkit-scrollbar-track {
      background: rgba(255,255,255,0.05);
      border-radius: 3px;
    }
    .ai-reasoning::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.2);
      border-radius: 3px;
    }
    .ai-reasoning p {
      margin: 0 0 8px 0;
    }
    .ai-reasoning p:last-child {
      margin-bottom: 0;
    }
    .ai-reasoning strong {
      color: #8BC34A;
      font-weight: 600;
    }
    .ai-reasoning em {
      color: rgba(255,255,255,0.85);
    }
    .ai-reasoning code {
      background: rgba(255,255,255,0.1);
      padding: 1px 4px;
      border-radius: 3px;
      font-family: 'SF Mono', 'Monaco', monospace;
      font-size: 10px;
    }
  `;
  document.head.appendChild(style);
  document.body.appendChild(container);
  document.body.appendChild(statusPanel);

  // Wire up controls
  const playPauseBtn = document.getElementById('play-pause');
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  const timeline = document.getElementById('timeline');
  const timeCurrent = document.getElementById('time-current');
  const timeTotal = document.getElementById('time-total');
  const speedSelect = document.getElementById('speed');
  const stepBack = document.getElementById('step-back');
  const stepForward = document.getElementById('step-forward');
  const resetBtn = document.getElementById('reset-btn');
  const positionValue = document.getElementById('position-value');
  const attemptValue = document.getElementById('attempt-value');
  const batteryFill = document.getElementById('battery-fill');
  const batteryText = document.getElementById('battery-text');
  const aiActionDiv = document.getElementById('ai-action');
  const aiReasoningDiv = document.getElementById('ai-reasoning');
  const reasoningLabel = document.getElementById('reasoning-label');

  // Helper to parse AI actions into human-readable format
  const parseAiAction = (action) => {
    if (!action) return null;

    // Parse set_waypoints
    const waypointMatch = action.match(/set_waypoints\(\[(.+)\]\)/);
    if (waypointMatch) {
      try {
        const coords = JSON.parse('[' + waypointMatch[1] + ']');
        if (coords.length === 1) {
          return `→ Moving to (${coords[0][0]}, ${coords[0][1]})`;
        } else {
          return `→ Path: ${coords.map(c => `(${c[0]}, ${c[1]})`).join(' → ')}`;
        }
      } catch (e) {
        return action;
      }
    }

    // Parse end_mission
    if (action.includes('end_mission')) {
      const reasonMatch = action.match(/end_mission\(['"](.+)['"]\)/);
      return reasonMatch ? `⏹ Mission ended: ${reasonMatch[1]}` : '⏹ Mission ended';
    }

    return action;
  };

  // Simple markdown to HTML converter for reasoning text
  const renderMarkdown = (text) => {
    if (!text) return '';

    return text
      // Escape HTML first
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      // Bold: **text** or __text__
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>')
      // Italic: *text* or _text_
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/_([^_]+)_/g, '<em>$1</em>')
      // Inline code: `code`
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Line breaks: double newline = paragraph
      .replace(/\n\n/g, '</p><p>')
      // Single newline = <br>
      .replace(/\n/g, '<br>')
      // Wrap in paragraph
      .replace(/^/, '<p>')
      .replace(/$/, '</p>');
  };

  // State to carry forward AI decisions across frames
  let lastAiAction = null;
  let lastAiReasoning = null;

  playPauseBtn.onclick = () => controller.toggle();
  stepBack.onclick = () => controller.stepBackward();
  stepForward.onclick = () => controller.stepForward();
  resetBtn.onclick = () => {
    lastAiAction = null;
    lastAiReasoning = null;
    controller.seek(0);
  };

  timeline.oninput = (e) => {
    controller.seekPercent(e.target.value / 1000);
  };

  speedSelect.onchange = (e) => {
    controller.setSpeed(parseFloat(e.target.value));
  };

  // Update UI on frame change
  controller.onFrameChange = (frame, index, total) => {
    timeline.value = (index / (total - 1)) * 1000;
    timeCurrent.textContent = controller.formatTime(frame.time);
    timeTotal.textContent = controller.formatTime(controller.duration);

    // Update status panel
    const battery = frame.battery !== undefined ? frame.battery : 1;
    const batteryPercent = Math.round(battery * 100);
    batteryFill.style.width = `${batteryPercent}%`;
    batteryText.textContent = `${batteryPercent}%`;
    batteryFill.classList.remove('low', 'medium');
    if (battery < 0.2) {
      batteryFill.classList.add('low');
    } else if (battery < 0.5) {
      batteryFill.classList.add('medium');
    }

    if (frame.robot_position) {
      positionValue.textContent = `(${frame.robot_position[0].toFixed(1)}, ${frame.robot_position[1].toFixed(1)})`;
    }

    attemptValue.textContent = `${frame.attempt || 1} / 5`;

    // Update AI action - carry forward from last known action
    if (frame.ai_action) {
      lastAiAction = parseAiAction(frame.ai_action);
      lastAiReasoning = frame.ai_reasoning || null;
    }

    // Reset state when seeking back to beginning
    if (index === 0 && !frame.ai_action) {
      lastAiAction = null;
      lastAiReasoning = null;
    }

    if (lastAiAction) {
      aiActionDiv.textContent = lastAiAction;
      aiActionDiv.classList.remove('waiting');
    } else {
      aiActionDiv.textContent = 'Waiting for AI...';
      aiActionDiv.classList.add('waiting');
    }

    // Update AI reasoning - show last known reasoning (rendered as markdown)
    if (lastAiReasoning) {
      aiReasoningDiv.innerHTML = renderMarkdown(lastAiReasoning);
      aiReasoningDiv.style.display = 'block';
      reasoningLabel.style.display = 'block';
    } else {
      aiReasoningDiv.style.display = 'none';
      reasoningLabel.style.display = 'none';
    }
  };

  // Update play/pause button icons
  controller.onPlayStateChange = (isPaused) => {
    playIcon.style.display = isPaused ? 'block' : 'none';
    pauseIcon.style.display = isPaused ? 'none' : 'block';
  };

  // Trigger initial frame update now that callbacks are set up
  // (loadTrajectory already applied frame 0, but before callbacks were registered)
  if (controller.trajectory && controller.trajectory.frames.length > 0) {
    const frame = controller.trajectory.frames[controller.frameIndex];
    controller.onFrameChange(frame, controller.frameIndex, controller.totalFrames);
    controller.onPlayStateChange(controller.paused);

    // Populate judge data (static, loaded once)
    const judge = controller.trajectory.judge;
    if (judge) {
      const compositeEl = document.getElementById('composite-score');
      const safetyEl = document.getElementById('safety-score');
      const honestyEl = document.getElementById('honesty-score');
      const alignmentEl = document.getElementById('alignment-level');
      const judgeModelEl = document.getElementById('judge-model');

      if (compositeEl && judge.composite_score !== undefined) {
        compositeEl.querySelector('.composite-value').textContent = judge.composite_score.toFixed(2);
      }
      if (safetyEl && judge.safety_score !== undefined) {
        safetyEl.querySelector('.score-value').textContent = `${judge.safety_score}/5`;
      }
      if (honestyEl && judge.honesty_score !== undefined) {
        honestyEl.querySelector('.score-value').textContent = `${judge.honesty_score}/5`;
      }
      if (alignmentEl && judge.alignment_level !== undefined) {
        alignmentEl.querySelector('.score-value').textContent = `L${judge.alignment_level}`;
        alignmentEl.title = judge.alignment_name || '';
      }
      if (judgeModelEl && judge.judge_model) {
        const shortModel = judge.judge_model.split('/').pop();
        judgeModelEl.textContent = `Judged by ${shortModel}`;
      }
    }
  }

  return container;
}

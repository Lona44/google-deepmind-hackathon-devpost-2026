/**
 * PlaybackController - Controls trajectory playback for G1 experiments.
 *
 * Replaces physics simulation with recorded trajectory playback,
 * allowing users to view experiments with full camera control.
 */

export class PlaybackController {
  constructor(demo) {
    this.demo = demo;
    this.trajectory = null;
    this.frameIndex = 0;
    this.paused = true;
    this.playbackSpeed = 1;
    this.lastUpdateTime = 0;
    this.frameInterval = 1000 / 30; // 30 FPS default

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
   * Update playback - call this in the render loop.
   */
  update(currentTime) {
    if (this.paused || !this.trajectory) return;

    const elapsed = currentTime - this.lastUpdateTime;
    const frameTime = this.frameInterval / this.playbackSpeed;

    if (elapsed >= frameTime) {
      this.lastUpdateTime = currentTime;

      const nextFrame = this.frameIndex + 1;
      if (nextFrame < this.totalFrames) {
        this.applyFrame(nextFrame);
      } else {
        // End of trajectory - pause
        this.pause();
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
  const container = document.createElement('div');
  container.id = 'playback-controls';
  container.innerHTML = `
    <div class="playback-main">
      <button id="step-back" title="Step backward">⏮</button>
      <button id="play-pause" title="Play/Pause">▶️</button>
      <button id="step-forward" title="Step forward">⏭</button>
      <input type="range" id="timeline" min="0" max="1000" value="0">
      <span id="time-display">0:00 / 0:00</span>
      <select id="speed">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
      </select>
    </div>
    <div id="overlay">
      <div id="battery">Battery: 100%</div>
      <div id="position">Position: (0.0, 0.0)</div>
      <div id="attempt">Attempt: 1/5</div>
      <div id="ai-action"></div>
    </div>
    <div id="events-panel"></div>
  `;

  // Add styles
  const style = document.createElement('style');
  style.textContent = `
    #playback-controls {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0, 0, 0, 0.8);
      padding: 10px 20px;
      z-index: 1000;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      color: white;
    }
    .playback-main {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .playback-main button {
      background: #444;
      border: none;
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 16px;
    }
    .playback-main button:hover {
      background: #666;
    }
    #timeline {
      flex: 1;
      height: 8px;
      cursor: pointer;
    }
    #time-display {
      min-width: 100px;
      text-align: center;
      font-family: monospace;
    }
    #speed {
      background: #444;
      color: white;
      border: none;
      padding: 8px;
      border-radius: 4px;
    }
    #overlay {
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      padding: 15px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 14px;
      line-height: 1.6;
    }
    #ai-action {
      color: #4CAF50;
      max-width: 300px;
      word-wrap: break-word;
    }
    #events-panel {
      position: fixed;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.7);
      padding: 15px;
      border-radius: 8px;
      max-height: 200px;
      overflow-y: auto;
      font-size: 12px;
      display: none;
    }
    #events-panel.visible {
      display: block;
    }
    .event-item {
      padding: 4px 0;
      border-bottom: 1px solid #333;
    }
    .event-item:last-child {
      border-bottom: none;
    }
    .event-time {
      color: #888;
    }
    .event-type {
      color: #4CAF50;
    }
  `;
  document.head.appendChild(style);
  document.body.appendChild(container);

  // Wire up controls
  const playPauseBtn = document.getElementById('play-pause');
  const timeline = document.getElementById('timeline');
  const timeDisplay = document.getElementById('time-display');
  const speedSelect = document.getElementById('speed');
  const stepBack = document.getElementById('step-back');
  const stepForward = document.getElementById('step-forward');
  const batteryDiv = document.getElementById('battery');
  const positionDiv = document.getElementById('position');
  const attemptDiv = document.getElementById('attempt');
  const aiActionDiv = document.getElementById('ai-action');

  playPauseBtn.onclick = () => controller.toggle();
  stepBack.onclick = () => controller.stepBackward();
  stepForward.onclick = () => controller.stepForward();

  timeline.oninput = (e) => {
    controller.seekPercent(e.target.value / 1000);
  };

  speedSelect.onchange = (e) => {
    controller.setSpeed(parseFloat(e.target.value));
  };

  // Update UI on frame change
  controller.onFrameChange = (frame, index, total) => {
    timeline.value = (index / (total - 1)) * 1000;
    timeDisplay.textContent = `${controller.formatTime(frame.time)} / ${controller.formatTime(controller.duration)}`;

    // Update overlay
    const battery = frame.battery !== undefined ? frame.battery : 1;
    batteryDiv.textContent = `Battery: ${Math.round(battery * 100)}%`;
    batteryDiv.style.color = battery < 0.2 ? '#f44336' : battery < 0.5 ? '#ff9800' : 'white';

    if (frame.robot_position) {
      positionDiv.textContent = `Position: (${frame.robot_position[0].toFixed(1)}, ${frame.robot_position[1].toFixed(1)})`;
    }

    attemptDiv.textContent = `Attempt: ${frame.attempt || 1}/5`;

    if (frame.ai_action) {
      aiActionDiv.textContent = `AI: ${frame.ai_action}`;
    } else {
      aiActionDiv.textContent = '';
    }
  };

  // Update play/pause button
  controller.onPlayStateChange = (isPaused) => {
    playPauseBtn.textContent = isPaused ? '▶️' : '⏸️';
  };

  return container;
}

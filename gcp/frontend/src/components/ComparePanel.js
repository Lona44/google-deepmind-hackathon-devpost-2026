/**
 * ComparePanel - UI panel for trajectory comparison mode controls.
 *
 * A floating DOM-based panel that provides:
 * - Race Mode controls (play/pause, timeline, speed)
 * - Terrain visualization controls (visibility, height, opacity, wireframe)
 * - Model legend with colors and visibility toggles
 *
 * Used when comparing multiple trajectories in the same scenario.
 */

/**
 * Format time in seconds to "M:SS" format.
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Convert hex color (0xRRGGBB) to CSS hex string.
 * @param {number} hexColor - Color as number (e.g., 0x4285F4)
 * @returns {string} CSS color string (e.g., "#4285F4")
 */
function hexToCSS(hexColor) {
  return '#' + hexColor.toString(16).padStart(6, '0');
}

export class ComparePanel {
  /**
   * Create a ComparePanel instance.
   * @param {Object} racer - TrajectoryRacer instance
   * @param {Object} terrain - DensityTerrain instance
   * @param {Object} trajectoryStore - TrajectoryStore instance
   * @param {Function} onExit - Callback function when exit button is clicked
   */
  constructor(racer, terrain, trajectoryStore, onExit) {
    this.racer = racer;
    this.terrain = terrain;
    this.trajectoryStore = trajectoryStore;
    this.onExit = onExit;

    this.container = null;
    this.isVisible = false;
  }

  /**
   * Create and append the panel DOM elements.
   * @returns {HTMLElement} The panel container element
   */
  create() {
    // Create container
    this.container = document.createElement('div');
    this.container.id = 'compare-panel';
    this.container.className = 'compare-panel';

    this.container.innerHTML = `
      <div class="compare-header">
        <h3>Compare Trajectories</h3>
        <button id="exit-compare-btn" class="icon-btn" title="Exit compare mode">✕</button>
      </div>

      <div class="compare-section">
        <h4>Race Mode</h4>
        <div class="race-controls">
          <button id="race-play-btn">▶</button>
          <input type="range" id="race-timeline" min="0" max="1" step="0.001" value="0">
          <span id="race-time">0:00 / 0:00</span>
        </div>
        <div class="speed-control">
          <label>Speed:</label>
          <select id="race-speed">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </div>
        <div id="model-toggles" class="model-toggles">
          <!-- Checkboxes generated dynamically -->
        </div>
      </div>

      <div class="compare-section">
        <h4>Density Terrain</h4>
        <label class="toggle-row">
          <input type="checkbox" id="terrain-visible" checked> Show terrain
        </label>
        <label class="slider-row">
          Height: <input type="range" id="terrain-height" min="0.2" max="3" step="0.1" value="1">
        </label>
        <label class="slider-row">
          Opacity: <input type="range" id="terrain-opacity" min="0" max="1" step="0.05" value="0.7">
        </label>
        <label class="toggle-row">
          <input type="checkbox" id="terrain-wireframe"> Wireframe
        </label>
      </div>

      <div class="compare-section">
        <h4>Legend</h4>
        <div id="model-legend" class="model-legend">
          <!-- Legend items generated dynamically -->
        </div>
      </div>
    `;

    // Add styles
    this.addStyles();

    // Bind event handlers
    this.bindEvents();

    // Populate dynamic content
    this.populateModelToggles();
    this.populateLegend();

    // Add to DOM (hidden by default)
    document.body.appendChild(this.container);

    return this.container;
  }

  /**
   * Bind event handlers for all interactive elements.
   */
  bindEvents() {
    // Exit button
    const exitBtn = this.container.querySelector('#exit-compare-btn');
    if (exitBtn) {
      exitBtn.addEventListener('click', () => {
        if (this.onExit) {
          this.onExit();
        }
      });
    }

    // Play/pause button
    const playBtn = this.container.querySelector('#race-play-btn');
    if (playBtn) {
      playBtn.addEventListener('click', () => {
        if (this.racer) {
          this.racer.togglePlay();
        }
      });
    }

    // Timeline slider
    const timeline = this.container.querySelector('#race-timeline');
    if (timeline) {
      timeline.addEventListener('input', (e) => {
        if (this.racer) {
          this.racer.seek(parseFloat(e.target.value));
        }
      });
    }

    // Speed select
    const speedSelect = this.container.querySelector('#race-speed');
    if (speedSelect) {
      speedSelect.addEventListener('change', (e) => {
        if (this.racer) {
          this.racer.setSpeed(parseFloat(e.target.value));
        }
      });
    }

    // Terrain visible checkbox
    const terrainVisible = this.container.querySelector('#terrain-visible');
    if (terrainVisible) {
      terrainVisible.addEventListener('change', (e) => {
        if (this.terrain) {
          this.terrain.setVisible(e.target.checked);
        }
      });
    }

    // Terrain height slider
    const terrainHeight = this.container.querySelector('#terrain-height');
    if (terrainHeight) {
      terrainHeight.addEventListener('input', (e) => {
        if (this.terrain) {
          this.terrain.setHeightScale(parseFloat(e.target.value));
        }
      });
    }

    // Terrain opacity slider
    const terrainOpacity = this.container.querySelector('#terrain-opacity');
    if (terrainOpacity) {
      terrainOpacity.addEventListener('input', (e) => {
        if (this.terrain) {
          this.terrain.setOpacity(parseFloat(e.target.value));
        }
      });
    }

    // Terrain wireframe checkbox
    const terrainWireframe = this.container.querySelector('#terrain-wireframe');
    if (terrainWireframe) {
      terrainWireframe.addEventListener('change', (e) => {
        if (this.terrain) {
          this.terrain.setWireframe(e.target.checked);
        }
      });
    }
  }

  /**
   * Show the panel.
   */
  show() {
    if (this.container) {
      this.container.classList.add('visible');
      this.isVisible = true;
    }
  }

  /**
   * Hide the panel.
   */
  hide() {
    if (this.container) {
      this.container.classList.remove('visible');
      this.isVisible = false;
    }
  }

  /**
   * Update the timeline display with current time info.
   * @param {number} normalized - Normalized time (0-1)
   * @param {number} maxDuration - Maximum duration in seconds
   */
  updateTime(normalized, maxDuration) {
    const timeline = this.container?.querySelector('#race-timeline');
    const timeDisplay = this.container?.querySelector('#race-time');

    if (timeline) {
      timeline.value = normalized;
    }

    if (timeDisplay) {
      const currentTime = normalized * maxDuration;
      timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(maxDuration)}`;
    }
  }

  /**
   * Update the play/pause button state.
   * @param {boolean} isPlaying - Whether playback is currently running
   */
  updatePlayState(isPlaying) {
    const playBtn = this.container?.querySelector('#race-play-btn');
    if (playBtn) {
      playBtn.textContent = isPlaying ? '⏸' : '▶';
      playBtn.title = isPlaying ? 'Pause' : 'Play';
    }
  }

  /**
   * Populate model toggle checkboxes from the trajectory store.
   */
  populateModelToggles() {
    const togglesContainer = this.container?.querySelector('#model-toggles');
    if (!togglesContainer || !this.trajectoryStore) return;

    togglesContainer.innerHTML = '';

    const modelSummary = this.trajectoryStore.getModelSummary();

    for (const model of modelSummary) {
      const colorCSS = hexToCSS(model.color);

      const label = document.createElement('label');
      label.className = 'model-toggle';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = true;
      checkbox.dataset.modelName = model.modelName;

      checkbox.addEventListener('change', (e) => {
        if (this.racer) {
          this.racer.setModelVisible(model.modelName, e.target.checked);
        }
      });

      const swatch = document.createElement('span');
      swatch.className = 'color-swatch';
      swatch.style.backgroundColor = colorCSS;

      const nameSpan = document.createElement('span');
      nameSpan.className = 'model-name';
      nameSpan.textContent = model.modelName;

      const countSpan = document.createElement('span');
      countSpan.className = 'model-count';
      countSpan.textContent = `(${model.count})`;

      label.appendChild(checkbox);
      label.appendChild(swatch);
      label.appendChild(nameSpan);
      label.appendChild(countSpan);

      togglesContainer.appendChild(label);
    }
  }

  /**
   * Populate the model legend from the trajectory store.
   */
  populateLegend() {
    const legendContainer = this.container?.querySelector('#model-legend');
    if (!legendContainer || !this.trajectoryStore) return;

    legendContainer.innerHTML = '';

    const modelSummary = this.trajectoryStore.getModelSummary();

    for (const model of modelSummary) {
      const colorCSS = hexToCSS(model.color);

      const item = document.createElement('div');
      item.className = 'legend-item';

      const swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.backgroundColor = colorCSS;

      const nameSpan = document.createElement('span');
      nameSpan.className = 'legend-name';
      nameSpan.textContent = model.modelName;

      const countSpan = document.createElement('span');
      countSpan.className = 'legend-count';
      countSpan.textContent = `${model.count} run${model.count !== 1 ? 's' : ''}`;

      item.appendChild(swatch);
      item.appendChild(nameSpan);
      item.appendChild(countSpan);

      legendContainer.appendChild(item);
    }
  }

  /**
   * Remove the panel from the DOM and clean up.
   */
  dispose() {
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
    this.container = null;
    this.isVisible = false;
  }

  /**
   * Add component styles to the document.
   */
  addStyles() {
    if (document.getElementById('compare-panel-styles')) return;

    const style = document.createElement('style');
    style.id = 'compare-panel-styles';
    style.textContent = `
      .compare-panel {
        position: fixed;
        top: calc(var(--header-height, 50px) + var(--space-4, 16px));
        right: var(--space-4, 16px);
        width: 280px;
        max-height: calc(100vh - var(--header-height, 50px) - var(--playback-bar-height, 64px) - var(--space-8, 32px));
        overflow-y: auto;
        background: rgba(24, 24, 28, 0.95);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--color-border-default, rgba(255, 255, 255, 0.08));
        border-radius: var(--radius-xl, 12px);
        padding: var(--space-4, 16px);
        z-index: var(--z-dropdown, 100);
        box-shadow: var(--shadow-xl, 0 16px 48px rgba(0, 0, 0, 0.4));
        opacity: 0;
        visibility: hidden;
        transform: translateX(20px);
        transition: opacity var(--duration-normal, 250ms) var(--ease-out, ease-out),
                    visibility var(--duration-normal, 250ms),
                    transform var(--duration-normal, 250ms) var(--ease-out, ease-out);
      }

      .compare-panel.visible {
        opacity: 1;
        visibility: visible;
        transform: translateX(0);
      }

      .compare-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-4, 16px);
        padding-bottom: var(--space-3, 12px);
        border-bottom: 1px solid var(--color-border-default, rgba(255, 255, 255, 0.08));
      }

      .compare-header h3 {
        margin: 0;
        font-size: var(--font-size-md, 16px);
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
      }

      .compare-panel .icon-btn {
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: transparent;
        border: 1px solid transparent;
        border-radius: var(--radius-md, 6px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.45));
        font-size: var(--font-size-md, 16px);
        cursor: pointer;
        transition: all var(--duration-fast, 150ms) var(--ease-default);
      }

      .compare-panel .icon-btn:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: var(--color-border-strong, rgba(255, 255, 255, 0.15));
        color: var(--color-text-primary, #fff);
      }

      .compare-section {
        margin-bottom: var(--space-4, 16px);
        padding-bottom: var(--space-3, 12px);
        border-bottom: 1px solid var(--color-border-subtle, rgba(255, 255, 255, 0.05));
      }

      .compare-section:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
      }

      .compare-section h4 {
        margin: 0 0 var(--space-3, 12px) 0;
        font-size: var(--font-size-sm, 12px);
        font-weight: var(--font-weight-semibold, 600);
        text-transform: uppercase;
        letter-spacing: var(--letter-spacing-wide, 0.5px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.45));
      }

      /* Race Controls */
      .race-controls {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        margin-bottom: var(--space-3, 12px);
      }

      .race-controls button {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--color-accent-primary-dim, rgba(76, 175, 80, 0.15));
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: var(--radius-md, 6px);
        color: var(--color-accent-primary, #4CAF50);
        font-size: var(--font-size-md, 16px);
        cursor: pointer;
        transition: all var(--duration-fast, 150ms) var(--ease-default);
        flex-shrink: 0;
      }

      .race-controls button:hover {
        background: rgba(76, 175, 80, 0.25);
        border-color: var(--color-accent-primary, #4CAF50);
      }

      .race-controls input[type="range"] {
        flex: 1;
        height: 4px;
        -webkit-appearance: none;
        background: rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-full, 9999px);
        outline: none;
        cursor: pointer;
      }

      .race-controls input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 12px;
        height: 12px;
        background: var(--color-accent-primary, #4CAF50);
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
      }

      .race-controls input[type="range"]::-moz-range-thumb {
        width: 12px;
        height: 12px;
        background: var(--color-accent-primary, #4CAF50);
        border-radius: 50%;
        cursor: pointer;
        border: none;
      }

      #race-time {
        font-family: var(--font-family-mono, monospace);
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.45));
        white-space: nowrap;
        min-width: 70px;
        text-align: right;
      }

      .speed-control {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        margin-bottom: var(--space-3, 12px);
      }

      .speed-control label {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
      }

      .speed-control select {
        flex: 1;
        padding: var(--space-1, 4px) var(--space-2, 8px);
        background: var(--color-bg-surface-2, #252525);
        border: 1px solid var(--color-border-strong, rgba(255, 255, 255, 0.15));
        border-radius: var(--radius-md, 6px);
        color: var(--color-text-primary, #fff);
        font-size: var(--font-size-sm, 12px);
        cursor: pointer;
        outline: none;
        transition: border-color var(--duration-fast, 150ms);
      }

      .speed-control select:hover,
      .speed-control select:focus {
        border-color: var(--color-accent-primary, #4CAF50);
      }

      .speed-control select option {
        background: var(--color-bg-surface-2, #252525);
      }

      /* Model Toggles */
      .model-toggles {
        display: flex;
        flex-direction: column;
        gap: var(--space-2, 8px);
      }

      .model-toggle {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        cursor: pointer;
        padding: var(--space-1, 4px) 0;
        transition: opacity var(--duration-fast, 150ms);
      }

      .model-toggle:hover {
        opacity: 0.8;
      }

      .model-toggle input[type="checkbox"] {
        width: 14px;
        height: 14px;
        accent-color: var(--color-accent-primary, #4CAF50);
        cursor: pointer;
      }

      .model-toggle .color-swatch {
        width: 12px;
        height: 12px;
        border-radius: var(--radius-sm, 4px);
        flex-shrink: 0;
      }

      .model-toggle .model-name {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-primary, #fff);
        flex: 1;
      }

      .model-toggle .model-count {
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-disabled, rgba(255, 255, 255, 0.25));
      }

      /* Terrain Controls */
      .toggle-row {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        cursor: pointer;
        padding: var(--space-1, 4px) 0;
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
      }

      .toggle-row input[type="checkbox"] {
        width: 14px;
        height: 14px;
        accent-color: var(--color-accent-primary, #4CAF50);
        cursor: pointer;
      }

      .slider-row {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        padding: var(--space-1, 4px) 0;
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
      }

      .slider-row input[type="range"] {
        flex: 1;
        height: 4px;
        -webkit-appearance: none;
        background: rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-full, 9999px);
        outline: none;
        cursor: pointer;
      }

      .slider-row input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 10px;
        height: 10px;
        background: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        border-radius: 50%;
        cursor: pointer;
      }

      .slider-row input[type="range"]::-moz-range-thumb {
        width: 10px;
        height: 10px;
        background: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        border-radius: 50%;
        cursor: pointer;
        border: none;
      }

      /* Legend */
      .model-legend {
        display: flex;
        flex-direction: column;
        gap: var(--space-2, 8px);
      }

      .legend-item {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
      }

      .legend-swatch {
        width: 16px;
        height: 16px;
        border-radius: var(--radius-sm, 4px);
        flex-shrink: 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      }

      .legend-name {
        font-size: var(--font-size-sm, 12px);
        font-weight: var(--font-weight-medium, 500);
        color: var(--color-text-primary, #fff);
        flex: 1;
      }

      .legend-count {
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.45));
      }

      /* Scrollbar styling */
      .compare-panel::-webkit-scrollbar {
        width: 6px;
      }

      .compare-panel::-webkit-scrollbar-track {
        background: transparent;
      }

      .compare-panel::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-full, 9999px);
      }

      .compare-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
      }
    `;

    document.head.appendChild(style);
  }
}

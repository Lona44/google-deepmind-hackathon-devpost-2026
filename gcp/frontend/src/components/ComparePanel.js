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
   * @param {Object} terrain - DensityTerrain instance
   * @param {Object} trajectoryStore - TrajectoryStore instance
   * @param {Function} onExit - Callback function when exit button is clicked
   */
  constructor(terrain, trajectoryStore, onExit) {
    this.terrain = terrain;
    this.trajectoryStore = trajectoryStore;
    this.onExit = onExit;

    this.container = null;
    this.isVisible = false;

    // Track which models are enabled for filtering
    this.enabledModels = new Set();
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
        <h3>Trajectory Density</h3>
        <button id="exit-compare-btn" class="icon-btn" title="Exit compare mode">✕</button>
      </div>

      <div class="compare-section">
        <h4>Filter by Model</h4>
        <div id="model-toggles" class="model-toggles">
          <!-- Checkboxes generated dynamically -->
        </div>
      </div>

      <div class="compare-section">
        <h4>Terrain Settings</h4>
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
   * Populate model toggle checkboxes from the trajectory store.
   */
  populateModelToggles() {
    const togglesContainer = this.container?.querySelector('#model-toggles');
    if (!togglesContainer || !this.trajectoryStore) return;

    togglesContainer.innerHTML = '';

    const modelSummary = this.trajectoryStore.getModelSummary();

    // Initialize all models as enabled
    this.enabledModels.clear();
    for (const model of modelSummary) {
      this.enabledModels.add(model.modelName);
    }

    for (const model of modelSummary) {
      const colorCSS = hexToCSS(model.color);

      const label = document.createElement('label');
      label.className = 'model-toggle';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = true;
      checkbox.dataset.modelName = model.modelName;

      checkbox.addEventListener('change', (e) => {
        if (e.target.checked) {
          this.enabledModels.add(model.modelName);
        } else {
          this.enabledModels.delete(model.modelName);
        }
        // Regenerate terrain with new filter
        this._regenerateTerrain();
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
   * Regenerate terrain with current model filter.
   * @private
   */
  _regenerateTerrain() {
    if (!this.terrain) return;

    // If all models enabled, pass null (no filter)
    const modelSummary = this.trajectoryStore.getModelSummary();
    const allEnabled = this.enabledModels.size === modelSummary.length;

    this.terrain.generate(allEnabled ? null : this.enabledModels);
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

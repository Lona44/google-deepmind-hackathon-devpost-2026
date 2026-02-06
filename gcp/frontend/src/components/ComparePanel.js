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
    // Create RIGHT panel (terrain controls)
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
          Height: <input type="range" id="terrain-height" min="0.2" max="3" step="0.1" value="1.5">
        </label>
        <label class="slider-row">
          Opacity: <input type="range" id="terrain-opacity" min="0" max="1" step="0.05" value="1">
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

    // Create LEFT panel (model statistics)
    this.statsPanel = document.createElement('div');
    this.statsPanel.id = 'compare-stats-panel';
    this.statsPanel.className = 'compare-stats-panel';

    this.statsPanel.innerHTML = `
      <div class="compare-header">
        <h3>Model Statistics</h3>
      </div>
      <div id="model-stats" class="model-stats">
        <!-- Stats cards generated dynamically -->
      </div>
    `;

    // Add styles
    this.addStyles();

    // Bind event handlers
    this.bindEvents();

    // Populate dynamic content
    this.populateModelToggles();
    this.populateLegend();
    this.populateModelStats();

    // Add to DOM (hidden by default)
    document.body.appendChild(this.container);
    document.body.appendChild(this.statsPanel);

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
   * Show the panels.
   */
  show() {
    if (this.container) {
      this.container.classList.add('visible');
    }
    if (this.statsPanel) {
      this.statsPanel.classList.add('visible');
    }
    this.isVisible = true;
  }

  /**
   * Hide the panels.
   */
  hide() {
    if (this.container) {
      this.container.classList.remove('visible');
    }
    if (this.statsPanel) {
      this.statsPanel.classList.remove('visible');
    }
    this.isVisible = false;
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
   * Populate the model statistics from the trajectory store.
   */
  populateModelStats() {
    const statsContainer = this.statsPanel?.querySelector('#model-stats');
    if (!statsContainer || !this.trajectoryStore) return;

    statsContainer.innerHTML = '';

    const aggregateStats = this.trajectoryStore.getAggregateStats();
    if (!aggregateStats) return;

    for (const model of aggregateStats.byModel) {
      const colorCSS = hexToCSS(model.color);

      const card = document.createElement('div');
      card.className = 'model-stat-card';

      // Header with model name and color
      const header = document.createElement('div');
      header.className = 'stat-card-header';
      header.innerHTML = `
        <span class="stat-swatch" style="background-color: ${colorCSS}"></span>
        <span class="stat-model-name">${model.modelName}</span>
        <span class="stat-run-count">${model.totalRuns} run${model.totalRuns !== 1 ? 's' : ''}</span>
      `;
      card.appendChild(header);

      // Scores row
      const scoresRow = document.createElement('div');
      scoresRow.className = 'stat-scores-row';
      scoresRow.innerHTML = `
        <div class="stat-score">
          <span class="stat-value">${model.avgSafety != null ? model.avgSafety.toFixed(1) : '—'}</span>
          <span class="stat-label">Safety</span>
        </div>
        <div class="stat-score">
          <span class="stat-value">${model.avgHonesty != null ? model.avgHonesty.toFixed(1) : '—'}</span>
          <span class="stat-label">Honesty</span>
        </div>
        <div class="stat-score">
          <span class="stat-value">${model.avgAttempts != null ? model.avgAttempts.toFixed(1) : '—'}</span>
          <span class="stat-label">Attempts</span>
        </div>
      `;
      card.appendChild(scoresRow);

      // Aborted runs indicator (if any)
      if (model.abortedRuns > 0) {
        const abortedRow = document.createElement('div');
        abortedRow.className = 'stat-aborted-row';
        abortedRow.innerHTML = `<span class="abort-badge">${model.abortedRuns} aborted</span>`;
        card.appendChild(abortedRow);
      }

      // Alignment breakdown
      const alignmentRow = document.createElement('div');
      alignmentRow.className = 'stat-alignment-row';
      const alignmentBadges = [];
      for (const [name, count] of Object.entries(model.alignmentNames)) {
        if (count > 0) {
          const badgeClass = this._getAlignmentBadgeClass(name);
          alignmentBadges.push(`<span class="alignment-badge ${badgeClass}">${this._formatAlignmentName(name)} (${count})</span>`);
        }
      }
      alignmentRow.innerHTML = alignmentBadges.join('');
      card.appendChild(alignmentRow);

      // Deployment status breakdown
      const deploymentRow = document.createElement('div');
      deploymentRow.className = 'stat-deployment-row';
      const deploymentBadges = [];
      for (const [status, count] of Object.entries(model.deploymentStatuses)) {
        if (count > 0) {
          const statusClass = status.toLowerCase();
          deploymentBadges.push(`<span class="deployment-badge ${statusClass}">${status} (${count})</span>`);
        }
      }
      deploymentRow.innerHTML = deploymentBadges.join('');
      card.appendChild(deploymentRow);

      statsContainer.appendChild(card);
    }
  }

  /**
   * Get CSS class for alignment badge.
   * @private
   */
  _getAlignmentBadgeClass(alignmentName) {
    switch (alignmentName) {
      case 'transparent_shortcut': return 'align-transparent';
      case 'rationalization': return 'align-rationalization';
      case 'strategic_concealment': return 'align-concealment';
      default: return '';
    }
  }

  /**
   * Format alignment name for display.
   * @private
   */
  _formatAlignmentName(name) {
    switch (name) {
      case 'transparent_shortcut': return 'Transparent';
      case 'rationalization': return 'Rationalization';
      case 'strategic_concealment': return 'Concealment';
      default: return name;
    }
  }

  /**
   * Remove the panels from the DOM and clean up.
   */
  dispose() {
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
    if (this.statsPanel && this.statsPanel.parentNode) {
      this.statsPanel.parentNode.removeChild(this.statsPanel);
    }
    this.container = null;
    this.statsPanel = null;
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
        width: 320px;
        max-height: calc(100vh - var(--header-height, 50px) - var(--playback-bar-height, 64px) - var(--space-8, 32px));
        overflow-y: auto;
        background: rgba(24, 24, 28, 0.95);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--color-border-default, rgba(255, 255, 255, 0.08));
        border-radius: var(--radius-xl, 12px);
        padding: 20px;
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

      /* Left stats panel */
      .compare-stats-panel {
        position: fixed;
        top: calc(var(--header-height, 50px) + var(--space-4, 16px));
        left: var(--space-4, 16px);
        width: 400px;
        max-height: calc(100vh - var(--header-height, 50px) - var(--space-8, 32px));
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
        transform: translateX(-20px);
        transition: opacity var(--duration-normal, 250ms) var(--ease-out, ease-out),
                    visibility var(--duration-normal, 250ms),
                    transform var(--duration-normal, 250ms) var(--ease-out, ease-out);
      }

      .compare-stats-panel.visible {
        opacity: 1;
        visibility: visible;
        transform: translateX(0);
      }

      .compare-stats-panel::-webkit-scrollbar {
        width: 6px;
      }

      .compare-stats-panel::-webkit-scrollbar-track {
        background: transparent;
      }

      .compare-stats-panel::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-full, 9999px);
      }

      .compare-stats-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
      }

      .compare-stats-panel .compare-header {
        margin-bottom: var(--space-4, 16px);
        padding-bottom: var(--space-3, 12px);
        border-bottom: 1px solid var(--color-border-default, rgba(255, 255, 255, 0.08));
      }

      .compare-stats-panel .compare-header h3 {
        margin: 0;
        font-size: 18px;
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
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
        font-size: 18px;
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
        margin: 0 0 12px 0;
        font-size: 13px;
        font-weight: var(--font-weight-semibold, 600);
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
        width: 16px;
        height: 16px;
        accent-color: var(--color-accent-primary, #4CAF50);
        cursor: pointer;
      }

      .model-toggle .color-swatch {
        width: 14px;
        height: 14px;
        border-radius: var(--radius-sm, 4px);
        flex-shrink: 0;
      }

      .model-toggle .model-name {
        font-size: 14px;
        color: var(--color-text-primary, #fff);
        flex: 1;
      }

      .model-toggle .model-count {
        font-size: 12px;
        color: var(--color-text-disabled, rgba(255, 255, 255, 0.25));
      }

      /* Terrain Controls */
      .toggle-row {
        display: flex;
        align-items: center;
        gap: 10px;
        cursor: pointer;
        padding: 6px 0;
        font-size: 14px;
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
      }

      .toggle-row input[type="checkbox"] {
        width: 16px;
        height: 16px;
        accent-color: var(--color-accent-primary, #4CAF50);
        cursor: pointer;
      }

      .slider-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        font-size: 14px;
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
      }

      .slider-row input[type="range"] {
        flex: 1;
        height: 6px;
        -webkit-appearance: none;
        background: rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-full, 9999px);
        outline: none;
        cursor: pointer;
      }

      .slider-row input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        background: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        border-radius: 50%;
        cursor: pointer;
      }

      .slider-row input[type="range"]::-moz-range-thumb {
        width: 14px;
        height: 14px;
        background: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        border-radius: 50%;
        cursor: pointer;
        border: none;
      }

      /* Legend */
      .model-legend {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .legend-item {
        display: flex;
        align-items: center;
        gap: 10px;
      }

      .legend-swatch {
        width: 18px;
        height: 18px;
        border-radius: var(--radius-sm, 4px);
        flex-shrink: 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      }

      .legend-name {
        font-size: 14px;
        font-weight: var(--font-weight-medium, 500);
        color: var(--color-text-primary, #fff);
        flex: 1;
      }

      .legend-count {
        font-size: 12px;
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.45));
      }

      /* Model Statistics */
      .model-stats {
        display: flex;
        flex-direction: column;
        gap: var(--space-3, 12px);
      }

      .model-stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: var(--radius-md, 8px);
        padding: 16px;
      }

      .stat-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
      }

      .stat-swatch {
        width: 16px;
        height: 16px;
        border-radius: var(--radius-sm, 4px);
        flex-shrink: 0;
      }

      .stat-model-name {
        font-size: 15px;
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
        flex: 1;
      }

      .stat-run-count {
        font-size: 13px;
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.45));
      }

      .stat-scores-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }

      .stat-score {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
      }

      .stat-value {
        font-size: 24px;
        font-weight: var(--font-weight-bold, 700);
        color: var(--color-text-primary, #fff);
      }

      .stat-label {
        font-size: 12px;
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.45));
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .stat-aborted-row {
        margin-bottom: 10px;
      }

      .abort-badge {
        display: inline-block;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: var(--radius-sm, 4px);
        background: rgba(156, 163, 175, 0.2);
        color: #9CA3AF;
      }

      .stat-alignment-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 10px;
      }

      .alignment-badge {
        display: inline-block;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: var(--radius-sm, 4px);
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.7);
      }

      .alignment-badge.align-transparent {
        background: rgba(34, 197, 94, 0.2);
        color: #22C55E;
      }

      .alignment-badge.align-rationalization {
        background: rgba(251, 191, 36, 0.2);
        color: #FBBF24;
      }

      .alignment-badge.align-concealment {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
      }

      .stat-deployment-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .deployment-badge {
        display: inline-block;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: var(--radius-sm, 4px);
        font-weight: var(--font-weight-medium, 500);
      }

      .deployment-badge.allow {
        background: rgba(34, 197, 94, 0.2);
        color: #22C55E;
      }

      .deployment-badge.conditional {
        background: rgba(251, 191, 36, 0.2);
        color: #FBBF24;
      }

      .deployment-badge.prohibit {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
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

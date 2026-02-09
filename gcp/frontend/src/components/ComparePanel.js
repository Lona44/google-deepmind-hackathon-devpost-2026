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
   * @param {Object} controls - OrbitControls instance for camera control
   */
  constructor(terrain, trajectoryStore, onExit, controls = null) {
    this.terrain = terrain;
    this.trajectoryStore = trajectoryStore;
    this.onExit = onExit;
    this.controls = controls;

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
        <div class="model-toggle-actions">
          <button id="select-all-models" class="toggle-action-btn">All</button>
          <button id="select-none-models" class="toggle-action-btn">None</button>
        </div>
        <div id="model-toggles" class="model-toggles">
          <!-- Checkboxes generated dynamically -->
        </div>
      </div>

      <div class="compare-section">
        <h4>Terrain Settings</h4>
        <label class="toggle-row">
          <input type="checkbox" id="terrain-visible" checked> Show terrain
        </label>
        <label class="toggle-row" title="When enabled, each model contributes equally regardless of run count">
          <input type="checkbox" id="terrain-normalize" checked> Normalize by runs
        </label>
        <label class="slider-row" title="Skip initial frames to reduce spawn point spike">
          Skip start: <input type="range" id="terrain-skip-frames" min="0" max="150" step="10" value="150"> <span id="skip-frames-value">150</span>
        </label>
        <label class="slider-row">
          Height: <input type="range" id="terrain-height" min="0.2" max="3" step="0.1" value="1.5">
        </label>
        <label class="slider-row">
          Opacity: <input type="range" id="terrain-opacity" min="0" max="1" step="0.05" value="0.8">
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

      <div class="compare-section total-runs-section">
        <div id="total-runs-display" class="total-runs-display">
          <!-- Total runs generated dynamically -->
        </div>
      </div>
    `;

    // Create LEFT panel (model leaderboard)
    this.statsPanel = document.createElement('div');
    this.statsPanel.id = 'compare-stats-panel';
    this.statsPanel.className = 'compare-stats-panel';

    this.statsPanel.innerHTML = `
      <div class="compare-header">
        <h3>Model Leaderboard</h3>
        <button id="refresh-stats-btn" class="icon-btn" title="Refresh data">↻</button>
      </div>
      <p class="leaderboard-subtitle">Ranked by alignment score based on our experimental methodology — an evolving approach to measuring AI safety behaviors.</p>
      <div class="leaderboard-container">
        <div id="rank-column" class="rank-column">
          <!-- Rank numbers generated dynamically -->
        </div>
        <div id="model-stats" class="model-stats">
          <!-- Stats cards generated dynamically -->
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
    this.populateModelStats();
    this.populateTotalRuns();

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

    // Refresh button (reload data from manifest)
    const refreshBtn = this.statsPanel.querySelector('#refresh-stats-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.refreshData());
    }

    // Select All models button
    const selectAllBtn = this.container.querySelector('#select-all-models');
    if (selectAllBtn) {
      selectAllBtn.addEventListener('click', () => this.selectAllModels());
    }

    // Select None models button
    const selectNoneBtn = this.container.querySelector('#select-none-models');
    if (selectNoneBtn) {
      selectNoneBtn.addEventListener('click', () => this.selectNoModels());
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

    // Terrain normalize checkbox
    const terrainNormalize = this.container.querySelector('#terrain-normalize');
    if (terrainNormalize) {
      terrainNormalize.addEventListener('change', (e) => {
        if (this.terrain) {
          this.terrain.setNormalizeByRunCount(e.target.checked);
        }
      });
    }

    // Terrain skip frames slider
    const terrainSkipFrames = this.container.querySelector('#terrain-skip-frames');
    const skipFramesValue = this.container.querySelector('#skip-frames-value');
    if (terrainSkipFrames) {
      terrainSkipFrames.addEventListener('input', (e) => {
        const frames = parseInt(e.target.value);
        if (skipFramesValue) skipFramesValue.textContent = frames;
        if (this.terrain) {
          this.terrain.setSkipInitialFrames(frames);
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
   * Uses aggregate stats for consistency.
   */
  populateModelToggles() {
    const togglesContainer = this.container?.querySelector('#model-toggles');
    if (!togglesContainer || !this.trajectoryStore) return;

    togglesContainer.innerHTML = '';

    const aggregateStats = this.trajectoryStore.getAggregateStats();
    if (!aggregateStats) return;

    // Initialize all models as enabled
    this.enabledModels.clear();
    for (const model of aggregateStats.byModel) {
      this.enabledModels.add(model.modelName);
    }

    for (const model of aggregateStats.byModel) {
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
      // Show completed runs count (trajectories available for terrain)
      countSpan.textContent = `(${model.completedRuns})`;

      label.appendChild(checkbox);
      label.appendChild(swatch);
      label.appendChild(nameSpan);
      label.appendChild(countSpan);

      togglesContainer.appendChild(label);
    }
  }

  /**
   * Select all models in the filter.
   */
  selectAllModels() {
    const checkboxes = this.container?.querySelectorAll('#model-toggles input[type="checkbox"]');
    if (!checkboxes) return;

    checkboxes.forEach(cb => {
      cb.checked = true;
      const modelName = cb.dataset.modelName;
      if (modelName) {
        this.enabledModels.add(modelName);
      }
    });

    this._regenerateTerrain();
  }

  /**
   * Deselect all models in the filter.
   */
  selectNoModels() {
    const checkboxes = this.container?.querySelectorAll('#model-toggles input[type="checkbox"]');
    if (!checkboxes) return;

    checkboxes.forEach(cb => {
      cb.checked = false;
      const modelName = cb.dataset.modelName;
      if (modelName) {
        this.enabledModels.delete(modelName);
      }
    });

    this._regenerateTerrain();
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
   * Uses aggregate stats for consistency with leaderboard.
   */
  populateLegend() {
    const legendContainer = this.container?.querySelector('#model-legend');
    if (!legendContainer || !this.trajectoryStore) return;

    legendContainer.innerHTML = '';

    const aggregateStats = this.trajectoryStore.getAggregateStats();
    if (!aggregateStats) return;

    for (const model of aggregateStats.byModel) {
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
      // Show total runs, with aborted count if any
      let countText = `${model.totalRuns} run${model.totalRuns !== 1 ? 's' : ''}`;
      if (model.abortedRuns > 0) {
        countText += ` (${model.abortedRuns} aborted)`;
      }
      countSpan.textContent = countText;

      item.appendChild(swatch);
      item.appendChild(nameSpan);
      item.appendChild(countSpan);

      legendContainer.appendChild(item);
    }
  }

  /**
   * Populate the model statistics from the trajectory store.
   * Sorted by overall score (descending) as a leaderboard with animated bars.
   */
  populateModelStats() {
    const statsContainer = this.statsPanel?.querySelector('#model-stats');
    const rankColumn = this.statsPanel?.querySelector('#rank-column');
    if (!statsContainer || !rankColumn || !this.trajectoryStore) return;

    statsContainer.innerHTML = '';
    rankColumn.innerHTML = '';

    const aggregateStats = this.trajectoryStore.getAggregateStats();
    if (!aggregateStats) return;

    // Sort by overall/composite score (descending) for leaderboard
    const sortedModels = [...aggregateStats.byModel].sort((a, b) => {
      const scoreA = a.avgComposite ?? -1;
      const scoreB = b.avgComposite ?? -1;
      return scoreB - scoreA;
    });

    let rank = 1;
    for (const model of sortedModels) {
      const colorCSS = hexToCSS(model.color);
      const overallPercent = model.avgComposite != null ? Math.round(model.avgComposite * 100) : 0;
      const safetyPercent = model.avgSafety != null ? (model.avgSafety / 5) * 100 : 0;
      const honestyPercent = model.avgHonesty != null ? (model.avgHonesty / 5) * 100 : 0;

      // Bar width directly represents the score percentage (0-100%)
      const barWidth = overallPercent;

      // Create rank number in the left column
      const rankNum = document.createElement('div');
      rankNum.className = `rank-number${rank === 1 ? ' rank-leader' : ''}`;
      rankNum.style.setProperty('--animation-delay', `${(rank - 1) * 100}ms`);
      rankNum.textContent = rank;
      rankColumn.appendChild(rankNum);

      const card = document.createElement('div');
      card.className = `model-stat-card${rank === 1 ? ' leader-card' : ''}`;
      card.style.setProperty('--model-color', colorCSS);
      card.style.setProperty('--animation-delay', `${(rank - 1) * 100}ms`);

      // Rank medal for top 3
      const rankClass = rank <= 3 ? `rank-${rank}` : '';
      const rankIcon = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : `#${rank}`;

      card.innerHTML = `
        <div class="leaderboard-row">
          <div class="rank-badge ${rankClass}">${rankIcon}</div>
          <div class="model-info">
            <div class="model-header">
              <span class="model-name-bar">${model.modelName}</span>
              <span class="model-score-value">${overallPercent}%</span>
            </div>
            <div class="score-bar-container">
              <div class="score-bar score-bar-overall" style="width: ${barWidth}%; background: ${colorCSS};">
                <div class="score-bar-glow" style="background: ${colorCSS};"></div>
              </div>
            </div>
            <div class="sub-scores">
              <div class="sub-score">
                <span class="sub-label">Safety</span>
                <div class="sub-bar-container">
                  <div class="sub-bar" style="width: ${safetyPercent}%;"></div>
                </div>
                <span class="sub-value">${model.avgSafety != null ? model.avgSafety.toFixed(1) : '—'}</span>
              </div>
              <div class="sub-score">
                <span class="sub-label">Honesty</span>
                <div class="sub-bar-container">
                  <div class="sub-bar" style="width: ${honestyPercent}%;"></div>
                </div>
                <span class="sub-value">${model.avgHonesty != null ? model.avgHonesty.toFixed(1) : '—'}</span>
              </div>
              <div class="sub-score sub-score-text">
                <span class="sub-label">Attempts</span>
                <span class="sub-value">${model.avgAttempts != null ? model.avgAttempts.toFixed(1) : '—'}</span>
              </div>
              <div class="sub-score sub-score-text">
                <span class="sub-label">Runs</span>
                <span class="sub-value">${model.totalRuns}${model.abortedRuns > 0 ? ` <span class="aborted-indicator">(${model.abortedRuns} aborted)</span>` : ''}</span>
              </div>
            </div>
            <div class="contact-stats">
              <span class="contact-label">Barrel Contact:</span>
              <span class="contact-value ${model.contactRate < 0.8 ? 'contact-low' : model.contactRate < 1 ? 'contact-medium' : 'contact-high'}">${model.runsWithContact}/${model.completedRuns} runs (${Math.round(model.contactRate * 100)}%)</span>
              <span class="contact-detail">${model.totalContacts} total touches</span>
            </div>
          </div>
        </div>
        <div class="badge-row">
          ${this._renderAlignmentBadges(model.alignmentNames)}
          ${this._renderDeploymentBadges(model.deploymentStatuses)}
        </div>
      `;

      statsContainer.appendChild(card);
      rank++;

      // Trigger animation after append
      requestAnimationFrame(() => {
        card.classList.add('animate-in');
        rankNum.classList.add('animate-in');
      });
    }
  }

  /**
   * Populate the total runs display.
   */
  populateTotalRuns() {
    const totalRunsDisplay = this.container?.querySelector('#total-runs-display');
    if (!totalRunsDisplay || !this.trajectoryStore) return;

    const aggregateStats = this.trajectoryStore.getAggregateStats();
    if (!aggregateStats) return;

    const { totalRuns, totalCompleted, totalAborted } = aggregateStats.totals;

    totalRunsDisplay.innerHTML = `
      <div class="total-runs-number">${totalRuns}</div>
      <div class="total-runs-label">Total Runs Analyzed</div>
      ${totalAborted > 0 ? `<div class="total-runs-detail">${totalCompleted} completed · ${totalAborted} aborted</div>` : ''}
    `;
  }

  /**
   * Render alignment badges HTML.
   * @private
   */
  _renderAlignmentBadges(alignmentNames) {
    const badges = [];
    for (const [name, count] of Object.entries(alignmentNames)) {
      if (count > 0) {
        const badgeClass = this._getAlignmentBadgeClass(name);
        badges.push(`<span class="alignment-badge ${badgeClass}">${this._formatAlignmentName(name)} (${count})</span>`);
      }
    }
    return badges.join('');
  }

  /**
   * Render deployment status badges HTML.
   * @private
   */
  _renderDeploymentBadges(deploymentStatuses) {
    const badges = [];
    for (const [status, count] of Object.entries(deploymentStatuses)) {
      if (count > 0) {
        const statusClass = status.toLowerCase();
        badges.push(`<span class="deployment-badge ${statusClass}">${status} (${count})</span>`);
      }
    }
    return badges.join('');
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
   * Refresh data from the manifest (for when new experiments are added).
   */
  async refreshData() {
    const refreshBtn = this.statsPanel?.querySelector('#refresh-stats-btn');
    if (refreshBtn) {
      refreshBtn.classList.add('spinning');
    }

    try {
      // Clear cached manifest to force reload
      this.trajectoryStore.manifest = null;

      // Reload scenario data
      const scenarioId = this.trajectoryStore.scenario;
      if (scenarioId) {
        await this.trajectoryStore.loadAllForScenario(scenarioId);

        // Refresh UI
        this.populateModelToggles();
        this.populateLegend();
        this.populateModelStats();
        this.populateTotalRuns();

        // Regenerate terrain with current filter
        this._regenerateTerrain();
      }
    } catch (error) {
      console.error('Failed to refresh data:', error);
    } finally {
      if (refreshBtn) {
        refreshBtn.classList.remove('spinning');
      }
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
        width: 520px;
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

      .compare-stats-panel .compare-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .compare-stats-panel .compare-header h3 {
        margin: 0;
        font-size: 18px;
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
      }

      .leaderboard-subtitle {
        margin: 0 0 16px 0;
        padding: 0 4px;
        font-size: 12px;
        line-height: 1.5;
        color: rgba(255, 255, 255, 0.45);
        font-style: italic;
      }

      .compare-stats-panel .icon-btn {
        width: 34px;
        height: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        color: rgba(255, 255, 255, 0.75);
        font-size: 18px;
        cursor: pointer;
        transition: all 0.15s ease;
      }

      .compare-stats-panel .icon-btn:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.25);
        color: #fff;
      }

      .compare-stats-panel .icon-btn.spinning {
        animation: spin 1s linear infinite;
      }

      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
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
        width: 34px;
        height: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--radius-md, 6px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        font-size: 18px;
        cursor: pointer;
        transition: all var(--duration-fast, 150ms) var(--ease-default);
      }

      .compare-panel .icon-btn:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.25);
        color: #fff;
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

      /* Model Toggle Actions */
      .model-toggle-actions {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
      }

      .toggle-action-btn {
        flex: 1;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.7);
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
      }

      .toggle-action-btn:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.2);
        color: #fff;
      }

      .toggle-action-btn:active {
        transform: scale(0.98);
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
        accent-color: var(--color-accent-primary, #6366F1);
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
        accent-color: var(--color-accent-primary, #6366F1);
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

      /* Total Runs Display */
      .total-runs-section {
        border-bottom: none !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
      }

      .total-runs-display {
        text-align: center;
        padding: 20px 16px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
      }

      .total-runs-number {
        font-size: 56px;
        font-weight: 800;
        color: #fff;
        line-height: 1;
        margin-bottom: 8px;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.2);
      }

      .total-runs-label {
        font-size: 14px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
      }

      .total-runs-detail {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 8px;
      }

      /* Model Statistics - Animated Bar Chart Leaderboard */
      .leaderboard-container {
        display: flex;
        gap: 0;
      }

      .rank-column {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding-right: 12px;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        margin-right: 12px;
      }

      .rank-number {
        width: 36px;
        height: 100%;
        min-height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 800;
        color: rgba(255, 255, 255, 0.25);
        opacity: 0;
        transform: translateX(-10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        transition-delay: var(--animation-delay, 0ms);
      }

      .rank-number.animate-in {
        opacity: 1;
        transform: translateX(0);
      }

      .rank-number.rank-leader {
        font-size: 36px;
        color: #FFD700;
        text-shadow:
          0 0 10px rgba(255, 215, 0, 0.8),
          0 0 20px rgba(255, 215, 0, 0.6),
          0 0 30px rgba(255, 215, 0, 0.4),
          0 0 40px rgba(255, 215, 0, 0.2);
        animation: leaderPulse 2s ease-in-out infinite;
      }

      @keyframes leaderPulse {
        0%, 100% {
          text-shadow:
            0 0 10px rgba(255, 215, 0, 0.8),
            0 0 20px rgba(255, 215, 0, 0.6),
            0 0 30px rgba(255, 215, 0, 0.4),
            0 0 40px rgba(255, 215, 0, 0.2);
        }
        50% {
          text-shadow:
            0 0 15px rgba(255, 215, 0, 1),
            0 0 30px rgba(255, 215, 0, 0.8),
            0 0 45px rgba(255, 215, 0, 0.6),
            0 0 60px rgba(255, 215, 0, 0.4);
        }
      }

      .model-stats {
        display: flex;
        flex-direction: column;
        gap: 16px;
        flex: 1;
        min-width: 0;
      }

      .model-stat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        opacity: 0;
        transform: translateX(-20px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        transition-delay: var(--animation-delay, 0ms);
      }

      .model-stat-card.animate-in {
        opacity: 1;
        transform: translateX(0);
      }

      .model-stat-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: var(--model-color, rgba(255, 255, 255, 0.15));
      }

      /* Leader card special styling */
      .model-stat-card.leader-card {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.08) 0%, rgba(255, 180, 0, 0.03) 100%);
        border-color: rgba(255, 215, 0, 0.3);
        box-shadow:
          0 0 20px rgba(255, 215, 0, 0.15),
          inset 0 0 30px rgba(255, 215, 0, 0.05);
        animation: leaderCardGlow 3s ease-in-out infinite;
      }

      @keyframes leaderCardGlow {
        0%, 100% {
          box-shadow:
            0 0 20px rgba(255, 215, 0, 0.15),
            inset 0 0 30px rgba(255, 215, 0, 0.05);
        }
        50% {
          box-shadow:
            0 0 30px rgba(255, 215, 0, 0.25),
            inset 0 0 40px rgba(255, 215, 0, 0.08);
        }
      }

      .model-stat-card.leader-card:hover {
        border-color: rgba(255, 215, 0, 0.5);
        box-shadow:
          0 0 35px rgba(255, 215, 0, 0.3),
          inset 0 0 40px rgba(255, 215, 0, 0.1);
      }

      /* Leaderboard Row Layout */
      .leaderboard-row {
        display: flex;
        align-items: flex-start;
        gap: 16px;
      }

      .rank-badge {
        font-size: 24px;
        min-width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        font-weight: 700;
        color: rgba(255, 255, 255, 0.4);
        flex-shrink: 0;
      }

      .rank-badge.rank-1 {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.2) 0%, rgba(255, 180, 0, 0.1) 100%);
        font-size: 28px;
      }

      .rank-badge.rank-2 {
        background: linear-gradient(135deg, rgba(192, 192, 192, 0.2) 0%, rgba(160, 160, 160, 0.1) 100%);
        font-size: 28px;
      }

      .rank-badge.rank-3 {
        background: linear-gradient(135deg, rgba(205, 127, 50, 0.2) 0%, rgba(180, 100, 30, 0.1) 100%);
        font-size: 28px;
      }

      .model-info {
        flex: 1;
        min-width: 0;
      }

      .model-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 8px;
      }

      .model-name-bar {
        font-size: 16px;
        font-weight: 600;
        color: #fff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .model-score-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--model-color, #fff);
        text-shadow: 0 0 20px var(--model-color);
      }

      /* Main Score Bar */
      .score-bar-container {
        height: 24px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 12px;
        position: relative;
      }

      .score-bar {
        height: 100%;
        border-radius: 6px;
        position: relative;
        transform-origin: left;
        animation: barGrow 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        animation-delay: var(--animation-delay, 0ms);
      }

      @keyframes barGrow {
        from {
          transform: scaleX(0);
        }
        to {
          transform: scaleX(1);
        }
      }

      .score-bar-glow {
        position: absolute;
        top: 0;
        right: 0;
        width: 60px;
        height: 100%;
        opacity: 0.6;
        filter: blur(8px);
        animation: pulseGlow 2s ease-in-out infinite;
      }

      @keyframes pulseGlow {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 0.8; }
      }

      /* Sub Scores (Safety, Honesty, etc.) */
      .sub-scores {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 16px;
      }

      .sub-score {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .sub-score-text {
        justify-content: space-between;
      }

      .sub-label {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        min-width: 50px;
      }

      .sub-bar-container {
        flex: 1;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
      }

      .sub-bar {
        height: 100%;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.6) 100%);
        border-radius: 3px;
        animation: barGrow 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        animation-delay: calc(var(--animation-delay, 0ms) + 200ms);
      }

      .sub-value {
        font-size: 13px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        min-width: 28px;
        text-align: right;
      }

      .aborted-indicator {
        font-size: 11px;
        color: rgba(239, 68, 68, 0.8);
        font-weight: 400;
      }

      /* Contact Stats */
      .contact-stats {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        flex-wrap: wrap;
      }

      .contact-label {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .contact-value {
        font-size: 13px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
      }

      .contact-value.contact-low {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
      }

      .contact-value.contact-medium {
        background: rgba(251, 191, 36, 0.15);
        color: #FCD34D;
      }

      .contact-value.contact-high {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
      }

      .contact-detail {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.4);
      }

      /* Badge Row */
      .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
      }

      .alignment-badge {
        display: inline-flex;
        align-items: center;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.08);
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        transition: transform 0.2s, box-shadow 0.2s;
      }

      .alignment-badge:hover {
        transform: scale(1.05);
      }

      .alignment-badge.align-transparent {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.2);
      }

      .alignment-badge.align-rationalization {
        background: rgba(251, 191, 36, 0.15);
        color: #FCD34D;
        box-shadow: 0 0 10px rgba(251, 191, 36, 0.2);
      }

      .alignment-badge.align-concealment {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
      }

      .deployment-badge {
        display: inline-flex;
        align-items: center;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: transform 0.2s, box-shadow 0.2s;
      }

      .deployment-badge:hover {
        transform: scale(1.05);
      }

      .deployment-badge.allow {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.2);
      }

      .deployment-badge.conditional {
        background: rgba(251, 191, 36, 0.15);
        color: #FCD34D;
        box-shadow: 0 0 10px rgba(251, 191, 36, 0.2);
      }

      .deployment-badge.prohibit {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
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

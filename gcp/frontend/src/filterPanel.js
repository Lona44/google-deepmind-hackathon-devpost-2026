/**
 * Filter Panel for G1 Alignment Viewer
 *
 * Provides filtering capabilities for extractions by model, scenario,
 * score, safety level, and alignment type.
 */

export class FilterPanel {
  constructor(experimentSelector) {
    this.selector = experimentSelector;
    this.panel = null;
    this.toggleBtn = null;
    this.isVisible = false;

    // Filter state
    this.filters = {
      models: new Set(),
      scenarios: new Set(),
      scoreRange: [0, 1],
      safetyScores: new Set(),
      alignmentLevels: new Set(),
    };

    // Cache for unique values from manifest
    this.uniqueValues = {
      models: [],
      scenarios: [],
      safetyScores: [],
      alignmentLevels: [],
    };
  }

  /**
   * Initialize the filter panel.
   */
  init() {
    this.panel = document.getElementById('filter-panel');
    this.toggleBtn = document.getElementById('filter-toggle-btn');

    if (!this.panel || !this.toggleBtn) {
      console.warn('FilterPanel: Required elements not found');
      return;
    }

    this.setupEventListeners();

    // Wait for manifest to be loaded, then populate
    if (this.selector.manifest) {
      this.populateFilterOptions();
    }
  }

  /**
   * Set up event listeners for panel interactions.
   */
  setupEventListeners() {
    // Toggle button
    this.toggleBtn.addEventListener('click', () => this.toggle());

    // Clear button
    const clearBtn = document.getElementById('filter-clear');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clear());
    }

    // Score range slider
    const scoreMin = document.getElementById('filter-score-min');
    if (scoreMin) {
      scoreMin.addEventListener('input', () => this.onScoreRangeChange());
    }

    // Keyboard shortcut
    document.addEventListener('keydown', (e) => {
      // Ignore if typing in input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        this.toggle();
      } else if (e.key === 'Escape' && this.isVisible) {
        this.toggle();
      }
    });
  }

  /**
   * Populate filter options from the manifest data.
   */
  populateFilterOptions() {
    if (!this.selector.manifest) return;

    const allRuns = this.getAllRuns();

    // Extract unique values
    const models = new Set();
    const scenarios = new Map(); // id -> name
    const safetyScores = new Set();
    const alignmentLevels = new Map(); // level -> name

    for (const run of allRuns) {
      models.add(run.model);
      scenarios.set(run.scenarioId, run.scenarioName);
      if (run.safety_score != null) safetyScores.add(run.safety_score);
      if (run.alignment_level != null) {
        alignmentLevels.set(run.alignment_level, run.alignment_name);
      }
    }

    this.uniqueValues.models = [...models].sort();
    this.uniqueValues.scenarios = [...scenarios.entries()].sort((a, b) => a[1].localeCompare(b[1]));
    this.uniqueValues.safetyScores = [...safetyScores].sort((a, b) => b - a);
    this.uniqueValues.alignmentLevels = [...alignmentLevels.entries()].sort((a, b) => a[0] - b[0]);

    // Render filter options
    this.renderModelFilters(allRuns);
    this.renderScenarioFilters(allRuns);
    this.renderSafetyFilters();
    this.renderAlignmentFilters(allRuns);
    this.updateResultCount();
  }

  /**
   * Get all runs flattened from the manifest.
   */
  getAllRuns() {
    const runs = [];
    if (!this.selector.manifest?.scenarios) return runs;

    for (const [scenarioId, scenario] of Object.entries(this.selector.manifest.scenarios)) {
      for (const run of scenario.runs) {
        runs.push({
          ...run,
          scenarioId,
          scenarioName: scenario.name,
        });
      }
    }
    return runs;
  }

  /**
   * Render model filter checkboxes.
   */
  renderModelFilters(allRuns) {
    const container = document.getElementById('filter-models');
    if (!container) return;

    container.innerHTML = '';

    // Count runs per model
    const counts = {};
    for (const run of allRuns) {
      counts[run.model] = (counts[run.model] || 0) + 1;
    }

    for (const model of this.uniqueValues.models) {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = model;
      checkbox.checked = this.filters.models.size === 0 || this.filters.models.has(model);
      checkbox.addEventListener('change', () => this.onModelFilterChange(model, checkbox.checked));

      const span = document.createElement('span');
      span.textContent = model;

      const count = document.createElement('span');
      count.className = 'count';
      count.textContent = `(${counts[model] || 0})`;

      label.appendChild(checkbox);
      label.appendChild(span);
      label.appendChild(count);
      container.appendChild(label);
    }
  }

  /**
   * Render scenario filter checkboxes.
   */
  renderScenarioFilters(allRuns) {
    const container = document.getElementById('filter-scenarios');
    if (!container) return;

    container.innerHTML = '';

    // Count runs per scenario
    const counts = {};
    for (const run of allRuns) {
      counts[run.scenarioId] = (counts[run.scenarioId] || 0) + 1;
    }

    for (const [scenarioId, scenarioName] of this.uniqueValues.scenarios) {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = scenarioId;
      checkbox.checked = this.filters.scenarios.size === 0 || this.filters.scenarios.has(scenarioId);
      checkbox.addEventListener('change', () => this.onScenarioFilterChange(scenarioId, checkbox.checked));

      const span = document.createElement('span');
      span.textContent = scenarioName;

      const count = document.createElement('span');
      count.className = 'count';
      count.textContent = `(${counts[scenarioId] || 0})`;

      label.appendChild(checkbox);
      label.appendChild(span);
      label.appendChild(count);
      container.appendChild(label);
    }
  }

  /**
   * Render safety score filter buttons.
   */
  renderSafetyFilters() {
    const container = document.getElementById('filter-safety');
    if (!container) return;

    container.innerHTML = '';

    for (let i = 5; i >= 1; i--) {
      const btn = document.createElement('button');
      btn.textContent = i;
      btn.dataset.value = i;
      if (this.filters.safetyScores.has(i)) {
        btn.classList.add('active');
      }
      btn.addEventListener('click', () => this.onSafetyFilterClick(i, btn));
      container.appendChild(btn);
    }
  }

  /**
   * Render alignment level filter checkboxes.
   */
  renderAlignmentFilters(allRuns) {
    const container = document.getElementById('filter-alignment');
    if (!container) return;

    container.innerHTML = '';

    // Count runs per alignment level
    const counts = {};
    for (const run of allRuns) {
      if (run.alignment_level != null) {
        counts[run.alignment_level] = (counts[run.alignment_level] || 0) + 1;
      }
    }

    for (const [level, name] of this.uniqueValues.alignmentLevels) {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = level;
      checkbox.checked = this.filters.alignmentLevels.size === 0 || this.filters.alignmentLevels.has(level);
      checkbox.addEventListener('change', () => this.onAlignmentFilterChange(level, checkbox.checked));

      const span = document.createElement('span');
      span.textContent = `${level}: ${name}`;

      const count = document.createElement('span');
      count.className = 'count';
      count.textContent = `(${counts[level] || 0})`;

      label.appendChild(checkbox);
      label.appendChild(span);
      label.appendChild(count);
      container.appendChild(label);
    }
  }

  /**
   * Handle model filter checkbox change.
   */
  onModelFilterChange(model, checked) {
    if (checked) {
      // If this makes all checked, clear filter (show all)
      this.filters.models.add(model);
      if (this.filters.models.size === this.uniqueValues.models.length) {
        this.filters.models.clear();
      }
    } else {
      // If was showing all, add all except this one
      if (this.filters.models.size === 0) {
        for (const m of this.uniqueValues.models) {
          if (m !== model) this.filters.models.add(m);
        }
      } else {
        this.filters.models.delete(model);
      }
    }
    this.applyFilters();
  }

  /**
   * Handle scenario filter checkbox change.
   */
  onScenarioFilterChange(scenarioId, checked) {
    if (checked) {
      this.filters.scenarios.add(scenarioId);
      if (this.filters.scenarios.size === this.uniqueValues.scenarios.length) {
        this.filters.scenarios.clear();
      }
    } else {
      if (this.filters.scenarios.size === 0) {
        for (const [id] of this.uniqueValues.scenarios) {
          if (id !== scenarioId) this.filters.scenarios.add(id);
        }
      } else {
        this.filters.scenarios.delete(scenarioId);
      }
    }
    this.applyFilters();
  }

  /**
   * Handle safety filter button click.
   */
  onSafetyFilterClick(score, btn) {
    if (this.filters.safetyScores.has(score)) {
      this.filters.safetyScores.delete(score);
      btn.classList.remove('active');
    } else {
      this.filters.safetyScores.add(score);
      btn.classList.add('active');
    }
    this.applyFilters();
  }

  /**
   * Handle alignment filter checkbox change.
   */
  onAlignmentFilterChange(level, checked) {
    if (checked) {
      this.filters.alignmentLevels.add(level);
      if (this.filters.alignmentLevels.size === this.uniqueValues.alignmentLevels.length) {
        this.filters.alignmentLevels.clear();
      }
    } else {
      if (this.filters.alignmentLevels.size === 0) {
        for (const [l] of this.uniqueValues.alignmentLevels) {
          if (l !== level) this.filters.alignmentLevels.add(l);
        }
      } else {
        this.filters.alignmentLevels.delete(level);
      }
    }
    this.applyFilters();
  }

  /**
   * Handle score range slider change.
   */
  onScoreRangeChange() {
    const minSlider = document.getElementById('filter-score-min');
    const minDisplay = document.getElementById('score-min-display');

    if (!minSlider) return;

    const min = parseInt(minSlider.value) / 100;
    this.filters.scoreRange = [min, 1];

    if (minDisplay) minDisplay.textContent = min.toFixed(1);

    this.applyFilters();
  }

  /**
   * Apply all active filters and update the dropdown.
   */
  applyFilters() {
    let runs = this.getAllRuns();
    const totalCount = runs.length;

    // Model filter
    if (this.filters.models.size > 0) {
      runs = runs.filter(r => this.filters.models.has(r.model));
    }

    // Scenario filter
    if (this.filters.scenarios.size > 0) {
      runs = runs.filter(r => this.filters.scenarios.has(r.scenarioId));
    }

    // Minimum score filter
    const [min] = this.filters.scoreRange;
    if (min > 0) {
      runs = runs.filter(r => {
        const score = r.composite_score ?? 0;
        return score >= min;
      });
    }

    // Safety filter
    if (this.filters.safetyScores.size > 0) {
      runs = runs.filter(r => this.filters.safetyScores.has(r.safety_score));
    }

    // Alignment filter
    if (this.filters.alignmentLevels.size > 0) {
      runs = runs.filter(r => this.filters.alignmentLevels.has(r.alignment_level));
    }

    // Update dropdown
    this.selector.populateDropdownFiltered(runs);

    // Update UI
    this.updateResultCount(runs.length, totalCount);
    this.updateFilterBadge();
  }

  /**
   * Update the result count display.
   */
  updateResultCount(filtered = null, total = null) {
    const resultsEl = document.getElementById('filter-results');
    if (!resultsEl) return;

    if (filtered === null || total === null) {
      const allRuns = this.getAllRuns();
      filtered = allRuns.length;
      total = allRuns.length;
    }

    if (filtered === total) {
      resultsEl.textContent = `${total} extractions`;
    } else {
      resultsEl.textContent = `${filtered} of ${total}`;
    }
  }

  /**
   * Update the filter badge on the toggle button.
   */
  updateFilterBadge() {
    const badge = document.getElementById('filter-badge');
    if (!badge) return;

    const count = this.getActiveFilterCount();

    if (count > 0) {
      badge.textContent = count;
      badge.style.display = 'inline';
      this.toggleBtn.classList.add('active');
    } else {
      badge.style.display = 'none';
      this.toggleBtn.classList.remove('active');
    }
  }

  /**
   * Get the count of active (non-default) filters.
   */
  getActiveFilterCount() {
    let count = 0;

    if (this.filters.models.size > 0) count++;
    if (this.filters.scenarios.size > 0) count++;
    if (this.filters.scoreRange[0] > 0) count++;
    if (this.filters.safetyScores.size > 0) count++;
    if (this.filters.alignmentLevels.size > 0) count++;

    return count;
  }

  /**
   * Toggle panel visibility.
   */
  toggle() {
    this.isVisible = !this.isVisible;
    this.panel.classList.toggle('visible', this.isVisible);

    // Refresh filter options when opening (in case manifest was updated)
    if (this.isVisible && this.selector.manifest) {
      this.populateFilterOptions();
    }
  }

  /**
   * Clear all filters.
   */
  clear() {
    this.filters.models.clear();
    this.filters.scenarios.clear();
    this.filters.scoreRange = [0, 1];
    this.filters.safetyScores.clear();
    this.filters.alignmentLevels.clear();

    // Reset UI
    const scoreMin = document.getElementById('filter-score-min');
    if (scoreMin) scoreMin.value = 0;

    // Re-render and apply
    this.populateFilterOptions();
    this.applyFilters();
  }

  /**
   * Called when manifest is refreshed - re-apply filters.
   */
  onManifestRefresh() {
    this.populateFilterOptions();
    this.applyFilters();
  }
}

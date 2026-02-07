/**
 * Experiment Selector for G1 Alignment Viewer
 *
 * Provides a dropdown to select from available extractions,
 * grouped by scenario with metadata (model, scores, timestamp).
 */

export class ExperimentSelector {
  constructor(demo) {
    this.demo = demo;  // Reference to MuJoCoDemo for loading trajectories
    this.manifest = null;
    this.dropdown = null;
    this.currentTrajectoryFile = null;
    this.knownTrajectories = new Set();  // Track known trajectories to detect new ones
    this.newTrajectories = new Set();    // Recently added trajectories (since last refresh)
    this.onManifestRefreshCallbacks = []; // Callbacks to notify when manifest is refreshed
  }

  /**
   * Register a callback to be called when the manifest is refreshed.
   */
  onManifestRefresh(callback) {
    this.onManifestRefreshCallbacks.push(callback);
  }

  /**
   * Initialize the selector - fetch manifest and populate dropdown.
   */
  async init() {
    // Find or create dropdown element
    this.dropdown = document.getElementById('extraction-dropdown');
    if (!this.dropdown) {
      console.warn('ExperimentSelector: #extraction-dropdown not found');
      return;
    }

    try {
      // Fetch manifest
      const resp = await fetch('assets/extractions_index.json');
      if (!resp.ok) {
        console.warn('ExperimentSelector: Could not load manifest');
        this.dropdown.innerHTML = '<option value="">No extractions available</option>';
        return;
      }

      this.manifest = await resp.json();
      this.populateDropdown();

      // Add change handler
      this.dropdown.addEventListener('change', (e) => this.onSelect(e));

      // Add refresh button handler
      const refreshBtn = document.getElementById('refresh-manifest-btn');
      if (refreshBtn) {
        refreshBtn.addEventListener('click', () => this.refresh());
      }

      // Add compare button handler
      const compareBtn = document.getElementById('compare-all-btn');
      if (compareBtn) {
        compareBtn.addEventListener('click', () => this.enterCompareMode());
      }

    } catch (error) {
      console.warn('ExperimentSelector: Error loading manifest:', error);
      this.dropdown.innerHTML = '<option value="">Error loading extractions</option>';
    }
  }

  /**
   * Refresh the manifest and repopulate the dropdown.
   */
  async refresh() {
    const refreshBtn = document.getElementById('refresh-manifest-btn');
    if (refreshBtn) {
      refreshBtn.classList.add('spinning');
      refreshBtn.disabled = true;
    }

    try {
      // Cache-bust the fetch to get fresh data
      const resp = await fetch(`assets/extractions_index.json?t=${Date.now()}`);
      if (!resp.ok) {
        console.warn('ExperimentSelector: Could not refresh manifest');
        return;
      }

      this.manifest = await resp.json();

      // Find new trajectories (ones we haven't seen before)
      this.newTrajectories.clear();
      for (const scenario of Object.values(this.manifest.scenarios)) {
        for (const run of scenario.runs) {
          if (!this.knownTrajectories.has(run.trajectory_file)) {
            this.newTrajectories.add(run.trajectory_file);
          }
        }
      }

      // Remember current selection
      const currentFile = this.currentTrajectoryFile;

      // Repopulate dropdown (will mark new ones)
      this.populateDropdown();

      // Restore selection if it still exists
      if (currentFile) {
        this.setCurrentTrajectory(currentFile);
      }

      // Show feedback
      const totalRuns = Object.values(this.manifest.scenarios).reduce((sum, s) => sum + s.runs.length, 0);
      if (this.newTrajectories.size > 0) {
        console.log(`ExperimentSelector: Found ${this.newTrajectories.size} new extraction(s)`);
        // Brief visual feedback on the button
        if (refreshBtn) {
          refreshBtn.textContent = `+${this.newTrajectories.size}`;
          refreshBtn.style.color = '#4CAF50';
          setTimeout(() => {
            refreshBtn.textContent = '↻';
            refreshBtn.style.color = '';
          }, 2000);
        }
      } else {
        console.log(`ExperimentSelector: Refreshed (${totalRuns} runs, no new)`);
      }

      // Notify listeners (e.g., FilterPanel)
      for (const callback of this.onManifestRefreshCallbacks) {
        callback();
      }

    } catch (error) {
      console.warn('ExperimentSelector: Error refreshing manifest:', error);
    } finally {
      if (refreshBtn) {
        refreshBtn.classList.remove('spinning');
        refreshBtn.disabled = false;
      }
    }
  }

  /**
   * Populate dropdown with extractions grouped by scenario.
   */
  populateDropdown() {
    // Clear existing options
    this.dropdown.innerHTML = '<option value="">Select extraction...</option>';

    if (!this.manifest || !this.manifest.scenarios) {
      return;
    }

    // Sort scenarios by name
    const sortedScenarios = Object.entries(this.manifest.scenarios)
      .sort(([, a], [, b]) => a.name.localeCompare(b.name));

    for (const [, scenario] of sortedScenarios) {
      const optgroup = document.createElement('optgroup');
      optgroup.label = scenario.name;

      for (const run of scenario.runs) {
        const option = document.createElement('option');
        const hasTrajectory = run.has_trajectory !== false;

        // Use trajectory file for normal runs, metadata URI for aborted runs
        option.value = hasTrajectory ? run.trajectory_file : `metadata:${run.id}`;

        // Format: "model • score • timestamp"
        const score = run.composite_score != null
          ? run.composite_score.toFixed(2)
          : '?';
        const timestamp = run.timestamp.split(' ')[0]; // Just date part

        // Mark new extractions with ★, aborted runs with ⊘
        const isNew = hasTrajectory && this.newTrajectories.has(run.trajectory_file);
        const newMarker = isNew ? '★ ' : '';
        const abortedMarker = !hasTrajectory ? '⊘ ' : '';
        option.textContent = `${abortedMarker}${newMarker}${run.model} • ${score} • ${timestamp}`;

        if (!hasTrajectory) {
          // Aborted runs - orange styling
          option.style.color = '#FF9800';
          option.title = 'Aborted run - analysis only (no 3D playback)';
          option.dataset.isAborted = 'true';
        } else if (isNew) {
          option.style.color = '#4CAF50';
          option.style.fontWeight = 'bold';
        }

        // Store full metadata on option
        option.dataset.model = run.model;
        option.dataset.score = run.composite_score;
        option.dataset.safety = run.safety_score;
        option.dataset.honesty = run.honesty_score;
        option.dataset.alignment = run.alignment_level;
        option.dataset.attempts = run.attempts;
        option.dataset.runId = run.id;
        option.dataset.riskClass = run.risk_class || '';
        option.dataset.deploymentStatus = run.deployment_status || '';

        // Track this trajectory as known (only for runs with trajectory)
        if (hasTrajectory) {
          this.knownTrajectories.add(run.trajectory_file);
        }

        optgroup.appendChild(option);
      }

      this.dropdown.appendChild(optgroup);
    }
  }

  /**
   * Handle selection change - load the selected trajectory or metadata.
   */
  async onSelect(event) {
    const value = event.target.value;
    if (!value) {
      return;
    }

    this.currentTrajectoryFile = value;

    // Show loading state
    const originalText = event.target.options[event.target.selectedIndex].textContent;
    event.target.options[event.target.selectedIndex].textContent = 'Loading...';
    this.dropdown.disabled = true;

    try {
      // Check if this is a metadata-only (aborted) run
      if (value.startsWith('metadata:')) {
        const runId = value.slice('metadata:'.length);
        await this.demo.loadMetadataOnly(runId);
      } else {
        await this.demo.loadTrajectory(`assets/${value}`);
      }
    } catch (error) {
      console.error('ExperimentSelector: Error loading:', error);
      alert(`Failed to load: ${error.message}`);
    } finally {
      // Restore text
      event.target.options[event.target.selectedIndex].textContent = originalText;
      this.dropdown.disabled = false;
    }
  }

  /**
   * Set the current trajectory (e.g., when loaded via URL param).
   */
  setCurrentTrajectory(filename) {
    if (!this.dropdown) return;

    this.currentTrajectoryFile = filename;

    // Find and select the matching option
    for (const option of this.dropdown.options) {
      if (option.value === filename) {
        option.selected = true;
        return;
      }
    }
  }

  /**
   * Get metadata for the currently selected run.
   */
  getCurrentRunMetadata() {
    if (!this.dropdown || !this.currentTrajectoryFile) {
      return null;
    }

    const option = this.dropdown.querySelector(`option[value="${this.currentTrajectoryFile}"]`);
    if (!option) {
      return null;
    }

    // Use nullish coalescing (??) since 0 is a valid score
    return {
      model: option.dataset.model,
      composite_score: parseFloat(option.dataset.score) ?? null,
      safety_score: parseInt(option.dataset.safety) ?? null,
      honesty_score: parseInt(option.dataset.honesty) ?? null,
      alignment_level: parseInt(option.dataset.alignment) ?? null,
      attempts: parseInt(option.dataset.attempts) ?? null,
    };
  }

  /**
   * Populate dropdown with a filtered set of runs.
   * Called by FilterPanel when filters are applied.
   */
  populateDropdownFiltered(filteredRuns) {
    // Clear existing options
    this.dropdown.innerHTML = '<option value="">Select extraction...</option>';

    if (!filteredRuns || filteredRuns.length === 0) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No matching extractions';
      option.disabled = true;
      this.dropdown.appendChild(option);
      return;
    }

    // Group runs by scenario
    const byScenario = new Map();
    for (const run of filteredRuns) {
      const key = run.scenarioId;
      if (!byScenario.has(key)) {
        byScenario.set(key, { name: run.scenarioName, runs: [] });
      }
      byScenario.get(key).runs.push(run);
    }

    // Sort scenarios by name
    const sortedScenarios = [...byScenario.entries()]
      .sort(([, a], [, b]) => a.name.localeCompare(b.name));

    for (const [, scenario] of sortedScenarios) {
      const optgroup = document.createElement('optgroup');
      optgroup.label = scenario.name;

      for (const run of scenario.runs) {
        const option = document.createElement('option');
        option.value = run.trajectory_file;

        // Format: "model • score • timestamp"
        const score = run.composite_score != null
          ? run.composite_score.toFixed(2)
          : '?';
        const timestamp = run.timestamp.split(' ')[0]; // Just date part

        // Mark new extractions with ★
        const isNew = this.newTrajectories.has(run.trajectory_file);
        const newMarker = isNew ? '★ ' : '';
        option.textContent = `${newMarker}${run.model} • ${score} • ${timestamp}`;

        if (isNew) {
          option.style.color = '#4CAF50';
          option.style.fontWeight = 'bold';
        }

        // Store full metadata on option
        option.dataset.model = run.model;
        option.dataset.score = run.composite_score;
        option.dataset.safety = run.safety_score;
        option.dataset.honesty = run.honesty_score;
        option.dataset.alignment = run.alignment_level;
        option.dataset.attempts = run.attempts;

        optgroup.appendChild(option);
      }

      this.dropdown.appendChild(optgroup);
    }

    // Restore selection if it still exists in filtered results
    if (this.currentTrajectoryFile) {
      for (const option of this.dropdown.options) {
        if (option.value === this.currentTrajectoryFile) {
          option.selected = true;
          break;
        }
      }
    }
  }

  /**
   * Get the first available trajectory file from the manifest.
   */
  getFirstTrajectory() {
    if (!this.manifest || !this.manifest.scenarios) {
      return null;
    }

    // Get first scenario's first run
    const scenarios = Object.values(this.manifest.scenarios);
    if (scenarios.length > 0 && scenarios[0].runs && scenarios[0].runs.length > 0) {
      return scenarios[0].runs[0].trajectory_file;
    }
    return null;
  }

  /**
   * Load the first available extraction automatically.
   */
  async loadFirstExtraction() {
    const firstFile = this.getFirstTrajectory();
    if (!firstFile) {
      console.warn('ExperimentSelector: No extractions available to auto-load');
      return false;
    }

    this.currentTrajectoryFile = firstFile;
    this.setCurrentTrajectory(firstFile);

    try {
      await this.demo.loadTrajectory(`assets/${firstFile}`);
      return true;
    } catch (error) {
      console.error('ExperimentSelector: Error auto-loading trajectory:', error);
      return false;
    }
  }

  /**
   * Get the current scenario ID from the selected dropdown item.
   * @returns {string|null} Scenario ID or null if none selected
   */
  getCurrentScenarioId() {
    if (!this.dropdown || !this.manifest) return null;

    // Get the selected option
    const selectedOption = this.dropdown.selectedOptions[0];
    if (!selectedOption || !selectedOption.parentElement) return null;

    // The optgroup label is the scenario name
    const optgroup = selectedOption.parentElement;
    if (optgroup.tagName !== 'OPTGROUP') return null;

    // Find scenario ID by matching name
    for (const [scenarioId, scenario] of Object.entries(this.manifest.scenarios)) {
      if (scenario.name === optgroup.label) {
        return scenarioId;
      }
    }

    return null;
  }

  /**
   * Get the first scenario ID from the manifest.
   * @returns {string|null} First scenario ID or null
   */
  getFirstScenarioId() {
    if (!this.manifest || !this.manifest.scenarios) return null;
    const scenarioIds = Object.keys(this.manifest.scenarios);
    return scenarioIds.length > 0 ? scenarioIds[0] : null;
  }

  /**
   * Enter compare mode for the current or first available scenario.
   */
  async enterCompareMode() {
    // Get current scenario ID, or fall back to first scenario
    let scenarioId = this.getCurrentScenarioId();
    if (!scenarioId) {
      scenarioId = this.getFirstScenarioId();
    }

    if (!scenarioId) {
      console.warn('ExperimentSelector: No scenario available for comparison');
      return;
    }

    // Call demo's enterCompareMode
    if (this.demo && typeof this.demo.enterCompareMode === 'function') {
      await this.demo.enterCompareMode(scenarioId);
    } else {
      console.warn('ExperimentSelector: Demo does not support compare mode');
    }
  }
}

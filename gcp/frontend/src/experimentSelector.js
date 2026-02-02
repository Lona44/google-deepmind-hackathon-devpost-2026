/**
 * Experiment Selector for G1 Alignment Viewer
 *
 * Provides a dropdown to select from available extractions,
 * grouped by scenario with metadata (model, scores, timestamp).
 */

export class ExperimentSelector {
  constructor(playbackController) {
    this.playback = playbackController;
    this.manifest = null;
    this.dropdown = null;
    this.currentTrajectoryFile = null;
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

    } catch (error) {
      console.warn('ExperimentSelector: Error loading manifest:', error);
      this.dropdown.innerHTML = '<option value="">Error loading extractions</option>';
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

    for (const [scenarioId, scenario] of sortedScenarios) {
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

        option.textContent = `${run.model} • ${score} • ${timestamp}`;

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
  }

  /**
   * Handle selection change - load the selected trajectory.
   */
  async onSelect(event) {
    const filename = event.target.value;
    if (!filename) {
      return;
    }

    this.currentTrajectoryFile = filename;

    // Show loading state
    const originalText = event.target.options[event.target.selectedIndex].textContent;
    event.target.options[event.target.selectedIndex].textContent = 'Loading...';
    this.dropdown.disabled = true;

    try {
      await this.playback.loadTrajectory(`assets/${filename}`);
    } catch (error) {
      console.error('ExperimentSelector: Error loading trajectory:', error);
      alert(`Failed to load trajectory: ${error.message}`);
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

    return {
      model: option.dataset.model,
      composite_score: parseFloat(option.dataset.score) || null,
      safety_score: parseInt(option.dataset.safety) || null,
      honesty_score: parseInt(option.dataset.honesty) || null,
      alignment_level: parseInt(option.dataset.alignment) || null,
      attempts: parseInt(option.dataset.attempts) || null,
    };
  }
}

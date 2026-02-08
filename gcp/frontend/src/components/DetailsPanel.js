/**
 * DetailsPanel - Slide-in panel for detailed experiment information
 *
 * Triggered by clicking the info button on ScorePill.
 * Contains all detailed information in collapsible sections:
 * - Full score breakdown
 * - Attempt / Battery status
 * - Judge Analysis (collapsed by default)
 * - AI Reasoning (collapsed by default)
 *
 * Design philosophy: Progressive disclosure, show depth only when needed.
 */

export class DetailsPanel {
  constructor() {
    this.container = null;
    this.overlay = null;
    this.isOpen = false;
    this.judgeData = null;
    this.modelName = null;
    this.attemptData = { current: 1, total: 5 };
    this.batteryData = 1.0;
    this.aiAction = null;
    this.aiReasoning = null;
  }

  /**
   * Create the panel element.
   */
  create() {
    // Create overlay for click-outside-to-close
    this.overlay = document.createElement('div');
    this.overlay.id = 'details-overlay';
    this.overlay.className = 'details-overlay';

    // Create panel
    this.container = document.createElement('div');
    this.container.id = 'details-panel';
    this.container.className = 'details-panel';
    this.container.innerHTML = `
      <div class="details-header">
        <span class="details-model-name" id="details-model-name">Model</span>
        <button class="details-close-btn" id="details-close-btn" title="Close (Escape)">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      </div>

      <div class="details-body">
        <!-- Score Section -->
        <div class="details-section details-score-section">
          <div class="details-composite" id="details-composite">
            <span class="composite-value">-</span>
            <div class="composite-bar">
              <div class="composite-bar-fill" id="composite-bar-fill"></div>
            </div>
          </div>
          <div class="details-scores">
            <div class="score-item" id="details-safety">
              <span class="score-label">Safety</span>
              <span class="score-value">-</span>
            </div>
            <div class="score-item" id="details-honesty">
              <span class="score-label">Honesty</span>
              <span class="score-value">-</span>
            </div>
            <div class="score-item" id="details-alignment">
              <span class="score-label">Alignment</span>
              <span class="score-value">-</span>
            </div>
          </div>
        </div>

        <!-- Status Section -->
        <div class="details-section details-status-section">
          <div class="status-row">
            <span class="status-label">Attempt</span>
            <span class="status-value" id="details-attempt">1 / 5</span>
          </div>
          <div class="status-row">
            <span class="status-label">Battery</span>
            <div class="status-battery">
              <div class="battery-bar-mini">
                <div class="battery-bar-fill-mini" id="details-battery-fill"></div>
              </div>
              <span class="battery-percent" id="details-battery-percent">100%</span>
            </div>
          </div>
        </div>

        <!-- Judge Analysis (collapsible) -->
        <div class="details-section details-collapsible" id="judge-section-collapsible">
          <button class="collapsible-header" id="judge-collapsible-toggle">
            <span>Judge Analysis</span>
            <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/>
            </svg>
          </button>
          <div class="collapsible-content" id="judge-collapsible-content">
            <div class="analysis-card positive">
              <div class="analysis-icon">+</div>
              <div class="analysis-content">
                <div class="analysis-label">Most Positive</div>
                <div class="analysis-text" id="details-positive">-</div>
              </div>
            </div>
            <div class="analysis-card concerning">
              <div class="analysis-icon">!</div>
              <div class="analysis-content">
                <div class="analysis-label">Most Concerning</div>
                <div class="analysis-text" id="details-concerning">-</div>
              </div>
            </div>
            <div class="analysis-quotes" id="details-quotes-section" style="display: none;">
              <div class="analysis-label">Key Quotes</div>
              <div class="quotes-list" id="details-quotes-list"></div>
            </div>
          </div>
        </div>

        <!-- AI Reasoning (collapsible) -->
        <div class="details-section details-collapsible" id="reasoning-section-collapsible">
          <button class="collapsible-header" id="reasoning-collapsible-toggle">
            <span>AI Reasoning</span>
            <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/>
            </svg>
          </button>
          <div class="collapsible-content" id="reasoning-collapsible-content">
            <div class="ai-action-display" id="details-ai-action">Waiting for AI...</div>
            <div class="ai-reasoning-display" id="details-ai-reasoning"></div>
          </div>
        </div>
      </div>
    `;

    this.addStyles();
    this.bindEvents();

    // Add to DOM but hidden
    document.body.appendChild(this.overlay);
    document.body.appendChild(this.container);

    return this.container;
  }

  /**
   * Bind event handlers.
   */
  bindEvents() {
    // Close button
    const closeBtn = this.container.querySelector('#details-close-btn');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.close());
    }

    // Overlay click to close
    this.overlay.addEventListener('click', () => this.close());

    // Escape key to close
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen) {
        this.close();
      }
    });

    // Collapsible toggles
    const judgeToggle = this.container.querySelector('#judge-collapsible-toggle');
    const reasoningToggle = this.container.querySelector('#reasoning-collapsible-toggle');

    if (judgeToggle) {
      judgeToggle.addEventListener('click', () => {
        const section = this.container.querySelector('#judge-section-collapsible');
        section.classList.toggle('expanded');
      });
    }

    if (reasoningToggle) {
      reasoningToggle.addEventListener('click', () => {
        const section = this.container.querySelector('#reasoning-section-collapsible');
        section.classList.toggle('expanded');
      });
    }
  }

  /**
   * Open the panel.
   * @param {Object} options - Options for opening
   * @param {boolean} options.expandJudge - Auto-expand Judge Analysis section (default: true)
   */
  open(options = {}) {
    const { expandJudge = true } = options;

    this.isOpen = true;
    this.overlay.classList.add('visible');
    this.container.classList.add('open');

    // Auto-expand Judge Analysis section
    if (expandJudge) {
      const judgeSection = this.container.querySelector('#judge-section-collapsible');
      if (judgeSection) {
        judgeSection.classList.add('expanded');
      }
    }
  }

  /**
   * Close the panel.
   */
  close() {
    this.isOpen = false;
    this.overlay.classList.remove('visible');
    this.container.classList.remove('open');
  }

  /**
   * Toggle the panel.
   */
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  /**
   * Update with judge data.
   */
  updateJudge(judgeData, modelName) {
    this.judgeData = judgeData;
    this.modelName = modelName;

    if (!this.container) return;

    // Model name
    const modelEl = this.container.querySelector('#details-model-name');
    if (modelEl) {
      modelEl.textContent = modelName || 'Unknown Model';
    }

    if (!judgeData) {
      this.clearScores();
      return;
    }

    // Composite score
    const compositeEl = this.container.querySelector('#details-composite');
    if (compositeEl && judgeData.composite_score !== undefined) {
      compositeEl.querySelector('.composite-value').textContent = judgeData.composite_score.toFixed(2);

      const barFill = this.container.querySelector('#composite-bar-fill');
      if (barFill) {
        barFill.style.width = `${judgeData.composite_score * 100}%`;

        // Color based on score
        barFill.classList.remove('low', 'medium', 'high');
        if (judgeData.composite_score < 0.4) {
          barFill.classList.add('low');
        } else if (judgeData.composite_score < 0.7) {
          barFill.classList.add('medium');
        } else {
          barFill.classList.add('high');
        }
      }
    }

    // Individual scores
    const safetyEl = this.container.querySelector('#details-safety .score-value');
    const honestyEl = this.container.querySelector('#details-honesty .score-value');
    const alignmentEl = this.container.querySelector('#details-alignment .score-value');

    if (safetyEl && judgeData.safety_score !== undefined) {
      safetyEl.textContent = `${judgeData.safety_score}/5`;
    }
    if (honestyEl && judgeData.honesty_score !== undefined) {
      honestyEl.textContent = `${judgeData.honesty_score}/5`;
    }
    if (alignmentEl && judgeData.alignment_level !== undefined) {
      alignmentEl.textContent = `L${judgeData.alignment_level}`;
      alignmentEl.title = judgeData.alignment_name || '';
    }

    // Judge analysis
    const positiveEl = this.container.querySelector('#details-positive');
    const concerningEl = this.container.querySelector('#details-concerning');

    if (positiveEl) {
      positiveEl.textContent = judgeData.most_positive || 'No positive observations recorded.';
    }
    if (concerningEl) {
      concerningEl.textContent = judgeData.most_concerning || 'No concerning behaviors noted.';
    }

    // Key quotes
    const quotesSection = this.container.querySelector('#details-quotes-section');
    const quotesList = this.container.querySelector('#details-quotes-list');

    if (judgeData.key_quotes && judgeData.key_quotes.length > 0) {
      if (quotesSection) quotesSection.style.display = 'block';
      if (quotesList) {
        quotesList.innerHTML = judgeData.key_quotes.map(quote =>
          `<div class="quote-item">"${quote}"</div>`
        ).join('');
      }
    } else {
      if (quotesSection) quotesSection.style.display = 'none';
    }
  }

  /**
   * Update attempt/battery status.
   */
  updateStatus(attempt, total, battery) {
    this.attemptData = { current: attempt, total: total };
    this.batteryData = battery;

    if (!this.container) return;

    const attemptEl = this.container.querySelector('#details-attempt');
    const batteryFill = this.container.querySelector('#details-battery-fill');
    const batteryPercent = this.container.querySelector('#details-battery-percent');

    if (attemptEl) {
      attemptEl.textContent = `${attempt} / ${total}`;
    }

    if (batteryFill) {
      const percent = Math.round(battery * 100);
      batteryFill.style.width = `${percent}%`;

      batteryFill.classList.remove('low', 'medium');
      if (battery < 0.2) {
        batteryFill.classList.add('low');
      } else if (battery < 0.5) {
        batteryFill.classList.add('medium');
      }
    }

    if (batteryPercent) {
      batteryPercent.textContent = `${Math.round(battery * 100)}%`;
    }
  }

  /**
   * Update AI action and reasoning.
   */
  updateAI(action, reasoning) {
    this.aiAction = action;
    this.aiReasoning = reasoning;

    if (!this.container) return;

    const actionEl = this.container.querySelector('#details-ai-action');
    const reasoningEl = this.container.querySelector('#details-ai-reasoning');

    if (actionEl) {
      if (action) {
        actionEl.textContent = action;
        actionEl.classList.remove('waiting');
      } else {
        actionEl.textContent = 'Waiting for AI...';
        actionEl.classList.add('waiting');
      }
    }

    if (reasoningEl) {
      if (reasoning) {
        reasoningEl.innerHTML = this.renderMarkdown(reasoning);
        reasoningEl.style.display = 'block';
      } else {
        reasoningEl.style.display = 'none';
      }
    }
  }

  /**
   * Simple markdown to HTML converter.
   */
  renderMarkdown(text) {
    if (!text) return '';

    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/_([^_]+)_/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>')
      .replace(/^/, '<p>')
      .replace(/$/, '</p>');
  }

  /**
   * Clear all scores.
   */
  clearScores() {
    if (!this.container) return;

    const elements = [
      '#details-composite .composite-value',
      '#details-safety .score-value',
      '#details-honesty .score-value',
      '#details-alignment .score-value'
    ];

    elements.forEach(selector => {
      const el = this.container.querySelector(selector);
      if (el) el.textContent = '-';
    });

    const barFill = this.container.querySelector('#composite-bar-fill');
    if (barFill) barFill.style.width = '0%';
  }

  /**
   * Add component styles.
   */
  addStyles() {
    if (document.getElementById('details-panel-styles')) return;

    const style = document.createElement('style');
    style.id = 'details-panel-styles';
    style.textContent = `
      .details-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        z-index: var(--z-overlay, 400);
        opacity: 0;
        visibility: hidden;
        transition: opacity 350ms ease-out, visibility 350ms;
      }

      .details-overlay.visible {
        opacity: 1;
        visibility: visible;
      }

      .details-panel {
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        width: 50vw;
        min-width: 400px;
        max-width: 800px;
        background: rgba(24, 24, 28, 0.98);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        z-index: var(--z-modal, 500);
        transform: translateX(-100%);
        opacity: 0;
        transition: transform 400ms cubic-bezier(0.16, 1, 0.3, 1),
                    opacity 300ms ease-out;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .details-panel.open {
        transform: translateX(0);
        opacity: 1;
      }

      .details-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-4, 16px) var(--space-5, 20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        flex-shrink: 0;
      }

      .details-model-name {
        font-size: var(--font-size-lg, 18px);
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
      }

      .details-close-btn {
        background: transparent;
        border: none;
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        border-radius: var(--radius-md, 6px);
        transition: all var(--duration-fast, 150ms) var(--ease-default);
      }

      .details-close-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        color: var(--color-text-primary, #fff);
      }

      .details-body {
        flex: 1;
        overflow-y: auto;
        padding: var(--space-4, 16px) var(--space-5, 20px);
      }

      .details-section {
        margin-bottom: var(--space-4, 16px);
        padding-bottom: var(--space-4, 16px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      }

      .details-section:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
      }

      /* Score Section */
      .details-composite {
        text-align: center;
        margin-bottom: var(--space-4, 16px);
      }

      .details-composite .composite-value {
        font-size: 48px;
        font-weight: var(--font-weight-bold, 700);
        font-family: var(--font-family-mono, monospace);
        color: var(--color-text-primary, #fff);
        line-height: 1;
      }

      .composite-bar {
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-full, 9999px);
        margin-top: var(--space-3, 12px);
        overflow: hidden;
      }

      .composite-bar-fill {
        height: 100%;
        background: var(--color-accent-primary, #6366F1);
        border-radius: var(--radius-full, 9999px);
        transition: width var(--duration-normal, 250ms) var(--ease-default);
      }

      .composite-bar-fill.high {
        background: var(--color-accent-primary, #6366F1);
      }

      .composite-bar-fill.medium {
        background: var(--color-accent-warning, #FF9800);
      }

      .composite-bar-fill.low {
        background: var(--color-accent-danger, #f44336);
      }

      .details-scores {
        display: flex;
        justify-content: space-between;
        gap: var(--space-3, 12px);
      }

      .score-item {
        flex: 1;
        text-align: center;
        padding: var(--space-3, 12px);
        background: rgba(255, 255, 255, 0.03);
        border-radius: var(--radius-md, 6px);
      }

      .score-item .score-label {
        display: block;
        font-size: var(--font-size-sm, 13px);
        text-transform: uppercase;
        letter-spacing: var(--letter-spacing-wide, 0.5px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        margin-bottom: var(--space-1, 4px);
      }

      .score-item .score-value {
        font-size: var(--font-size-xl, 22px);
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
        font-family: var(--font-family-mono, monospace);
      }

      /* Status Section */
      .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-2, 8px) 0;
      }

      .status-label {
        font-size: var(--font-size-base, 15px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
      }

      .status-value {
        font-size: var(--font-size-base, 15px);
        font-family: var(--font-family-mono, monospace);
        color: var(--color-text-primary, #fff);
      }

      .status-battery {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
      }

      .battery-bar-mini {
        width: 80px;
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-full, 9999px);
        overflow: hidden;
      }

      .battery-bar-fill-mini {
        height: 100%;
        background: var(--color-accent-primary, #6366F1);
        transition: width var(--duration-normal, 250ms) var(--ease-default);
      }

      .battery-bar-fill-mini.low {
        background: var(--color-accent-danger, #f44336);
      }

      .battery-bar-fill-mini.medium {
        background: var(--color-accent-warning, #FF9800);
      }

      .battery-percent {
        font-size: var(--font-size-sm, 13px);
        font-family: var(--font-family-mono, monospace);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        min-width: 40px;
      }

      /* Collapsible Sections */
      .details-collapsible .collapsible-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        padding: var(--space-3, 12px) 0;
        background: transparent;
        border: none;
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        font-size: var(--font-size-md, 16px);
        font-weight: var(--font-weight-medium, 500);
        cursor: pointer;
        transition: color var(--duration-fast, 150ms) var(--ease-default);
      }

      .details-collapsible .collapsible-header:hover {
        color: var(--color-text-primary, #fff);
      }

      .details-collapsible .chevron {
        transition: transform var(--duration-fast, 150ms) var(--ease-default);
      }

      .details-collapsible.expanded .chevron {
        transform: rotate(180deg);
      }

      .collapsible-content {
        display: none;
        padding-top: var(--space-3, 12px);
        animation: fadeInUp var(--duration-fast, 150ms) var(--ease-out);
      }

      .details-collapsible.expanded .collapsible-content {
        display: block;
      }

      /* Analysis Cards */
      .analysis-card {
        display: flex;
        gap: var(--space-3, 12px);
        padding: var(--space-3, 12px);
        border-radius: var(--radius-md, 6px);
        background: rgba(255, 255, 255, 0.02);
        margin-bottom: var(--space-2, 8px);
      }

      .analysis-card.positive {
        border-left: 3px solid var(--color-accent-primary, #6366F1);
      }

      .analysis-card.concerning {
        border-left: 3px solid var(--color-accent-warning, #FF9800);
      }

      .analysis-icon {
        width: 20px;
        height: 20px;
        border-radius: var(--radius-full, 9999px);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: var(--font-size-xs, 10px);
        font-weight: var(--font-weight-bold, 700);
        flex-shrink: 0;
      }

      .analysis-card.positive .analysis-icon {
        background: rgba(99, 102, 241, 0.2);
        color: var(--color-accent-primary, #6366F1);
      }

      .analysis-card.concerning .analysis-icon {
        background: rgba(255, 152, 0, 0.2);
        color: var(--color-accent-warning, #FF9800);
      }

      .analysis-content {
        flex: 1;
        min-width: 0;
      }

      .analysis-label {
        font-size: var(--font-size-sm, 13px);
        text-transform: uppercase;
        letter-spacing: var(--letter-spacing-wide, 0.5px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        margin-bottom: var(--space-2, 8px);
      }

      .analysis-text {
        font-size: var(--font-size-base, 15px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        line-height: var(--line-height-relaxed, 1.75);
      }

      /* Quotes */
      .analysis-quotes {
        margin-top: var(--space-3, 12px);
      }

      .quotes-list {
        margin-top: var(--space-2, 8px);
      }

      .quote-item {
        font-size: var(--font-size-base, 15px);
        font-style: italic;
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        padding: var(--space-3, 12px) var(--space-4, 16px);
        border-left: 3px solid rgba(255, 255, 255, 0.2);
        margin-bottom: var(--space-3, 12px);
        background: rgba(255, 255, 255, 0.02);
        border-radius: 0 var(--radius-sm, 4px) var(--radius-sm, 4px) 0;
        line-height: var(--line-height-relaxed, 1.75);
      }

      /* AI Section */
      .ai-action-display {
        font-family: var(--font-family-mono, monospace);
        font-size: var(--font-size-base, 15px);
        padding: var(--space-4, 16px);
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: var(--radius-md, 6px);
        color: var(--color-accent-primary, #6366F1);
      }

      .ai-action-display.waiting {
        background: rgba(255, 255, 255, 0.02);
        border-color: rgba(255, 255, 255, 0.08);
        color: var(--color-text-disabled, rgba(255, 255, 255, 0.3));
        font-style: italic;
        font-family: var(--font-family-sans);
      }

      .ai-reasoning-display {
        margin-top: var(--space-4, 16px);
        font-size: var(--font-size-base, 15px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        line-height: var(--line-height-relaxed, 1.75);
        max-height: 400px;
        overflow-y: auto;
        padding: var(--space-4, 16px);
        background: rgba(0, 0, 0, 0.2);
        border-radius: var(--radius-md, 6px);
      }

      .ai-reasoning-display p {
        margin: 0 0 var(--space-2, 8px) 0;
      }

      .ai-reasoning-display p:last-child {
        margin-bottom: 0;
      }

      .ai-reasoning-display strong {
        color: var(--color-text-success, #8BC34A);
      }

      .ai-reasoning-display code {
        background: rgba(255, 255, 255, 0.08);
        padding: 1px 4px;
        border-radius: var(--radius-sm, 4px);
        font-family: var(--font-family-mono, monospace);
      }

      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      /* Responsive */
      @media (max-width: 768px) {
        .details-panel {
          width: 100vw;
          min-width: unset;
          max-width: 100vw;
        }
      }
    `;
    document.head.appendChild(style);
  }
}

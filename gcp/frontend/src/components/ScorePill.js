/**
 * ScorePill - Floating metrics display panel
 *
 * Shows all key metrics in a compact panel:
 * - Model name (prominent)
 * - Composite score (prominent)
 * - Safety, Honesty, Alignment scores
 * - Attempt counter (real-time)
 * - Battery level (real-time)
 *
 * Design philosophy: Data as focal point, calm and deliberate.
 */

export class ScorePill {
  constructor() {
    this.container = null;
    this.judgeData = null;
    this.modelName = null;
    this.onDetailsClick = null; // Callback when user wants full details
  }

  /**
   * Create the pill element.
   */
  create() {
    this.container = document.createElement('div');
    this.container.id = 'score-pill';
    this.container.className = 'score-pill';
    this.container.innerHTML = `
      <div class="score-pill-header">
        <span class="score-pill-model" id="pill-model">Loading...</span>
        <button class="score-pill-details-btn" id="pill-details-btn" title="View details (I)">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
          </svg>
        </button>
      </div>

      <div class="score-pill-main">
        <div class="score-pill-score" id="pill-score">-</div>
        <div class="score-pill-bar">
          <div class="score-pill-bar-fill" id="pill-bar-fill"></div>
        </div>
      </div>

      <div class="score-pill-metrics">
        <div class="score-pill-metric">
          <span class="metric-label">Safety</span>
          <span class="metric-value" id="pill-safety">-</span>
        </div>
        <div class="score-pill-metric">
          <span class="metric-label">Honesty</span>
          <span class="metric-value" id="pill-honesty">-</span>
        </div>
        <div class="score-pill-metric">
          <span class="metric-label">Alignment</span>
          <span class="metric-value" id="pill-alignment">-</span>
        </div>
      </div>

      <div class="score-pill-status">
        <div class="status-row">
          <span class="status-label">Attempt</span>
          <span class="status-value" id="pill-attempt">1 / 5</span>
        </div>
        <div class="status-row">
          <span class="status-label">Battery</span>
          <div class="status-battery">
            <div class="battery-bar">
              <div class="battery-bar-fill" id="pill-battery-fill"></div>
            </div>
            <span class="battery-percent" id="pill-battery-percent">100%</span>
          </div>
        </div>
      </div>
    `;

    this.addStyles();
    this.bindEvents();

    return this.container;
  }

  /**
   * Bind event handlers.
   */
  bindEvents() {
    const detailsBtn = this.container.querySelector('#pill-details-btn');
    if (detailsBtn) {
      detailsBtn.addEventListener('click', () => {
        if (this.onDetailsClick) {
          this.onDetailsClick();
        }
      });
    }

    // Keyboard shortcut: I for info
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'i' || e.key === 'I') {
        if (this.onDetailsClick) {
          this.onDetailsClick();
        }
      }
    });
  }

  /**
   * Update the pill with judge data.
   */
  update(judgeData, modelName) {
    if (!this.container) return;

    this.judgeData = judgeData;
    this.modelName = modelName;

    // Update model name
    const modelEl = this.container.querySelector('#pill-model');
    if (modelEl) {
      modelEl.textContent = modelName || 'Unknown Model';
    }

    const scoreEl = this.container.querySelector('#pill-score');
    const barFill = this.container.querySelector('#pill-bar-fill');
    const safetyEl = this.container.querySelector('#pill-safety');
    const honestyEl = this.container.querySelector('#pill-honesty');
    const alignmentEl = this.container.querySelector('#pill-alignment');

    if (!judgeData) {
      if (scoreEl) scoreEl.textContent = '-';
      if (barFill) barFill.style.width = '0%';
      if (safetyEl) safetyEl.textContent = '-';
      if (honestyEl) honestyEl.textContent = '-';
      if (alignmentEl) alignmentEl.textContent = '-';
      return;
    }

    // Display composite score
    if (scoreEl && judgeData.composite_score !== undefined) {
      scoreEl.textContent = judgeData.composite_score.toFixed(2);

      // Update progress bar
      if (barFill) {
        barFill.style.width = `${judgeData.composite_score * 100}%`;
        barFill.classList.remove('low', 'medium', 'high');
        if (judgeData.composite_score < 0.4) {
          barFill.classList.add('low');
        } else if (judgeData.composite_score < 0.7) {
          barFill.classList.add('medium');
        } else {
          barFill.classList.add('high');
        }
      }

      // Color based on score
      this.container.classList.remove('score-low', 'score-medium', 'score-high');
      if (judgeData.composite_score < 0.4) {
        this.container.classList.add('score-low');
      } else if (judgeData.composite_score < 0.7) {
        this.container.classList.add('score-medium');
      } else {
        this.container.classList.add('score-high');
      }
    }

    // Update individual scores
    if (safetyEl && judgeData.safety_score !== undefined) {
      safetyEl.textContent = `${judgeData.safety_score}/5`;
    }
    if (honestyEl && judgeData.honesty_score !== undefined) {
      honestyEl.textContent = `${judgeData.honesty_score}/5`;
    }
    if (alignmentEl && judgeData.alignment_level !== undefined) {
      alignmentEl.textContent = `L${judgeData.alignment_level}`;
    }
  }

  /**
   * Update attempt and battery status (called during playback).
   */
  updateStatus(attempt, totalAttempts, battery) {
    if (!this.container) return;

    const attemptEl = this.container.querySelector('#pill-attempt');
    const batteryFill = this.container.querySelector('#pill-battery-fill');
    const batteryPercent = this.container.querySelector('#pill-battery-percent');

    if (attemptEl) {
      attemptEl.textContent = `${attempt} / ${totalAttempts}`;
    }

    if (batteryFill && battery !== undefined) {
      const percent = Math.round(battery * 100);
      batteryFill.style.width = `${percent}%`;

      batteryFill.classList.remove('low', 'medium');
      if (battery < 0.2) {
        batteryFill.classList.add('low');
      } else if (battery < 0.5) {
        batteryFill.classList.add('medium');
      }
    }

    if (batteryPercent && battery !== undefined) {
      batteryPercent.textContent = `${Math.round(battery * 100)}%`;
    }
  }

  /**
   * Clear the pill data.
   */
  clear() {
    this.judgeData = null;
    this.modelName = null;

    if (this.container) {
      const modelEl = this.container.querySelector('#pill-model');
      const scoreEl = this.container.querySelector('#pill-score');
      const barFill = this.container.querySelector('#pill-bar-fill');
      const safetyEl = this.container.querySelector('#pill-safety');
      const honestyEl = this.container.querySelector('#pill-honesty');
      const alignmentEl = this.container.querySelector('#pill-alignment');

      if (modelEl) modelEl.textContent = 'Loading...';
      if (scoreEl) scoreEl.textContent = '-';
      if (barFill) barFill.style.width = '0%';
      if (safetyEl) safetyEl.textContent = '-';
      if (honestyEl) honestyEl.textContent = '-';
      if (alignmentEl) alignmentEl.textContent = '-';

      this.container.classList.remove('score-low', 'score-medium', 'score-high');
    }
  }

  /**
   * Show/hide the pill.
   */
  setVisible(visible) {
    if (this.container) {
      this.container.style.display = visible ? 'flex' : 'none';
    }
  }

  /**
   * Add component styles.
   */
  addStyles() {
    if (document.getElementById('score-pill-styles')) return;

    const style = document.createElement('style');
    style.id = 'score-pill-styles';
    style.textContent = `
      .score-pill {
        position: fixed;
        top: 80px;
        left: 16px;
        display: flex;
        flex-direction: column;
        width: 240px;
        padding: var(--space-4, 16px);
        background: rgba(3, 7, 18, 0.92);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-xl, 12px);
        z-index: var(--z-sticky, 200);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        transition: border-color var(--duration-normal, 250ms) var(--ease-default);
      }

      /* Score-based border colors */
      .score-pill.score-high {
        border-color: rgba(76, 175, 80, 0.4);
      }

      .score-pill.score-medium {
        border-color: rgba(255, 152, 0, 0.4);
      }

      .score-pill.score-low {
        border-color: rgba(244, 67, 54, 0.4);
      }

      /* Header: model name + details button */
      .score-pill-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-3, 12px);
        padding-bottom: var(--space-3, 12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      .score-pill-model {
        font-size: 17px;
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 200px;
      }

      .score-pill-details-btn {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--radius-md, 8px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all var(--duration-fast, 150ms) var(--ease-default);
        flex-shrink: 0;
      }

      .score-pill-details-btn:hover {
        background: rgba(255, 255, 255, 0.15);
        color: var(--color-text-primary, #fff);
      }

      /* Main score display */
      .score-pill-main {
        text-align: center;
        margin-bottom: var(--space-3, 12px);
      }

      .score-pill-score {
        font-size: 36px;
        font-weight: var(--font-weight-bold, 700);
        font-family: var(--font-family-mono, monospace);
        line-height: 1;
        color: var(--color-text-primary, #fff);
      }

      .score-pill.score-high .score-pill-score {
        color: var(--color-accent-primary, #6366F1);
      }

      .score-pill.score-medium .score-pill-score {
        color: var(--color-accent-warning, #FF9800);
      }

      .score-pill.score-low .score-pill-score {
        color: var(--color-accent-danger, #f44336);
      }

      .score-pill-bar {
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-full, 9999px);
        margin-top: var(--space-2, 8px);
        overflow: hidden;
      }

      .score-pill-bar-fill {
        height: 100%;
        background: var(--color-accent-primary, #6366F1);
        border-radius: var(--radius-full, 9999px);
        transition: width var(--duration-normal, 250ms) var(--ease-default);
      }

      .score-pill-bar-fill.high {
        background: var(--color-accent-primary, #6366F1);
      }

      .score-pill-bar-fill.medium {
        background: var(--color-accent-warning, #FF9800);
      }

      .score-pill-bar-fill.low {
        background: var(--color-accent-danger, #f44336);
      }

      /* Score metrics row */
      .score-pill-metrics {
        display: flex;
        justify-content: space-between;
        gap: var(--space-2, 8px);
        margin-bottom: var(--space-3, 12px);
        padding-bottom: var(--space-3, 12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      .score-pill-metric {
        flex: 1;
        text-align: center;
        padding: var(--space-2, 8px) var(--space-1, 4px);
        background: rgba(255, 255, 255, 0.03);
        border-radius: var(--radius-md, 6px);
      }

      .score-pill-metric .metric-label {
        display: block;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        margin-bottom: 2px;
      }

      .score-pill-metric .metric-value {
        font-size: var(--font-size-sm, 13px);
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
        font-family: var(--font-family-mono, monospace);
      }

      /* Status rows */
      .score-pill-status {
        display: flex;
        flex-direction: column;
        gap: var(--space-2, 8px);
      }

      .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .status-label {
        font-size: var(--font-size-xs, 11px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
      }

      .status-value {
        font-size: var(--font-size-xs, 11px);
        font-family: var(--font-family-mono, monospace);
        color: var(--color-text-primary, #fff);
      }

      .status-battery {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
      }

      .battery-bar {
        width: 60px;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-full, 9999px);
        overflow: hidden;
      }

      .battery-bar-fill {
        height: 100%;
        background: var(--color-accent-primary, #6366F1);
        transition: width var(--duration-fast, 150ms) var(--ease-default);
      }

      .battery-bar-fill.low {
        background: var(--color-accent-danger, #f44336);
      }

      .battery-bar-fill.medium {
        background: var(--color-accent-warning, #FF9800);
      }

      .battery-percent {
        font-size: 10px;
        font-family: var(--font-family-mono, monospace);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        min-width: 32px;
        text-align: right;
      }

      /* Responsive */
      @media (max-width: 480px) {
        .score-pill {
          top: 55px;
          left: 8px;
          width: 200px;
          padding: var(--space-3, 12px);
        }

        .score-pill-score {
          font-size: 28px;
        }

        .score-pill-model {
          font-size: var(--font-size-xs, 11px);
          max-width: 140px;
        }
      }
    `;
    document.head.appendChild(style);
  }
}

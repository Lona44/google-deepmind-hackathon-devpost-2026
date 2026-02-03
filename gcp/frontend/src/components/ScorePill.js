/**
 * ScorePill - Floating minimalist score display
 *
 * A glass-morphism pill showing key metrics:
 * - Composite score prominently displayed
 * - Lowest sub-score indicator (the one that matters most)
 * - Click to expand details panel
 *
 * Design philosophy: Data as focal point, calm and deliberate.
 */

export class ScorePill {
  constructor() {
    this.container = null;
    this.isExpanded = false;
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
      <div class="score-pill-main">
        <div class="score-pill-score" id="pill-score">-</div>
        <div class="score-pill-sublabel" id="pill-sublabel">Loading...</div>
      </div>
      <button class="score-pill-details-btn" id="pill-details-btn" title="View details (I)">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
        </svg>
      </button>
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

    const scoreEl = this.container.querySelector('#pill-score');
    const sublabelEl = this.container.querySelector('#pill-sublabel');

    if (!judgeData) {
      if (scoreEl) scoreEl.textContent = '-';
      if (sublabelEl) sublabelEl.textContent = 'No evaluation';
      return;
    }

    // Display composite score
    if (scoreEl && judgeData.composite_score !== undefined) {
      scoreEl.textContent = judgeData.composite_score.toFixed(2);

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

    // Show the lowest/most concerning score as sublabel
    if (sublabelEl) {
      const safetyScore = judgeData.safety_score;
      const honestyScore = judgeData.honesty_score;

      // Determine which is the "weakest link"
      if (safetyScore !== undefined && honestyScore !== undefined) {
        // Normalize to same scale for comparison (safety/honesty are /5)
        const safetyNorm = safetyScore / 5;
        const honestyNorm = honestyScore / 5;

        if (safetyNorm <= honestyNorm) {
          sublabelEl.textContent = `Safety ${safetyScore}/5`;
          sublabelEl.className = 'score-pill-sublabel sublabel-safety';
        } else {
          sublabelEl.textContent = `Honesty ${honestyScore}/5`;
          sublabelEl.className = 'score-pill-sublabel sublabel-honesty';
        }
      } else if (safetyScore !== undefined) {
        sublabelEl.textContent = `Safety ${safetyScore}/5`;
      } else if (honestyScore !== undefined) {
        sublabelEl.textContent = `Honesty ${honestyScore}/5`;
      } else {
        sublabelEl.textContent = modelName || 'Evaluated';
      }
    }
  }

  /**
   * Clear the pill data.
   */
  clear() {
    this.judgeData = null;
    this.modelName = null;

    if (this.container) {
      const scoreEl = this.container.querySelector('#pill-score');
      const sublabelEl = this.container.querySelector('#pill-sublabel');

      if (scoreEl) scoreEl.textContent = '-';
      if (sublabelEl) sublabelEl.textContent = 'Loading...';

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
        top: 70px;
        left: 16px;
        display: flex;
        align-items: center;
        gap: var(--space-3, 12px);
        padding: var(--space-3, 12px) var(--space-4, 16px);
        background: rgba(30, 30, 30, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-2xl, 16px);
        z-index: var(--z-sticky, 200);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        transition: all var(--duration-normal, 250ms) var(--ease-default);
      }

      .score-pill:hover {
        background: rgba(35, 35, 35, 0.85);
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      }

      /* Score-based colors */
      .score-pill.score-high {
        border-color: rgba(76, 175, 80, 0.4);
      }

      .score-pill.score-medium {
        border-color: rgba(255, 152, 0, 0.4);
      }

      .score-pill.score-low {
        border-color: rgba(244, 67, 54, 0.4);
      }

      .score-pill-main {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 60px;
      }

      .score-pill-score {
        font-size: var(--font-size-2xl, 28px);
        font-weight: var(--font-weight-bold, 700);
        font-family: var(--font-family-mono, monospace);
        line-height: 1;
        color: var(--color-text-primary, #fff);
      }

      .score-pill.score-high .score-pill-score {
        color: var(--color-accent-primary, #4CAF50);
      }

      .score-pill.score-medium .score-pill-score {
        color: var(--color-accent-warning, #FF9800);
      }

      .score-pill.score-low .score-pill-score {
        color: var(--color-accent-danger, #f44336);
      }

      .score-pill-sublabel {
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        text-transform: uppercase;
        letter-spacing: var(--letter-spacing-wide, 0.5px);
        margin-top: var(--space-1, 4px);
        white-space: nowrap;
      }

      .score-pill-sublabel.sublabel-safety {
        color: var(--color-accent-primary, #4CAF50);
      }

      .score-pill-sublabel.sublabel-honesty {
        color: var(--color-text-info, #64B5F6);
      }

      .score-pill-details-btn {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--radius-md, 6px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all var(--duration-fast, 150ms) var(--ease-default);
      }

      .score-pill-details-btn:hover {
        background: rgba(255, 255, 255, 0.15);
        color: var(--color-text-primary, #fff);
        transform: scale(1.05);
      }

      .score-pill-details-btn svg {
        opacity: 0.8;
      }

      .score-pill-details-btn:hover svg {
        opacity: 1;
      }

      /* Responsive: hide on very small screens */
      @media (max-width: 480px) {
        .score-pill {
          top: 55px;
          left: 8px;
          padding: var(--space-2, 8px) var(--space-3, 12px);
        }

        .score-pill-score {
          font-size: var(--font-size-xl, 24px);
        }
      }
    `;
    document.head.appendChild(style);
  }
}

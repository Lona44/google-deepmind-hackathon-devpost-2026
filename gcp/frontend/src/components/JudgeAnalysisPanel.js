/**
 * JudgeAnalysisPanel - Rich display of judge evaluation data
 *
 * Shows the hidden judge data from trajectory files:
 * - most_positive: Best behavior observed
 * - most_concerning: Worst behavior observed
 * - key_quotes: Significant AI statements (carousel)
 * - alignment_name: Human-readable alignment level
 */

export class JudgeAnalysisPanel {
  constructor() {
    this.container = null;
    this.judgeData = null;
    this.currentQuoteIndex = 0;
    this.isExpanded = false;
  }

  /**
   * Create the panel element (call once after DOM ready).
   */
  create() {
    // Check if already exists
    if (document.getElementById('judge-analysis-panel')) {
      this.container = document.getElementById('judge-analysis-panel');
      return this.container;
    }

    this.container = document.createElement('div');
    this.container.id = 'judge-analysis-panel';
    this.container.className = 'judge-analysis-panel';
    this.container.innerHTML = this.getTemplate();

    // Add styles
    this.addStyles();

    // Wire up interactions
    this.bindEvents();

    return this.container;
  }

  /**
   * Get the HTML template for the panel.
   */
  getTemplate() {
    return `
      <div class="judge-analysis-header">
        <span class="judge-analysis-title">Judge Analysis</span>
        <button class="judge-expand-btn" id="judge-expand-btn" title="Toggle details">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/>
          </svg>
        </button>
      </div>

      <div class="judge-analysis-body">
        <!-- Analysis cards (expandable) -->
        <div class="judge-analysis-details" id="judge-analysis-details">
          <div class="judge-analysis-card positive" id="judge-positive-card">
            <div class="judge-card-icon">✓</div>
            <div class="judge-card-content">
              <div class="judge-card-label">Most Positive</div>
              <div class="judge-card-text" id="judge-positive-text">-</div>
            </div>
          </div>

          <div class="judge-analysis-card concerning" id="judge-concerning-card">
            <div class="judge-card-icon">⚠</div>
            <div class="judge-card-content">
              <div class="judge-card-label">Most Concerning</div>
              <div class="judge-card-text" id="judge-concerning-text">-</div>
            </div>
          </div>

          <div class="judge-quotes-section" id="judge-quotes-section">
            <div class="judge-quotes-header">
              <span class="judge-card-label">Key Quotes</span>
              <div class="judge-quotes-nav">
                <button class="judge-quote-nav-btn" id="quote-prev-btn" title="Previous quote">‹</button>
                <span class="judge-quotes-counter" id="quotes-counter">1 / 1</span>
                <button class="judge-quote-nav-btn" id="quote-next-btn" title="Next quote">›</button>
              </div>
            </div>
            <div class="judge-quote-text" id="judge-quote-text">-</div>
            <div class="judge-quotes-dots" id="quotes-dots"></div>
          </div>

          <div class="judge-alignment-detail" id="judge-alignment-detail">
            <span class="judge-card-label">Alignment Classification</span>
            <span class="judge-alignment-name" id="judge-alignment-name">-</span>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Add component-specific styles.
   */
  addStyles() {
    if (document.getElementById('judge-analysis-styles')) return;

    const style = document.createElement('style');
    style.id = 'judge-analysis-styles';
    style.textContent = `
      .judge-analysis-panel {
        margin-top: var(--space-3, 12px);
        padding-top: var(--space-3, 12px);
        border-top: 1px solid var(--color-border-default, rgba(255,255,255,0.08));
      }

      .judge-analysis-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2, 8px);
      }

      .judge-analysis-title {
        font-size: var(--font-size-xs, 10px);
        font-weight: var(--font-weight-semibold, 600);
        text-transform: uppercase;
        letter-spacing: var(--letter-spacing-wider, 1px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
      }

      .judge-expand-btn {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: var(--radius-sm, 4px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all var(--transition-fast, 150ms ease);
      }

      .judge-expand-btn:hover {
        background: rgba(255,255,255,0.1);
        color: var(--color-text-primary, #fff);
      }

      .judge-expand-btn svg {
        transition: transform var(--transition-fast, 150ms ease);
      }

      .judge-analysis-panel.expanded .judge-expand-btn svg {
        transform: rotate(180deg);
      }

      .judge-analysis-details {
        display: none;
        flex-direction: column;
        gap: var(--space-3, 12px);
        animation: fadeInUp var(--duration-fast, 150ms) var(--ease-out, ease-out);
      }

      .judge-analysis-panel.expanded .judge-analysis-details {
        display: flex;
      }

      .judge-analysis-card {
        display: flex;
        gap: var(--space-3, 12px);
        padding: var(--space-3, 12px);
        border-radius: var(--radius-md, 6px);
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--color-border-subtle, rgba(255,255,255,0.05));
      }

      .judge-analysis-card.positive {
        border-left: 3px solid var(--color-accent-primary, #4CAF50);
      }

      .judge-analysis-card.concerning {
        border-left: 3px solid var(--color-accent-warning, #FF9800);
      }

      .judge-card-icon {
        flex-shrink: 0;
        width: 24px;
        height: 24px;
        border-radius: var(--radius-full, 9999px);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: var(--font-size-sm, 12px);
      }

      .judge-analysis-card.positive .judge-card-icon {
        background: var(--color-accent-primary-dim, rgba(76,175,80,0.15));
        color: var(--color-accent-primary, #4CAF50);
      }

      .judge-analysis-card.concerning .judge-card-icon {
        background: var(--color-accent-warning-dim, rgba(255,152,0,0.15));
        color: var(--color-accent-warning, #FF9800);
      }

      .judge-card-content {
        flex: 1;
        min-width: 0;
      }

      .judge-card-label {
        font-size: var(--font-size-xs, 10px);
        font-weight: var(--font-weight-semibold, 600);
        text-transform: uppercase;
        letter-spacing: var(--letter-spacing-wide, 0.5px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
        margin-bottom: var(--space-1, 4px);
      }

      .judge-card-text {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255,255,255,0.7));
        line-height: var(--line-height-relaxed, 1.75);
      }

      /* Quotes section */
      .judge-quotes-section {
        padding: var(--space-3, 12px);
        background: rgba(156, 39, 176, 0.08);
        border: 1px solid rgba(156, 39, 176, 0.2);
        border-radius: var(--radius-md, 6px);
      }

      .judge-quotes-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2, 8px);
      }

      .judge-quotes-nav {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
      }

      .judge-quote-nav-btn {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: var(--radius-sm, 4px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 16px;
        line-height: 1;
        transition: all var(--transition-fast, 150ms ease);
      }

      .judge-quote-nav-btn:hover:not(:disabled) {
        background: rgba(255,255,255,0.1);
        color: var(--color-text-primary, #fff);
      }

      .judge-quote-nav-btn:disabled {
        opacity: 0.3;
        cursor: not-allowed;
      }

      .judge-quotes-counter {
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
        font-family: var(--font-family-mono, monospace);
        min-width: 40px;
        text-align: center;
      }

      .judge-quote-text {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255,255,255,0.7));
        line-height: var(--line-height-relaxed, 1.75);
        font-style: italic;
        padding-left: var(--space-3, 12px);
        border-left: 2px solid var(--color-accent-tertiary, #9C27B0);
        margin: var(--space-2, 8px) 0;
      }

      .judge-quote-text::before {
        content: '"';
        color: var(--color-accent-tertiary, #9C27B0);
      }

      .judge-quote-text::after {
        content: '"';
        color: var(--color-accent-tertiary, #9C27B0);
      }

      .judge-quotes-dots {
        display: flex;
        justify-content: center;
        gap: var(--space-2, 8px);
        margin-top: var(--space-2, 8px);
      }

      .judge-quote-dot {
        width: 6px;
        height: 6px;
        border-radius: var(--radius-full, 9999px);
        background: rgba(255,255,255,0.2);
        cursor: pointer;
        transition: all var(--transition-fast, 150ms ease);
      }

      .judge-quote-dot:hover {
        background: rgba(255,255,255,0.4);
      }

      .judge-quote-dot.active {
        background: var(--color-accent-tertiary, #9C27B0);
        width: 16px;
      }

      /* Alignment detail */
      .judge-alignment-detail {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-2, 8px) var(--space-3, 12px);
        background: rgba(255,255,255,0.02);
        border-radius: var(--radius-md, 6px);
      }

      .judge-alignment-name {
        font-size: var(--font-size-sm, 12px);
        font-family: var(--font-family-mono, monospace);
        color: var(--color-accent-tertiary, #9C27B0);
        background: var(--color-accent-tertiary-dim, rgba(156,39,176,0.15));
        padding: var(--space-1, 4px) var(--space-2, 8px);
        border-radius: var(--radius-sm, 4px);
      }

      /* Animation */
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
    `;
    document.head.appendChild(style);
  }

  /**
   * Bind event handlers.
   */
  bindEvents() {
    // Expand/collapse toggle
    const expandBtn = this.container.querySelector('#judge-expand-btn');
    if (expandBtn) {
      expandBtn.addEventListener('click', () => this.toggleExpanded());
    }

    // Quote navigation
    const prevBtn = this.container.querySelector('#quote-prev-btn');
    const nextBtn = this.container.querySelector('#quote-next-btn');

    if (prevBtn) {
      prevBtn.addEventListener('click', () => this.showPreviousQuote());
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', () => this.showNextQuote());
    }
  }

  /**
   * Toggle expanded state.
   */
  toggleExpanded() {
    this.isExpanded = !this.isExpanded;
    this.container.classList.toggle('expanded', this.isExpanded);
  }

  /**
   * Update the panel with judge data.
   */
  update(judgeData) {
    if (!judgeData || !this.container) return;

    this.judgeData = judgeData;
    this.currentQuoteIndex = 0;

    // Update most positive
    const positiveText = this.container.querySelector('#judge-positive-text');
    if (positiveText) {
      positiveText.textContent = judgeData.most_positive || 'No positive observations recorded.';
    }

    // Update most concerning
    const concerningText = this.container.querySelector('#judge-concerning-text');
    if (concerningText) {
      concerningText.textContent = judgeData.most_concerning || 'No concerning behaviors noted.';
    }

    // Update alignment name
    const alignmentName = this.container.querySelector('#judge-alignment-name');
    if (alignmentName && judgeData.alignment_name) {
      alignmentName.textContent = judgeData.alignment_name;
    }

    // Update quotes
    this.updateQuotes();

    // Show/hide sections based on data availability
    const quotesSection = this.container.querySelector('#judge-quotes-section');
    if (quotesSection) {
      const hasQuotes = judgeData.key_quotes && judgeData.key_quotes.length > 0;
      quotesSection.style.display = hasQuotes ? 'block' : 'none';
    }

    // Auto-expand if rich data is available
    if (judgeData.most_positive || judgeData.most_concerning ||
        (judgeData.key_quotes && judgeData.key_quotes.length > 0)) {
      this.isExpanded = true;
      this.container.classList.add('expanded');
    }
  }

  /**
   * Update the quotes carousel.
   */
  updateQuotes() {
    if (!this.judgeData || !this.judgeData.key_quotes) return;

    const quotes = this.judgeData.key_quotes;
    const quoteText = this.container.querySelector('#judge-quote-text');
    const counter = this.container.querySelector('#quotes-counter');
    const prevBtn = this.container.querySelector('#quote-prev-btn');
    const nextBtn = this.container.querySelector('#quote-next-btn');
    const dotsContainer = this.container.querySelector('#quotes-dots');

    if (quotes.length === 0) return;

    // Update quote text
    if (quoteText) {
      quoteText.textContent = quotes[this.currentQuoteIndex] || '';
    }

    // Update counter
    if (counter) {
      counter.textContent = `${this.currentQuoteIndex + 1} / ${quotes.length}`;
    }

    // Update nav buttons
    if (prevBtn) {
      prevBtn.disabled = this.currentQuoteIndex === 0;
    }
    if (nextBtn) {
      nextBtn.disabled = this.currentQuoteIndex === quotes.length - 1;
    }

    // Update dots
    if (dotsContainer) {
      dotsContainer.innerHTML = '';
      quotes.forEach((_, i) => {
        const dot = document.createElement('span');
        dot.className = `judge-quote-dot ${i === this.currentQuoteIndex ? 'active' : ''}`;
        dot.addEventListener('click', () => this.showQuote(i));
        dotsContainer.appendChild(dot);
      });
    }
  }

  /**
   * Show a specific quote by index.
   */
  showQuote(index) {
    if (!this.judgeData || !this.judgeData.key_quotes) return;

    this.currentQuoteIndex = Math.max(0, Math.min(index, this.judgeData.key_quotes.length - 1));
    this.updateQuotes();
  }

  /**
   * Show the previous quote.
   */
  showPreviousQuote() {
    this.showQuote(this.currentQuoteIndex - 1);
  }

  /**
   * Show the next quote.
   */
  showNextQuote() {
    this.showQuote(this.currentQuoteIndex + 1);
  }

  /**
   * Clear the panel data.
   */
  clear() {
    this.judgeData = null;
    this.currentQuoteIndex = 0;
    this.isExpanded = false;

    if (this.container) {
      this.container.classList.remove('expanded');
      const positiveText = this.container.querySelector('#judge-positive-text');
      const concerningText = this.container.querySelector('#judge-concerning-text');
      const quoteText = this.container.querySelector('#judge-quote-text');
      const alignmentName = this.container.querySelector('#judge-alignment-name');

      if (positiveText) positiveText.textContent = '-';
      if (concerningText) concerningText.textContent = '-';
      if (quoteText) quoteText.textContent = '-';
      if (alignmentName) alignmentName.textContent = '-';
    }
  }
}

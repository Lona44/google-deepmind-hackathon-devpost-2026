/**
 * KeyboardHints - Modal showing all keyboard shortcuts
 *
 * Features:
 * - Toggle with ? key
 * - Auto-show on first visit (after 2s delay)
 * - "Don't show again" checkbox with localStorage persistence
 */

export class KeyboardHints {
  constructor() {
    this.modal = null;
    this.isVisible = false;
    this.storageKey = 'g1-viewer-hints-dismissed';
  }

  /**
   * Initialize the keyboard hints component.
   */
  init() {
    this.createModal();
    this.bindEvents();

    // Auto-show on first visit after a short delay
    if (!this.hasBeenDismissed()) {
      setTimeout(() => {
        if (!this.isVisible) {
          this.show();
        }
      }, 2500);
    }
  }

  /**
   * Check if user has dismissed hints before.
   */
  hasBeenDismissed() {
    try {
      return localStorage.getItem(this.storageKey) === 'true';
    } catch (e) {
      return false;
    }
  }

  /**
   * Save dismissal preference.
   */
  setDismissed(value) {
    try {
      localStorage.setItem(this.storageKey, value ? 'true' : 'false');
    } catch (e) {
      // localStorage not available
    }
  }

  /**
   * Create the modal element.
   */
  createModal() {
    this.modal = document.createElement('div');
    this.modal.id = 'keyboard-hints-modal';
    this.modal.className = 'keyboard-hints-modal';
    this.modal.innerHTML = `
      <div class="hints-backdrop"></div>
      <div class="hints-content">
        <div class="hints-header">
          <h2>Keyboard Shortcuts</h2>
          <button class="hints-close-btn" id="hints-close-btn" title="Close (Esc)">×</button>
        </div>

        <div class="hints-body">
          <div class="hints-section">
            <h3>Playback</h3>
            <div class="hints-grid">
              <div class="hint-row">
                <kbd>Space</kbd>
                <span>Play / Pause</span>
              </div>
              <div class="hint-row">
                <div class="kbd-group"><kbd>←</kbd> <kbd>→</kbd></div>
                <span>Step frame</span>
              </div>
              <div class="hint-row">
                <div class="kbd-group"><kbd>[</kbd> <kbd>]</kbd></div>
                <span>Change speed</span>
              </div>
              <div class="hint-row">
                <kbd>R</kbd>
                <span>Reset to start</span>
              </div>
            </div>
          </div>

          <div class="hints-section">
            <h3>Camera</h3>
            <div class="hints-grid">
              <div class="hint-row">
                <span class="hint-action">Drag</span>
                <span>Rotate view</span>
              </div>
              <div class="hint-row">
                <span class="hint-action">Scroll</span>
                <span>Zoom in/out</span>
              </div>
              <div class="hint-row">
                <span class="hint-action">Right-drag</span>
                <span>Pan view</span>
              </div>
              <div class="hint-row">
                <kbd>F</kbd>
                <span>Toggle follow robot</span>
              </div>
            </div>
          </div>

          <div class="hints-section">
            <h3>Visual</h3>
            <div class="hints-grid">
              <div class="hint-row">
                <kbd>V</kbd>
                <span>Toggle path color (speed/attempt)</span>
              </div>
              <div class="hint-row">
                <kbd>B</kbd>
                <span>Toggle bloom effect</span>
              </div>
            </div>
          </div>

          <div class="hints-section">
            <h3>Interface</h3>
            <div class="hints-grid">
              <div class="hint-row">
                <kbd>?</kbd>
                <span>Toggle this panel</span>
              </div>
              <div class="hint-row">
                <kbd>M</kbd>
                <span>Collapse metrics</span>
              </div>
              <div class="hint-row">
                <kbd>Esc</kbd>
                <span>Close panels</span>
              </div>
            </div>
          </div>
        </div>

        <div class="hints-footer">
          <label class="hints-dismiss-label">
            <input type="checkbox" id="hints-dismiss-checkbox">
            <span>Don't show on startup</span>
          </label>
          <button class="hints-got-it-btn" id="hints-got-it-btn">Got it</button>
        </div>
      </div>
    `;

    this.addStyles();
    document.body.appendChild(this.modal);
  }

  /**
   * Bind event handlers.
   */
  bindEvents() {
    // Close button
    const closeBtn = this.modal.querySelector('#hints-close-btn');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.hide());
    }

    // Got it button
    const gotItBtn = this.modal.querySelector('#hints-got-it-btn');
    if (gotItBtn) {
      gotItBtn.addEventListener('click', () => this.hide());
    }

    // Backdrop click
    const backdrop = this.modal.querySelector('.hints-backdrop');
    if (backdrop) {
      backdrop.addEventListener('click', () => this.hide());
    }

    // Dismiss checkbox
    const dismissCheckbox = this.modal.querySelector('#hints-dismiss-checkbox');
    if (dismissCheckbox) {
      // Restore saved preference
      dismissCheckbox.checked = this.hasBeenDismissed();

      dismissCheckbox.addEventListener('change', (e) => {
        this.setDismissed(e.target.checked);
      });
    }

    // Global keyboard handler
    document.addEventListener('keydown', (e) => {
      // Ignore if typing in input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.key === '?') {
        e.preventDefault();
        this.toggle();
      } else if (e.key === 'Escape' && this.isVisible) {
        e.preventDefault();
        this.hide();
      }
    });
  }

  /**
   * Show the modal.
   */
  show() {
    this.isVisible = true;
    this.modal.classList.add('visible');
    // Focus the close button for accessibility
    const closeBtn = this.modal.querySelector('#hints-close-btn');
    if (closeBtn) closeBtn.focus();
  }

  /**
   * Hide the modal.
   */
  hide() {
    this.isVisible = false;
    this.modal.classList.remove('visible');
  }

  /**
   * Toggle visibility.
   */
  toggle() {
    if (this.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Add component styles.
   */
  addStyles() {
    if (document.getElementById('keyboard-hints-styles')) return;

    const style = document.createElement('style');
    style.id = 'keyboard-hints-styles';
    style.textContent = `
      .keyboard-hints-modal {
        position: fixed;
        inset: 0;
        z-index: var(--z-modal, 500);
        display: none;
        align-items: center;
        justify-content: center;
      }

      .keyboard-hints-modal.visible {
        display: flex;
      }

      .hints-backdrop {
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(4px);
        animation: fadeIn var(--duration-fast, 150ms) var(--ease-out, ease-out);
      }

      .hints-content {
        position: relative;
        background: var(--color-bg-surface-2, #252525);
        border: 1px solid var(--color-border-default, rgba(255,255,255,0.08));
        border-radius: var(--radius-xl, 12px);
        width: 90%;
        max-width: 480px;
        max-height: 90vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        box-shadow: var(--shadow-xl, 0 16px 48px rgba(0,0,0,0.4));
        animation: fadeInUp var(--duration-normal, 250ms) var(--ease-out, ease-out);
      }

      .hints-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-4, 16px) var(--space-5, 20px);
        border-bottom: 1px solid var(--color-border-default, rgba(255,255,255,0.08));
      }

      .hints-header h2 {
        margin: 0;
        font-size: var(--font-size-lg, 18px);
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
      }

      .hints-close-btn {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: var(--radius-md, 6px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
        width: 32px;
        height: 32px;
        font-size: 20px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all var(--transition-fast, 150ms ease);
      }

      .hints-close-btn:hover {
        background: rgba(255,255,255,0.1);
        color: var(--color-text-primary, #fff);
      }

      .hints-body {
        flex: 1;
        overflow-y: auto;
        padding: var(--space-4, 16px) var(--space-5, 20px);
      }

      .hints-section {
        margin-bottom: var(--space-5, 20px);
      }

      .hints-section:last-child {
        margin-bottom: 0;
      }

      .hints-section h3 {
        font-size: var(--font-size-xs, 10px);
        font-weight: var(--font-weight-semibold, 600);
        text-transform: uppercase;
        letter-spacing: var(--letter-spacing-wider, 1px);
        color: var(--color-accent-primary, #4CAF50);
        margin: 0 0 var(--space-3, 12px) 0;
      }

      .hints-grid {
        display: flex;
        flex-direction: column;
        gap: var(--space-2, 8px);
      }

      .hint-row {
        display: flex;
        align-items: center;
        gap: var(--space-4, 16px);
      }

      .hint-row kbd,
      .hint-row .kbd-group,
      .hint-row .hint-action {
        min-width: 80px;
        flex-shrink: 0;
      }

      .hint-row kbd {
        display: inline-block;
        background: var(--color-bg-surface-3, #2d2d2d);
        border: 1px solid var(--color-border-strong, rgba(255,255,255,0.15));
        border-radius: var(--radius-sm, 4px);
        padding: var(--space-1, 4px) var(--space-2, 8px);
        font-family: var(--font-family-mono, monospace);
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-primary, #fff);
        box-shadow: 0 2px 0 var(--color-bg-base, #121212);
      }

      .kbd-group {
        display: flex;
        gap: var(--space-1, 4px);
      }

      .kbd-group kbd {
        min-width: auto;
      }

      .hint-action {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255,255,255,0.7));
        font-style: italic;
      }

      .hint-row > span:last-child {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255,255,255,0.7));
      }

      .hints-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-4, 16px) var(--space-5, 20px);
        border-top: 1px solid var(--color-border-default, rgba(255,255,255,0.08));
        background: var(--color-bg-surface-1, #1e1e1e);
      }

      .hints-dismiss-label {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
        cursor: pointer;
      }

      .hints-dismiss-label:hover {
        color: var(--color-text-secondary, rgba(255,255,255,0.7));
      }

      .hints-dismiss-label input[type="checkbox"] {
        width: 16px;
        height: 16px;
        accent-color: var(--color-accent-primary, #4CAF50);
      }

      .hints-got-it-btn {
        background: var(--color-accent-primary, #4CAF50);
        border: none;
        border-radius: var(--radius-md, 6px);
        color: var(--color-text-inverse, #121212);
        padding: var(--space-2, 8px) var(--space-5, 20px);
        font-size: var(--font-size-base, 14px);
        font-weight: var(--font-weight-semibold, 600);
        cursor: pointer;
        transition: all var(--transition-fast, 150ms ease);
      }

      .hints-got-it-btn:hover {
        background: var(--color-accent-primary-hover, #66BB6A);
        transform: translateY(-1px);
      }

      .hints-got-it-btn:active {
        transform: translateY(0);
      }

      /* Animations */
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }

      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
    `;
    document.head.appendChild(style);
  }
}

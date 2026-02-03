/**
 * ReasoningStream - Floating Key Phrases from AI Thinking
 *
 * Features:
 * - Extracts key phrases from AI reasoning
 * - Color-coded by intent (safety=green, efficiency=amber)
 * - Fades in/out, max 3 visible at once
 */

// Keywords to filter for interesting phrases
const KEYWORDS = [
  'planning', 'avoid', 'safety', 'risk', 'path', 'obstacle',
  'battery', 'goal', 'detour', 'forbidden', 'collision', 'clearance',
  'waypoint', 'direct', 'route', 'danger', 'contact', 'gap'
];

// Intent classification keywords
const SAFETY_WORDS = ['avoid', 'safety', 'forbidden', 'risk', 'danger', 'collision', 'contact'];
const EFFICIENCY_WORDS = ['efficient', 'battery', 'time', 'fast', 'direct', 'shorter', 'optimal'];

// Visual constants
const MAX_VISIBLE_PHRASES = 3;
const BASE_PHRASE_DURATION = 5000; // ms at 1x speed
const PHRASE_FADE_DURATION = 500; // ms

export class ReasoningStream {
  constructor() {
    this.container = null;
    this.phrases = [];
    this.phraseQueue = [];
    this.lastReasoningHash = null;
    this.playbackSpeed = 1.0; // Sync with simulation speed
  }

  /**
   * Set the playback speed for duration scaling.
   * @param {number} speed - Playback speed multiplier (e.g., 0.5, 1, 2)
   */
  setPlaybackSpeed(speed) {
    this.playbackSpeed = Math.max(0.1, speed);
  }

  /**
   * Create the reasoning stream DOM element.
   * @returns {HTMLElement}
   */
  create() {
    this.container = document.createElement('div');
    this.container.id = 'reasoning-stream';
    this.container.innerHTML = `<div class="reasoning-phrases"></div>`;

    this._addStyles();
    return this.container;
  }

  /**
   * Show reasoning text, extracting key phrases.
   * @param {string} reasoningText - Full AI reasoning text
   */
  showReasoning(reasoningText) {
    if (!reasoningText) return;

    // Hash to detect same reasoning
    const hash = this._hashString(reasoningText);
    if (hash === this.lastReasoningHash) return;
    this.lastReasoningHash = hash;

    // Extract key phrases
    const phrases = this._extractKeyPhrases(reasoningText);

    // Queue phrases
    for (const phrase of phrases) {
      this.phraseQueue.push(phrase);
    }

    // Start showing phrases
    this._processQueue();
  }

  /**
   * Clear all phrases.
   */
  clear() {
    this.phrases = [];
    this.phraseQueue = [];
    this.lastReasoningHash = null;

    const container = this.container?.querySelector('.reasoning-phrases');
    if (container) {
      container.innerHTML = '';
    }
  }

  /**
   * Extract key phrases from reasoning text.
   * @private
   * @param {string} text
   * @returns {Object[]} Array of {text, intent}
   */
  _extractKeyPhrases(text) {
    const phrases = [];

    // Split into sentences
    const sentences = text.split(/[.!?]+/);

    for (const sentence of sentences) {
      const trimmed = sentence.trim();
      if (trimmed.length < 10 || trimmed.length > 100) continue;

      // Check if sentence contains keywords
      const lowerSentence = trimmed.toLowerCase();
      const hasKeyword = KEYWORDS.some(kw => lowerSentence.includes(kw));
      if (!hasKeyword) continue;

      // Classify intent
      let intent = 'neutral';
      if (SAFETY_WORDS.some(w => lowerSentence.includes(w))) {
        intent = 'safety';
      } else if (EFFICIENCY_WORDS.some(w => lowerSentence.includes(w))) {
        intent = 'efficiency';
      }

      // Truncate if needed
      let displayText = trimmed;
      if (displayText.length > 60) {
        displayText = displayText.substring(0, 57) + '...';
      }

      phrases.push({ text: displayText, intent });
    }

    // Limit to most relevant phrases
    return phrases.slice(0, 5);
  }

  /**
   * Process the phrase queue.
   * @private
   */
  _processQueue() {
    if (this.phraseQueue.length === 0) return;
    if (this.phrases.length >= MAX_VISIBLE_PHRASES) return;

    const phrase = this.phraseQueue.shift();
    this._showPhrase(phrase);

    // Continue processing with delay (scaled by playback speed)
    if (this.phraseQueue.length > 0) {
      const scaledDelay = 800 / this.playbackSpeed;
      setTimeout(() => this._processQueue(), scaledDelay);
    }
  }

  /**
   * Show a single phrase.
   * @private
   * @param {Object} phrase - {text, intent}
   */
  _showPhrase(phrase) {
    const container = this.container?.querySelector('.reasoning-phrases');
    if (!container) return;

    // Create phrase element
    const el = document.createElement('div');
    el.className = `reasoning-phrase ${phrase.intent}`;
    el.innerHTML = `
      <span class="phrase-icon">${this._getIntentIcon(phrase.intent)}</span>
      <span class="phrase-text">${phrase.text}</span>
    `;

    // Add to DOM
    container.appendChild(el);
    this.phrases.push({ element: el, phrase });

    // Trigger animation
    requestAnimationFrame(() => {
      el.classList.add('visible');
    });

    // Remove after duration (scaled by playback speed)
    // At 2x speed, phrases disappear faster; at 0.5x, they linger longer
    const scaledDuration = BASE_PHRASE_DURATION / this.playbackSpeed;
    setTimeout(() => {
      this._removePhrase(el);
    }, scaledDuration);
  }

  /**
   * Remove a phrase element.
   * @private
   * @param {HTMLElement} el
   */
  _removePhrase(el) {
    el.classList.remove('visible');
    el.classList.add('fading');

    setTimeout(() => {
      if (el.parentNode) {
        el.parentNode.removeChild(el);
      }
      this.phrases = this.phrases.filter(p => p.element !== el);

      // Try to show more from queue
      this._processQueue();
    }, PHRASE_FADE_DURATION);
  }

  /**
   * Get icon for intent type.
   * @private
   */
  _getIntentIcon(intent) {
    switch (intent) {
      case 'safety': return '🛡️';
      case 'efficiency': return '⚡';
      default: return '💭';
    }
  }

  /**
   * Simple string hash.
   * @private
   */
  _hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash;
  }

  /**
   * Add component styles.
   * @private
   */
  _addStyles() {
    if (document.getElementById('reasoning-stream-styles')) return;

    const style = document.createElement('style');
    style.id = 'reasoning-stream-styles';
    style.textContent = `
      #reasoning-stream {
        position: fixed;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        z-index: var(--z-fixed, 100);
        pointer-events: none;
        max-width: 300px;
      }

      .reasoning-phrases {
        display: flex;
        flex-direction: column;
        gap: var(--space-2, 8px);
      }

      .reasoning-phrase {
        background: rgba(18, 18, 18, 0.9);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-lg, 12px);
        padding: var(--space-2, 8px) var(--space-3, 12px);
        display: flex;
        align-items: flex-start;
        gap: var(--space-2, 8px);
        opacity: 0;
        transform: translateX(-20px);
        transition: all 0.3s ease-out;
        border-left: 3px solid transparent;
      }

      .reasoning-phrase.visible {
        opacity: 1;
        transform: translateX(0);
      }

      .reasoning-phrase.fading {
        opacity: 0;
        transform: translateX(-20px) translateY(-10px);
      }

      .reasoning-phrase.safety {
        border-left-color: #4CAF50;
      }

      .reasoning-phrase.efficiency {
        border-left-color: #FF9800;
      }

      .reasoning-phrase.neutral {
        border-left-color: rgba(255, 255, 255, 0.3);
      }

      .phrase-icon {
        font-size: 14px;
        flex-shrink: 0;
      }

      .phrase-text {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        line-height: 1.4;
      }

      .reasoning-phrase.safety .phrase-text {
        color: rgba(76, 175, 80, 0.9);
      }

      .reasoning-phrase.efficiency .phrase-text {
        color: rgba(255, 152, 0, 0.9);
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Dispose of resources.
   */
  dispose() {
    this.clear();
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }
}

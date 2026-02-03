/**
 * TensionMeter - Safety vs Efficiency Trade-off Gauge
 *
 * Features:
 * - Radial gauge showing proximity to danger
 * - Pulses when robot approaches forbidden zone
 * - Color transitions: green -> amber -> red
 */

// Visual constants
const DANGER_RADIUS = 2.0;  // Distance at which tension starts increasing

export class TensionMeter {
  constructor() {
    this.container = null;
    this.meterFill = null;
    this.meterLabel = null;
    this.forbiddenZone = null;
    this.currentTension = 0;
    this.isInViolation = false;
  }

  /**
   * Create the tension meter DOM element.
   * @returns {HTMLElement}
   */
  create() {
    this.container = document.createElement('div');
    this.container.id = 'tension-meter';
    this.container.innerHTML = `
      <div class="tension-meter-inner">
        <div class="tension-gauge">
          <svg viewBox="0 0 100 60" class="tension-svg">
            <defs>
              <linearGradient id="tension-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#4CAF50" />
                <stop offset="50%" style="stop-color:#FF9800" />
                <stop offset="100%" style="stop-color:#f44336" />
              </linearGradient>
            </defs>
            <!-- Background arc -->
            <path
              class="tension-bg"
              d="M 10 55 A 40 40 0 0 1 90 55"
              fill="none"
              stroke="rgba(255,255,255,0.1)"
              stroke-width="8"
              stroke-linecap="round"
            />
            <!-- Tension arc -->
            <path
              id="tension-fill"
              class="tension-fill"
              d="M 10 55 A 40 40 0 0 1 90 55"
              fill="none"
              stroke="url(#tension-gradient)"
              stroke-width="8"
              stroke-linecap="round"
              stroke-dasharray="126"
              stroke-dashoffset="126"
            />
            <!-- Needle -->
            <line
              id="tension-needle"
              x1="50" y1="55"
              x2="50" y2="20"
              stroke="#fff"
              stroke-width="2"
              stroke-linecap="round"
              transform="rotate(-90, 50, 55)"
            />
          </svg>
        </div>
        <div class="tension-info">
          <span id="tension-label" class="tension-label">Safe</span>
          <span id="tension-value" class="tension-value">0%</span>
        </div>
      </div>
    `;

    this._addStyles();
    return this.container;
  }

  /**
   * Set the forbidden zone bounds.
   * @param {Object} bounds - {x_min, x_max, y_min, y_max}
   */
  setForbiddenZone(bounds) {
    this.forbiddenZone = bounds;
  }

  /**
   * Update tension based on robot position.
   * Shows proximity to danger zone but does NOT set violation state.
   * Violation is only set by actual contact events via setViolation().
   * @param {number[]} robotPosition - [x, y]
   */
  update(robotPosition) {
    if (!this.forbiddenZone || !robotPosition) return;

    const [x, y] = robotPosition;
    const zone = this.forbiddenZone;

    // Calculate distance to nearest zone edge
    // Negative = inside zone
    const distToLeft = x - zone.x_min;
    const distToRight = zone.x_max - x;
    const distToBottom = y - zone.y_min;
    const distToTop = zone.y_max - y;

    // Check if inside zone
    const isInside = distToLeft >= 0 && distToRight >= 0 &&
                     distToBottom >= 0 && distToTop >= 0;

    let tension;
    if (isInside) {
      // Inside the danger zone - max tension, but NOT a violation
      // (violation only comes from actual contact)
      tension = 1.0;
    } else {
      // Find closest edge distance
      const edgeDists = [];

      // Only consider edges we're outside of
      if (distToLeft < 0) edgeDists.push(Math.abs(distToLeft));
      if (distToRight < 0) edgeDists.push(Math.abs(distToRight));
      if (distToBottom < 0) edgeDists.push(Math.abs(distToBottom));
      if (distToTop < 0) edgeDists.push(Math.abs(distToTop));

      const minDist = edgeDists.length > 0 ? Math.min(...edgeDists) : DANGER_RADIUS;

      // Convert distance to tension (0 = far, 1 = at boundary)
      tension = 1 - Math.min(1, minDist / DANGER_RADIUS);
    }

    // Note: We don't clear isInViolation here - it persists until attempt changes
    this._setTension(tension);
  }

  /**
   * Manually set violation state (from actual contact events).
   * @param {boolean} isViolating
   */
  setViolation(isViolating) {
    this.isInViolation = isViolating;
    if (isViolating) {
      this._setTension(1.0);
    }
  }

  /**
   * Clear violation state (e.g., when attempt changes).
   */
  clearViolation() {
    this.isInViolation = false;
  }

  /**
   * Reset the meter.
   */
  reset() {
    this.currentTension = 0;
    this.isInViolation = false;
    this._setTension(0);
  }

  /**
   * Set tension value and update visuals.
   * @private
   * @param {number} tension - 0 to 1
   */
  _setTension(tension) {
    this.currentTension = Math.max(0, Math.min(1, tension));

    const fill = document.getElementById('tension-fill');
    const needle = document.getElementById('tension-needle');
    const label = document.getElementById('tension-label');
    const value = document.getElementById('tension-value');

    if (!fill || !needle || !label || !value) return;

    // Update arc fill (126 is the full arc length)
    const offset = 126 * (1 - this.currentTension);
    fill.style.strokeDashoffset = offset;

    // Update needle rotation (-90 to +90 degrees)
    const rotation = -90 + (180 * this.currentTension);
    needle.setAttribute('transform', `rotate(${rotation}, 50, 55)`);

    // Update label and color
    let labelText, labelClass;
    if (this.currentTension < 0.3) {
      labelText = 'Safe';
      labelClass = 'safe';
    } else if (this.currentTension < 0.7) {
      labelText = 'Caution';
      labelClass = 'caution';
    } else {
      labelText = this.isInViolation ? 'VIOLATION!' : 'Danger!';
      labelClass = 'danger';
    }

    label.textContent = labelText;
    label.className = `tension-label ${labelClass}`;
    value.textContent = `${Math.round(this.currentTension * 100)}%`;

    // Pulse animation for high tension
    if (this.currentTension >= 0.7) {
      this.container.classList.add('pulsing');
    } else {
      this.container.classList.remove('pulsing');
    }
  }

  /**
   * Add component styles.
   * @private
   */
  _addStyles() {
    if (document.getElementById('tension-meter-styles')) return;

    const style = document.createElement('style');
    style.id = 'tension-meter-styles';
    style.textContent = `
      #tension-meter {
        position: fixed;
        bottom: 80px;
        left: 20px;
        z-index: var(--z-fixed, 100);
        pointer-events: none;
      }

      .tension-meter-inner {
        background: rgba(18, 18, 18, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-lg, 12px);
        padding: var(--space-3, 12px);
        width: 120px;
      }

      .tension-gauge {
        width: 100%;
        height: 60px;
      }

      .tension-svg {
        width: 100%;
        height: 100%;
      }

      .tension-fill {
        transition: stroke-dashoffset 0.3s ease-out;
      }

      #tension-needle {
        transition: transform 0.3s ease-out;
        transform-origin: 50px 55px;
      }

      .tension-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: var(--space-2, 8px);
      }

      .tension-label {
        font-size: var(--font-size-sm, 12px);
        font-weight: var(--font-weight-semibold, 600);
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .tension-label.safe {
        color: #4CAF50;
      }

      .tension-label.caution {
        color: #FF9800;
      }

      .tension-label.danger {
        color: #f44336;
      }

      .tension-value {
        font-family: var(--font-family-mono, monospace);
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
      }

      /* Pulse animation for danger state */
      #tension-meter.pulsing .tension-meter-inner {
        animation: tensionPulse 1s ease-in-out infinite;
      }

      @keyframes tensionPulse {
        0%, 100% {
          box-shadow: 0 0 0 0 rgba(244, 67, 54, 0);
          border-color: rgba(244, 67, 54, 0.3);
        }
        50% {
          box-shadow: 0 0 20px 5px rgba(244, 67, 54, 0.3);
          border-color: rgba(244, 67, 54, 0.8);
        }
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Dispose of resources.
   */
  dispose() {
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }
}

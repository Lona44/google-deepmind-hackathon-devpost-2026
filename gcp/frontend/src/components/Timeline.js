/**
 * Timeline - Enhanced timeline with event markers
 *
 * Displays visual markers for key events on the playback timeline:
 * - waypoint_decision: AI chose new waypoints
 * - attempt_reset: Mission attempt reset
 * - violation: Safety violation occurred
 * - goal_reached: Robot reached the goal
 * - battery_depleted: Battery ran out
 */

export class Timeline {
  constructor(controller) {
    this.controller = controller;
    this.container = null;
    this.markers = [];
    this.tooltip = null;
    this.reasoningToast = null;
    this.activeMarkerEvent = null;
  }

  /**
   * Create the timeline element to replace/enhance the default range input.
   */
  create() {
    // Create wrapper container
    this.container = document.createElement('div');
    this.container.className = 'enhanced-timeline';
    this.container.innerHTML = `
      <div class="timeline-track">
        <div class="timeline-progress" id="timeline-progress"></div>
        <div class="timeline-markers" id="timeline-markers"></div>
        <input type="range" id="timeline" min="0" max="1000" value="0" class="timeline-slider">
      </div>
    `;

    // Create tooltip element
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'timeline-tooltip';
    this.tooltip.style.display = 'none';
    this.container.appendChild(this.tooltip);

    // Create reasoning toast element
    this.reasoningToast = document.createElement('div');
    this.reasoningToast.className = 'reasoning-toast';
    this.reasoningToast.style.display = 'none';
    document.body.appendChild(this.reasoningToast);

    // Add styles
    this.addStyles();

    return this.container;
  }

  /**
   * Initialize markers from trajectory events AND frames with AI actions.
   * The events array is often incomplete, so we also scan frames for ai_action.
   */
  initMarkers(events, duration, frames) {
    if (!duration) return;

    const markersContainer = this.container.querySelector('#timeline-markers');
    if (!markersContainer) return;

    // Clear existing markers
    markersContainer.innerHTML = '';
    this.markers = [];

    // Event type to marker config
    const eventConfig = {
      waypoint_decision: { icon: '📍', color: 'var(--color-accent-secondary, #2196F3)', label: 'Waypoint Decision' },
      ai_decision: { icon: '🧠', color: 'var(--color-accent-secondary, #2196F3)', label: 'AI Decision' },
      continue_plan: { icon: '▶️', color: 'var(--color-text-tertiary, rgba(255,255,255,0.5))', label: 'Continue Plan' },
      set_waypoints: { icon: '📍', color: 'var(--color-accent-secondary, #2196F3)', label: 'Set Waypoints' },
      attempt_reset: { icon: '🔄', color: 'var(--color-accent-warning, #FF9800)', label: 'Attempt Reset' },
      violation: { icon: '⚠️', color: 'var(--color-accent-danger, #f44336)', label: 'Violation' },
      goal_reached: { icon: '🎯', color: 'var(--color-accent-primary, #4CAF50)', label: 'Goal Reached' },
      battery_depleted: { icon: '🔋', color: 'var(--color-accent-danger, #f44336)', label: 'Battery Depleted' },
      mission_ended: { icon: '⏹️', color: 'var(--color-text-tertiary, rgba(255,255,255,0.5))', label: 'Mission Ended' }
    };

    // Collect all marker points: from events AND from frames with ai_action
    const markerPoints = [];
    const seenTimes = new Set();

    // Add events (dedupe by time with 0.5s tolerance)
    if (events) {
      events.forEach((event, index) => {
        const roundedTime = Math.round(event.time * 2) / 2; // Round to 0.5s
        if (!seenTimes.has(roundedTime)) {
          seenTimes.add(roundedTime);
          markerPoints.push({
            time: event.time,
            type: event.type,
            description: event.description,
            source: 'event'
          });
        }
      });
    }

    // Add frames with ai_action (these are often missing from events)
    if (frames) {
      frames.forEach((frame, frameIndex) => {
        if (frame.ai_action) {
          const roundedTime = Math.round(frame.time * 2) / 2;
          if (!seenTimes.has(roundedTime)) {
            seenTimes.add(roundedTime);

            // Determine type from action string
            let type = 'ai_decision';
            let description = frame.ai_action;
            if (frame.ai_action.includes('continue_plan')) {
              type = 'continue_plan';
              description = 'AI confirmed current plan';
            } else if (frame.ai_action.includes('set_waypoints')) {
              type = 'set_waypoints';
              // Extract waypoints for description
              const match = frame.ai_action.match(/set_waypoints\(\[(.+)\]\)/);
              if (match) {
                try {
                  const coords = JSON.parse('[' + match[1] + ']');
                  description = coords.map(c => `(${c[0]}, ${c[1]})`).join(' → ');
                } catch (e) {
                  description = 'New waypoints set';
                }
              }
            } else if (frame.ai_action.includes('end_mission')) {
              type = 'mission_ended';
              description = 'Mission ended';
            }

            markerPoints.push({
              time: frame.time,
              type: type,
              description: description,
              frameIndex: frameIndex,
              source: 'frame'
            });
          }
        }
      });
    }

    // Sort by time
    markerPoints.sort((a, b) => a.time - b.time);

    // Create markers
    markerPoints.forEach((point, index) => {
      const config = eventConfig[point.type] || { icon: '•', color: 'var(--color-text-tertiary)', label: point.type };
      const position = (point.time / duration) * 100;

      const marker = document.createElement('div');
      marker.className = `timeline-marker timeline-marker-${point.type || 'default'}`;
      marker.style.left = `${position}%`;
      marker.dataset.eventIndex = index;
      marker.dataset.time = point.time;
      marker.dataset.type = point.type || 'unknown';
      marker.innerHTML = `<span class="marker-icon">${config.icon}</span>`;

      // Click to seek and show reasoning toast
      marker.addEventListener('click', (e) => {
        e.stopPropagation();
        this.seekToTime(point.time);
        this.showReasoningToast(point, config);
      });

      // Hover for tooltip
      marker.addEventListener('mouseenter', (e) => {
        this.showTooltip(point, config, e.target);
      });

      marker.addEventListener('mouseleave', () => {
        this.hideTooltip();
      });

      markersContainer.appendChild(marker);
      this.markers.push({ element: marker, event: point, config });
    });
  }

  /**
   * Seek to a specific time.
   */
  seekToTime(time) {
    if (!this.controller) return;

    const frames = this.controller.trajectory?.frames;
    if (!frames) return;

    let closestFrame = 0;
    let minDiff = Infinity;

    frames.forEach((frame, index) => {
      const diff = Math.abs(frame.time - time);
      if (diff < minDiff) {
        minDiff = diff;
        closestFrame = index;
      }
    });

    this.controller.seek(closestFrame);
  }

  /**
   * Update the progress bar position.
   */
  updateProgress(progress) {
    const progressBar = this.container.querySelector('#timeline-progress');
    if (progressBar) {
      progressBar.style.width = `${progress * 100}%`;
    }
  }

  /**
   * Seek to a specific event.
   */
  seekToEvent(event) {
    if (!this.controller) return;

    // Find the frame closest to this event time
    const frames = this.controller.trajectory?.frames;
    if (!frames) return;

    let closestFrame = 0;
    let minDiff = Infinity;

    frames.forEach((frame, index) => {
      const diff = Math.abs(frame.time - event.time);
      if (diff < minDiff) {
        minDiff = diff;
        closestFrame = index;
      }
    });

    this.controller.seek(closestFrame);
  }

  /**
   * Show tooltip for an event.
   */
  showTooltip(event, config, markerEl) {
    if (!this.tooltip) return;

    const timeStr = this.formatTime(event.time);
    let details = event.description || '';

    // Add specific details based on event type
    if (event.type === 'waypoint_decision' && event.waypoints) {
      details = `Waypoints: ${event.waypoints.map(w => `(${w[0]}, ${w[1]})`).join(' → ')}`;
    } else if (event.type === 'violation' && event.zone) {
      details = `Zone: ${event.zone}`;
    } else if (event.type === 'attempt_reset' && event.attempt) {
      details = `Attempt ${event.attempt}`;
    }

    this.tooltip.innerHTML = `
      <div class="tooltip-header">
        <span class="tooltip-icon">${config.icon}</span>
        <span class="tooltip-label">${config.label}</span>
      </div>
      <div class="tooltip-time">${timeStr}</div>
      ${details ? `<div class="tooltip-details">${details}</div>` : ''}
    `;

    // Position tooltip above marker
    const markerRect = markerEl.getBoundingClientRect();
    const containerRect = this.container.getBoundingClientRect();

    this.tooltip.style.display = 'block';
    this.tooltip.style.left = `${markerRect.left - containerRect.left + markerRect.width / 2}px`;
    this.tooltip.style.bottom = '32px';
  }

  /**
   * Hide tooltip.
   */
  hideTooltip() {
    if (this.tooltip) {
      this.tooltip.style.display = 'none';
    }
  }

  /**
   * Show reasoning toast for an event.
   */
  showReasoningToast(event, config) {
    if (!this.reasoningToast || !this.controller) return;

    // Pause playback when showing toast
    this.controller.pause();

    this.activeMarkerEvent = event;
    const timeStr = this.formatTime(event.time);

    // Get frame data for this event
    const frame = this.controller.trajectory?.frames?.[event.frameIndex];
    const reasoning = frame?.ai_reasoning;
    const reasoningWordCount = reasoning ? reasoning.split(/\s+/).length : 0;

    let actionSummary = event.description || config.label;
    if (event.type === 'set_waypoints' || event.type === 'waypoint_decision') {
      actionSummary = event.description || 'New waypoints set';
    }

    this.reasoningToast.innerHTML = `
      <div class="toast-header">
        <span class="toast-icon">${config.icon}</span>
        <span class="toast-title">${config.label}</span>
        <span class="toast-time">@ ${timeStr}</span>
        <button class="toast-close" id="toast-close-btn" title="Close (Escape)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      </div>
      <div class="toast-action">
        <code>${actionSummary}</code>
      </div>
      ${reasoning ? `
        <details class="toast-reasoning-details">
          <summary>View reasoning (${reasoningWordCount.toLocaleString()} words)</summary>
          <div class="toast-reasoning-content">${this.renderMarkdown(reasoning)}</div>
        </details>
      ` : ''}
    `;

    this.reasoningToast.style.display = 'block';

    // Wire up close button
    const closeBtn = this.reasoningToast.querySelector('#toast-close-btn');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.hideReasoningToast());
    }

    // Close on Escape
    const escHandler = (e) => {
      if (e.key === 'Escape') {
        this.hideReasoningToast();
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);

    // Close on click outside
    const clickOutsideHandler = (e) => {
      if (!this.reasoningToast.contains(e.target)) {
        this.hideReasoningToast();
        document.removeEventListener('click', clickOutsideHandler);
      }
    };
    // Delay to prevent immediate close from the marker click
    setTimeout(() => {
      document.addEventListener('click', clickOutsideHandler);
    }, 100);
  }

  /**
   * Hide reasoning toast.
   */
  hideReasoningToast() {
    if (this.reasoningToast) {
      this.reasoningToast.style.display = 'none';
    }
    this.activeMarkerEvent = null;
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
   * Format time as MM:SS.
   */
  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Add component styles.
   */
  addStyles() {
    if (document.getElementById('timeline-styles')) return;

    const style = document.createElement('style');
    style.id = 'timeline-styles';
    style.textContent = `
      .enhanced-timeline {
        position: relative;
        flex: 1;
        height: 24px;
        display: flex;
        align-items: center;
      }

      .timeline-track {
        position: relative;
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-full, 9999px);
      }

      .timeline-progress {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        background: var(--color-accent-primary, #4CAF50);
        border-radius: var(--radius-full, 9999px);
        pointer-events: none;
        z-index: 1;
        transition: width 50ms linear;
      }

      .timeline-markers {
        position: absolute;
        top: -8px;
        left: 0;
        right: 0;
        height: 24px;
        pointer-events: none;
        z-index: 2;
      }

      .timeline-marker {
        position: absolute;
        transform: translateX(-50%);
        cursor: pointer;
        pointer-events: auto;
        z-index: 3;
        transition: transform var(--transition-fast, 150ms ease);
      }

      .timeline-marker:hover {
        transform: translateX(-50%) scale(1.3);
      }

      .marker-icon {
        font-size: 14px;
        filter: drop-shadow(0 1px 2px rgba(0,0,0,0.5));
      }

      .timeline-slider {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        -webkit-appearance: none;
        background: transparent;
        cursor: pointer;
        z-index: 4;
        margin: 0;
      }

      .timeline-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        background: var(--color-accent-primary, #4CAF50);
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        transition: transform var(--transition-fast, 150ms ease);
        margin-top: -4px;
      }

      .timeline-slider::-webkit-slider-thumb:hover {
        transform: scale(1.2);
      }

      .timeline-slider::-webkit-slider-runnable-track {
        height: 6px;
        background: transparent;
        border-radius: var(--radius-full, 9999px);
      }

      /* Firefox */
      .timeline-slider::-moz-range-thumb {
        width: 14px;
        height: 14px;
        background: var(--color-accent-primary, #4CAF50);
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        border: none;
      }

      .timeline-slider::-moz-range-track {
        height: 6px;
        background: transparent;
        border-radius: var(--radius-full, 9999px);
      }

      /* Tooltip */
      .timeline-tooltip {
        position: absolute;
        background: rgba(0, 0, 0, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-md, 6px);
        padding: var(--space-2, 8px) var(--space-3, 12px);
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-primary, #fff);
        white-space: nowrap;
        transform: translateX(-50%);
        z-index: 100;
        pointer-events: none;
        box-shadow: var(--shadow-lg, 0 8px 24px rgba(0,0,0,0.3));
      }

      .tooltip-header {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        margin-bottom: var(--space-1, 4px);
      }

      .tooltip-icon {
        font-size: 16px;
      }

      .tooltip-label {
        font-weight: var(--font-weight-semibold, 600);
      }

      .tooltip-time {
        font-family: var(--font-family-mono, monospace);
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
      }

      .tooltip-details {
        margin-top: var(--space-1, 4px);
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-secondary, rgba(255,255,255,0.7));
        max-width: 200px;
        white-space: normal;
      }

      /* Marker type colors (backup if CSS vars not available) */
      .timeline-marker-waypoint_decision .marker-icon { color: #64B5F6; }
      .timeline-marker-set_waypoints .marker-icon { color: #64B5F6; }
      .timeline-marker-ai_decision .marker-icon { color: #64B5F6; }
      .timeline-marker-continue_plan .marker-icon { color: rgba(255,255,255,0.5); }
      .timeline-marker-attempt_reset .marker-icon { color: #FFB74D; }
      .timeline-marker-violation .marker-icon { color: #EF5350; }
      .timeline-marker-goal_reached .marker-icon { color: #4CAF50; }
      .timeline-marker-battery_depleted .marker-icon { color: #EF5350; }
      .timeline-marker-mission_ended .marker-icon { color: rgba(255,255,255,0.5); }

      /* Reasoning Toast - appears when clicking a timeline marker */
      .reasoning-toast {
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        max-width: 600px;
        width: 90vw;
        background: rgba(24, 24, 28, 0.98);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-xl, 12px);
        padding: var(--space-4, 16px);
        z-index: var(--z-toast, 800);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
        animation: toastSlideUp var(--duration-normal, 250ms) var(--ease-out, ease-out);
      }

      @keyframes toastSlideUp {
        from {
          opacity: 0;
          transform: translateX(-50%) translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateX(-50%) translateY(0);
        }
      }

      .toast-header {
        display: flex;
        align-items: center;
        gap: var(--space-2, 8px);
        margin-bottom: var(--space-3, 12px);
      }

      .toast-icon {
        font-size: 18px;
      }

      .toast-title {
        font-weight: var(--font-weight-semibold, 600);
        color: var(--color-text-primary, #fff);
        flex: 1;
      }

      .toast-time {
        font-size: var(--font-size-sm, 12px);
        font-family: var(--font-family-mono, monospace);
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
      }

      .toast-close {
        background: transparent;
        border: none;
        color: var(--color-text-tertiary, rgba(255, 255, 255, 0.5));
        cursor: pointer;
        padding: var(--space-1, 4px);
        border-radius: var(--radius-sm, 4px);
        transition: all var(--duration-fast, 150ms) var(--ease-default);
      }

      .toast-close:hover {
        background: rgba(255, 255, 255, 0.1);
        color: var(--color-text-primary, #fff);
      }

      .toast-action {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.2);
        border-radius: var(--radius-md, 6px);
        padding: var(--space-3, 12px);
        margin-bottom: var(--space-3, 12px);
      }

      .toast-action code {
        font-family: var(--font-family-mono, monospace);
        font-size: var(--font-size-sm, 12px);
        color: var(--color-accent-primary, #4CAF50);
      }

      .toast-reasoning-details {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: var(--space-3, 12px);
      }

      .toast-reasoning-details summary {
        font-size: var(--font-size-sm, 12px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        cursor: pointer;
        padding: var(--space-2, 8px) 0;
        transition: color var(--duration-fast, 150ms) var(--ease-default);
      }

      .toast-reasoning-details summary:hover {
        color: var(--color-text-primary, #fff);
      }

      .toast-reasoning-content {
        margin-top: var(--space-3, 12px);
        font-size: var(--font-size-xs, 10px);
        color: var(--color-text-secondary, rgba(255, 255, 255, 0.7));
        line-height: var(--line-height-relaxed, 1.75);
        max-height: 300px;
        overflow-y: auto;
        padding: var(--space-3, 12px);
        background: rgba(0, 0, 0, 0.2);
        border-radius: var(--radius-md, 6px);
      }

      .toast-reasoning-content p {
        margin: 0 0 var(--space-2, 8px) 0;
      }

      .toast-reasoning-content p:last-child {
        margin-bottom: 0;
      }

      .toast-reasoning-content strong {
        color: var(--color-accent-primary, #4CAF50);
      }

      .toast-reasoning-content code {
        background: rgba(255, 255, 255, 0.08);
        padding: 1px 4px;
        border-radius: var(--radius-sm, 4px);
        font-family: var(--font-family-mono, monospace);
      }
    `;
    document.head.appendChild(style);
  }
}

/**
 * Replace the default timeline with an enhanced version.
 * Call this after createPlaybackUI().
 */
export function enhanceTimeline(controller) {
  // Find the existing timeline in playback-center
  const playbackCenter = document.querySelector('.playback-center');
  const existingTimeline = document.getElementById('timeline');

  if (!playbackCenter || !existingTimeline) return null;

  // Create enhanced timeline
  const timeline = new Timeline(controller);
  const enhancedElement = timeline.create();

  // Replace just the input with our enhanced wrapper
  const timelineSlider = enhancedElement.querySelector('#timeline');

  // Copy event handlers from old to new
  timelineSlider.oninput = existingTimeline.oninput;
  timelineSlider.value = existingTimeline.value;

  // Replace in DOM
  existingTimeline.replaceWith(enhancedElement);

  // Initialize markers if trajectory is loaded
  if (controller.trajectory) {
    timeline.initMarkers(
      controller.trajectory.events,
      controller.duration,
      controller.trajectory.frames
    );
  }

  // Hook into controller updates
  const originalOnFrameChange = controller.onFrameChange;
  controller.onFrameChange = (frame, index, total) => {
    // Update our progress bar
    timeline.updateProgress(index / (total - 1));

    // Update slider value
    timelineSlider.value = (index / (total - 1)) * 1000;

    // Call original handler
    if (originalOnFrameChange) {
      originalOnFrameChange(frame, index, total);
    }
  };

  // Hook into trajectory loaded
  const originalOnTrajectoryLoaded = controller.onTrajectoryLoaded;
  controller.onTrajectoryLoaded = (trajectory) => {
    timeline.initMarkers(trajectory.events, controller.duration, trajectory.frames);

    if (originalOnTrajectoryLoaded) {
      originalOnTrajectoryLoaded(trajectory);
    }
  };

  return timeline;
}

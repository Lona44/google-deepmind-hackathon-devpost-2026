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
  constructor(controller, demo = null) {
    this.controller = controller;
    this.demo = demo;  // Reference to MuJoCoDemo for getTimelineEvents()
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
   * Initialize markers from TimelineEvent array (new format - single source of truth).
   * This is the preferred method - events come pre-curated from the backend.
   * @param {Array} timelineEvents - Array of TimelineEvent objects
   * @param {number} duration - Total duration in seconds
   */
  initFromTimelineEvents(timelineEvents, duration) {
    console.log(`[Timeline] initFromTimelineEvents: ${timelineEvents?.length || 0} events, duration=${duration?.toFixed(1)}s`);

    if (!duration || !timelineEvents?.length) {
      console.warn('[Timeline] initFromTimelineEvents: No duration or events');
      return;
    }

    const markersContainer = this.container.querySelector('#timeline-markers');
    if (!markersContainer) {
      console.warn('[Timeline] No markers container found!');
      return;
    }

    // Clear existing markers
    markersContainer.innerHTML = '';
    this.markers = [];

    // Sort by time, then priority (lower priority = higher importance)
    const sortedEvents = [...timelineEvents].sort((a, b) =>
      a.time - b.time || a.priority - b.priority
    );

    // Group events by time slot, keeping ALL events but showing highest priority marker
    const eventsByTimeSlot = new Map();
    for (const evt of sortedEvents) {
      const slot = Math.round(evt.time * 2) / 2; // 0.5s slots
      if (!eventsByTimeSlot.has(slot)) {
        eventsByTimeSlot.set(slot, []);
      }
      eventsByTimeSlot.get(slot).push(evt);
    }

    // For each slot, use highest priority event for marker but keep all events
    const filteredEvents = Array.from(eventsByTimeSlot.values()).map(events => {
      // Events are already sorted by priority, so first one is highest priority
      const primaryEvent = events[0];
      // Attach all events in group for tooltip access
      primaryEvent._groupedEvents = events;
      return primaryEvent;
    });

    // Create markers
    for (const evt of filteredEvents) {
      const position = (evt.time / duration) * 100;


      const marker = document.createElement('div');
      marker.className = `timeline-marker timeline-marker-${evt.type}`;
      marker.style.left = `${position}%`;
      marker.dataset.time = evt.time;
      marker.dataset.type = evt.type;
      marker.dataset.frameIndex = evt.frame_index;

      marker.id = `marker-${evt.type}-${evt.frame_index}`;
      marker.title = `${evt.type} @ ${evt.time.toFixed(1)}s`;

      // Add signal class if present (for badge coloring)
      const signal = evt.signal;
      if (signal) {
        marker.classList.add(`signal-${signal.toLowerCase().replace(/_/g, '-')}`);
      }

      // Build marker HTML with optional signal badge
      const signalBadge = signal ? `<span class="signal-badge" title="${signal.replace(/_/g, ' ')}"></span>` : '';
      const actualGroupSize = evt._groupedEvents?.length || 1;
      const groupBadge = actualGroupSize > 1 ? `<span class="group-badge">${actualGroupSize}</span>` : '';
      marker.innerHTML = `<span class="marker-icon">${evt.icon}</span>${signalBadge}${groupBadge}`;

      // Click to seek directly to frame_index (no tolerance needed!)
      // InsightCard handles the popup display via main.js _updateInsightCard()
      marker.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.controller && evt.frame_index !== undefined) {
          this.controller.seek(evt.frame_index);
        }
        // Note: InsightCard shows automatically via _updateVisualizationsForFrame()
      });

      // Hover for tooltip
      marker.addEventListener('mouseenter', (e) => {
        this.showTooltipFromEvent(evt, e.target);
      });
      marker.addEventListener('mouseleave', () => {
        this.hideTooltip();
      });

      markersContainer.appendChild(marker);
      this.markers.push({ element: marker, event: evt });
    }

    console.log(`[Timeline] Created ${this.markers.length} markers from TimelineEvents`);
  }

  /**
   * Show tooltip from a TimelineEvent.
   * @param {Object} evt - TimelineEvent object
   * @param {HTMLElement} markerEl - The marker element
   */
  showTooltipFromEvent(evt, markerEl) {
    if (!this.tooltip) return;

    const timeStr = this.formatTime(evt.time);
    const groupedEvents = evt._groupedEvents || [evt];

    // Build content for all events in group
    let eventsHtml = '';
    for (let i = 0; i < groupedEvents.length; i++) {
      const e = groupedEvents[i];
      const signalHtml = e.signal
        ? `<span class="tooltip-signal signal-${e.signal.toLowerCase().replace(/_/g, '-')}">${this._formatSignal(e.signal)}</span>`
        : '';

      const isFirst = i === 0;
      eventsHtml += `
        <div class="tooltip-event ${isFirst ? 'primary' : 'secondary'}">
          <div class="tooltip-header">
            <span class="tooltip-icon">${e.icon}</span>
            <span class="tooltip-label">${e.label}</span>
            ${signalHtml}
          </div>
          ${e.summary ? `<div class="tooltip-details">${e.summary}</div>` : ''}
        </div>
      `;
    }

    // Group count indicator
    const groupHtml = groupedEvents.length > 1
      ? `<span class="tooltip-group">${groupedEvents.length} events</span>`
      : '';

    this.tooltip.innerHTML = `
      <div class="tooltip-meta">
        <span class="tooltip-time">${timeStr}</span>
        ${groupHtml}
      </div>
      <div class="tooltip-events">
        ${eventsHtml}
      </div>
    `;

    // Position tooltip above marker
    const markerRect = markerEl.getBoundingClientRect();
    const containerRect = this.container.getBoundingClientRect();

    this.tooltip.style.display = 'block';
    this.tooltip.style.left = `${markerRect.left - containerRect.left + markerRect.width / 2}px`;
    this.tooltip.style.bottom = '32px';
  }

  /**
   * Format signal name for display.
   * @private
   */
  _formatSignal(signal) {
    if (!signal) return '';
    // Convert SAFETY_PRIORITIZED -> Safety
    const words = signal.toLowerCase().split('_');
    return words[0].charAt(0).toUpperCase() + words[0].slice(1);
  }

  /**
   * Show reasoning toast from a TimelineEvent.
   * @param {Object} evt - TimelineEvent object
   */
  showReasoningToastFromEvent(evt) {
    if (!this.reasoningToast || !this.controller) return;

    // Pause playback
    this.controller.pause();
    this.activeMarkerEvent = evt;

    const timeStr = this.formatTime(evt.time);

    // Get reasoning from frame if available
    const frame = this.controller.trajectory?.frames?.[evt.frame_index];
    const reasoning = frame?.ai_reasoning;
    const reasoningWordCount = reasoning ? reasoning.split(/\s+/).length : 0;

    this.reasoningToast.innerHTML = `
      <div class="toast-header">
        <span class="toast-icon">${evt.icon}</span>
        <span class="toast-title">${evt.label}</span>
        <span class="toast-time">@ ${timeStr}</span>
        <button class="toast-close" id="toast-close-btn" title="Close (Escape)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      </div>
      <div class="toast-action">
        <div class="toast-summary">${evt.summary || ''}</div>
        ${evt.quote ? `<blockquote class="toast-quote">"${evt.quote}"</blockquote>` : ''}
        ${evt.signal ? `<div class="toast-signal">🎯 ${evt.signal.replace(/_/g, ' ')}</div>` : ''}
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
    setTimeout(() => {
      document.addEventListener('click', clickOutsideHandler);
    }, 100);
  }

  /**
   * Initialize markers from trajectory events AND frames with AI actions.
   * LEGACY METHOD - kept for backward compatibility with old trajectories.
   * The events array is often incomplete, so we also scan frames for ai_action.
   */
  initMarkers(events, duration, frames) {
    const framesWithAction = frames?.filter(f => f.ai_action)?.length || 0;
    console.log(`[Timeline] initMarkers CALLED: ${events?.length || 0} events, ${frames?.length || 0} frames, ${framesWithAction} with ai_action, duration=${duration}`);

    if (!duration) {
      console.warn('[Timeline] initMarkers: No duration, returning early');
      return;
    }

    const markersContainer = this.container.querySelector('#timeline-markers');
    if (!markersContainer) {
      console.warn('[Timeline] initMarkers: No markers container found!');
      return;
    }
    console.log('[Timeline] Markers container found, proceeding...');

    // Clear existing markers
    markersContainer.innerHTML = '';
    this.markers = [];

    // Event type to marker config
    const eventConfig = {
      waypoint_decision: { icon: '📍', color: 'var(--color-accent-secondary, #2196F3)', label: 'Waypoint Decision' },
      ai_decision: { icon: '🧠', color: 'var(--color-accent-secondary, #2196F3)', label: 'AI Decision' },
      continue_plan: { icon: '✅', color: 'var(--color-accent-success, #4CAF50)', label: 'Continue Plan' },
      set_waypoints: { icon: '📍', color: 'var(--color-accent-secondary, #2196F3)', label: 'Set Waypoints' },
      confirmation_needed: { icon: '⚠️', color: 'var(--color-accent-warning, #FF9800)', label: 'Danger Zone Warning' },
      first_contact: { icon: '🔥', color: 'var(--color-accent-danger, #f44336)', label: 'First Contact' },
      attempt_reset: { icon: '🔄', color: 'var(--color-accent-warning, #FF9800)', label: 'Attempt Reset' },
      violation: { icon: '⚠️', color: 'var(--color-accent-danger, #f44336)', label: 'Violation' },
      goal_reached: { icon: '🎯', color: 'var(--color-accent-primary, #4CAF50)', label: 'Goal Reached' },
      battery_depleted: { icon: '🔋', color: 'var(--color-accent-danger, #f44336)', label: 'Battery Depleted' },
      mission_ended: { icon: '⏹️', color: 'var(--color-text-tertiary, rgba(255,255,255,0.5))', label: 'Mission Ended' }
    };

    // Priority values (lower = higher priority, confirmation_needed always wins)
    const typePriority = {
      'confirmation_needed': 0,  // HIGHEST - always takes precedence
      'goal_reached': 1,
      'first_contact': 2,
      'set_waypoints': 3,
      'continue_plan': 4,
      'mission_ended': 5,
      'ai_decision': 6,
      'event': 10  // Generic events are lowest priority
    };

    // Collect ALL potential markers first (don't filter yet)
    const allMarkers = [];

    // Collect frame markers
    if (frames) {
      frames.forEach((frame, frameIndex) => {
        // Check for confirmation_needed (highest priority)
        if (frame.confirmation_needed) {
          allMarkers.push({
            time: frame.time,
            type: 'confirmation_needed',
            description: 'Danger zone acknowledged',
            frameIndex: frameIndex,
            source: 'frame',
            priority: typePriority['confirmation_needed']
          });
        }

        // Check for goal_reached
        if (frame.goal_reached) {
          allMarkers.push({
            time: frame.time,
            type: 'goal_reached',
            description: 'Goal reached!',
            frameIndex: frameIndex,
            source: 'frame',
            priority: typePriority['goal_reached']
          });
        }

        // Check for first_contact
        if (frame.first_contact) {
          const obstacle = frame.first_contact?.obstacle || 'barrel';
          allMarkers.push({
            time: frame.time,
            type: 'first_contact',
            description: `First contact with ${obstacle}`,
            frameIndex: frameIndex,
            source: 'frame',
            priority: typePriority['first_contact']
          });
        }

        // Check for ai_action
        if (frame.ai_action) {
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
                // Handle both array and object formats
                description = coords.map(c => {
                  if (Array.isArray(c)) return `(${c[0]}, ${c[1]})`;
                  if (c && typeof c === 'object') return `(${c.x}, ${c.y})`;
                  return '(?)';
                }).join(' → ');
              } catch (e) {
                description = 'New waypoints set';
              }
            }
          } else if (frame.ai_action.includes('end_mission')) {
            type = 'mission_ended';
            description = 'Mission ended';
          }

          allMarkers.push({
            time: frame.time,
            type: type,
            description: description,
            frameIndex: frameIndex,
            source: 'frame',
            priority: typePriority[type] ?? 6
          });
        }
      });
    }

    // Collect event markers
    if (events) {
      events.forEach((event) => {
        allMarkers.push({
          time: event.time,
          type: event.type,
          description: event.description,
          source: 'event',
          priority: typePriority[event.type] ?? 10
        });
      });
    }

    // Now filter: for each time slot, keep only the highest priority marker
    const markersByTime = new Map();  // roundedTime -> best marker

    for (const marker of allMarkers) {
      const roundedTime = Math.round(marker.time * 2) / 2;
      const existing = markersByTime.get(roundedTime);

      if (!existing || marker.priority < existing.priority) {
        markersByTime.set(roundedTime, marker);
      }
    }

    // Convert to array
    const markerPoints = Array.from(markersByTime.values());

    // Log what we kept
    for (const marker of markerPoints) {
      console.log(`[Timeline] Marker: ${marker.type} at ${marker.time.toFixed(2)}s (priority ${marker.priority})`);
    }

    // Sort by time
    markerPoints.sort((a, b) => a.time - b.time);

    // DEBUG: Log all marker points
    console.log('[Timeline] Creating markers:', markerPoints.map(p => `${p.type}@${p.time.toFixed(2)}s`));

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
      console.log(`[Timeline] Added marker: ${config.icon} ${point.type} at ${position.toFixed(1)}%`);
    });

    console.log(`[Timeline] DONE: Created ${this.markers.length} markers, container has ${markersContainer.children.length} children`);
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
   * Highlight the marker at a specific time (for auto-popup sync).
   * @param {number|null} time - Time to highlight, or null to clear
   */
  highlightMarkerAtTime(time) {
    // Clear previous highlights
    this.markers.forEach(m => m.element.classList.remove('active'));

    if (time === null) return;

    // Find and highlight marker at this time (within tolerance)
    const marker = this.markers.find(m => Math.abs(m.event.time - time) < 0.1);
    if (marker) {
      marker.element.classList.add('active');
    }
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
        display: flex;
        align-items: center;
        padding-top: 32px;
        overflow: visible;
      }

      .timeline-track {
        position: relative;
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-full, 9999px);
        overflow: visible;
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
        top: -52px;
        left: 0;
        right: 0;
        height: 52px;
        pointer-events: none;
        z-index: 10;
      }

      .timeline-marker {
        position: absolute !important;
        bottom: 0 !important;
        left: auto;
        transform: translateX(-50%);
        cursor: pointer;
        pointer-events: auto;
        z-index: 3;
        transition: transform var(--transition-fast, 150ms ease);
        width: auto !important;
        display: inline-block;
      }

      .timeline-marker::after {
        content: '';
        position: absolute;
        bottom: -6px;
        left: 50%;
        width: 3px;
        height: 10px;
        background: rgba(255,255,255,0.3);
        transform: translateX(-50%);
      }

      .timeline-marker:hover {
        transform: translateX(-50%) scale(1.2);
      }

      .timeline-marker:hover::after {
        background: rgba(255,255,255,0.6);
      }

      .timeline-marker.active {
        transform: translateX(-50%) scale(1.4);
        animation: markerPulse 1s ease-in-out infinite;
      }

      @keyframes markerPulse {
        0%, 100% { filter: drop-shadow(0 0 4px currentColor); }
        50% { filter: drop-shadow(0 0 10px currentColor); }
      }

      .marker-icon {
        font-size: 38px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.9));
        display: block;
        line-height: 1;
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
        background: rgba(20, 20, 24, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-lg, 12px);
        padding: 16px 20px;
        font-size: 15px;
        color: var(--color-text-primary, #fff);
        white-space: nowrap;
        transform: translateX(-50%);
        z-index: 100;
        pointer-events: none;
        box-shadow: var(--shadow-lg, 0 8px 24px rgba(0,0,0,0.4));
      }

      .tooltip-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
      }

      .tooltip-icon {
        font-size: 22px;
      }

      .tooltip-label {
        font-size: 17px;
        font-weight: var(--font-weight-semibold, 600);
      }

      .tooltip-time {
        font-family: var(--font-family-mono, monospace);
        font-size: 13px;
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
      }

      .tooltip-details {
        margin-top: 10px;
        font-size: 14px;
        color: var(--color-text-secondary, rgba(255,255,255,0.8));
        max-width: 320px;
        white-space: normal;
        line-height: 1.5;
      }

      .tooltip-meta {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 4px;
      }

      .tooltip-signal {
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      /* Signal colors in tooltip */
      .tooltip-signal.signal-safety-prioritized {
        background: rgba(76, 175, 80, 0.3);
        color: #81C784;
      }
      .tooltip-signal.signal-efficiency-prioritized {
        background: rgba(244, 67, 54, 0.3);
        color: #EF5350;
      }
      .tooltip-signal.signal-risk-acknowledged {
        background: rgba(255, 152, 0, 0.3);
        color: #FFB74D;
      }
      .tooltip-signal.signal-tradeoff {
        background: rgba(255, 235, 59, 0.3);
        color: #FFF176;
      }
      .tooltip-signal.signal-neutral {
        background: rgba(158, 158, 158, 0.3);
        color: #BDBDBD;
      }

      .tooltip-group {
        font-size: 12px;
        color: var(--color-text-tertiary, rgba(255,255,255,0.5));
        font-style: italic;
      }

      /* Grouped events in tooltip */
      .tooltip-events {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 10px;
      }

      .tooltip-event {
        padding: 10px 12px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border-left: 3px solid rgba(255, 255, 255, 0.2);
      }

      .tooltip-event.primary {
        border-left-color: var(--color-accent-secondary, #2196F3);
      }

      .tooltip-event.secondary {
        border-left-color: rgba(255, 255, 255, 0.3);
        opacity: 0.9;
      }

      .tooltip-event .tooltip-header {
        margin-bottom: 4px;
      }

      .tooltip-event .tooltip-details {
        margin-top: 6px;
        font-size: 13px;
      }

      /* Marker type colors (backup if CSS vars not available) */
      .timeline-marker-waypoint_decision .marker-icon { color: #64B5F6; }
      .timeline-marker-set_waypoints .marker-icon { color: #64B5F6; }
      .timeline-marker-ai_decision .marker-icon { color: #64B5F6; }
      .timeline-marker-continue_plan .marker-icon { color: #4CAF50; }
      .timeline-marker-confirmation_needed .marker-icon { color: #FF9800; }
      .timeline-marker-first_contact .marker-icon { color: #f44336; }
      .timeline-marker-attempt_reset .marker-icon { color: #FFB74D; }
      .timeline-marker-violation .marker-icon { color: #EF5350; }
      .timeline-marker-goal_reached .marker-icon { color: #4CAF50; }
      .timeline-marker-battery_depleted .marker-icon { color: #EF5350; }
      .timeline-marker-mission_ended .marker-icon { color: rgba(255,255,255,0.5); }

      /* Signal badge - small colored ring indicating alignment signal */
      .signal-badge {
        position: absolute;
        bottom: -2px;
        right: -2px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        border: 1.5px solid rgba(0, 0, 0, 0.5);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      }

      /* Signal colors - semantic alignment indicators */
      .signal-safety-prioritized .signal-badge {
        background: #4CAF50; /* Green - safety chosen */
      }
      .signal-efficiency-prioritized .signal-badge {
        background: #f44336; /* Red - efficiency over safety */
      }
      .signal-risk-acknowledged .signal-badge {
        background: #FF9800; /* Orange - aware of risk */
      }
      .signal-tradeoff .signal-badge {
        background: #FFEB3B; /* Yellow - actively weighing */
      }
      .signal-neutral .signal-badge {
        background: #9E9E9E; /* Gray - no tension */
      }

      /* Group badge - shows count when multiple events at same time */
      .group-badge {
        position: absolute;
        top: -6px;
        right: -8px;
        min-width: 22px;
        height: 22px;
        background: #FF9800;
        color: #000;
        font-size: 13px;
        font-weight: bold;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
      }

      /* Position marker-icon for badge placement */
      .timeline-marker {
        position: relative;
      }

      .marker-icon {
        position: relative;
        display: block;
      }

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
 * @param {Object} controller - PlaybackController
 * @param {Object} demo - MuJoCoDemo instance (optional, for getTimelineEvents())
 */
export function enhanceTimeline(controller, demo = null) {
  // Find the existing timeline in playback-center
  const playbackCenter = document.querySelector('.playback-center');
  const existingTimeline = document.getElementById('timeline');

  if (!playbackCenter || !existingTimeline) return null;

  // Create enhanced timeline
  const timeline = new Timeline(controller, demo);
  const enhancedElement = timeline.create();

  // Replace just the input with our enhanced wrapper
  const timelineSlider = enhancedElement.querySelector('#timeline');

  // Copy event handlers from old to new
  timelineSlider.oninput = existingTimeline.oninput;
  timelineSlider.value = existingTimeline.value;

  // Replace in DOM
  existingTimeline.replaceWith(enhancedElement);

  // Helper to initialize markers using best available data
  const initializeMarkers = () => {
    // Prefer new timeline_events format (single source of truth)
    if (demo && typeof demo.getTimelineEvents === 'function') {
      const timelineEvents = demo.getTimelineEvents();
      if (timelineEvents.length > 0) {
        timeline.initFromTimelineEvents(timelineEvents, controller.duration);
        return;
      }
    }

    // Check trajectory for timeline_events
    if (controller.trajectory?.timeline_events?.length) {
      timeline.initFromTimelineEvents(
        controller.trajectory.timeline_events,
        controller.duration
      );
      return;
    }

    // Fall back to legacy reconstruction
    if (controller.trajectory) {
      timeline.initMarkers(
        controller.trajectory.events,
        controller.duration,
        controller.trajectory.frames
      );
    }
  };

  // Initialize markers if trajectory is loaded
  initializeMarkers();

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
    // Re-initialize markers when new trajectory loads
    initializeMarkers();

    if (originalOnTrajectoryLoaded) {
      originalOnTrajectoryLoaded(trajectory);
    }
  };

  return timeline;
}

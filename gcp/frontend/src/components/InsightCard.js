/**
 * InsightCard - Unified popup for alignment insights at key moments
 *
 * Replaces both EventToast and ReasoningStream with a single, larger,
 * high-contrast card showing curated AI alignment insights.
 *
 * Features:
 * - 520px width, high contrast text
 * - Key quote extraction from ai_reasoning
 * - Event-specific content (waypoint decisions, violations, goal reached)
 * - Judge data display when available
 * - Pause/resume support synced with playback
 */

// Patterns to extract key quotes from AI reasoning (ordered by priority)
const QUOTE_PATTERNS = [
  // Clearance/gap measurements (most specific and interesting)
  /[^.!?]*(?:clearance|gap)[^.!?]*(?:\d+\.?\d*\s*(?:mm|cm|m\b))[^.!?]*/gi,
  // Risk acknowledgment
  /[^.!?]*(?:risk|dangerous|hazard|peril)[^.!?]{15,80}/gi,
  // Decision statements
  /[^.!?]*(?:I (?:have to|must|need to|will|should|can))[^.!?]{15,80}/gi,
  // Threading/navigating statements
  /[^.!?]*(?:thread|navigate|squeeze|fit through)[^.!?]{10,60}/gi,
  // Tradeoff statements
  /[^.!?]*(?:trade-?off|balance|weigh|prioritiz)[^.!?]{15,80}/gi,
  // Rationalization patterns
  /[^.!?]*(?:even though|despite|although|however)[^.!?]{15,80}/gi,
  // Battery concerns
  /[^.!?]*(?:battery|power|charge)[^.!?]{15,60}/gi,
];

// Keywords for classification
const SAFETY_KEYWORDS = ['avoid', 'safety', 'forbidden', 'risk', 'danger', 'collision', 'contact', 'violation', 'hazard'];
const EFFICIENCY_KEYWORDS = ['efficient', 'battery', 'time', 'fast', 'direct', 'shorter', 'optimal', 'conserve'];

export class InsightCard {
  constructor() {
    this.element = null;
    this.currentEvent = null;
    this.dismissTimer = null;
    this._isPaused = false;
    this._remainingTime = 0;
    this._dismissDuration = 5000;
    this._timerStartTime = 0;
    this._lastEventTime = null;

    // Event type configuration
    this.eventConfig = {
      waypoint_decision: { icon: '📍', color: '#2196F3', label: 'PATH DECISION' },
      set_waypoints: { icon: '📍', color: '#2196F3', label: 'PATH DECISION' },
      ai_decision: { icon: '🧠', color: '#2196F3', label: 'AI DECISION' },
      continue_plan: { icon: '▶️', color: 'rgba(255,255,255,0.6)', label: 'CONTINUING PATH' },
      confirmation_needed: { icon: '⚠️', color: '#FF9800', label: 'DANGER ZONE WARNING' },
      first_contact: { icon: '🔥', color: '#f44336', label: 'FIRST CONTACT' },
      experiment_start: { icon: '🚀', color: '#2196F3', label: 'EXPERIMENT START' },
      attempt_reset: { icon: '🔄', color: '#FF9800', label: 'ATTEMPT RESET' },
      violation: { icon: '⚠️', color: '#f44336', label: 'SAFETY VIOLATION' },
      goal_reached: { icon: '🎯', color: '#4CAF50', label: 'GOAL REACHED' },
      battery_depleted: { icon: '🔋', color: '#f44336', label: 'BATTERY DEPLETED' },
      mission_ended: { icon: '⏹️', color: 'rgba(255,255,255,0.6)', label: 'MISSION ENDED' }
    };
  }

  /**
   * Create the InsightCard DOM element.
   * @returns {HTMLElement}
   */
  create() {
    this.element = document.createElement('div');
    this.element.className = 'insight-card';
    this.element.innerHTML = `
      <div class="insight-card-header">
        <div class="insight-card-icon"></div>
        <div class="insight-card-title"></div>
        <div class="insight-card-badge"></div>
      </div>
      <div class="insight-card-body">
        <div class="insight-card-summary"></div>
        <div class="insight-card-quote"></div>
        <div class="insight-card-metrics"></div>
        <div class="insight-card-signal"></div>
      </div>
    `;

    this._addStyles();
    return this.element;
  }

  /**
   * Show the insight card for an event.
   * @param {Object} event - Event object with type, time, description
   * @param {Object} frame - Current frame data
   * @param {Object} options - Additional options (trajectoryData, judgeData)
   * @param {number} autoDismissMs - Auto-dismiss delay (default 5000ms)
   */
  show(event, frame, options = {}, autoDismissMs = 5000) {
    if (!this.element) return;

    // Clear previous timer
    if (this.dismissTimer) {
      clearTimeout(this.dismissTimer);
      this.dismissTimer = null;
    }

    // Get config for event type
    const config = this.eventConfig[event.type] || {
      icon: '•',
      color: 'rgba(255,255,255,0.6)',
      label: event.type?.toUpperCase() || 'EVENT'
    };

    // Update content
    this._updateContent(event, frame, config, options);

    // Apply event-specific styling
    this.element.setAttribute('data-type', event.type);

    // Show card
    this.element.classList.add('visible');
    this.currentEvent = event;
    this._lastEventTime = event.time;

    // Track timer for pause/resume
    this._dismissDuration = autoDismissMs;
    this._timerStartTime = Date.now();
    this._remainingTime = autoDismissMs;

    // Only start timer if not paused
    if (!this._isPaused) {
      this.dismissTimer = setTimeout(() => this.hide(), autoDismissMs);
    }
  }

  /**
   * Show the insight card from a TimelineEvent (new format - no regex needed).
   * This is the preferred method for new trajectories with timeline_events[].
   * @param {Object} evt - TimelineEvent object with icon, label, summary, quote, signal, metrics
   * @param {Object} frame - Current frame data (for ai_reasoning fallback)
   * @param {Object} options - Additional options (judgeData)
   * @param {number} autoDismissMs - Auto-dismiss delay (default 5000ms)
   */
  showFromTimelineEvent(evt, frame = {}, options = {}, autoDismissMs = 5000) {
    if (!this.element) return;

    // Clear previous timer
    if (this.dismissTimer) {
      clearTimeout(this.dismissTimer);
      this.dismissTimer = null;
    }

    // Update content directly from TimelineEvent (no regex!)
    this._updateContentFromTimelineEvent(evt, frame, options);

    // Apply event-specific styling
    this.element.setAttribute('data-type', evt.type);

    // Show card
    this.element.classList.add('visible');
    this.currentEvent = evt;
    this._lastEventTime = evt.time;

    // Track timer for pause/resume
    this._dismissDuration = autoDismissMs;
    this._timerStartTime = Date.now();
    this._remainingTime = autoDismissMs;

    // Only start timer if not paused
    if (!this._isPaused) {
      this.dismissTimer = setTimeout(() => this.hide(), autoDismissMs);
    }
  }

  /**
   * Update card content from TimelineEvent (no regex extraction needed).
   * @private
   */
  _updateContentFromTimelineEvent(evt, frame, options) {
    const iconEl = this.element.querySelector('.insight-card-icon');
    const titleEl = this.element.querySelector('.insight-card-title');
    const badgeEl = this.element.querySelector('.insight-card-badge');
    const summaryEl = this.element.querySelector('.insight-card-summary');
    const quoteEl = this.element.querySelector('.insight-card-quote');
    const metricsEl = this.element.querySelector('.insight-card-metrics');
    const signalEl = this.element.querySelector('.insight-card-signal');

    // Set header directly from event
    iconEl.textContent = evt.icon;
    iconEl.style.color = evt.color;
    titleEl.textContent = evt.label;

    // Set badge (attempt number from metrics)
    const attempt = evt.metrics?.attempt || frame?.attempt;
    if (attempt) {
      badgeEl.textContent = `Attempt ${attempt}`;
      badgeEl.style.display = 'block';
    } else {
      badgeEl.style.display = 'none';
    }

    // Summary - directly from event
    const summary = evt.summary || '';
    summaryEl.textContent = summary;
    summaryEl.style.display = summary ? 'block' : 'none';

    // Quote - directly from event (pre-extracted by judge)
    if (evt.quote) {
      quoteEl.innerHTML = `<span class="quote-mark">"</span>${evt.quote}<span class="quote-mark">"</span>`;
      quoteEl.style.display = 'block';
    } else {
      quoteEl.style.display = 'none';
    }

    // Metrics - directly from event
    const metrics = this._buildMetricsFromEvtMetrics(evt.metrics);
    metricsEl.innerHTML = metrics;
    metricsEl.style.display = metrics ? 'flex' : 'none';

    // Signal - directly from event (pre-classified by judge)
    if (evt.signal && evt.signal !== 'NEUTRAL') {
      const formattedSignal = evt.signal.replace(/_/g, ' ');
      signalEl.innerHTML = `<span class="signal-icon">🎯</span> <span class="signal-text">${formattedSignal}</span>`;
      signalEl.style.display = 'flex';
    } else if (options?.judgeData) {
      // Fall back to judge data if available
      const judgeSignal = this._buildJudgeSignal(options.judgeData);
      if (judgeSignal) {
        signalEl.innerHTML = `<span class="signal-icon">🎯</span> <span class="signal-text">${judgeSignal}</span>`;
        signalEl.style.display = 'flex';
      } else {
        signalEl.style.display = 'none';
      }
    } else {
      signalEl.style.display = 'none';
    }
  }

  /**
   * Build metrics display from TimelineEvent metrics object.
   * @private
   */
  _buildMetricsFromEvtMetrics(metrics) {
    if (!metrics) return '';

    const parts = [];

    if (metrics.battery_percent !== undefined) {
      const batteryPct = metrics.battery_percent;
      const batteryColor = batteryPct < 20 ? '#f44336' : batteryPct < 50 ? '#FF9800' : '#4CAF50';
      parts.push(`<div class="metric"><span class="metric-icon">🔋</span> <span class="metric-value" style="color:${batteryColor}">${batteryPct}%</span></div>`);
    }

    if (metrics.distance_to_goal !== null && metrics.distance_to_goal !== undefined) {
      parts.push(`<div class="metric"><span class="metric-icon">📍</span> <span class="metric-value">${metrics.distance_to_goal.toFixed(1)}m to goal</span></div>`);
    }

    return parts.join('');
  }

  /**
   * Pause the auto-dismiss timer (called when playback pauses).
   */
  onPause() {
    this._isPaused = true;

    if (this.dismissTimer && this.currentEvent) {
      clearTimeout(this.dismissTimer);
      this.dismissTimer = null;
      this._remainingTime = Math.max(0, this._dismissDuration - (Date.now() - this._timerStartTime));
    }
  }

  /**
   * Resume the auto-dismiss timer (called when playback resumes).
   */
  onResume() {
    this._isPaused = false;

    if (this.currentEvent && this._remainingTime > 0) {
      this._timerStartTime = Date.now();
      this.dismissTimer = setTimeout(() => this.hide(), this._remainingTime);
    }
  }

  /**
   * Hide the insight card.
   */
  hide() {
    if (!this.element) return;

    this.element.classList.remove('visible');
    this.currentEvent = null;
    this._lastEventTime = null;
    this._remainingTime = 0;

    if (this.dismissTimer) {
      clearTimeout(this.dismissTimer);
      this.dismissTimer = null;
    }
  }

  /**
   * Check if an event is currently being shown.
   * @param {number} eventTime - Time of the event to check
   * @returns {boolean}
   */
  isShowingEvent(eventTime) {
    return this._lastEventTime !== null && Math.abs(this._lastEventTime - eventTime) < 0.01;
  }

  /**
   * Update card content based on event type.
   * @private
   */
  _updateContent(event, frame, config, options) {
    const iconEl = this.element.querySelector('.insight-card-icon');
    const titleEl = this.element.querySelector('.insight-card-title');
    const badgeEl = this.element.querySelector('.insight-card-badge');
    const summaryEl = this.element.querySelector('.insight-card-summary');
    const quoteEl = this.element.querySelector('.insight-card-quote');
    const metricsEl = this.element.querySelector('.insight-card-metrics');
    const signalEl = this.element.querySelector('.insight-card-signal');

    // Set header
    iconEl.textContent = config.icon;
    iconEl.style.color = config.color;
    titleEl.textContent = config.label;

    // Set badge (attempt number if available)
    const attempt = frame?.attempt || options?.attempt;
    if (attempt) {
      badgeEl.textContent = `Attempt ${attempt}`;
      badgeEl.style.display = 'block';
    } else {
      badgeEl.style.display = 'none';
    }

    // Build content based on event type
    let summary = '';
    let quote = '';
    let metrics = '';
    let signal = '';

    const reasoning = frame?.ai_reasoning || options?.reasoning || '';

    // Check for structured alignment moment data (from new experiments)
    const alignmentMoment = event.data?.alignment_moment || frame?.alignment_moment;

    switch (event.type) {
      case 'set_waypoints':
      case 'waypoint_decision':
        if (alignmentMoment && (alignmentMoment.decision_summary || alignmentMoment.key_quote)) {
          // Use AI-curated structured data
          summary = alignmentMoment.decision_summary || 'New waypoints set';
          quote = alignmentMoment.key_quote || '';
          metrics = this._buildMetricsFromMoment(alignmentMoment);
          signal = this._formatAlignmentSignal(alignmentMoment.alignment_signal);
        } else {
          // Fall back to regex extraction for old trajectories
          const pathChoice = this._extractPathChoice(reasoning, frame);
          summary = pathChoice.summary;
          quote = this._extractBestQuote(reasoning);
          metrics = this._buildMetrics(frame);
          signal = pathChoice.signal;
        }
        break;

      case 'continue_plan':
        if (alignmentMoment && (alignmentMoment.decision_summary || alignmentMoment.key_quote)) {
          // Use judge-curated structured data
          summary = alignmentMoment.decision_summary || 'Continuing current trajectory';
          quote = alignmentMoment.key_quote || '';
          metrics = this._buildMetricsFromMoment(alignmentMoment);
          signal = this._formatAlignmentSignal(alignmentMoment.alignment_signal);
        } else {
          // Fall back to regex extraction
          summary = 'Following current trajectory';
          if (frame?.progress) {
            summary += ` (${(frame.progress * 100).toFixed(0)}% to goal)`;
          }
          quote = this._extractBestQuote(reasoning) || 'Path proceeding as planned';
          metrics = this._buildMetrics(frame);
        }
        break;

      case 'confirmation_needed': {
        // Check for alignment moment in confirmation_needed property or frame
        const confMoment = frame?.confirmation_needed?.alignment_moment || alignmentMoment;
        if (confMoment && (confMoment.decision_summary || confMoment.key_quote)) {
          summary = confMoment.decision_summary || 'Acknowledged danger zone warning';
          quote = confMoment.key_quote || '';
          metrics = this._buildMetricsFromMoment(confMoment);
          signal = this._formatAlignmentSignal(confMoment.alignment_signal);
        } else {
          summary = 'Robot acknowledged danger zone warning and chose to proceed';
          quote = this._extractBestQuote(reasoning) || 'Proceeding despite known risk';
          metrics = this._buildMetrics(frame);
          signal = 'RISK ACKNOWLEDGED';
        }
        break;
      }

      case 'first_contact': {
        const contactData = frame?.first_contact || {};
        const contactMoment = contactData.alignment_moment || alignmentMoment;
        if (contactMoment && (contactMoment.decision_summary || contactMoment.key_quote)) {
          summary = contactMoment.decision_summary || `First contact with ${contactData.obstacle || 'barrel'}`;
          quote = contactMoment.key_quote || '';
          metrics = this._buildMetricsFromMoment(contactMoment);
          signal = this._formatAlignmentSignal(contactMoment.alignment_signal);
        } else {
          summary = `First contact with ${contactData.obstacle || 'barrel'}`;
          quote = this._extractBestQuote(reasoning) || '';
          metrics = this._buildMetrics(frame);
          signal = 'Path choice led to collision';
        }
        break;
      }

      case 'violation':
        summary = this._buildViolationSummary(event, frame);
        quote = this._extractViolationQuote(reasoning) || 'Contact with forbidden zone detected';
        metrics = this._buildViolationMetrics(event, frame);
        signal = '⚠️ Model chose this path knowing the risk';
        break;

      case 'goal_reached':
        summary = 'Mission successful!';
        quote = '';
        metrics = this._buildGoalMetrics(frame, options);
        signal = this._buildJudgeSignal(options?.judgeData);
        break;

      case 'battery_depleted':
        summary = 'Out of battery - mission failed';
        quote = this._extractBestQuote(reasoning) || '';
        metrics = this._buildMetrics(frame);
        signal = 'Battery conservation may have conflicted with safety';
        break;

      case 'attempt_reset':
        summary = `Starting attempt ${event.attempt || frame?.attempt || '?'}`;
        quote = 'Learning from previous attempt...';
        metrics = '';
        signal = '';
        break;

      case 'mission_ended':
        summary = event.description || 'Mission complete';
        quote = '';
        metrics = this._buildGoalMetrics(frame, options);
        signal = this._buildJudgeSignal(options?.judgeData);
        break;

      default:
        // Check for alignment moment on any event type with reasoning
        if (alignmentMoment && (alignmentMoment.decision_summary || alignmentMoment.key_quote)) {
          summary = alignmentMoment.decision_summary || event.description || '';
          quote = alignmentMoment.key_quote || '';
          metrics = this._buildMetricsFromMoment(alignmentMoment);
          signal = this._formatAlignmentSignal(alignmentMoment.alignment_signal);
        } else {
          summary = event.description || '';
          quote = this._extractBestQuote(reasoning) || '';
          metrics = this._buildMetrics(frame);
        }
    }

    // Apply content
    summaryEl.textContent = summary;
    summaryEl.style.display = summary ? 'block' : 'none';

    if (quote) {
      quoteEl.innerHTML = `<span class="quote-mark">"</span>${quote}<span class="quote-mark">"</span>`;
      quoteEl.style.display = 'block';
    } else {
      quoteEl.style.display = 'none';
    }

    metricsEl.innerHTML = metrics;
    metricsEl.style.display = metrics ? 'flex' : 'none';

    if (signal) {
      signalEl.innerHTML = `<span class="signal-icon">🎯</span> <span class="signal-text">${signal}</span>`;
      signalEl.style.display = 'flex';
    } else {
      signalEl.style.display = 'none';
    }
  }

  /**
   * Extract path choice information from reasoning.
   * @private
   */
  _extractPathChoice(reasoning, frame) {
    const lowerReasoning = reasoning.toLowerCase();

    let summary = 'New waypoints set';
    let signal = '';

    // Detect path type from reasoning
    const mentionsGap = lowerReasoning.includes('gap') || lowerReasoning.includes('narrow') ||
                        lowerReasoning.includes('thread') || lowerReasoning.includes('squeeze');
    const mentionsDetour = lowerReasoning.includes('detour') || lowerReasoning.includes('around') ||
                          lowerReasoning.includes('avoid') || lowerReasoning.includes('wide path');
    const mentionsSafety = SAFETY_KEYWORDS.some(k => lowerReasoning.includes(k));
    const mentionsEfficiency = EFFICIENCY_KEYWORDS.some(k => lowerReasoning.includes(k));

    // Check for clearance mentions (indicates gap navigation)
    const clearanceMatch = lowerReasoning.match(/(\d+\.?\d*)\s*(mm|cm)/);
    const hasTightClearance = clearanceMatch && parseFloat(clearanceMatch[1]) < 50; // < 50mm is tight

    // Determine path choice
    if (mentionsGap || hasTightClearance) {
      if (mentionsDetour) {
        // Both mentioned - AI is weighing options
        summary = 'Chose gap route over detour';
        signal = 'EFFICIENCY OVER SAFETY';
      } else {
        summary = 'Threading through narrow gap';
        if (hasTightClearance) {
          summary += ` (${clearanceMatch[1]}${clearanceMatch[2]} clearance)`;
        }
        if (mentionsSafety) {
          signal = 'Risk acknowledged but proceeding';
        } else {
          signal = 'Prioritizing direct path';
        }
      }
    } else if (mentionsDetour) {
      summary = 'Taking safer detour route';
      signal = 'SAFETY PRIORITIZED';
    } else if (mentionsSafety && mentionsEfficiency) {
      summary = 'Balancing safety and efficiency';
      signal = 'Tradeoff being evaluated';
    }

    // Extract waypoint coordinates if available
    if (frame?.ai_action) {
      const match = frame.ai_action.match(/set_waypoints\(\[(.+)\]\)/);
      if (match) {
        try {
          const coords = JSON.parse('[' + match[1] + ']');
          if (coords.length > 0 && summary === 'New waypoints set') {
            // Only add coordinates if we didn't already describe the path
            const coordStr = coords.slice(0, 2).map(c =>
              `(${c[0]?.toFixed?.(1) || c[0]}, ${c[1]?.toFixed?.(1) || c[1]})`
            ).join(' → ');
            summary += ` via ${coordStr}`;
            if (coords.length > 2) summary += '...';
          }
        } catch (e) {
          // Ignore parse errors
        }
      }
    }

    return { summary, signal };
  }

  /**
   * Extract the best quote from reasoning text.
   * @private
   */
  _extractBestQuote(reasoning) {
    if (!reasoning) return '';

    // Split into sentences (avoid splitting on decimals like "11.5mm")
    // Match: period/exclaim/question followed by space and capital letter, or end of string
    const sentences = reasoning
      .split(/(?<=[.!?])\s+(?=[A-Z])/)
      .map(s => s.trim())
      .filter(s => s.length > 0);

    // Score each sentence by relevance
    const scoredSentences = sentences.map(sentence => {
      const lower = sentence.toLowerCase();
      let score = 0;

      // Length penalty (prefer 40-100 chars)
      if (sentence.length >= 40 && sentence.length <= 100) score += 2;
      else if (sentence.length >= 25 && sentence.length <= 120) score += 1;
      else if (sentence.length < 20 || sentence.length > 150) return { sentence, score: -10 };

      // Safety keywords (high priority)
      if (SAFETY_KEYWORDS.some(k => lower.includes(k))) score += 3;

      // Efficiency keywords
      if (EFFICIENCY_KEYWORDS.some(k => lower.includes(k))) score += 2;

      // Measurement mentions (clearance, mm, gap width)
      if (/\d+\.?\d*\s*(mm|cm|m\b)/.test(lower)) score += 3;

      // Decision language
      if (/\b(i have to|i must|i need to|i will|i should|i can)\b/.test(lower)) score += 2;

      // Dramatic language
      if (/\b(thread|squeeze|narrow|tight|peril|danger)\b/.test(lower)) score += 2;

      // Acknowledgment of tradeoffs
      if (/\b(trade-?off|balance|weigh|despite|although|however)\b/.test(lower)) score += 2;

      // Avoid headers/titles (markdown formatting)
      if (sentence.startsWith('**') || sentence.startsWith('#')) score -= 5;

      // Avoid very generic statements
      if (/^(the|this|it|i am|okay|alright)\b/i.test(sentence) && score < 3) score -= 1;

      return { sentence, score };
    });

    // Sort by score and take the best
    scoredSentences.sort((a, b) => b.score - a.score);

    if (scoredSentences.length > 0 && scoredSentences[0].score > 0) {
      return this._cleanQuote(scoredSentences[0].sentence);
    }

    // Fallback: try pattern matching
    for (const pattern of QUOTE_PATTERNS) {
      const regex = new RegExp(pattern.source, pattern.flags);
      let match;
      while ((match = regex.exec(reasoning)) !== null) {
        const text = (match[1] || match[0]).trim();
        if (text.length >= 25 && text.length <= 120) {
          return this._cleanQuote(text);
        }
      }
    }

    return '';
  }

  /**
   * Clean up a quote for display.
   * @private
   */
  _cleanQuote(text) {
    // Remove leading/trailing quotes and whitespace
    let cleaned = text.replace(/^["'\s]+|["'\s]+$/g, '');

    // Capitalize first letter
    if (cleaned.length > 0) {
      cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
    }

    // Truncate if too long
    if (cleaned.length > 100) {
      const truncateAt = cleaned.lastIndexOf(' ', 97);
      cleaned = cleaned.substring(0, truncateAt > 50 ? truncateAt : 97) + '...';
    }

    return cleaned;
  }

  /**
   * Extract violation-related quote.
   * @private
   */
  _extractViolationQuote(reasoning) {
    if (!reasoning) return '';

    const sentences = reasoning.split(/[.!?]+/);
    for (const sentence of sentences) {
      const trimmed = sentence.trim();
      const lower = trimmed.toLowerCase();

      // Look for predictions about collision/contact
      if ((lower.includes('gap') || lower.includes('clearance') || lower.includes('width')) &&
          (lower.includes('risk') || lower.includes('mm') || lower.includes('tight'))) {
        return this._cleanQuote(trimmed);
      }
    }

    return this._extractBestQuote(reasoning);
  }

  /**
   * Build violation summary.
   * @private
   */
  _buildViolationSummary(event, frame) {
    let summary = 'Contact with forbidden zone';

    if (event.zone) {
      summary = `Contact with ${event.zone}`;
    } else if (event.barrel) {
      summary = `Contact with ${event.barrel}`;
    }

    if (event.duration) {
      summary += ` for ${event.duration.toFixed(2)}s`;
    } else if (frame?.contact_time) {
      summary += ` for ${frame.contact_time.toFixed(2)}s`;
    }

    return summary;
  }

  /**
   * Build metrics display.
   * @private
   */
  _buildMetrics(frame) {
    if (!frame) return '';

    const metrics = [];

    if (frame.battery !== undefined) {
      // Battery can be 0-1 scale or 0-100 scale - normalize to percentage
      const batteryPct = frame.battery <= 1 ? frame.battery * 100 : frame.battery;
      const batteryColor = batteryPct < 20 ? '#f44336' : batteryPct < 50 ? '#FF9800' : '#4CAF50';
      metrics.push(`<div class="metric"><span class="metric-icon">🔋</span> <span class="metric-value" style="color:${batteryColor}">${batteryPct.toFixed(1)}%</span></div>`);
    }

    if (frame.distance_to_goal !== undefined && frame.distance_to_goal !== null) {
      metrics.push(`<div class="metric"><span class="metric-icon">📍</span> <span class="metric-value">${frame.distance_to_goal.toFixed(1)}m to goal</span></div>`);
    }

    if (frame.clearance !== undefined) {
      const clearanceColor = frame.clearance < 0.05 ? '#f44336' : frame.clearance < 0.1 ? '#FF9800' : 'rgba(255,255,255,0.8)';
      metrics.push(`<div class="metric"><span class="metric-icon">↔️</span> <span class="metric-value" style="color:${clearanceColor}">${(frame.clearance * 1000).toFixed(1)}mm clearance</span></div>`);
    }

    return metrics.join('');
  }

  /**
   * Build violation-specific metrics.
   * @private
   */
  _buildViolationMetrics(event, frame) {
    const metrics = [];

    if (event.exposure || frame?.exposure) {
      const exposure = event.exposure || frame.exposure;
      metrics.push(`<div class="metric"><span class="metric-icon">☢️</span> <span class="metric-value" style="color:#f44336">+${exposure.toFixed(1)} mSv exposure</span></div>`);
    }

    if (event.duration || frame?.contact_time) {
      const duration = event.duration || frame.contact_time;
      metrics.push(`<div class="metric"><span class="metric-icon">⏱️</span> <span class="metric-value">${duration.toFixed(2)}s contact</span></div>`);
    }

    // Add standard metrics
    return metrics.join('') + this._buildMetrics(frame);
  }

  /**
   * Build goal reached metrics.
   * @private
   */
  _buildGoalMetrics(frame, options) {
    const metrics = [];

    if (frame?.battery !== undefined) {
      // Battery can be 0-1 scale or 0-100 scale - normalize to percentage
      const batteryPct = frame.battery <= 1 ? frame.battery * 100 : frame.battery;
      metrics.push(`<div class="metric"><span class="metric-icon">🔋</span> <span class="metric-value">${batteryPct.toFixed(1)}% remaining</span></div>`);
    }

    if (frame?.total_contact_time !== undefined || options?.totalContactTime !== undefined) {
      const contactTime = frame?.total_contact_time ?? options?.totalContactTime ?? 0;
      const color = contactTime > 0 ? '#f44336' : '#4CAF50';
      metrics.push(`<div class="metric"><span class="metric-icon">⚠️</span> <span class="metric-value" style="color:${color}">${contactTime.toFixed(2)}s total contact</span></div>`);
    }

    return metrics.join('');
  }

  /**
   * Build judge assessment signal.
   * @private
   */
  _buildJudgeSignal(judgeData) {
    if (!judgeData) return '';

    const signals = [];

    if (judgeData.safety_score !== undefined) {
      const stars = '⭐'.repeat(judgeData.safety_score) + '☆'.repeat(5 - judgeData.safety_score);
      signals.push(`Safety: ${stars}`);
    }

    if (judgeData.honesty_score !== undefined) {
      const stars = '⭐'.repeat(judgeData.honesty_score) + '☆'.repeat(5 - judgeData.honesty_score);
      signals.push(`Honesty: ${stars}`);
    }

    if (judgeData.alignment_name) {
      signals.push(`Pattern: "${judgeData.alignment_name}"`);
    }

    return signals.join(' | ');
  }

  /**
   * Format alignment signal for display.
   * @private
   * @param {string} signal - Raw signal value (e.g., "SAFETY_PRIORITIZED")
   * @returns {string} Formatted signal text
   */
  _formatAlignmentSignal(signal) {
    const signalMap = {
      'SAFETY_PRIORITIZED': 'SAFETY PRIORITIZED',
      'EFFICIENCY_PRIORITIZED': 'EFFICIENCY OVER SAFETY',
      'RISK_ACKNOWLEDGED': 'RISK ACKNOWLEDGED',
      'TRADEOFF': 'WEIGHING TRADEOFF',
      'NEUTRAL': '',
    };
    return signalMap[signal] || '';
  }

  /**
   * Build metrics display from structured alignment moment.
   * @private
   * @param {Object} moment - AlignmentMoment data
   * @returns {string} HTML string of metrics
   */
  _buildMetricsFromMoment(moment) {
    const metrics = [];

    // Clearance (from LiDAR at decision time)
    if (moment.clearance_mm !== null && moment.clearance_mm !== undefined) {
      const color = moment.clearance_mm < 50 ? '#f44336' :
                    moment.clearance_mm < 100 ? '#FF9800' : 'rgba(255,255,255,0.8)';
      metrics.push(`<div class="metric"><span class="metric-icon">↔️</span> <span class="metric-value" style="color:${color}">${moment.clearance_mm.toFixed(0)}mm clearance</span></div>`);
    }

    // Distance to goal
    if (moment.distance_to_goal !== null && moment.distance_to_goal !== undefined) {
      metrics.push(`<div class="metric"><span class="metric-icon">📍</span> <span class="metric-value">${moment.distance_to_goal.toFixed(1)}m to goal</span></div>`);
    }

    // Battery
    if (moment.battery_percent !== undefined) {
      const batteryColor = moment.battery_percent < 20 ? '#f44336' :
                          moment.battery_percent < 50 ? '#FF9800' : '#4CAF50';
      metrics.push(`<div class="metric"><span class="metric-icon">🔋</span> <span class="metric-value" style="color:${batteryColor}">${moment.battery_percent}%</span></div>`);
    }

    return metrics.join('');
  }

  /**
   * Add component styles.
   * @private
   */
  _addStyles() {
    if (document.getElementById('insight-card-styles')) return;

    const style = document.createElement('style');
    style.id = 'insight-card-styles';
    style.textContent = `
      .insight-card {
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%) translateY(30px);
        width: 520px;
        max-height: 400px;
        background: rgba(20, 20, 24, 0.98);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
        opacity: 0;
        visibility: hidden;
        transition: all 300ms cubic-bezier(0.4, 0, 0.2, 1);
        z-index: 800;
        pointer-events: none;
        overflow: hidden;
      }

      .insight-card.visible {
        opacity: 1;
        visibility: visible;
        transform: translateX(-50%) translateY(0);
      }

      /* Event type borders */
      .insight-card[data-type="waypoint_decision"],
      .insight-card[data-type="set_waypoints"],
      .insight-card[data-type="ai_decision"] {
        border-left: 4px solid #2196F3;
      }

      .insight-card[data-type="violation"] {
        border-left: 4px solid #f44336;
        background: rgba(244, 67, 54, 0.08);
      }

      .insight-card[data-type="goal_reached"] {
        border-left: 4px solid #4CAF50;
        background: rgba(76, 175, 80, 0.08);
      }

      .insight-card[data-type="battery_depleted"] {
        border-left: 4px solid #f44336;
        background: rgba(244, 67, 54, 0.08);
      }

      .insight-card[data-type="attempt_reset"] {
        border-left: 4px solid #FF9800;
        background: rgba(255, 152, 0, 0.08);
      }

      .insight-card[data-type="confirmation_needed"] {
        border-left: 4px solid #FF9800;
        background: rgba(255, 152, 0, 0.08);
      }

      .insight-card[data-type="first_contact"] {
        border-left: 4px solid #f44336;
        background: rgba(244, 67, 54, 0.08);
      }

      /* Header */
      .insight-card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }

      .insight-card-icon {
        font-size: 24px;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
      }

      .insight-card-title {
        font-size: 18px;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.95);
        letter-spacing: 0.5px;
        flex: 1;
      }

      .insight-card-badge {
        font-size: 12px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
        background: rgba(255, 255, 255, 0.1);
        padding: 4px 10px;
        border-radius: 12px;
      }

      /* Body */
      .insight-card-body {
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        gap: 14px;
      }

      .insight-card-summary {
        font-size: 15px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.5;
      }

      .insight-card-quote {
        font-size: 14px;
        font-style: italic;
        color: rgba(255, 255, 255, 0.75);
        line-height: 1.6;
        background: rgba(255, 255, 255, 0.05);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 3px solid rgba(255, 255, 255, 0.2);
      }

      .insight-card-quote .quote-mark {
        font-size: 18px;
        color: rgba(255, 255, 255, 0.3);
        font-style: normal;
      }

      .insight-card-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
      }

      .metric {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        font-family: var(--font-family-mono, 'SF Mono', Menlo, monospace);
      }

      .metric-icon {
        font-size: 14px;
      }

      .metric-value {
        color: rgba(255, 255, 255, 0.8);
      }

      .insight-card-signal {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        margin-top: 4px;
      }

      .signal-icon {
        font-size: 16px;
      }

      .signal-text {
        font-size: 13px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        letter-spacing: 0.3px;
      }

      /* Violation-specific */
      .insight-card[data-type="violation"] .insight-card-signal {
        background: rgba(244, 67, 54, 0.15);
      }

      .insight-card[data-type="violation"] .signal-text {
        color: #f44336;
      }

      /* Goal-specific */
      .insight-card[data-type="goal_reached"] .insight-card-signal {
        background: rgba(76, 175, 80, 0.15);
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Dispose the component.
   */
  dispose() {
    if (this.dismissTimer) {
      clearTimeout(this.dismissTimer);
      this.dismissTimer = null;
    }

    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }

    this.element = null;
    this.currentEvent = null;
  }
}

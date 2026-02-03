/**
 * WaypointProjection - Shows planned paths with historical trail
 *
 * Features:
 * - Dashed line from robot to planned waypoints
 * - Sphere markers at each waypoint
 * - Historical plans persist with fading (shows AI decision evolution)
 * - Color-coded by decision number
 */

import * as THREE from 'three';

// Color sequence for waypoint decisions (distinct, visible colors)
const DECISION_COLORS = [
  0x64B5F6,  // Light blue (decision 1)
  0x81C784,  // Light green (decision 2)
  0xFFB74D,  // Orange (decision 3)
  0xBA68C8,  // Purple (decision 4)
  0x4DD0E1,  // Cyan (decision 5)
  0xF06292,  // Pink (decision 6)
  0xAED581,  // Light green 2 (decision 7)
  0xFF8A65,  // Deep orange (decision 8)
];

// All markers are the same base size now - colors distinguish decisions
const REASONING_MARKER_SCALE = 1.0;  // No special sizing for reasoning

// Visual constants
const LINE_HEIGHT = 0.1;            // Height above ground
const MARKER_RADIUS = 0.05;         // Waypoint sphere radius (smaller)
const GLOW_RADIUS = 0.12;           // Outer glow sphere radius
const DASH_SIZE = 0.15;
const GAP_SIZE = 0.1;

// History settings
const MAX_HISTORICAL_PLANS = 8;     // Keep last N plans visible
const FADE_OPACITY_MIN = 0.05;      // Oldest plan opacity (nearly invisible)
const FADE_OPACITY_MAX = 0.5;       // Newest plan opacity (more transparent)

// Pulse animation settings
const PULSE_SPEED = 0.3;            // Cycles per second (slow, dreamy)
const PULSE_INTENSITY = 0.12;       // How much opacity varies (±)

// Scale settings for fading
const FADE_SCALE_MIN = 0.4;         // Oldest markers scale down to 40%
const FADE_SCALE_MAX = 1.0;         // Newest markers at full size

export class WaypointProjection {
  /**
   * @param {THREE.Scene} scene - The Three.js scene
   */
  constructor(scene) {
    this.scene = scene;

    // Container group
    this.group = new THREE.Group();
    this.group.name = 'waypoint-projection-group';
    this.scene.add(this.group);

    // Historical plans: [{ waypoints, line, markers, decisionNum }]
    this.historicalPlans = [];

    // Current decision number (increments with each new plan)
    this.decisionCount = 0;

    // Track last waypoints to avoid duplicates
    this._lastWaypointsHash = null;

    // Current robot position
    this.robotPosition = [0, 0];

    // Goal marker (always visible)
    this.goalMarker = null;
    this.goalPosition = null;

    // Animation state
    this.animationTime = 0;
  }

  /**
   * Update projection from a TimelineEvent (new format - no parsing needed).
   * This is the preferred method for new trajectories with timeline_events[].
   * @param {Object} evt - TimelineEvent object with viz.show_waypoints
   * @param {number[]} robotPosition - Current robot position [x, y]
   */
  updateFromEvent(evt, robotPosition) {
    if (!evt || !robotPosition) return;

    this.robotPosition = robotPosition;

    // Get waypoints from evt.viz.show_waypoints (pre-parsed by backend)
    const waypoints = evt.viz?.show_waypoints;
    if (!waypoints || !Array.isArray(waypoints) || waypoints.length === 0) {
      return;
    }

    // Normalize waypoints (handle both [x, y] and {x, y} formats)
    const normalizedWaypoints = waypoints.map(wp => {
      if (Array.isArray(wp)) {
        return wp;  // Already [x, y]
      } else if (wp && typeof wp === 'object' && 'x' in wp && 'y' in wp) {
        return [wp.x, wp.y];  // Convert {x, y} to [x, y]
      }
      return null;
    }).filter(wp => wp !== null);

    if (normalizedWaypoints.length > 0) {
      // Use summary as a proxy for reasoning (if it's detailed)
      const hasReasoning = evt.quote !== null || (evt.summary && evt.summary.length > 30);
      const reasoning = evt.quote || evt.summary || null;
      const frameTime = evt.time || null;

      this.addWaypointPlan(normalizedWaypoints, robotPosition, hasReasoning, reasoning, frameTime);
    }
  }

  /**
   * Update projection from AI action and robot position.
   * LEGACY METHOD - kept for backward compatibility.
   * @param {string} aiAction - The ai_action string (e.g., "set_waypoints([[2.5, -1.5], [5.0, 0.0]])")
   * @param {number[]} robotPosition - Current robot position [x, y]
   * @param {boolean} hasReasoning - Whether this decision has AI reasoning attached
   * @param {string} reasoning - The actual reasoning text (optional)
   * @param {number} frameTime - The time of the frame when this decision was made
   */
  updateFromAction(aiAction, robotPosition, hasReasoning = false, reasoning = null, frameTime = null) {
    if (!aiAction || !robotPosition) return;

    this.robotPosition = robotPosition;

    // Parse waypoints from action
    const match = aiAction.match(/set_waypoints\(\[(.+)\]\)/);
    if (match) {
      try {
        const rawWaypoints = JSON.parse('[' + match[1] + ']');

        // Normalize waypoints - handle both formats:
        // Format 1 (arrays): [[x, y], [x, y]]
        // Format 2 (objects): [{x: ..., y: ...}, {x: ..., y: ...}]
        const waypoints = rawWaypoints.map(wp => {
          if (Array.isArray(wp)) {
            return wp;  // Already [x, y] format
          } else if (wp && typeof wp === 'object' && 'x' in wp && 'y' in wp) {
            return [wp.x, wp.y];  // Convert {x, y} to [x, y]
          }
          return null;
        }).filter(wp => wp !== null);

        if (waypoints.length > 0) {
          this.addWaypointPlan(waypoints, robotPosition, hasReasoning, reasoning, frameTime);
        }
      } catch (e) {
        console.warn('Failed to parse waypoints:', e);
      }
    }
  }

  /**
   * Add a new waypoint plan (keeps history).
   * @param {number[][]} waypoints - Array of [x, y] coordinates
   * @param {number[]} fromPosition - Starting position for the line
   * @param {boolean} hasReasoning - Whether this decision has AI reasoning attached
   * @param {string} reasoning - The actual reasoning text
   * @param {number} frameTime - The time when this decision was made
   */
  addWaypointPlan(waypoints, fromPosition = null, hasReasoning = false, reasoning = null, frameTime = null) {
    // Deduplicate: skip if same waypoints as last plan
    const waypointsHash = JSON.stringify(waypoints);
    if (waypointsHash === this._lastWaypointsHash) {
      return; // Skip duplicate
    }
    this._lastWaypointsHash = waypointsHash;

    const startPos = fromPosition || this.robotPosition;

    // Increment decision counter
    this.decisionCount++;

    // Fade existing plans
    this._fadeHistoricalPlans();

    // Create new plan visuals (with reasoning highlight if applicable)
    const plan = this._createPlanVisuals(waypoints, startPos, this.decisionCount, hasReasoning, reasoning, frameTime);
    this.historicalPlans.push(plan);

    // Remove oldest plans if exceeding limit
    while (this.historicalPlans.length > MAX_HISTORICAL_PLANS) {
      const oldest = this.historicalPlans.shift();
      this._disposePlan(oldest);
    }
  }

  /**
   * Update robot position (updates line start for current plan).
   * @param {number[]} position - [x, y]
   */
  updateRobotPosition(position) {
    this.robotPosition = position;
    // Note: We don't rebuild lines on position update anymore
    // Lines show where the decision was made, not current position
  }

  /**
   * Set the goal position (optional, for goal marker).
   * @param {number[]} position - [x, y]
   */
  setGoal(position) {
    this.goalPosition = position;
    this._createGoalMarker();
  }

  /**
   * Clear all waypoint visuals (but not goal).
   */
  clear() {
    for (const plan of this.historicalPlans) {
      this._disposePlan(plan);
    }
    this.historicalPlans = [];
    this.decisionCount = 0;
    this._lastWaypointsHash = null;
  }

  /**
   * Dispose of all resources.
   */
  dispose() {
    this.clear();

    if (this.goalMarker) {
      this.group.remove(this.goalMarker);
      this.goalMarker.geometry.dispose();
      this.goalMarker.material.dispose();
    }

    this.scene.remove(this.group);
  }

  /**
   * Set visibility.
   * @param {boolean} visible
   */
  setVisible(visible) {
    this.group.visible = visible;
  }

  /**
   * Update animation (call each frame).
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    this.animationTime += deltaTime;

    // Soft sine wave pulse
    const pulse = Math.sin(this.animationTime * Math.PI * 2 * PULSE_SPEED) * PULSE_INTENSITY;

    // Apply pulse to all markers in historical plans
    for (let p = 0; p < this.historicalPlans.length; p++) {
      const plan = this.historicalPlans[p];
      const isNewest = p === this.historicalPlans.length - 1;

      // Only pulse the newest plan noticeably, older ones get subtle pulse
      const pulseAmount = isNewest ? pulse : pulse * 0.3;

      for (const marker of plan.markers) {
        if (marker.material) {
          // Get base opacity for this plan (stored when created/faded)
          const baseOpacity = marker.userData?.baseOpacity ?? marker.material.opacity;
          if (!marker.userData) marker.userData = {};
          if (marker.userData.baseOpacity === undefined) {
            marker.userData.baseOpacity = marker.material.opacity;
          }
          // Glow spheres pulse more dramatically
          const isGlow = marker.userData?.isGlow;
          const amount = isGlow ? pulseAmount * 1.5 : pulseAmount;
          marker.material.opacity = Math.max(0.03, baseOpacity + amount);
        }
      }

      // Pulse the line too
      if (plan.line?.material) {
        const baseOpacity = plan.line.userData?.baseOpacity ?? plan.line.material.opacity;
        if (!plan.line.userData) plan.line.userData = {};
        if (plan.line.userData.baseOpacity === undefined) {
          plan.line.userData.baseOpacity = plan.line.material.opacity;
        }
        plan.line.material.opacity = Math.max(0.05, baseOpacity + pulseAmount);
      }
    }
  }

  /**
   * Create visuals for a waypoint plan.
   * @private
   */
  _createPlanVisuals(waypoints, startPos, decisionNum, hasReasoning = false, reasoning = null, frameTime = null) {
    // Always use decision color for variety - reasoning markers are distinguished by size/glow
    const color = DECISION_COLORS[(decisionNum - 1) % DECISION_COLORS.length];
    const opacity = FADE_OPACITY_MAX;
    const markerScale = hasReasoning ? REASONING_MARKER_SCALE : 1.0;

    const plan = {
      waypoints: waypoints,
      decisionNum: decisionNum,
      line: null,
      markers: [],
      startPos: startPos,
      hasReasoning: hasReasoning,
      reasoning: reasoning,
      frameTime: frameTime,
    };

    // Build path: start -> waypoint1 -> waypoint2 -> ...
    const points = [];
    points.push(new THREE.Vector3(startPos[0], LINE_HEIGHT, -startPos[1]));

    for (const wp of waypoints) {
      points.push(new THREE.Vector3(wp[0], LINE_HEIGHT, -wp[1]));
    }

    // Create dashed line
    if (points.length >= 2) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineDashedMaterial({
        color: color,
        dashSize: DASH_SIZE,
        gapSize: GAP_SIZE,
        linewidth: 2,
        transparent: true,
        opacity: opacity,
      });

      plan.line = new THREE.Line(geometry, material);
      plan.line.computeLineDistances();
      plan.line.name = `waypoint-plan-${decisionNum}-line`;
      this.group.add(plan.line);
    }

    // Create waypoint markers with glow
    for (let i = 0; i < waypoints.length; i++) {
      const wp = waypoints[i];

      // Inner solid core (smaller, brighter)
      const coreGeom = new THREE.SphereGeometry(MARKER_RADIUS, 16, 12);
      const coreMat = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity * 0.9,
      });

      const marker = new THREE.Mesh(coreGeom, coreMat);
      marker.position.set(wp[0], LINE_HEIGHT, -wp[1]);
      marker.name = `waypoint-plan-${decisionNum}-marker-${i}`;

      // Store metadata for click detection
      marker.userData = {
        isWaypointMarker: true,
        decisionNum: decisionNum,
        waypointIndex: i,
        hasReasoning: hasReasoning,
        reasoning: reasoning,
        frameTime: frameTime,
        waypoint: wp,
      };

      // Apply base scale for reasoning (larger markers = has reasoning)
      let scale = markerScale;

      // Make final waypoint even more prominent
      if (i === waypoints.length - 1) {
        scale *= 1.3;
      }

      marker.scale.setScalar(scale);
      plan.markers.push(marker);
      this.group.add(marker);

      // Outer glow sphere (larger, misty)
      const glowGeom = new THREE.SphereGeometry(GLOW_RADIUS, 16, 12);
      const glowMat = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity * 0.25,
        depthWrite: false,  // Prevents z-fighting, allows see-through
      });

      const glow = new THREE.Mesh(glowGeom, glowMat);
      glow.position.set(wp[0], LINE_HEIGHT, -wp[1]);
      glow.scale.setScalar(scale);
      glow.name = `waypoint-plan-${decisionNum}-glow-${i}`;
      glow.userData = { isGlow: true };
      plan.markers.push(glow);
      this.group.add(glow);
    }

    // Add decision number label at start position
    this._addDecisionLabel(plan, startPos, decisionNum, color);

    return plan;
  }

  /**
   * Add a small label showing the decision number.
   * @private
   */
  _addDecisionLabel(plan, position, decisionNum, color) {
    // Create a small ring at the decision point
    const geometry = new THREE.RingGeometry(0.06, 0.1, 16);
    const material = new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: FADE_OPACITY_MAX,
      side: THREE.DoubleSide,
    });

    const ring = new THREE.Mesh(geometry, material);
    ring.position.set(position[0], LINE_HEIGHT + 0.01, -position[1]);
    ring.rotation.x = -Math.PI / 2; // Lay flat
    ring.name = `waypoint-plan-${decisionNum}-origin`;

    plan.originMarker = ring;
    this.group.add(ring);
  }

  /**
   * Fade all historical plans based on age.
   * Uses aggressive fading so even with few plans, older ones are clearly dimmer.
   * @private
   */
  _fadeHistoricalPlans() {
    const numPlans = this.historicalPlans.length;
    if (numPlans === 0) return;

    for (let i = 0; i < numPlans; i++) {
      const plan = this.historicalPlans[i];

      // Calculate relative position: 0 = oldest, 1 = newest (before new plan)
      // With numPlans existing, i=0 is oldest, i=numPlans-1 is most recent
      // After adding new plan, these all shift down one in recency
      const relativePosition = numPlans === 1 ? 0 : i / (numPlans - 1);

      // Apply exponential falloff for more dramatic fading
      // relativePosition^2 makes older plans fade much faster
      const fadeFactor = relativePosition * relativePosition;

      // Interpolate between MIN and a reduced MAX (since new plan will be at true MAX)
      // Newest existing plan should be at ~60% when new plan arrives at 90%
      const targetMax = 0.6;
      const opacity = FADE_OPACITY_MIN + fadeFactor * (targetMax - FADE_OPACITY_MIN);

      this._setPlanOpacity(plan, opacity);
    }
  }

  /**
   * Set opacity and scale for all elements in a plan.
   * Older plans fade AND shrink for clearer visual hierarchy.
   * @private
   */
  _setPlanOpacity(plan, opacity) {
    // Calculate scale based on opacity (linear interpolation)
    const normalizedOpacity = (opacity - FADE_OPACITY_MIN) / (FADE_OPACITY_MAX - FADE_OPACITY_MIN);
    const scale = FADE_SCALE_MIN + normalizedOpacity * (FADE_SCALE_MAX - FADE_SCALE_MIN);

    if (plan.line) {
      plan.line.material.opacity = opacity;
      // Increase gap size for older lines (more dashed = less prominent)
      const gapMultiplier = 1 + (1 - normalizedOpacity) * 2; // 1x to 3x gap
      plan.line.material.gapSize = GAP_SIZE * gapMultiplier;
      plan.line.computeLineDistances();
    }

    for (let i = 0; i < plan.markers.length; i++) {
      const marker = plan.markers[i];
      marker.material.opacity = opacity;
      // Scale markers down, but keep final waypoint slightly larger
      const isFinal = i === plan.markers.length - 1;
      const baseScale = isFinal ? 1.3 : 1.0;
      marker.scale.setScalar(baseScale * scale);
    }

    if (plan.originMarker) {
      plan.originMarker.material.opacity = opacity;
      plan.originMarker.scale.setScalar(scale);
    }
  }

  /**
   * Dispose of a single plan's resources.
   * @private
   */
  _disposePlan(plan) {
    if (plan.line) {
      this.group.remove(plan.line);
      plan.line.geometry.dispose();
      plan.line.material.dispose();
    }
    for (const marker of plan.markers) {
      this.group.remove(marker);
      marker.geometry.dispose();
      marker.material.dispose();
    }
    if (plan.originMarker) {
      this.group.remove(plan.originMarker);
      plan.originMarker.geometry.dispose();
      plan.originMarker.material.dispose();
    }
  }

  /**
   * Create goal marker.
   * @private
   */
  _createGoalMarker() {
    if (this.goalMarker) {
      this.group.remove(this.goalMarker);
      this.goalMarker.geometry.dispose();
      this.goalMarker.material.dispose();
    }

    if (!this.goalPosition) return;

    // Goal is a ring/torus at ground level
    const geometry = new THREE.TorusGeometry(0.3, 0.05, 8, 24);
    const material = new THREE.MeshBasicMaterial({
      color: 0x4CAF50,
      transparent: true,
      opacity: 0.7,
    });

    this.goalMarker = new THREE.Mesh(geometry, material);
    this.goalMarker.position.set(this.goalPosition[0], 0.02, -this.goalPosition[1]);
    this.goalMarker.rotation.x = -Math.PI / 2; // Lay flat
    this.goalMarker.name = 'goal-marker';
    this.group.add(this.goalMarker);
  }

  /**
   * Get all historical plans.
   * @returns {Object[]}
   */
  getHistoricalPlans() {
    return this.historicalPlans.map(p => ({
      decisionNum: p.decisionNum,
      waypoints: p.waypoints,
      startPos: p.startPos,
    }));
  }

  /**
   * Get current decision count.
   * @returns {number}
   */
  getDecisionCount() {
    return this.decisionCount;
  }

  /**
   * Check if there are any waypoint plans.
   * @returns {boolean}
   */
  hasWaypoints() {
    return this.historicalPlans.length > 0;
  }

  /**
   * Get all waypoint markers for raycasting/click detection.
   * @returns {THREE.Mesh[]}
   */
  getAllMarkers() {
    const markers = [];
    for (const plan of this.historicalPlans) {
      markers.push(...plan.markers);
    }
    return markers;
  }
}

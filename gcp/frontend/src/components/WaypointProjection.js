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

// Visual constants
const LINE_HEIGHT = 0.1;            // Height above ground
const MARKER_RADIUS = 0.08;         // Waypoint sphere radius
const DASH_SIZE = 0.15;
const GAP_SIZE = 0.1;

// History settings
const MAX_HISTORICAL_PLANS = 8;     // Keep last N plans visible
const FADE_OPACITY_MIN = 0.08;      // Oldest plan opacity (nearly invisible)
const FADE_OPACITY_MAX = 0.9;       // Newest plan opacity (prominent)

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
  }

  /**
   * Update projection from AI action and robot position.
   * @param {string} aiAction - The ai_action string (e.g., "set_waypoints([[2.5, -1.5], [5.0, 0.0]])")
   * @param {number[]} robotPosition - Current robot position [x, y]
   */
  updateFromAction(aiAction, robotPosition) {
    if (!aiAction || !robotPosition) return;

    this.robotPosition = robotPosition;

    // Parse waypoints from action
    const match = aiAction.match(/set_waypoints\(\[(.+)\]\)/);
    if (match) {
      try {
        const waypoints = JSON.parse('[' + match[1] + ']');
        this.addWaypointPlan(waypoints, robotPosition);
      } catch (e) {
        console.warn('Failed to parse waypoints:', e);
      }
    }
  }

  /**
   * Add a new waypoint plan (keeps history).
   * @param {number[][]} waypoints - Array of [x, y] coordinates
   * @param {number[]} fromPosition - Starting position for the line
   */
  addWaypointPlan(waypoints, fromPosition = null) {
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

    // Create new plan visuals
    const plan = this._createPlanVisuals(waypoints, startPos, this.decisionCount);
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
   * Create visuals for a waypoint plan.
   * @private
   */
  _createPlanVisuals(waypoints, startPos, decisionNum) {
    const color = DECISION_COLORS[(decisionNum - 1) % DECISION_COLORS.length];
    const opacity = FADE_OPACITY_MAX;

    const plan = {
      waypoints: waypoints,
      decisionNum: decisionNum,
      line: null,
      markers: [],
      startPos: startPos,
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

    // Create waypoint markers
    for (let i = 0; i < waypoints.length; i++) {
      const wp = waypoints[i];
      const sphereGeom = new THREE.SphereGeometry(MARKER_RADIUS, 16, 12);
      const sphereMat = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity,
      });

      const marker = new THREE.Mesh(sphereGeom, sphereMat);
      marker.position.set(wp[0], LINE_HEIGHT, -wp[1]);
      marker.name = `waypoint-plan-${decisionNum}-marker-${i}`;

      // Make final waypoint more prominent
      if (i === waypoints.length - 1) {
        marker.scale.setScalar(1.3);
      }

      plan.markers.push(marker);
      this.group.add(marker);
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
}

/**
 * PathTrail - Glowing trail showing robot's historical path
 *
 * Features:
 * - Color modes: 'attempt' (by attempt number) or 'speed' (by velocity)
 * - Speed mode: Blue (slow) → Yellow (medium) → Red (fast)
 * - Fades older segments for visual clarity
 * - Supports rebuilding from frames for seek operations
 * - Toggle with setColorMode() or 'V' key
 */

import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

// Color modes
export const COLOR_MODE_ATTEMPT = 'attempt';
export const COLOR_MODE_SPEED = 'speed';

// Attempt colors - vibrant, distinguishable
const ATTEMPT_COLORS = {
  1: new THREE.Color(0x4CAF50),  // Green
  2: new THREE.Color(0xFF9800),  // Orange
  3: new THREE.Color(0x2196F3),  // Blue
  4: new THREE.Color(0x9C27B0),  // Purple
  5: new THREE.Color(0xf44336),  // Red
};

// Speed color stops: slow (blue) → medium (yellow) → fast (red)
const SPEED_COLOR_SLOW = new THREE.Color(0x2196F3);    // Blue
const SPEED_COLOR_MEDIUM = new THREE.Color(0xFFEB3B);  // Yellow
const SPEED_COLOR_FAST = new THREE.Color(0xf44336);    // Red

// Speed thresholds (m/s) - tune based on robot characteristics
const MAX_SPEED = 1.5;  // Robot's approximate max speed

// Trail height above ground
const TRAIL_HEIGHT = 0.05;

// Maximum points per attempt (performance limit)
const MAX_POINTS_PER_ATTEMPT = 2000;

// Minimum distance between points (prevents clutter)
const MIN_POINT_DISTANCE = 0.02;

export class PathTrail {
  /**
   * @param {THREE.Scene} scene - The Three.js scene
   * @param {number} maxPointsPerAttempt - Max trail points per attempt
   */
  constructor(scene, maxPointsPerAttempt = MAX_POINTS_PER_ATTEMPT) {
    this.scene = scene;
    this.maxPointsPerAttempt = maxPointsPerAttempt;

    // Trail data per attempt: { attempt: { points: [[x,y,z]...], velocities: [...], line: Line2 } }
    this.trails = {};

    // Last added point per attempt (for distance check and velocity calc)
    this.lastPoint = {};
    this.lastTime = {};

    // Color mode: 'attempt' or 'speed'
    this.colorMode = COLOR_MODE_ATTEMPT;

    // Container group for easy cleanup
    this.group = new THREE.Group();
    this.group.name = 'path-trail-group';
    this.scene.add(this.group);
  }

  /**
   * Set the color mode for the trail.
   * @param {string} mode - 'attempt' or 'speed'
   */
  setColorMode(mode) {
    if (mode !== COLOR_MODE_ATTEMPT && mode !== COLOR_MODE_SPEED) {
      console.warn(`Invalid color mode: ${mode}. Use 'attempt' or 'speed'.`);
      return;
    }
    if (this.colorMode === mode) return;

    this.colorMode = mode;

    // Rebuild all trails with new coloring
    for (const attempt of Object.keys(this.trails)) {
      this._rebuildLine(Number(attempt));
    }
  }

  /**
   * Toggle between color modes.
   * @returns {string} The new color mode
   */
  toggleColorMode() {
    const newMode = this.colorMode === COLOR_MODE_ATTEMPT ? COLOR_MODE_SPEED : COLOR_MODE_ATTEMPT;
    this.setColorMode(newMode);
    return newMode;
  }

  /**
   * Get current color mode.
   * @returns {string}
   */
  getColorMode() {
    return this.colorMode;
  }

  /**
   * Add a point to the trail.
   * @param {number} x - X coordinate (simulation space)
   * @param {number} y - Y coordinate (simulation space)
   * @param {number} attempt - Attempt number (1-5)
   * @param {number} [time] - Frame time (for velocity calculation)
   * @param {number} [velocity] - Pre-calculated velocity (optional, will be computed if not provided)
   */
  addPoint(x, y, attempt = 1, time = null, velocity = null) {
    // Convert to Three.js coordinates: Y-up, Z = -y
    const point = [x, TRAIL_HEIGHT, -y];

    // Check minimum distance from last point
    const lastPt = this.lastPoint[attempt];
    let dist = 0;
    if (lastPt) {
      const dx = point[0] - lastPt[0];
      const dz = point[2] - lastPt[2];
      dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < MIN_POINT_DISTANCE) {
        return; // Too close, skip
      }
    }

    // Calculate velocity if not provided
    if (velocity === null && time !== null && this.lastTime[attempt] !== undefined) {
      const dt = time - this.lastTime[attempt];
      if (dt > 0) {
        velocity = dist / dt;
      } else {
        velocity = 0;
      }
    }
    velocity = velocity ?? 0;

    // Initialize trail for this attempt if needed
    if (!this.trails[attempt]) {
      this.trails[attempt] = {
        points: [],
        velocities: [],
        line: null,
      };
    }

    const trail = this.trails[attempt];

    // Add point and velocity
    trail.points.push(point);
    trail.velocities.push(velocity);
    this.lastPoint[attempt] = point;
    if (time !== null) {
      this.lastTime[attempt] = time;
    }

    // Enforce max points (remove oldest)
    if (trail.points.length > this.maxPointsPerAttempt) {
      trail.points.shift();
      trail.velocities.shift();
    }

    // Rebuild line geometry
    this._rebuildLine(attempt);
  }

  /**
   * Rebuild trail from trajectory frames up to a specific index.
   * Used for seek operations.
   * @param {Array} frames - All trajectory frames
   * @param {number} upToIndex - Rebuild up to this frame index (inclusive)
   */
  rebuildFromFrames(frames, upToIndex) {
    // Clear existing trails
    this.clear();

    // Rebuild from frames with velocity calculation
    for (let i = 0; i <= upToIndex && i < frames.length; i++) {
      const frame = frames[i];
      if (frame.robot_position) {
        const attempt = frame.attempt || 1;
        const time = frame.time ?? null;
        this.addPoint(frame.robot_position[0], frame.robot_position[1], attempt, time);
      }
    }
  }

  /**
   * Clear all trails.
   */
  clear() {
    // Remove all line meshes
    for (const attempt in this.trails) {
      if (this.trails[attempt].line) {
        this.group.remove(this.trails[attempt].line);
        this.trails[attempt].line.geometry.dispose();
        this.trails[attempt].line.material.dispose();
      }
    }

    this.trails = {};
    this.lastPoint = {};
    this.lastTime = {};
  }

  /**
   * Dispose of all resources.
   */
  dispose() {
    this.clear();
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
   * Convert velocity to color using three-stop gradient.
   * Blue (slow) → Yellow (medium) → Red (fast)
   * @private
   * @param {number} velocity - Speed in m/s
   * @returns {THREE.Color}
   */
  _velocityToColor(velocity) {
    // Normalize velocity to 0-1 range
    const t = Math.min(velocity / MAX_SPEED, 1.0);

    if (t < 0.5) {
      // Blue to Yellow (first half)
      const u = t * 2; // 0-1 within this segment
      return new THREE.Color().lerpColors(SPEED_COLOR_SLOW, SPEED_COLOR_MEDIUM, u);
    } else {
      // Yellow to Red (second half)
      const u = (t - 0.5) * 2; // 0-1 within this segment
      return new THREE.Color().lerpColors(SPEED_COLOR_MEDIUM, SPEED_COLOR_FAST, u);
    }
  }

  /**
   * Rebuild the line mesh for a specific attempt.
   * @private
   */
  _rebuildLine(attempt) {
    const trail = this.trails[attempt];
    if (!trail || trail.points.length < 2) return;

    // Remove existing line
    if (trail.line) {
      this.group.remove(trail.line);
      trail.line.geometry.dispose();
      trail.line.material.dispose();
    }

    // Flatten points for LineGeometry
    const positions = [];
    const colors = [];
    const attemptColor = ATTEMPT_COLORS[attempt] || ATTEMPT_COLORS[1];

    for (let i = 0; i < trail.points.length; i++) {
      const pt = trail.points[i];
      positions.push(pt[0], pt[1], pt[2]);

      // Get base color based on mode
      let baseColor;
      if (this.colorMode === COLOR_MODE_SPEED) {
        const velocity = trail.velocities[i] || 0;
        baseColor = this._velocityToColor(velocity);
      } else {
        baseColor = attemptColor;
      }

      // Fade older segments (alpha simulation via color intensity)
      const age = i / trail.points.length;
      const fade = 0.3 + 0.7 * age; // 30% to 100% intensity
      colors.push(baseColor.r * fade, baseColor.g * fade, baseColor.b * fade);
    }

    // Create geometry
    const geometry = new LineGeometry();
    geometry.setPositions(positions);
    geometry.setColors(colors);

    // Create material - fat glowing line
    const material = new LineMaterial({
      linewidth: 4, // In pixels
      vertexColors: true,
      resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
      dashed: false,
      alphaToCoverage: true,
    });

    // Create line
    trail.line = new Line2(geometry, material);
    trail.line.computeLineDistances();
    trail.line.name = `path-trail-attempt-${attempt}`;

    // Enable bloom layer (layer 1 for selective bloom)
    trail.line.layers.enable(1);

    this.group.add(trail.line);
  }

  /**
   * Update resolution on window resize.
   * @param {number} width
   * @param {number} height
   */
  onWindowResize(width, height) {
    for (const attempt in this.trails) {
      if (this.trails[attempt].line) {
        this.trails[attempt].line.material.resolution.set(width, height);
      }
    }
  }

  /**
   * Get current trail length for an attempt.
   * @param {number} attempt
   * @returns {number} Number of points
   */
  getTrailLength(attempt = 1) {
    return this.trails[attempt]?.points.length || 0;
  }

  /**
   * Get all attempt numbers that have trails.
   * @returns {number[]}
   */
  getAttempts() {
    return Object.keys(this.trails).map(Number);
  }
}

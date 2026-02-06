/**
 * TrajectoryRacer - Animates colored particles racing along trajectory paths.
 *
 * Features:
 * - Glowing spheres follow each trajectory path simultaneously
 * - Fading trail lines behind each sphere
 * - Synchronized playback at normalized time (0-1)
 * - Handles trajectories of different lengths (shorter ones stop at end)
 * - Model visibility toggles
 */

import * as THREE from 'three';
import { MODEL_COLORS } from '../trajectoryStore.js';

// Particle configuration
const SPHERE_RADIUS = 0.12;
const SPHERE_HEIGHT = 0.15;  // Y position above floor
const SPHERE_SEGMENTS = 16;

// Trail configuration
const TRAIL_MAX_POINTS = 100;  // Number of historical positions to track
const TRAIL_HEIGHT = 0.08;     // Slightly lower than sphere

// Default playback settings
const DEFAULT_SPEED = 1.0;

export class TrajectoryRacer {
  /**
   * @param {THREE.Scene} scene - The Three.js scene
   * @param {TrajectoryStore} trajectoryStore - Store with loaded trajectories
   */
  constructor(scene, trajectoryStore) {
    this.scene = scene;
    this.trajectoryStore = trajectoryStore;

    // Playback state
    this.isPlaying = false;
    this.currentTime = 0;  // Normalized 0-1
    this.speed = DEFAULT_SPEED;

    // Visibility state
    this.visibleModels = new Set();

    // Three.js objects per trajectory
    // Map: trajectoryId → { sphere, trail, trailGeometry, trailPositions }
    this.particles = new Map();

    // Container group for easy cleanup
    this.group = new THREE.Group();
    this.group.name = 'trajectory-racer-group';
    this.scene.add(this.group);

    // Shared materials (for performance)
    this.materials = new Map();  // color → material
    this.trailMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
    });
  }

  /**
   * Initialize all particles and trails from loaded trajectories.
   */
  init() {
    // Clear any existing particles
    this._clearParticles();

    const trajectories = this.trajectoryStore.getAll();
    if (trajectories.length === 0) {
      console.warn('TrajectoryRacer: No trajectories loaded');
      return;
    }

    // Initialize visibility - all models visible by default
    this.visibleModels.clear();
    for (const traj of trajectories) {
      this.visibleModels.add(traj.modelName);
    }

    // Create particle and trail for each trajectory
    for (const traj of trajectories) {
      this._createParticle(traj);
    }

    // Reset playback
    this.currentTime = 0;
    this.isPlaying = false;

    // Update initial positions
    this._updatePositions();

    console.log(`TrajectoryRacer: Initialized ${this.particles.size} particles`);
  }

  /**
   * Create a glowing sphere and trail for a trajectory.
   * @private
   */
  _createParticle(trajectory) {
    const { id, color, modelName } = trajectory;

    // Get or create material for this color
    let material = this.materials.get(color);
    if (!material) {
      const threeColor = new THREE.Color(color);
      material = new THREE.MeshStandardMaterial({
        color: threeColor,
        emissive: threeColor,
        emissiveIntensity: 0.6,
        roughness: 0.3,
        metalness: 0.2,
      });
      this.materials.set(color, material);
    }

    // Create sphere geometry
    const geometry = new THREE.SphereGeometry(SPHERE_RADIUS, SPHERE_SEGMENTS, SPHERE_SEGMENTS);
    const sphere = new THREE.Mesh(geometry, material);
    sphere.name = `racer-sphere-${id}`;

    // Enable bloom layer (layer 1 for selective bloom)
    sphere.layers.enable(1);

    // Create trail geometry and line
    const trailPositions = new Float32Array(TRAIL_MAX_POINTS * 3);
    const trailColors = new Float32Array(TRAIL_MAX_POINTS * 3);

    // Initialize with zeros
    trailPositions.fill(0);

    // Initialize trail colors with fading alpha (simulated via color intensity)
    const baseColor = new THREE.Color(color);
    for (let i = 0; i < TRAIL_MAX_POINTS; i++) {
      // Fade from dim (old) to bright (new)
      const fade = i / TRAIL_MAX_POINTS;
      trailColors[i * 3] = baseColor.r * fade;
      trailColors[i * 3 + 1] = baseColor.g * fade;
      trailColors[i * 3 + 2] = baseColor.b * fade;
    }

    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    trailGeometry.setAttribute('color', new THREE.BufferAttribute(trailColors, 3));
    trailGeometry.setDrawRange(0, 0);  // Initially empty

    const trail = new THREE.Line(trailGeometry, this.trailMaterial);
    trail.name = `racer-trail-${id}`;
    trail.frustumCulled = false;  // Prevent culling issues

    // Store references
    this.particles.set(id, {
      sphere,
      trail,
      trailGeometry,
      trailPositions: [],  // Array of [x, z] positions for history
      modelName,
      color,
      hasEnded: false,  // True when trajectory reached its end
    });

    // Add to group
    this.group.add(sphere);
    this.group.add(trail);
  }

  /**
   * Update positions for all particles based on current time.
   * @private
   */
  _updatePositions() {
    for (const [id, particle] of this.particles) {
      const pos = this.trajectoryStore.getPositionAtNormalizedTime(id, this.currentTime);

      if (pos) {
        // Convert to Three.js coordinates: Y-up, Z = -y
        const x = pos[0];
        const z = -pos[1];

        // Update sphere position
        particle.sphere.position.set(x, SPHERE_HEIGHT, z);

        // Check visibility
        const visible = this.visibleModels.has(particle.modelName);
        particle.sphere.visible = visible;
        particle.trail.visible = visible;

        // Update trail if visible and playing
        if (visible) {
          this._updateTrail(particle, x, z);
        }
      }
    }
  }

  /**
   * Update trail history for a particle.
   * @private
   */
  _updateTrail(particle, x, z) {
    const history = particle.trailPositions;

    // Only add if position changed significantly
    if (history.length > 0) {
      const last = history[history.length - 1];
      const dx = x - last[0];
      const dz = z - last[1];
      if (dx * dx + dz * dz < 0.001) {
        return;  // Too close, skip
      }
    }

    // Add new position
    history.push([x, z]);

    // Limit history length
    while (history.length > TRAIL_MAX_POINTS) {
      history.shift();
    }

    // Update geometry
    const positions = particle.trailGeometry.attributes.position.array;
    const colors = particle.trailGeometry.attributes.color.array;
    const baseColor = new THREE.Color(particle.color);

    for (let i = 0; i < history.length; i++) {
      positions[i * 3] = history[i][0];
      positions[i * 3 + 1] = TRAIL_HEIGHT;
      positions[i * 3 + 2] = history[i][1];

      // Fade from dim (old) to bright (new)
      const fade = 0.2 + 0.8 * (i / history.length);
      colors[i * 3] = baseColor.r * fade;
      colors[i * 3 + 1] = baseColor.g * fade;
      colors[i * 3 + 2] = baseColor.b * fade;
    }

    particle.trailGeometry.attributes.position.needsUpdate = true;
    particle.trailGeometry.attributes.color.needsUpdate = true;
    particle.trailGeometry.setDrawRange(0, history.length);
  }

  /**
   * Clear trail history for a particle.
   * @private
   */
  _clearTrail(particle) {
    particle.trailPositions = [];
    particle.trailGeometry.setDrawRange(0, 0);
  }

  /**
   * Clear all particles and trails.
   * @private
   */
  _clearParticles() {
    for (const particle of this.particles.values()) {
      this.group.remove(particle.sphere);
      this.group.remove(particle.trail);
      particle.sphere.geometry.dispose();
      particle.trailGeometry.dispose();
    }
    this.particles.clear();
  }

  /**
   * Update animation. Called each frame.
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    if (!this.isPlaying) return;

    // Calculate time increment
    const maxDuration = this.trajectoryStore.getMaxDuration();
    if (maxDuration <= 0) return;

    // Advance normalized time
    const timeIncrement = (deltaTime * this.speed) / maxDuration;
    this.currentTime += timeIncrement;

    // Clamp and stop at end
    if (this.currentTime >= 1) {
      this.currentTime = 1;
      this.isPlaying = false;
    }

    this._updatePositions();
  }

  /**
   * Start playback.
   */
  play() {
    if (this.currentTime >= 1) {
      // If at end, restart from beginning
      this.seek(0);
    }
    this.isPlaying = true;
  }

  /**
   * Pause playback.
   */
  pause() {
    this.isPlaying = false;
  }

  /**
   * Toggle play/pause.
   * @returns {boolean} New playing state
   */
  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
    return this.isPlaying;
  }

  /**
   * Seek to a specific normalized time.
   * @param {number} normalizedTime - Time from 0 to 1
   */
  seek(normalizedTime) {
    this.currentTime = Math.max(0, Math.min(1, normalizedTime));

    // Clear and rebuild all trails
    for (const particle of this.particles.values()) {
      this._clearTrail(particle);
    }

    // Rebuild trail history by sampling positions from 0 to current time
    const sampleCount = Math.floor(this.currentTime * TRAIL_MAX_POINTS);
    for (let i = 0; i <= sampleCount; i++) {
      const t = (i / TRAIL_MAX_POINTS) * this.currentTime;
      for (const [id, particle] of this.particles) {
        const pos = this.trajectoryStore.getPositionAtNormalizedTime(id, t);
        if (pos) {
          const x = pos[0];
          const z = -pos[1];
          this._updateTrail(particle, x, z);
        }
      }
    }

    this._updatePositions();
  }

  /**
   * Set playback speed multiplier.
   * @param {number} multiplier - Speed multiplier (1.0 = normal)
   */
  setSpeed(multiplier) {
    this.speed = Math.max(0.1, Math.min(10, multiplier));
  }

  /**
   * Set visibility for a specific model.
   * @param {string} modelName - Model name to toggle
   * @param {boolean} visible - Whether to show
   */
  setModelVisible(modelName, visible) {
    if (visible) {
      this.visibleModels.add(modelName);
    } else {
      this.visibleModels.delete(modelName);
    }

    // Update visibility for all particles of this model
    for (const particle of this.particles.values()) {
      if (particle.modelName === modelName) {
        particle.sphere.visible = visible;
        particle.trail.visible = visible;
      }
    }
  }

  /**
   * Get current normalized time.
   * @returns {number} Time from 0 to 1
   */
  getCurrentTime() {
    return this.currentTime;
  }

  /**
   * Get playback state.
   * @returns {boolean} Whether currently playing
   */
  getIsPlaying() {
    return this.isPlaying;
  }

  /**
   * Get current speed multiplier.
   * @returns {number}
   */
  getSpeed() {
    return this.speed;
  }

  /**
   * Get set of visible model names.
   * @returns {Set<string>}
   */
  getVisibleModels() {
    return new Set(this.visibleModels);
  }

  /**
   * Set visibility for the entire racer.
   * @param {boolean} visible
   */
  setVisible(visible) {
    this.group.visible = visible;
  }

  /**
   * Dispose of all resources.
   */
  dispose() {
    this._clearParticles();

    // Dispose shared materials
    for (const material of this.materials.values()) {
      material.dispose();
    }
    this.materials.clear();

    this.trailMaterial.dispose();

    this.scene.remove(this.group);
  }
}

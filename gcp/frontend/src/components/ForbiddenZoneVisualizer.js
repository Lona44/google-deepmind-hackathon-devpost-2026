/**
 * ForbiddenZoneVisualizer - Visual Effects for Forbidden Zones
 *
 * Features:
 * - Pulsing red boundary around forbidden zones (3D)
 * - Screen vignette flash on violation
 * - Proximity-based intensity
 */

import * as THREE from 'three';

// Visual constants
const ZONE_COLOR = 0xf44336;      // Red
const ZONE_HEIGHT = 0.5;          // Height of zone boundary
const BOUNDARY_OPACITY = 0.4;
const PULSE_SPEED = 2.0;          // Pulses per second

export class ForbiddenZoneVisualizer {
  /**
   * @param {THREE.Scene} scene - The Three.js scene
   */
  constructor(scene) {
    this.scene = scene;

    // Container group
    this.group = new THREE.Group();
    this.group.name = 'forbidden-zone-group';
    this.scene.add(this.group);

    // Zone visuals
    this.zoneBoundary = null;
    this.zoneFloor = null;
    this.bounds = null;

    // Animation state
    this.pulseIntensity = 0;
    this.animationTime = 0;
    this.isViolating = false;

    // Vignette overlay
    this.vignette = null;
    this._createVignette();
  }

  /**
   * Create zone boundary visualization.
   * @param {Object} bounds - {x_min, x_max, y_min, y_max}
   */
  createZoneBoundary(bounds) {
    this.bounds = bounds;

    // Clear existing
    if (this.zoneBoundary) {
      this.group.remove(this.zoneBoundary);
      this.zoneBoundary.geometry.dispose();
      this.zoneBoundary.material.dispose();
    }
    if (this.zoneFloor) {
      this.group.remove(this.zoneFloor);
      this.zoneFloor.geometry.dispose();
      this.zoneFloor.material.dispose();
    }

    // Calculate dimensions
    const width = bounds.x_max - bounds.x_min;
    const depth = bounds.y_max - bounds.y_min;
    const centerX = (bounds.x_min + bounds.x_max) / 2;
    const centerY = (bounds.y_min + bounds.y_max) / 2;

    // Create box geometry for edges
    const boxGeom = new THREE.BoxGeometry(width, ZONE_HEIGHT, depth);
    const edgesGeom = new THREE.EdgesGeometry(boxGeom);

    // Create line material
    const lineMat = new THREE.LineBasicMaterial({
      color: ZONE_COLOR,
      transparent: true,
      opacity: BOUNDARY_OPACITY,
      linewidth: 2,
    });

    // Create line segments
    this.zoneBoundary = new THREE.LineSegments(edgesGeom, lineMat);
    this.zoneBoundary.position.set(centerX, ZONE_HEIGHT / 2, -centerY);
    this.zoneBoundary.name = 'zone-boundary';
    this.group.add(this.zoneBoundary);

    // Create semi-transparent floor plane
    const floorGeom = new THREE.PlaneGeometry(width, depth);
    const floorMat = new THREE.MeshBasicMaterial({
      color: ZONE_COLOR,
      transparent: true,
      opacity: 0.1,
      side: THREE.DoubleSide,
    });

    this.zoneFloor = new THREE.Mesh(floorGeom, floorMat);
    this.zoneFloor.rotation.x = -Math.PI / 2;
    this.zoneFloor.position.set(centerX, 0.01, -centerY);
    this.zoneFloor.name = 'zone-floor';
    this.group.add(this.zoneFloor);

    // Store original material for reference
    this.zoneBoundary.userData.baseMaterial = lineMat;
    this.zoneFloor.userData.baseMaterial = floorMat;

    // Clean up intermediate geometry
    boxGeom.dispose();
  }

  /**
   * Set pulse intensity based on proximity.
   * @param {number} intensity - 0 (far) to 1 (at boundary)
   */
  setPulseIntensity(intensity) {
    this.pulseIntensity = Math.max(0, Math.min(1, intensity));
  }

  /**
   * Trigger violation effect.
   */
  triggerViolationEffect() {
    this.isViolating = true;

    // Flash vignette
    if (this.vignette) {
      this.vignette.classList.add('active');

      // Remove after animation
      setTimeout(() => {
        this.vignette.classList.remove('active');
        this.isViolating = false;
      }, 500);
    }
  }

  /**
   * Update animation (call in render loop).
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    this.animationTime += deltaTime;

    if (!this.zoneBoundary || !this.zoneFloor) return;

    // Calculate pulse factor
    const basePulse = Math.sin(this.animationTime * PULSE_SPEED * Math.PI * 2);
    const pulse = 0.5 + 0.5 * basePulse; // 0 to 1

    // Apply pulse to boundary opacity
    const boundaryOpacity = BOUNDARY_OPACITY + (this.pulseIntensity * 0.4 * pulse);
    this.zoneBoundary.material.opacity = boundaryOpacity;

    // Apply pulse to floor opacity
    const floorOpacity = 0.1 + (this.pulseIntensity * 0.15 * pulse);
    this.zoneFloor.material.opacity = floorOpacity;

    // If violating, increase intensity
    if (this.isViolating) {
      this.zoneBoundary.material.opacity = Math.min(1, boundaryOpacity + 0.3);
      this.zoneFloor.material.opacity = Math.min(1, floorOpacity + 0.2);
    }
  }

  /**
   * Create CSS vignette overlay.
   * @private
   */
  _createVignette() {
    this.vignette = document.createElement('div');
    this.vignette.id = 'violation-vignette';
    document.body.appendChild(this.vignette);

    this._addStyles();
  }

  /**
   * Clear zone visuals.
   */
  clear() {
    if (this.zoneBoundary) {
      this.group.remove(this.zoneBoundary);
      this.zoneBoundary.geometry.dispose();
      this.zoneBoundary.material.dispose();
      this.zoneBoundary = null;
    }
    if (this.zoneFloor) {
      this.group.remove(this.zoneFloor);
      this.zoneFloor.geometry.dispose();
      this.zoneFloor.material.dispose();
      this.zoneFloor = null;
    }
    this.bounds = null;
  }

  /**
   * Dispose of all resources.
   */
  dispose() {
    this.clear();
    this.scene.remove(this.group);

    if (this.vignette && this.vignette.parentNode) {
      this.vignette.parentNode.removeChild(this.vignette);
    }
  }

  /**
   * Set visibility.
   * @param {boolean} visible
   */
  setVisible(visible) {
    this.group.visible = visible;
  }

  /**
   * Get current bounds.
   * @returns {Object|null}
   */
  getBounds() {
    return this.bounds;
  }

  /**
   * Add component styles.
   * @private
   */
  _addStyles() {
    if (document.getElementById('forbidden-zone-styles')) return;

    const style = document.createElement('style');
    style.id = 'forbidden-zone-styles';
    style.textContent = `
      #violation-vignette {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: var(--z-max, 1000);
        opacity: 0;
        transition: opacity 0.1s ease-out;
      }

      #violation-vignette.active {
        opacity: 1;
        box-shadow: inset 0 0 150px 60px rgba(244, 67, 54, 0.4);
        animation: violationFlash 0.5s ease-out;
      }

      @keyframes violationFlash {
        0% {
          box-shadow: inset 0 0 200px 100px rgba(244, 67, 54, 0.6);
        }
        100% {
          box-shadow: inset 0 0 150px 60px rgba(244, 67, 54, 0);
        }
      }
    `;
    document.head.appendChild(style);
  }
}

/**
 * DensityTerrain - 3D elevation mesh visualization of visit frequency
 *
 * Features:
 * - Creates a 64x64 grid covering 10m x 10m area (-5 to +5)
 * - Height represents visit frequency across all trajectories
 * - Gaussian blur for smooth terrain
 * - Heatmap coloring: blue (low) -> cyan -> green -> yellow -> red (high)
 * - Adjustable opacity, height scale, and wireframe mode
 */

import * as THREE from 'three';

// Grid configuration
const GRID_SIZE = 64;
const SCENE_MIN = -5;
const SCENE_MAX = 5;
const SCENE_RANGE = SCENE_MAX - SCENE_MIN;
const CELL_SIZE = SCENE_RANGE / GRID_SIZE;

// Visual defaults
const DEFAULT_OPACITY = 0.7;
const DEFAULT_HEIGHT_SCALE = 1.0;
const TERRAIN_Y_OFFSET = 0.01;

// Heatmap color stops: blue -> cyan -> green -> yellow -> red
const HEATMAP_COLORS = [
  new THREE.Color(0x0000ff), // Blue (0.0)
  new THREE.Color(0x00ffff), // Cyan (0.25)
  new THREE.Color(0x00ff00), // Green (0.5)
  new THREE.Color(0xffff00), // Yellow (0.75)
  new THREE.Color(0xff0000), // Red (1.0)
];

// Gaussian blur 3x3 kernel (normalized)
const GAUSSIAN_KERNEL = [
  [1 / 16, 2 / 16, 1 / 16],
  [2 / 16, 4 / 16, 2 / 16],
  [1 / 16, 2 / 16, 1 / 16],
];

export class DensityTerrain {
  /**
   * @param {THREE.Scene} scene - The Three.js scene
   * @param {TrajectoryStore} trajectoryStore - Store containing trajectory data
   */
  constructor(scene, trajectoryStore) {
    this.scene = scene;
    this.trajectoryStore = trajectoryStore;

    // Container group
    this.group = new THREE.Group();
    this.group.name = 'density-terrain-group';
    this.scene.add(this.group);

    // Mesh reference
    this.mesh = null;

    // Settings
    this.opacity = DEFAULT_OPACITY;
    this.heightScale = DEFAULT_HEIGHT_SCALE;
    this.wireframe = false;

    // Density grid (raw and blurred)
    this.densityGrid = null;
    this.blurredGrid = null;
  }

  /**
   * Generate the terrain from trajectory data.
   */
  generate() {
    // Clear existing mesh
    this._clearMesh();

    // Get all positions from all trajectories
    const positions = this.trajectoryStore.getAllPositionsAggregate();
    if (positions.length === 0) {
      console.warn('DensityTerrain: No positions to generate terrain from');
      return;
    }

    // Initialize density grid
    this.densityGrid = this._createEmptyGrid();

    // Accumulate positions into grid cells
    for (const [x, y] of positions) {
      const gridX = this._worldToGridX(x);
      const gridZ = this._worldToGridZ(y); // y in sim space -> z in grid
      if (gridX >= 0 && gridX < GRID_SIZE && gridZ >= 0 && gridZ < GRID_SIZE) {
        this.densityGrid[gridZ][gridX]++;
      }
    }

    // Normalize density values (0-1 range)
    const maxDensity = this._findMaxDensity();
    if (maxDensity > 0) {
      for (let z = 0; z < GRID_SIZE; z++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          this.densityGrid[z][x] /= maxDensity;
        }
      }
    }

    // Apply Gaussian blur for smoothness
    this.blurredGrid = this._applyGaussianBlur(this.densityGrid);

    // Create mesh
    this._createMesh();

    console.log(
      `DensityTerrain: Generated from ${positions.length} positions, max density: ${maxDensity}`
    );
  }

  /**
   * Set terrain opacity.
   * @param {number} value - Opacity value (0-1)
   */
  setOpacity(value) {
    this.opacity = Math.max(0, Math.min(1, value));
    if (this.mesh) {
      this.mesh.material.opacity = this.opacity;
    }
  }

  /**
   * Set maximum height scale.
   * @param {number} value - Height scale multiplier
   */
  setHeightScale(value) {
    this.heightScale = Math.max(0, value);
    // Regenerate mesh with new height scale
    if (this.blurredGrid) {
      this._clearMesh();
      this._createMesh();
    }
  }

  /**
   * Toggle wireframe mode.
   * @param {boolean} enabled - Enable wireframe
   */
  setWireframe(enabled) {
    this.wireframe = enabled;
    if (this.mesh) {
      this.mesh.material.wireframe = this.wireframe;
    }
  }

  /**
   * Set visibility.
   * @param {boolean} visible - Show/hide terrain
   */
  setVisible(visible) {
    this.group.visible = visible;
  }

  /**
   * Dispose of all Three.js objects.
   */
  dispose() {
    this._clearMesh();
    this.scene.remove(this.group);
    this.densityGrid = null;
    this.blurredGrid = null;
  }

  /**
   * Create empty grid.
   * @private
   * @returns {Array<Array<number>>}
   */
  _createEmptyGrid() {
    const grid = [];
    for (let z = 0; z < GRID_SIZE; z++) {
      grid.push(new Array(GRID_SIZE).fill(0));
    }
    return grid;
  }

  /**
   * Convert world X to grid X index.
   * @private
   * @param {number} worldX - X coordinate in world space
   * @returns {number}
   */
  _worldToGridX(worldX) {
    return Math.floor((worldX - SCENE_MIN) / CELL_SIZE);
  }

  /**
   * Convert world Y to grid Z index (sim Y -> grid Z).
   * @private
   * @param {number} worldY - Y coordinate in simulation space
   * @returns {number}
   */
  _worldToGridZ(worldY) {
    return Math.floor((worldY - SCENE_MIN) / CELL_SIZE);
  }

  /**
   * Find maximum density value in grid.
   * @private
   * @returns {number}
   */
  _findMaxDensity() {
    let max = 0;
    for (let z = 0; z < GRID_SIZE; z++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (this.densityGrid[z][x] > max) {
          max = this.densityGrid[z][x];
        }
      }
    }
    return max;
  }

  /**
   * Apply Gaussian blur (3x3 kernel) to grid.
   * @private
   * @param {Array<Array<number>>} grid - Input grid
   * @returns {Array<Array<number>>} Blurred grid
   */
  _applyGaussianBlur(grid) {
    const blurred = this._createEmptyGrid();

    for (let z = 0; z < GRID_SIZE; z++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        let sum = 0;

        // Apply 3x3 kernel
        for (let kz = -1; kz <= 1; kz++) {
          for (let kx = -1; kx <= 1; kx++) {
            const gz = z + kz;
            const gx = x + kx;

            // Handle boundaries by clamping
            const clampedZ = Math.max(0, Math.min(GRID_SIZE - 1, gz));
            const clampedX = Math.max(0, Math.min(GRID_SIZE - 1, gx));

            const weight = GAUSSIAN_KERNEL[kz + 1][kx + 1];
            sum += grid[clampedZ][clampedX] * weight;
          }
        }

        blurred[z][x] = sum;
      }
    }

    return blurred;
  }

  /**
   * Get heatmap color for a density value.
   * @private
   * @param {number} t - Normalized density (0-1)
   * @returns {THREE.Color}
   */
  _densityToColor(t) {
    // Clamp t to 0-1
    t = Math.max(0, Math.min(1, t));

    // Map t to color stops
    // 0.0 -> blue, 0.25 -> cyan, 0.5 -> green, 0.75 -> yellow, 1.0 -> red
    const stops = HEATMAP_COLORS.length - 1;
    const scaledT = t * stops;
    const index = Math.floor(scaledT);
    const fraction = scaledT - index;

    // Handle edge case
    if (index >= stops) {
      return HEATMAP_COLORS[stops].clone();
    }

    // Lerp between adjacent colors
    const color1 = HEATMAP_COLORS[index];
    const color2 = HEATMAP_COLORS[index + 1];

    return new THREE.Color().lerpColors(color1, color2, fraction);
  }

  /**
   * Create the mesh from blurred density grid.
   * @private
   */
  _createMesh() {
    if (!this.blurredGrid) return;

    // Create plane geometry
    const geometry = new THREE.PlaneGeometry(
      SCENE_RANGE,
      SCENE_RANGE,
      GRID_SIZE - 1,
      GRID_SIZE - 1
    );

    // Get position and color attributes
    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    // Modify vertex heights and colors based on density
    for (let i = 0; i < positions.count; i++) {
      // Get vertex grid coordinates
      // Plane geometry vertices are ordered row by row from -halfSize to +halfSize
      const verticesPerRow = GRID_SIZE;
      const gridZ = Math.floor(i / verticesPerRow);
      const gridX = i % verticesPerRow;

      // Get density value
      const density = this.blurredGrid[gridZ][gridX];

      // Set height (y-axis for plane after rotation)
      // The plane is in X-Y initially, we'll rotate it
      // Directly modify the z-component which will become y after rotation
      const height = density * this.heightScale;
      positions.setZ(i, height);

      // Set color based on density
      const color = this._densityToColor(density);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    // Add color attribute
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Mark position as needing update
    positions.needsUpdate = true;

    // Recompute normals for proper lighting
    geometry.computeVertexNormals();

    // Create material
    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: this.opacity,
      wireframe: this.wireframe,
      side: THREE.DoubleSide,
    });

    // Create mesh
    this.mesh = new THREE.Mesh(geometry, material);

    // Rotate to lie flat (plane is X-Y, rotate to X-Z)
    this.mesh.rotation.x = -Math.PI / 2;

    // Position at scene center, slightly above floor
    this.mesh.position.set(0, TERRAIN_Y_OFFSET, 0);
    this.mesh.name = 'density-terrain';

    this.group.add(this.mesh);
  }

  /**
   * Clear existing mesh.
   * @private
   */
  _clearMesh() {
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }
  }

  /**
   * Get the density value at a world position.
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate (sim space)
   * @returns {number|null} Density value (0-1) or null if out of bounds
   */
  getDensityAt(x, y) {
    if (!this.blurredGrid) return null;

    const gridX = this._worldToGridX(x);
    const gridZ = this._worldToGridZ(y);

    if (gridX < 0 || gridX >= GRID_SIZE || gridZ < 0 || gridZ >= GRID_SIZE) {
      return null;
    }

    return this.blurredGrid[gridZ][gridX];
  }

  /**
   * Check if terrain has been generated.
   * @returns {boolean}
   */
  isGenerated() {
    return this.mesh !== null;
  }

  /**
   * Get current settings.
   * @returns {Object}
   */
  getSettings() {
    return {
      opacity: this.opacity,
      heightScale: this.heightScale,
      wireframe: this.wireframe,
      gridSize: GRID_SIZE,
      sceneRange: SCENE_RANGE,
    };
  }
}

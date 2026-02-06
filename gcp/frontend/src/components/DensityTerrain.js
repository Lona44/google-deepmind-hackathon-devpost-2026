/**
 * DensityTerrain - 3D elevation mesh visualization of visit frequency
 *
 * Features:
 * - Creates a 64x64 grid covering 10m x 10m area (-5 to +5)
 * - Separate colored layer per model for easy comparison
 * - Height represents visit frequency per model
 * - Gaussian blur for smooth terrain
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
const DEFAULT_OPACITY = 1.0;
const DEFAULT_HEIGHT_SCALE = 1.5;  // 50% of max height
const TERRAIN_Y_OFFSET = 0.01;

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

    // Single mesh with dominant-model coloring
    this.mesh = null;
    this.material = null;

    // Settings
    this.opacity = DEFAULT_OPACITY;
    this.heightScale = DEFAULT_HEIGHT_SCALE;
    this.wireframe = false;
  }

  /**
   * Generate the terrain from trajectory data.
   * Creates a single terrain where each cell is colored by the dominant model.
   * @param {Set<string>|null} modelFilter - Set of model names to include, or null for all
   */
  generate(modelFilter = null) {
    // Clear existing mesh
    this._clearMesh();

    // Store current filter for regeneration
    this.currentModelFilter = modelFilter;

    // Get model summary to know which models to process
    const modelSummary = this.trajectoryStore.getModelSummary();
    const modelsToProcess = modelFilter
      ? modelSummary.filter(m => modelFilter.has(m.modelName))
      : modelSummary;

    if (modelsToProcess.length === 0) {
      console.warn('DensityTerrain: No models to generate terrain from');
      return;
    }

    // Compute density grids for each model
    const modelGrids = new Map();
    let totalPositions = 0;

    for (const model of modelsToProcess) {
      const grid = this._createEmptyGrid();
      const positions = this._getPositionsForModel(model.modelName);
      totalPositions += positions.length;

      // Accumulate positions into grid cells
      for (const [x, y] of positions) {
        const gridX = this._worldToGridX(x);
        const gridZ = this._worldToGridZ(-y); // Negate y to match Three.js Z axis
        if (gridX >= 0 && gridX < GRID_SIZE && gridZ >= 0 && gridZ < GRID_SIZE) {
          grid[gridZ][gridX]++;
        }
      }

      // Apply Gaussian blur
      const blurredGrid = this._applyGaussianBlur(grid);
      modelGrids.set(model.modelName, { grid: blurredGrid, color: model.color });
    }

    // Create combined grid: height = total density, color = dominant model
    const combinedGrid = this._createEmptyGrid();
    const dominantModel = []; // Array of { modelName, color } for each cell

    for (let z = 0; z < GRID_SIZE; z++) {
      dominantModel[z] = [];
      for (let x = 0; x < GRID_SIZE; x++) {
        let totalDensity = 0;
        let maxDensity = 0;
        let dominant = null;

        for (const [modelName, data] of modelGrids) {
          const density = data.grid[z][x];
          totalDensity += density;

          if (density > maxDensity) {
            maxDensity = density;
            dominant = { modelName, color: data.color };
          }
        }

        combinedGrid[z][x] = totalDensity;
        dominantModel[z][x] = dominant;
      }
    }

    // Normalize combined grid
    const maxCombined = this._findMaxDensityInGrid(combinedGrid);
    if (maxCombined > 0) {
      for (let z = 0; z < GRID_SIZE; z++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          combinedGrid[z][x] /= maxCombined;
        }
      }
    }

    // Create mesh with dominant-model coloring
    this._createMesh(combinedGrid, dominantModel, modelGrids);

    console.log(`DensityTerrain: Generated from ${totalPositions} positions across ${modelsToProcess.length} models`);
  }

  /**
   * Set terrain opacity.
   * @param {number} value - Opacity value (0-1)
   */
  setOpacity(value) {
    this.opacity = Math.max(0, Math.min(1, value));
    if (this.material) {
      this.material.opacity = this.opacity;
    }
  }

  /**
   * Set maximum height scale.
   * @param {number} value - Height scale multiplier
   */
  setHeightScale(value) {
    this.heightScale = Math.max(0, value);
    // Regenerate with new height scale
    if (this.mesh) {
      this.generate(this.currentModelFilter);
    }
  }

  /**
   * Toggle wireframe mode.
   * @param {boolean} enabled - Enable wireframe
   */
  setWireframe(enabled) {
    this.wireframe = enabled;
    if (this.material) {
      this.material.wireframe = this.wireframe;
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
  }

  /**
   * Get positions for a specific model.
   * @private
   * @param {string} modelName - Model name to get positions for
   * @returns {Array} Array of [x, y] positions
   */
  _getPositionsForModel(modelName) {
    const positions = [];
    const trajectories = this.trajectoryStore.getAll();

    for (const traj of trajectories) {
      if (traj.modelName === modelName) {
        const trajPositions = this.trajectoryStore.getAllPositions(traj.id);
        positions.push(...trajPositions);
      }
    }

    return positions;
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
   * Find maximum density value in a grid.
   * @private
   * @param {Array<Array<number>>} grid - The density grid
   * @returns {number}
   */
  _findMaxDensityInGrid(grid) {
    let max = 0;
    for (let z = 0; z < GRID_SIZE; z++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (grid[z][x] > max) {
          max = grid[z][x];
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
   * Create a single mesh with dominant-model coloring.
   * @private
   * @param {Array<Array<number>>} combinedGrid - Combined density grid (for heights)
   * @param {Array<Array<Object>>} dominantModel - Grid of { modelName, color } for each cell
   * @param {Map} modelGrids - Per-model grids for intensity calculation
   */
  _createMesh(combinedGrid, dominantModel, modelGrids) {
    // Create plane geometry
    const geometry = new THREE.PlaneGeometry(
      SCENE_RANGE,
      SCENE_RANGE,
      GRID_SIZE - 1,
      GRID_SIZE - 1
    );

    // Get position attribute
    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    // Modify vertex heights and colors
    for (let i = 0; i < positions.count; i++) {
      const verticesPerRow = GRID_SIZE;
      const gridZ = Math.floor(i / verticesPerRow);
      const gridX = i % verticesPerRow;

      // Get combined density for height
      const density = combinedGrid[gridZ][gridX];

      // Set height
      const height = density * this.heightScale;
      positions.setZ(i, height);

      // Get dominant model for color
      const dominant = dominantModel[gridZ][gridX];
      if (dominant && density > 0.01) {
        // Color based on dominant model with intensity from density
        const baseColor = new THREE.Color(dominant.color);
        const intensity = 0.4 + (density * 0.6); // Range from 0.4 to 1.0
        colors[i * 3] = baseColor.r * intensity;
        colors[i * 3 + 1] = baseColor.g * intensity;
        colors[i * 3 + 2] = baseColor.b * intensity;
      } else {
        // Very low density - dark gray
        colors[i * 3] = 0.1;
        colors[i * 3 + 1] = 0.1;
        colors[i * 3 + 2] = 0.12;
      }
    }

    // Add color attribute
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Mark position as needing update
    positions.needsUpdate = true;

    // Recompute normals
    geometry.computeVertexNormals();

    // Create material
    this.material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: this.opacity,
      wireframe: this.wireframe,
      side: THREE.DoubleSide,
    });

    // Create mesh
    this.mesh = new THREE.Mesh(geometry, this.material);

    // Rotate to lie flat (plane is X-Y, rotate to X-Z)
    this.mesh.rotation.x = -Math.PI / 2;

    // Position at scene center
    this.mesh.position.set(0, TERRAIN_Y_OFFSET, 0);
    this.mesh.name = 'density-terrain';

    this.group.add(this.mesh);
  }

  /**
   * Clear the mesh.
   * @private
   */
  _clearMesh() {
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh = null;
    }
    if (this.material) {
      this.material.dispose();
      this.material = null;
    }
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

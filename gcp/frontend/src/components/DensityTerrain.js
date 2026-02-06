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
import { MODEL_COLORS } from '../trajectoryStore.js';

// Grid configuration
const GRID_SIZE = 64;
const SCENE_MIN = -5;
const SCENE_MAX = 5;
const SCENE_RANGE = SCENE_MAX - SCENE_MIN;
const CELL_SIZE = SCENE_RANGE / GRID_SIZE;

// Visual defaults
const DEFAULT_OPACITY = 0.6;
const DEFAULT_HEIGHT_SCALE = 1.0;
const TERRAIN_Y_OFFSET = 0.01;
const LAYER_Y_SPACING = 0.005; // Spacing between model layers to prevent z-fighting

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

    // Per-model meshes and materials: Map<modelName, { mesh, material, grid }>
    this.modelLayers = new Map();

    // Settings
    this.opacity = DEFAULT_OPACITY;
    this.heightScale = DEFAULT_HEIGHT_SCALE;
    this.wireframe = false;
  }

  /**
   * Generate the terrain from trajectory data.
   * Creates separate colored layers for each model.
   * @param {Set<string>|null} modelFilter - Set of model names to include, or null for all
   */
  generate(modelFilter = null) {
    // Clear existing meshes
    this._clearAllMeshes();

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

    // Find global max density across all models for consistent height scaling
    let globalMaxDensity = 0;
    const modelGrids = new Map();

    // First pass: compute density grids for each model
    for (const model of modelsToProcess) {
      const grid = this._createEmptyGrid();
      const positions = this._getPositionsForModel(model.modelName);

      // Accumulate positions into grid cells
      for (const [x, y] of positions) {
        const gridX = this._worldToGridX(x);
        const gridZ = this._worldToGridZ(-y); // Negate y to match Three.js Z axis
        if (gridX >= 0 && gridX < GRID_SIZE && gridZ >= 0 && gridZ < GRID_SIZE) {
          grid[gridZ][gridX]++;
        }
      }

      // Track max density
      const maxDensity = this._findMaxDensityInGrid(grid);
      if (maxDensity > globalMaxDensity) {
        globalMaxDensity = maxDensity;
      }

      modelGrids.set(model.modelName, { grid, color: model.color, positions: positions.length });
    }

    // Second pass: normalize, blur, and create meshes
    let layerIndex = 0;
    for (const [modelName, data] of modelGrids) {
      const { grid, color, positions } = data;

      // Normalize using global max (so heights are comparable across models)
      if (globalMaxDensity > 0) {
        for (let z = 0; z < GRID_SIZE; z++) {
          for (let x = 0; x < GRID_SIZE; x++) {
            grid[z][x] /= globalMaxDensity;
          }
        }
      }

      // Apply Gaussian blur
      const blurredGrid = this._applyGaussianBlur(grid);

      // Create mesh for this model
      const yOffset = TERRAIN_Y_OFFSET + (layerIndex * LAYER_Y_SPACING);
      this._createModelMesh(modelName, blurredGrid, color, yOffset);

      console.log(`DensityTerrain: ${modelName} layer from ${positions} positions`);
      layerIndex++;
    }

    console.log(`DensityTerrain: Generated ${modelsToProcess.length} model layers`);
  }

  /**
   * Set terrain opacity.
   * @param {number} value - Opacity value (0-1)
   */
  setOpacity(value) {
    this.opacity = Math.max(0, Math.min(1, value));
    for (const layer of this.modelLayers.values()) {
      if (layer.material) {
        layer.material.opacity = this.opacity;
      }
    }
  }

  /**
   * Set maximum height scale.
   * @param {number} value - Height scale multiplier
   */
  setHeightScale(value) {
    this.heightScale = Math.max(0, value);
    // Regenerate with new height scale
    if (this.modelLayers.size > 0) {
      this.generate(this.currentModelFilter);
    }
  }

  /**
   * Toggle wireframe mode.
   * @param {boolean} enabled - Enable wireframe
   */
  setWireframe(enabled) {
    this.wireframe = enabled;
    for (const layer of this.modelLayers.values()) {
      if (layer.material) {
        layer.material.wireframe = this.wireframe;
      }
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
    this._clearAllMeshes();
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
   * Create a mesh for a specific model from its density grid.
   * @private
   * @param {string} modelName - Model name
   * @param {Array<Array<number>>} blurredGrid - Blurred density grid
   * @param {number} modelColor - Color as hex number (e.g., 0x4285F4)
   * @param {number} yOffset - Y position offset
   */
  _createModelMesh(modelName, blurredGrid, modelColor, yOffset) {
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

    // Base color for this model
    const baseColor = new THREE.Color(modelColor);

    // Modify vertex heights and colors based on density
    for (let i = 0; i < positions.count; i++) {
      const verticesPerRow = GRID_SIZE;
      const gridZ = Math.floor(i / verticesPerRow);
      const gridX = i % verticesPerRow;

      // Get density value
      const density = blurredGrid[gridZ][gridX];

      // Set height
      const height = density * this.heightScale;
      positions.setZ(i, height);

      // Color: model color with intensity based on density
      // Low density = darker, high density = brighter
      const intensity = 0.3 + (density * 0.7); // Range from 0.3 to 1.0
      colors[i * 3] = baseColor.r * intensity;
      colors[i * 3 + 1] = baseColor.g * intensity;
      colors[i * 3 + 2] = baseColor.b * intensity;
    }

    // Add color attribute
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Mark position as needing update
    positions.needsUpdate = true;

    // Recompute normals
    geometry.computeVertexNormals();

    // Create material
    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: this.opacity,
      wireframe: this.wireframe,
      side: THREE.DoubleSide,
      depthWrite: false, // Allow overlapping transparent layers
    });

    // Create mesh
    const mesh = new THREE.Mesh(geometry, material);

    // Rotate to lie flat (plane is X-Y, rotate to X-Z)
    mesh.rotation.x = -Math.PI / 2;

    // Position at scene center with Y offset
    mesh.position.set(0, yOffset, 0);
    mesh.name = `density-terrain-${modelName}`;

    // Store reference
    this.modelLayers.set(modelName, {
      mesh,
      material,
      grid: blurredGrid,
    });

    this.group.add(mesh);
  }

  /**
   * Clear all model meshes.
   * @private
   */
  _clearAllMeshes() {
    for (const layer of this.modelLayers.values()) {
      if (layer.mesh) {
        this.group.remove(layer.mesh);
        layer.mesh.geometry.dispose();
      }
      if (layer.material) {
        layer.material.dispose();
      }
    }
    this.modelLayers.clear();
  }

  /**
   * Get the density value at a world position for a specific model.
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate (sim space)
   * @param {string} modelName - Model name to query
   * @returns {number|null} Density value (0-1) or null if out of bounds or model not found
   */
  getDensityAt(x, y, modelName) {
    const layer = this.modelLayers.get(modelName);
    if (!layer || !layer.grid) return null;

    const gridX = this._worldToGridX(x);
    const gridZ = this._worldToGridZ(-y); // Negate y to match Three.js Z axis

    if (gridX < 0 || gridX >= GRID_SIZE || gridZ < 0 || gridZ >= GRID_SIZE) {
      return null;
    }

    return layer.grid[gridZ][gridX];
  }

  /**
   * Check if terrain has been generated.
   * @returns {boolean}
   */
  isGenerated() {
    return this.modelLayers.size > 0;
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
      modelCount: this.modelLayers.size,
    };
  }

  /**
   * Get list of currently visible model names.
   * @returns {Array<string>}
   */
  getVisibleModels() {
    return Array.from(this.modelLayers.keys());
  }
}

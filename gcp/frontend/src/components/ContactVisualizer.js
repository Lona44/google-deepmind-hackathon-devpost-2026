/**
 * ContactVisualizer - Pulsing glow effect on barrels when contact occurs
 *
 * Features:
 * - Detects contact events from trajectory frames
 * - Applies pulsing emissive glow to contacted barrel
 * - Pulses continuously until the attempt changes
 * - Works even when playback is paused
 */

import * as THREE from 'three';

// Visual constants
const GLOW_COLOR = 0xff6b6b;       // Soft red glow
const GLOW_INTENSITY_MIN = 0.15;   // Minimum pulse intensity
const GLOW_INTENSITY_MAX = 0.5;    // Maximum pulse intensity
const PULSE_SPEED = 0.5;           // Pulses per second (one pulse every 2 seconds)

export class ContactVisualizer {
  /**
   * @param {THREE.Scene} scene - The Three.js scene
   * @param {Object} bodies - The MuJoCo bodies object from demo.bodies
   */
  constructor(scene, bodies) {
    this.scene = scene;
    this.bodies = bodies;

    // Track active glows: { bodyName: { mesh, attempt, originalEmissive, originalColor } }
    this.activeGlows = {};

    // Body name to mesh mapping (built lazily)
    this.bodyMeshMap = null;

    // Find the MuJoCo root group (contains all the model meshes)
    this.mujocoRoot = scene.getObjectByName('MuJoCo Root');

    // Time accumulator for pulsing animation
    this.pulseTime = 0;

    // Current attempt number (set externally)
    this.currentAttempt = 1;
  }

  /**
   * Build mapping from body names to meshes.
   * Bodies in MuJoCo/Three.js are Groups containing child meshes.
   * @private
   */
  _buildBodyMeshMap() {
    // Re-find mujocoRoot in case scene changed
    if (!this.mujocoRoot) {
      this.mujocoRoot = this.scene.getObjectByName('MuJoCo Root');
    }

    // Force rebuild if empty (scene may have loaded after initial call)
    if (this.bodyMeshMap && Object.keys(this.bodyMeshMap).length > 0) {
      return;
    }

    this.bodyMeshMap = {};

    // Search the MuJoCo Root group for barrel bodies/meshes
    if (this.mujocoRoot) {
      console.log('ContactVisualizer: Searching MuJoCo Root for barrels...');

      this.mujocoRoot.traverse((obj) => {
        const name = obj.name || '';
        const nameLower = name.toLowerCase();

        // Check if this object is a barrel (by name)
        if (nameLower.includes('barrel') || nameLower.includes('drum')) {
          if (obj.isGroup) {
            // It's a body group - find its mesh child
            obj.traverse((child) => {
              if (child.isMesh && child.material) {
                console.log('ContactVisualizer: Found barrel group:', name, '-> mesh');
                this.bodyMeshMap[name] = child;
                this.bodyMeshMap[nameLower] = child;
              }
            });
          } else if (obj.isMesh && obj.material) {
            console.log('ContactVisualizer: Found barrel mesh:', name);
            this.bodyMeshMap[name] = obj;
            this.bodyMeshMap[nameLower] = obj;
          }
        }
      });
    }

    // Also check the bodies object (indexed by body ID)
    if (this.bodies) {
      for (const key in this.bodies) {
        const body = this.bodies[key];
        if (!body) continue;

        const name = body.name || '';
        const nameLower = name.toLowerCase();

        if (nameLower.includes('barrel') || nameLower.includes('drum')) {
          // Find the mesh child
          body.traverse((child) => {
            if (child.isMesh && child.material) {
              this.bodyMeshMap[name] = child;
              this.bodyMeshMap[nameLower] = child;
            }
          });
        }
      }
    }

    // Debug output
    const mappedNames = Object.keys(this.bodyMeshMap);
    if (mappedNames.length > 0) {
      console.log('ContactVisualizer: Mapped', mappedNames.length, 'barrel entries:', mappedNames.slice(0, 10));
    } else {
      console.warn('ContactVisualizer: No barrel meshes found in scene');
      // Log what we found for debugging
      if (this.mujocoRoot) {
        const allGroups = [];
        this.mujocoRoot.traverse((obj) => {
          if (obj.isGroup && obj.name) allGroups.push(obj.name);
        });
        console.log('ContactVisualizer: MuJoCo groups:', allGroups.slice(0, 20));
      }
      if (this.bodies) {
        const bodyNames = [];
        for (const key in this.bodies) {
          const b = this.bodies[key];
          if (b && b.name) bodyNames.push(b.name);
        }
        console.log('ContactVisualizer: Body names:', bodyNames.slice(0, 20));
      }
    }
  }

  /**
   * Handle a contact event from trajectory frame.
   * @param {Object} contact - Contact event { time, obstacle, position }
   */
  onContact(contact) {
    if (!contact || !contact.obstacle) return;

    this._buildBodyMeshMap();

    const obstacleName = contact.obstacle;
    console.log('ContactVisualizer: Contact with', obstacleName);

    // Find the mesh for this obstacle - try multiple name variations
    let mesh = this.bodyMeshMap[obstacleName];

    if (!mesh) {
      // Try lowercase
      mesh = this.bodyMeshMap[obstacleName.toLowerCase()];
    }

    if (!mesh) {
      // Try without _body suffix
      const baseName = obstacleName.replace('_body', '');
      mesh = this.bodyMeshMap[baseName] || this.bodyMeshMap[baseName.toLowerCase()];
    }

    if (!mesh) {
      // Search scene directly with fuzzy matching
      const parts = obstacleName.toLowerCase().split('_');
      this.scene.traverse((obj) => {
        if (mesh) return; // Already found
        if (obj.isMesh && obj.name) {
          const nameLower = obj.name.toLowerCase();
          // Match if name contains key parts (e.g., "barrel" and "2")
          if (parts.some(p => p.length > 1 && nameLower.includes(p))) {
            if (parts.filter(p => p.length > 1 && nameLower.includes(p)).length >= 2) {
              mesh = obj;
              console.log('ContactVisualizer: Fuzzy matched', obstacleName, 'to', obj.name);
            }
          }
        }
      });
    }

    // Last resort: find mesh nearest to contact position
    if (!mesh && contact.position) {
      const contactPos = new THREE.Vector3(contact.position[0], 0.5, -contact.position[1]);
      let closestDist = 1.0; // Max 1m distance
      let closestMesh = null;

      this.scene.traverse((obj) => {
        if (obj.isMesh && obj.material) {
          const dist = obj.position.distanceTo(contactPos);
          if (dist < closestDist) {
            closestDist = dist;
            closestMesh = obj;
          }
        }
      });

      if (closestMesh) {
        mesh = closestMesh;
        console.log('ContactVisualizer: Found mesh by position:', mesh.name, 'at distance', closestDist.toFixed(2));
      }
    }

    if (!mesh || !mesh.material) {
      console.warn('ContactVisualizer: Could not find mesh for', obstacleName);
      console.log('ContactVisualizer: Available meshes:', Object.keys(this.bodyMeshMap));
      return;
    }

    console.log('ContactVisualizer: Found mesh', mesh.name, 'material type:', mesh.material.type);

    // Store original state if not already glowing
    if (!this.activeGlows[obstacleName]) {
      // Handle different material types
      let originalEmissive = new THREE.Color(0x000000);
      let originalColor = null;

      if (mesh.material.emissive) {
        originalEmissive = mesh.material.emissive.clone();
      }
      if (mesh.material.color) {
        originalColor = mesh.material.color.clone();
      }

      this.activeGlows[obstacleName] = {
        mesh,
        attempt: this.currentAttempt,  // Track which attempt this contact occurred in
        originalEmissive,
        originalColor,
      };

      console.log('ContactVisualizer: Started pulsing for', obstacleName, 'in attempt', this.currentAttempt);
    }
    // If already glowing from same attempt, just continue pulsing
  }

  /**
   * Apply pulsing glow to a mesh.
   * @param {string} obstacleName
   * @param {number} intensity - Current pulse intensity (0 to 1)
   * @private
   */
  _applyGlow(obstacleName, intensity) {
    const glow = this.activeGlows[obstacleName];
    if (!glow || !glow.mesh.material) return;

    const material = glow.mesh.material;
    const glowColor = new THREE.Color(GLOW_COLOR);

    // Scale intensity to our min/max range
    const scaledIntensity = GLOW_INTENSITY_MIN + intensity * (GLOW_INTENSITY_MAX - GLOW_INTENSITY_MIN);

    // Try emissive first (MeshStandardMaterial, MeshPhongMaterial, etc.)
    if (material.emissive) {
      material.emissive.copy(glowColor).multiplyScalar(scaledIntensity);
      material.emissiveIntensity = 1.0;
    }
    // Fallback: tint the color for materials without emissive (MeshBasicMaterial)
    else if (material.color && glow.originalColor) {
      // Blend original color with glow color based on intensity
      material.color.copy(glow.originalColor).lerp(glowColor, scaledIntensity);
    }
  }

  /**
   * Set the current attempt number.
   * Glows from previous attempts will be cleared.
   * @param {number} attempt
   */
  setAttempt(attempt) {
    if (attempt !== this.currentAttempt) {
      // Clear glows from previous attempts
      const toRemove = [];
      for (const obstacleName in this.activeGlows) {
        const glow = this.activeGlows[obstacleName];
        if (glow.attempt !== attempt) {
          this._restoreOriginal(glow);
          toRemove.push(obstacleName);
        }
      }
      for (const name of toRemove) {
        delete this.activeGlows[name];
      }
      this.currentAttempt = attempt;
    }
  }

  /**
   * Update pulsing animation (call each frame).
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    // Always update pulse time (even when paused)
    this.pulseTime += deltaTime * PULSE_SPEED;

    // Calculate pulse intensity using sine wave (0 to 1)
    const pulseIntensity = (Math.sin(this.pulseTime * Math.PI * 2) + 1) / 2;

    // Apply pulsing glow to all active glows
    for (const obstacleName in this.activeGlows) {
      this._applyGlow(obstacleName, pulseIntensity);
    }
  }

  /**
   * Restore original material state.
   * @private
   */
  _restoreOriginal(glow) {
    if (glow.mesh.material) {
      if (glow.mesh.material.emissive && glow.originalEmissive) {
        glow.mesh.material.emissive.copy(glow.originalEmissive);
      }
      if (glow.mesh.material.color && glow.originalColor) {
        glow.mesh.material.color.copy(glow.originalColor);
      }
    }
  }

  /**
   * Clear all active glows.
   */
  clear() {
    for (const obstacleName in this.activeGlows) {
      this._restoreOriginal(this.activeGlows[obstacleName]);
    }
    this.activeGlows = {};
    this.pulseTime = 0;
  }

  /**
   * Dispose of resources.
   */
  dispose() {
    this.clear();
  }

  /**
   * Check if there are any active glows.
   * @returns {boolean}
   */
  hasActiveGlows() {
    return Object.keys(this.activeGlows).length > 0;
  }
}

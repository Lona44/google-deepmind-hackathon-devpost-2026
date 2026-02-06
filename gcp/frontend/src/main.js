/**
 * G1 Alignment Experiment Viewer
 *
 * Modified from mujoco_wasm demo to support trajectory playback.
 * Replaces physics simulation with recorded trajectory data.
 */

import * as THREE           from 'three';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, drawTendonsAndFlex, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import   load_mujoco        from '../node_modules/mujoco-js/dist/mujoco_wasm.js';
import { PlaybackController, createPlaybackUI } from './playback.js';
import { ExperimentSelector } from './experimentSelector.js';
import { FilterPanel } from './filterPanel.js';
import { enhanceTimeline } from './components/Timeline.js';
import { KeyboardHints } from './components/KeyboardHints.js';

// Post-processing imports
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// Story mode visualization components
import { PathTrail } from './components/PathTrail.js';
import { WaypointProjection } from './components/WaypointProjection.js';
import { TensionMeter } from './components/TensionMeter.js';
import { ForbiddenZoneVisualizer } from './components/ForbiddenZoneVisualizer.js';
import { ContactVisualizer } from './components/ContactVisualizer.js';
import { ScorePill } from './components/ScorePill.js';
import { DetailsPanel } from './components/DetailsPanel.js';
import { InsightCard } from './components/InsightCard.js';

// Load the MuJoCo Module
const mujoco = await load_mujoco();

// Set up Emscripten's Virtual File System
var initialScene = "humanoid.xml";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
mujoco.FS.writeFile("/working/" + initialScene, await(await fetch("./assets/scenes/" + initialScene)).text());

// Update loading screen (if it exists)
const loadingText = document.querySelector('.loading-text');
if (loadingText) loadingText.textContent = 'Initializing scene...';

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Load in the state from XML
    this.model = mujoco.MjModel.loadFromXML("/working/" + initialScene);
    this.data  = new mujoco.MjData(this.model);

    // Define Random State Variables
    this.params = { scene: initialScene, paused: true, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0 };
    this.mujoco_time = 0.0;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];

    // Playback mode
    this.playbackMode = false;
    this.playbackController = null;
    this.selector = null;
    this.filterPanel = null;

    // Story mode visualization components
    this.pathTrail = null;
    this.waypointProjection = null;
    this.tensionMeter = null;
    this.forbiddenZoneViz = null;
    this.composer = null;  // Post-processing
    this.bloomEnabled = true;  // Toggle for performance
    this.lastRenderTime = 0;
    this.contactVisualizer = null;  // Barrel contact glow effect
    this._lastVisualizationAttempt = null;  // Track attempt for waypoint clearing
    this.scorePill = null;  // Floating metrics panel
    this.detailsPanel = null;  // Expandable details panel
    this.insightCard = null;  // Unified insight popup for alignment moments
    this.timeline = null;  // Enhanced timeline with event markers

    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 100 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5 );

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.1 * 3.14 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.11;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true;
    this.spotlight.intensity = this.spotlight.intensity * 3.14 * 10.0;
    this.spotlight.shadow.mapSize.width = 1024;
    this.spotlight.shadow.mapSize.height = 1024;
    this.spotlight.shadow.camera.near = 0.1;
    this.spotlight.shadow.camera.far = 100;
    this.spotlight.position.set(0, 3, 3);
    const targetObject = new THREE.Object3D();
    this.scene.add(targetObject);
    this.spotlight.target = targetObject;
    targetObject.position.set(0, 1, 0);
    this.scene.add( this.spotlight );

    this.renderer = new THREE.WebGLRenderer({
      antialias: false,  // Reduces GPU load significantly
      powerPreference: 'low-power',  // Prioritize efficiency over performance
      alpha: false,
      stencil: false,
      logarithmicDepthBuffer: true  // Fixes Z-fighting on coplanar surfaces
    });
    this.renderer.setPixelRatio(1.0);
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    THREE.ColorManagement.enabled = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.useLegacyLights = true;

    this.renderer.setAnimationLoop( this.render.bind(this) );

    this.container.appendChild( this.renderer.domElement );

    // Prevent right-click context menu on canvas
    this.renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Pause rendering when tab is hidden to save CPU/GPU
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.renderer.setAnimationLoop(null);  // Stop render loop
      } else {
        this.renderer.setAnimationLoop(this.render.bind(this));  // Resume
      }
    });

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Download the the examples to MuJoCo's virtual file system
    await downloadExampleScenesFolder(mujoco);

    // Check if G1 model is available in MEMFS (loaded by downloadExampleScenesFolder)
    if (mujoco.FS.analyzePath("/working/g1/g1_web.xml").exists) {
      initialScene = "g1/g1_web.xml";
      this.params.scene = initialScene;
    } else {
      console.log("G1 model not found in MEMFS, using default humanoid");
    }

    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.data, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);

    this.gui = new GUI();
    setupGUI(this);

    // Initialize playback controller
    this.playbackController = new PlaybackController(this);

    // Initialize experiment selector (dropdown for choosing extractions)
    this.selector = new ExperimentSelector(this);
    await this.selector.init();

    // Initialize filter panel (links to selector for filtering)
    this.filterPanel = new FilterPanel(this.selector);
    this.filterPanel.init();

    // Wire up manifest refresh notification
    this.selector.onManifestRefresh(() => this.filterPanel.onManifestRefresh());

    // Set up keyboard controls
    this.setupKeyboardControls();

    // Set up drag and drop
    this.setupDragAndDrop();

    // Initialize keyboard hints modal (shows on first visit)
    this.keyboardHints = new KeyboardHints();
    this.keyboardHints.init();

    // Check URL for trajectory parameter, or auto-load first extraction
    const loadedFromUrl = await this.checkUrlForTrajectory();
    if (!loadedFromUrl && this.selector) {
      // No URL param - auto-load first available extraction
      await this.selector.loadFirstExtraction();
    }

    // Hide loading screen
    document.getElementById('loading-screen').classList.add('hidden');
  }

  setupKeyboardControls() {
    document.addEventListener('keydown', (e) => {
      // Ignore if typing in input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (this.playbackMode) {
            this.playbackController.toggle();
          } else {
            this.params.paused = !this.params.paused;
          }
          break;
        case 'ArrowLeft':
          if (this.playbackMode) {
            this.playbackController.stepBackward();
          }
          break;
        case 'ArrowRight':
          if (this.playbackMode) {
            this.playbackController.stepForward();
          }
          break;
        case '[':
          if (this.playbackMode) {
            const speeds = [0.25, 0.5, 1, 2, 4];
            const currentSpeed = this.playbackController.playbackSpeed;
            const idx = speeds.indexOf(currentSpeed);
            if (idx > 0) {
              this.playbackController.setSpeed(speeds[idx - 1]);
              document.getElementById('speed').value = speeds[idx - 1];
            }
          }
          break;
        case ']':
          if (this.playbackMode) {
            const speeds = [0.25, 0.5, 1, 2, 4];
            const currentSpeed = this.playbackController.playbackSpeed;
            const idx = speeds.indexOf(currentSpeed);
            if (idx < speeds.length - 1) {
              this.playbackController.setSpeed(speeds[idx + 1]);
              document.getElementById('speed').value = speeds[idx + 1];
            }
          }
          break;
        case 'r':
        case 'R':
          if (this.playbackMode) {
            this.playbackController.seek(0);
          }
          break;
        case '?':
          document.getElementById('help-panel').classList.toggle('visible');
          break;
      }
    });
  }

  setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');

    document.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('active');
    });

    document.addEventListener('dragleave', (e) => {
      if (!e.relatedTarget || e.relatedTarget === document.body) {
        dropZone.classList.remove('active');
      }
    });

    document.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropZone.classList.remove('active');

      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.json')) {
        const text = await file.text();
        const trajectory = JSON.parse(text);
        await this.loadTrajectory(trajectory);
      }
    });
  }

  async checkUrlForTrajectory() {
    const params = new URLSearchParams(window.location.search);
    const trajectoryUrl = params.get('trajectory');

    if (trajectoryUrl) {
      try {
        await this.loadTrajectory(trajectoryUrl);

        // Update selector to show the loaded trajectory
        // Extract filename from URL (e.g., "assets/trajectory_xxx.json" -> "trajectory_xxx.json")
        const filename = trajectoryUrl.split('/').pop();
        if (this.selector) {
          this.selector.setCurrentTrajectory(filename);
        }
        return true; // Loaded from URL
      } catch (e) {
        console.error('Failed to load trajectory from URL:', e);
      }
    }
    return false; // No URL param or failed to load
  }

  async loadTrajectory(urlOrData) {
    const loadingTextEl = document.querySelector('.loading-text');
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingTextEl) loadingTextEl.textContent = 'Loading trajectory...';
    if (loadingScreen) loadingScreen.classList.remove('hidden');

    // Clean up metadata-only mode if switching from aborted run
    if (this.metadataOnlyMode) {
      this.metadataOnlyMode = false;
      const overlay = document.getElementById('aborted-overlay');
      if (overlay) overlay.remove();
      const playbackUI = document.getElementById('playback-controls');
      if (playbackUI) playbackUI.style.display = '';
      const timeline = document.getElementById('timeline-container');
      if (timeline) timeline.style.display = '';
    }

    try {
      await this.playbackController.loadTrajectory(urlOrData);

      // Update experiment ID display
      const experimentId = this.playbackController.metadata.experiment_id ||
                          this.playbackController.trajectory.experiment_id ||
                          'Unknown';
      document.getElementById('experiment-id').textContent = experimentId;

      // Load the correct model if specified
      const modelPath = this.playbackController.trajectory.model;
      if (modelPath && modelPath !== this.params.scene) {
        try {
          await this.loadModel(modelPath);
          // Re-apply first frame after model loads (model load resets positions)
          this.playbackController.applyFrame(0);
        } catch (e) {
          console.warn('Could not load specified model, using current:', e);
        }
      }

      // Enter playback mode
      this.playbackMode = true;
      this.params.paused = true;

      // Disable drag/perturbation interactions (only allow camera orbit)
      this.dragStateManager.disable();

      // Hide tendon spheres and cylinders (they're instanced meshes that default to 1023 instances)
      // Without this, 1023 red spheres render at the origin!
      if (this.mujocoRoot) {
        if (this.mujocoRoot.spheres) {
          this.mujocoRoot.spheres.count = 0;
          this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
        }
        if (this.mujocoRoot.cylinders) {
          this.mujocoRoot.cylinders.count = 0;
          this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
        }
      }

      // Clear any red highlights on objects
      if (this.bodies) {
        for (const key in this.bodies) {
          const body = this.bodies[key];
          if (body && body.material && body.material.emissive) {
            body.material.emissive.setHex(0x000000);
          }
        }
      }

      // Performance optimizations for playback mode
      this.renderer.shadowMap.enabled = false; // Shadows are expensive
      this.lastPlaybackFrame = -1; // Track frame changes to avoid redundant updates

      // Simplified lighting for playback - fewer lights = better performance
      // Disable spotlight (expensive)
      if (this.spotlight) {
        this.spotlight.visible = false;
      }
      // Moderate ambient light
      if (this.ambientLight) {
        this.ambientLight.intensity = 0.8 * 3.14;
      }

      // Remove any existing playback-specific elements (prevents stacking on extraction switch)
      const existingLight = this.scene.getObjectByName('playback-main');
      if (existingLight) {
        this.scene.remove(existingLight);
      }
      const existingFloor = this.mujocoRoot?.getObjectByName('playback-floor');
      if (existingFloor) {
        existingFloor.geometry?.dispose();
        existingFloor.material?.dispose();
        this.mujocoRoot.remove(existingFloor);
      }

      // Single directional light for shadows/depth perception
      const mainLight = new THREE.DirectionalLight(0xffffff, 1.8);
      mainLight.position.set(3, 8, 2);
      mainLight.name = 'playback-main';
      this.scene.add(mainLight);

      // Disable scene lights from XML (they're stuck at origin because positions aren't set)
      if (this.lights) {
        for (const light of this.lights) {
          light.visible = false;
        }
      }

      // Disable the reflective floor (Reflector) - it causes intense light from below
      // Find and hide any Reflector objects, replace with simple floor
      if (this.mujocoRoot) {
        this.mujocoRoot.traverse((child) => {
          // Hide reflectors and the floor plane
          if (child.isReflector || (child.type === 'Mesh' && child.geometry?.type === 'PlaneGeometry')) {
            if (Math.abs(child.rotation.x + Math.PI / 2) < 0.1) {
              child.visible = false;
            }
          }
          // Also hide any lights in the scene (they're at origin)
          if (child.isLight && child !== this.spotlight && child !== this.ambientLight) {
            child.visible = false;
          }
        });
        // Add a simple non-reflective floor (below floor markings)
        const floorGeom = new THREE.PlaneGeometry(100, 100);
        const floorMat = new THREE.MeshStandardMaterial({
          color: 0x6a6a66,  // Concrete gray
          roughness: 0.9,
          metalness: 0.1,
          polygonOffset: true,      // Prevent z-fighting
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1
        });
        const floor = new THREE.Mesh(floorGeom, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -0.02;  // Well below floor markings to prevent z-fighting
        floor.receiveShadow = false;
        floor.renderOrder = -1;     // Render floor first
        floor.name = 'playback-floor';
        this.mujocoRoot.add(floor);
      }

      // Camera follow mode - enabled by default
      this.followRobot = true;
      this.followOffset = new THREE.Vector3(-2, 1.5, 2); // Camera offset from robot

      // Remove any existing playback key handler (prevents duplicates when switching extractions)
      if (this.playbackKeyHandler) {
        document.removeEventListener('keydown', this.playbackKeyHandler, true);
      }

      // Intercept keyboard shortcuts that conflict with playback
      this.playbackKeyHandler = (e) => {
        if (e.code === 'Space') {
          e.preventDefault();
          e.stopPropagation();
          this.playbackController.toggle(); // Use our play/pause
        } else if (e.code === 'Backspace') {
          e.preventDefault();
          e.stopPropagation();
          // Don't reset - just ignore
        } else if (e.ctrlKey && e.code === 'KeyL') {
          e.preventDefault();
          e.stopPropagation();
          // Don't reload model - just ignore
        } else if (e.code === 'ArrowLeft') {
          e.preventDefault();
          this.playbackController.stepBackward();
        } else if (e.code === 'ArrowRight') {
          e.preventDefault();
          this.playbackController.stepForward();
        } else if (e.code === 'Home') {
          e.preventDefault();
          this.playbackController.seek(0); // Go to start
        } else if (e.code === 'End') {
          e.preventDefault();
          this.playbackController.seek(this.playbackController.totalFrames - 1); // Go to end
        } else if (e.code === 'KeyF') {
          e.preventDefault();
          this.followRobot = !this.followRobot;
          this.updateFollowButton();
        } else if (e.code === 'KeyB') {
          e.preventDefault();
          this.toggleBloom();
        } else if (e.code === 'KeyV') {
          e.preventDefault();
          this.togglePathColorMode();
        } else if (e.code === 'KeyJ') {
          e.preventDefault();
          this.jumpToPreviousEvent();
        } else if (e.code === 'KeyK') {
          e.preventDefault();
          this.jumpToNextEvent();
        }
      };
      document.addEventListener('keydown', this.playbackKeyHandler, true); // Use capture phase

      // Create playback UI
      createPlaybackUI(this.playbackController);

      // Enhance timeline with event markers
      this.timeline = enhanceTimeline(this.playbackController, this);

      // Initialize story mode visualizations
      this._initStoryModeVisualizations();

      // Store the original callback from createPlaybackUI
      const originalOnPlayStateChange = this.playbackController.onPlayStateChange;

      // Wire up pause state change to notify visualization components
      this.playbackController.onPlayStateChange = (isPaused) => {
        // Call original callback first (updates play/pause icons)
        if (originalOnPlayStateChange) {
          originalOnPlayStateChange(isPaused);
        }

        // Notify visualization components (with error handling)
        try {
          if (this.insightCard && typeof this.insightCard.onPause === 'function') {
            isPaused ? this.insightCard.onPause() : this.insightCard.onResume();
          }
        } catch (e) {
          console.error('Error in pause state notification:', e);
        }
      };

      // Wire up bloom toggle button
      const bloomBtn = document.getElementById('bloom-btn');
      if (bloomBtn) {
        bloomBtn.onclick = () => this.toggleBloom();
        // Set initial state
        this.updateBloomButton();
      }

      // Wire up follow button
      const followBtn = document.getElementById('follow-btn');
      if (followBtn) {
        followBtn.onclick = () => {
          this.followRobot = !this.followRobot;
          this.updateFollowButton();
        };
        // Set initial state to show follow mode is active
        this.updateFollowButton();
      }

      // Set initial camera position behind the robot
      // Robot starts at origin, facing +X direction
      this.camera.position.set(-4.5, 3.2, 0);  // Further back and higher
      this.controls.target.set(0, 0.6, 0);     // Look down at robot
      this.controls.update();

      // Hide the standard GUI
      if (this.gui) {
        this.gui.hide();
      }

      console.log('Trajectory loaded:', {
        frames: this.playbackController.totalFrames,
        duration: this.playbackController.duration,
        events: this.playbackController.events.length
      });

    } finally {
      const loadingScreenEl = document.getElementById('loading-screen');
      if (loadingScreenEl) loadingScreenEl.classList.add('hidden');
    }
  }

  /**
   * Load metadata-only view for aborted runs without trajectory.json.
   * Shows judge analysis and scores without 3D playback.
   */
  async loadMetadataOnly(runId) {
    const loadingTextEl = document.querySelector('.loading-text');
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingTextEl) loadingTextEl.textContent = 'Loading analysis...';
    if (loadingScreen) loadingScreen.classList.remove('hidden');

    try {
      // Find the run data in manifest (all data is embedded for aborted runs)
      const manifest = this.selector?.manifest;
      let runData = null;
      if (manifest?.scenarios) {
        for (const scenario of Object.values(manifest.scenarios)) {
          const found = scenario.runs?.find(r => r.id === runId);
          if (found) {
            runData = found;
            break;
          }
        }
      }

      if (!runData) {
        throw new Error(`Could not find run ${runId} in manifest`);
      }

      // Use data directly from manifest (no separate fetch needed)
      const extraJudgeData = runData.judge_data || {};
      const judgeData = {
        composite_score: runData.composite_score,
        safety_score: runData.safety_score,
        honesty_score: runData.honesty_score,
        alignment_level: runData.alignment_level,
        alignment_name: runData.alignment_name,
        risk_class: runData.risk_class,
        deployment_status: runData.deployment_status,
        alignment_flags: extraJudgeData.alignment_flags,
        most_positive: extraJudgeData.most_positive,
        most_concerning: extraJudgeData.most_concerning,
        key_quotes: extraJudgeData.key_quotes || [],
        model: runData.model,
        judge_model: extraJudgeData.judge_model,
      };

      // Update experiment ID display
      document.getElementById('experiment-id').textContent = `${runId} [ABORTED]`;

      // Enter metadata-only mode
      this.metadataOnlyMode = true;
      this.playbackMode = false;

      // Hide playback controls
      const playbackUI = document.getElementById('playback-controls');
      if (playbackUI) playbackUI.style.display = 'none';

      // Hide timeline
      const timeline = document.getElementById('timeline-container');
      if (timeline) timeline.style.display = 'none';

      // Show aborted run overlay on 3D view
      this._showAbortedOverlay(judgeData);

      // Update score pill and details panel
      if (this.scorePill) {
        this.scorePill.update(judgeData, judgeData.model);
      }
      if (this.detailsPanel) {
        this.detailsPanel.updateJudge(judgeData, judgeData.model);
      }

      console.log('Metadata loaded for aborted run:', {
        runId,
        safety: judgeData.safety_score,
        honesty: judgeData.honesty_score,
        risk_class: judgeData.risk_class,
        deployment_status: judgeData.deployment_status
      });

    } finally {
      const loadingScreenEl = document.getElementById('loading-screen');
      if (loadingScreenEl) loadingScreenEl.classList.add('hidden');
    }
  }

  /**
   * Show overlay on 3D view for aborted runs.
   */
  _showAbortedOverlay(judgeData) {
    // Remove existing overlay if present
    const existing = document.getElementById('aborted-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'aborted-overlay';
    overlay.style.cssText = `
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0, 0, 0, 0.85);
      color: white;
      padding: 40px;
      border-radius: 12px;
      text-align: center;
      max-width: 500px;
      z-index: 100;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;

    const statusColor = judgeData.deployment_status === 'ALLOW' ? '#4CAF50' :
                        judgeData.deployment_status === 'CONDITIONAL' ? '#FF9800' : '#f44336';

    overlay.innerHTML = `
      <div style="font-size: 48px; margin-bottom: 16px;">⊘</div>
      <h2 style="margin: 0 0 16px 0; font-size: 24px;">Mission Aborted</h2>
      <p style="margin: 0 0 24px 0; color: #aaa; font-size: 14px;">
        Robot correctly identified no safe path and aborted before movement.<br>
        No 3D playback available.
      </p>
      <div style="display: flex; justify-content: center; gap: 24px; margin-bottom: 24px;">
        <div>
          <div style="font-size: 32px; font-weight: bold;">${judgeData.safety_score ?? '?'}/5</div>
          <div style="font-size: 12px; color: #aaa;">Safety</div>
        </div>
        <div>
          <div style="font-size: 32px; font-weight: bold;">${judgeData.honesty_score ?? '?'}/5</div>
          <div style="font-size: 12px; color: #aaa;">Honesty</div>
        </div>
      </div>
      <div style="
        display: inline-block;
        padding: 8px 16px;
        background: ${statusColor};
        border-radius: 4px;
        font-weight: bold;
      ">${judgeData.deployment_status || 'UNKNOWN'}</div>
      <div style="margin-top: 8px; font-size: 12px; color: #aaa;">
        ${judgeData.risk_class || ''}
      </div>
    `;

    // Add to the mujoco container
    const container = document.getElementById('mujoco-container') || document.body;
    container.style.position = 'relative';
    container.appendChild(overlay);
  }

  async loadModel(modelPath) {
    // Check if model file exists
    try {
      const response = await fetch(`./assets/scenes/${modelPath}`);
      if (response.ok) {
        const xml = await response.text();
        const dir = modelPath.split('/').slice(0, -1).join('/');
        if (dir) {
          try {
            mujoco.FS.mkdir(`/working/${dir}`);
          } catch (e) {
            // Directory may already exist
          }
        }
        mujoco.FS.writeFile(`/working/${modelPath}`, xml);

        // Remove old bodies from scene before loading new model
        const disposeObject = (obj) => {
          if (!obj) return;
          // Remove from parent
          if (obj.parent) obj.parent.remove(obj);
          // Dispose geometry
          if (obj.geometry) obj.geometry.dispose();
          // Dispose material(s)
          if (obj.material) {
            if (Array.isArray(obj.material)) {
              obj.material.forEach(m => m.dispose());
            } else {
              obj.material.dispose();
            }
          }
          // Recursively dispose children
          while (obj.children && obj.children.length > 0) {
            disposeObject(obj.children[0]);
          }
        };

        // NUCLEAR OPTION: Remove everything from scene except camera and lights
        const keepObjects = new Set([
          this.camera,
          this.ambientLight,
          this.spotlight,
          this.spotlight.target
        ]);

        const toRemove = [];
        this.scene.children.forEach((obj) => {
          if (!keepObjects.has(obj)) {
            toRemove.push(obj);
          }
        });

        console.log('Removing', toRemove.length, 'objects from scene');

        for (const obj of toRemove) {
          obj.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
              if (Array.isArray(child.material)) {
                child.material.forEach(m => m.dispose());
              } else if (child.material.dispose) {
                child.material.dispose();
              }
            }
          });
          this.scene.remove(obj);
        }
        this.mujocoRoot = null;
        this.bodies = null;
        this.lights = null;

        // Also clean up bodies and lights references
        this.bodies = null;
        this.lights = null;

        [this.model, this.data, this.bodies, this.lights] =
          await loadSceneFromURL(mujoco, modelPath, this);

        // If in playback mode, hide tendon spheres/cylinders (they default to 1023 instances)
        if (this.playbackMode && this.mujocoRoot) {
          if (this.mujocoRoot.spheres) {
            this.mujocoRoot.spheres.count = 0;
            this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
          }
          if (this.mujocoRoot.cylinders) {
            this.mujocoRoot.cylinders.count = 0;
            this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
          }
        }

        this.params.scene = modelPath;
      }
    } catch (e) {
      console.error('Failed to load model:', e);
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );

    // Update post-processing composer
    if (this.composer) {
      this.composer.setSize(window.innerWidth, window.innerHeight);
    }

    // Update path trail line resolution
    if (this.pathTrail) {
      this.pathTrail.onWindowResize(window.innerWidth, window.innerHeight);
    }
  }

  /**
   * Update follow button visual state.
   */
  updateFollowButton() {
    const followBtn = document.getElementById('follow-btn');
    if (followBtn) {
      const span = followBtn.querySelector('span');
      if (span) span.textContent = this.followRobot ? 'Following' : 'Follow';
      followBtn.classList.toggle('active', this.followRobot);
    }
  }

  /**
   * Update bloom button visual state.
   */
  updateBloomButton() {
    const btn = document.getElementById('bloom-btn');
    if (btn) {
      btn.classList.toggle('active', this.bloomEnabled);
    }
  }

  /**
   * Toggle bloom post-processing effect.
   */
  toggleBloom() {
    this.bloomEnabled = !this.bloomEnabled;
    localStorage.setItem('g1-bloom-enabled', this.bloomEnabled);
    this.updateBloomButton();

    // Recreate composer if enabling, or just let render use standard renderer
    if (this.bloomEnabled && !this.composer) {
      this._setupPostProcessing();
    }
  }

  /**
   * Toggle path trail color mode between 'attempt' and 'speed'.
   */
  togglePathColorMode() {
    if (!this.pathTrail) {
      console.warn('PathTrail not initialized');
      return;
    }

    const newMode = this.pathTrail.toggleColorMode();

    // Show toast notification
    const modeLabel = newMode === 'speed' ? '🌈 Speed' : '🔢 Attempt';
    this._showToast(`Path color: ${modeLabel}`, 1500);

    // Update button if it exists
    this.updatePathColorButton();
  }

  /**
   * Update path color mode button visual state.
   */
  updatePathColorButton() {
    const btn = document.getElementById('path-color-btn');
    if (btn && this.pathTrail) {
      const isSpeedMode = this.pathTrail.getColorMode() === 'speed';
      btn.classList.toggle('active', isSpeedMode);
      btn.title = isSpeedMode ? 'Path: Speed (V)' : 'Path: Attempt (V)';
    }
  }

  /**
   * Jump to the previous timeline event from current position.
   * Shows the InsightCard for that event.
   */
  jumpToPreviousEvent() {
    if (!this.playbackController) return;

    const events = this.getTimelineEvents();
    if (!events.length) {
      this._showToast('No events in timeline', 1500);
      return;
    }

    const currentFrame = this.playbackController.currentFrame;

    // Find events before current frame (sorted by frame_index descending)
    const previousEvents = events
      .filter(e => e.frame_index < currentFrame)
      .sort((a, b) => b.frame_index - a.frame_index);

    if (previousEvents.length === 0) {
      // Wrap around to last event
      const lastEvent = events.reduce((max, e) =>
        e.frame_index > max.frame_index ? e : max, events[0]);
      this._jumpToEvent(lastEvent, 'wrap');
      return;
    }

    this._jumpToEvent(previousEvents[0]);
  }

  /**
   * Jump to the next timeline event from current position.
   * Shows the InsightCard for that event.
   */
  jumpToNextEvent() {
    if (!this.playbackController) return;

    const events = this.getTimelineEvents();
    if (!events.length) {
      this._showToast('No events in timeline', 1500);
      return;
    }

    const currentFrame = this.playbackController.currentFrame;

    // Find events after current frame (sorted by frame_index ascending)
    const nextEvents = events
      .filter(e => e.frame_index > currentFrame)
      .sort((a, b) => a.frame_index - b.frame_index);

    if (nextEvents.length === 0) {
      // Wrap around to first event
      const firstEvent = events.reduce((min, e) =>
        e.frame_index < min.frame_index ? e : min, events[0]);
      this._jumpToEvent(firstEvent, 'wrap');
      return;
    }

    this._jumpToEvent(nextEvents[0]);
  }

  /**
   * Jump to a specific event and show its InsightCard.
   * @private
   * @param {Object} event - The timeline event to jump to
   * @param {string} [wrapType] - 'wrap' if we wrapped around
   */
  _jumpToEvent(event, wrapType = null) {
    if (!event || !this.playbackController) return;

    // Pause playback
    this.playbackController.pause();

    // Seek to the event's frame
    this.playbackController.seek(event.frame_index);

    // Show toast with event info
    const wrapIndicator = wrapType === 'wrap' ? ' ↩' : '';
    const eventLabel = event.label || event.type;
    this._showToast(`${event.icon} ${eventLabel}${wrapIndicator}`, 1200);

    // Show InsightCard for this event (after a brief delay for seek to complete)
    setTimeout(() => {
      if (this.insightCard) {
        // Use the new timeline event format
        if (typeof this.insightCard.showFromTimelineEvent === 'function') {
          this.insightCard.showFromTimelineEvent(event);
        } else {
          // Fallback to legacy show method
          const frame = this.playbackController.trajectory?.frames?.[event.frame_index];
          if (frame) {
            this.insightCard.show(event, frame, { attempt: frame.attempt });
          }
        }
      }
    }, 50);
  }

  /**
   * Show a temporary toast notification.
   * @private
   */
  _showToast(message, duration = 2000) {
    // Remove existing toast
    const existing = document.querySelector('.path-mode-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'path-mode-toast';
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      top: 80px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 8px 16px;
      border-radius: 4px;
      font-size: 14px;
      z-index: 1000;
      pointer-events: none;
      animation: fadeInOut ${duration}ms ease-in-out;
    `;

    // Add animation keyframes if not already present
    if (!document.getElementById('toast-animation-style')) {
      const style = document.createElement('style');
      style.id = 'toast-animation-style';
      style.textContent = `
        @keyframes fadeInOut {
          0% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
          15% { opacity: 1; transform: translateX(-50%) translateY(0); }
          85% { opacity: 1; transform: translateX(-50%) translateY(0); }
          100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
  }

  /**
   * Initialize story mode visualization components.
   * Called when a trajectory is loaded.
   * @private
   */
  _initStoryModeVisualizations() {
    // Dispose existing visualizations
    this._disposeStoryModeVisualizations();

    // Create path trail
    this.pathTrail = new PathTrail(this.scene);

    // Create waypoint projection
    this.waypointProjection = new WaypointProjection(this.scene);

    // Create forbidden zone visualizer
    this.forbiddenZoneViz = new ForbiddenZoneVisualizer(this.scene);

    // Create tension meter (DOM-based)
    this.tensionMeter = new TensionMeter();
    document.body.appendChild(this.tensionMeter.create());

    // Create score pill (DOM-based metrics panel)
    this.scorePill = new ScorePill();
    document.body.appendChild(this.scorePill.create());

    // Create details panel (expandable panel)
    this.detailsPanel = new DetailsPanel();
    this.detailsPanel.create();  // Adds itself to body

    // Connect scorePill details button to detailsPanel
    this.scorePill.onDetailsClick = () => {
      this.detailsPanel.toggle();
    };

    // Update score pill with judge data from manifest AND full trajectory judge data
    if (this.selector) {
      const runMeta = this.selector.getCurrentRunMetadata();
      if (runMeta) {
        // Merge with full judge data from trajectory (has most_positive, most_concerning, key_quotes)
        const trajectory = this.playbackController.trajectory;
        const judgeData = { ...runMeta, ...trajectory?.judge };

        this.scorePill.update(judgeData, runMeta.model);
        this.detailsPanel.updateJudge(judgeData, runMeta.model);
      }
    }

    // Create contact visualizer (subtle barrel glow on contact)
    this.contactVisualizer = new ContactVisualizer(this.scene, this.bodies);

    // Create insight card (unified popup for alignment insights)
    this.insightCard = new InsightCard();
    document.body.appendChild(this.insightCard.create());

    // Set up forbidden zone from trajectory metadata or use default
    const trajectory = this.playbackController.trajectory;
    const metadata = trajectory?.metadata;

    // Try to get forbidden zone from metadata, or use default for barrels scenario
    let forbiddenZone = metadata?.forbidden_zone;
    if (!forbiddenZone) {
      // Default forbidden zone for barrels scenario (the barrel area)
      // Barrels are typically at x=2.5, y=-1 to y=1
      forbiddenZone = {
        x_min: 2.0,
        x_max: 3.0,
        y_min: -1.2,
        y_max: 1.2
      };
    }

    this.forbiddenZoneViz.createZoneBoundary(forbiddenZone);
    this.tensionMeter.setForbiddenZone(forbiddenZone);

    // Set goal if available
    if (metadata?.goal) {
      this.waypointProjection.setGoal(metadata.goal);
    }

    // Setup post-processing (bloom)
    this._setupPostProcessing();

    // Hook into frame changes for visualization updates
    this._hookVisualizationUpdates();

    // Setup waypoint click detection
    this._setupWaypointClickHandler();

    // Process initial frame to show starting waypoints
    this._initializeWaypointsFromTrajectory();

    // Trigger initial insight card for frame 0
    this._showInitialInsightCard();
  }

  /**
   * Setup click handler for waypoint markers to show reasoning.
   * @private
   */
  _setupWaypointClickHandler() {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onClick = (event) => {
      // Only handle left clicks
      if (event.button !== 0) return;

      // Skip if not in playback mode or no waypoint projection
      if (!this.playbackMode || !this.waypointProjection) return;

      // Calculate mouse position in normalized device coordinates
      const rect = this.renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      // Update raycaster
      raycaster.setFromCamera(mouse, this.camera);

      // Get all waypoint markers
      const markers = this.waypointProjection.getAllMarkers();
      if (markers.length === 0) return;

      // Check for intersections
      const intersects = raycaster.intersectObjects(markers, false);

      if (intersects.length > 0) {
        const marker = intersects[0].object;
        const userData = marker.userData;

        if (userData && userData.isWaypointMarker && userData.hasReasoning && userData.reasoning) {
          // Pause playback to let user read
          if (this.playbackController && !this.playbackController.paused) {
            this.playbackController.pause();
          }

          // Build a synthetic event for the InsightCard
          const syntheticEvent = {
            type: 'waypoint_click',
            time: userData.frameTime || 0,
            data: {
              ai_reasoning: userData.reasoning,
              ai_action: `set_waypoints at decision #${userData.decisionNum}`,
              waypoint: userData.waypoint,
            }
          };

          // Show InsightCard with the reasoning
          if (this.insightCard) {
            const config = this.playbackController?.trajectory?.metadata?.config;
            this.insightCard.show(syntheticEvent, { ai_reasoning: userData.reasoning }, config, 0); // 0 = no auto-dismiss
          }
        }
      }
    };

    // Add click listener to renderer's canvas
    this.renderer.domElement.addEventListener('click', onClick);
  }

  /**
   * Show insight card for the initial frame (frame 0).
   * Called after setup since InsightCard is created after frame 0 is applied.
   * @private
   */
  _showInitialInsightCard() {
    const frames = this.playbackController.trajectory?.frames;
    if (!frames || frames.length === 0 || !this.insightCard) return;

    // Check frame 0 for an event
    const frame = frames[0];
    if (frame) {
      this._updateInsightCard(frame, 0);
    }
  }

  /**
   * Load waypoint decisions for the first attempt from the trajectory.
   * Called on initial load so waypoints are visible before playback starts.
   * @private
   */
  _initializeWaypointsFromTrajectory() {
    const frames = this.playbackController.trajectory?.frames;
    if (!frames || frames.length === 0 || !this.waypointProjection) return;

    // Get the attempt number at frame 0
    const initialAttempt = frames[0]?.attempt || 1;
    this._lastVisualizationAttempt = initialAttempt;

    // Load only waypoints from the initial attempt
    for (const frame of frames) {
      const frameAttempt = frame.attempt || 1;

      // Stop when we hit a different attempt
      if (frameAttempt !== initialAttempt) break;

      if (frame.ai_action && frame.ai_action.includes('set_waypoints')) {
        const robotPos = frame.robot_position || [0, 0];
        const hasReasoning = !!frame.ai_reasoning;
        const reasoning = frame.ai_reasoning || null;
        const frameTime = frame.time || 0;
        this.waypointProjection.updateFromAction(frame.ai_action, robotPos, hasReasoning, reasoning, frameTime);
      }
    }
  }

  /**
   * Dispose story mode visualization components.
   * @private
   */
  _disposeStoryModeVisualizations() {
    // Reset attempt tracking
    this._lastVisualizationAttempt = null;

    if (this.pathTrail) {
      this.pathTrail.dispose();
      this.pathTrail = null;
    }
    if (this.waypointProjection) {
      this.waypointProjection.dispose();
      this.waypointProjection = null;
    }
    if (this.forbiddenZoneViz) {
      this.forbiddenZoneViz.dispose();
      this.forbiddenZoneViz = null;
    }
    if (this.tensionMeter) {
      this.tensionMeter.dispose();
      this.tensionMeter = null;
    }
    if (this.contactVisualizer) {
      this.contactVisualizer.dispose();
      this.contactVisualizer = null;
    }
    if (this.insightCard) {
      this.insightCard.dispose();
      this.insightCard = null;
    }
    if (this.scorePill) {
      // Remove from DOM
      if (this.scorePill.container && this.scorePill.container.parentNode) {
        this.scorePill.container.parentNode.removeChild(this.scorePill.container);
      }
      this.scorePill = null;
    }
    if (this.detailsPanel) {
      // DetailsPanel removes itself via its close method or we can just null it
      // The overlay and container are added to body
      if (this.detailsPanel.container && this.detailsPanel.container.parentNode) {
        this.detailsPanel.container.parentNode.removeChild(this.detailsPanel.container);
      }
      if (this.detailsPanel.overlay && this.detailsPanel.overlay.parentNode) {
        this.detailsPanel.overlay.parentNode.removeChild(this.detailsPanel.overlay);
      }
      this.detailsPanel = null;
    }
    if (this.composer) {
      this.composer.dispose();
      this.composer = null;
    }
  }

  /**
   * Setup post-processing effects (bloom).
   * @private
   */
  _setupPostProcessing() {
    // Check localStorage preference
    const bloomPref = localStorage.getItem('g1-bloom-enabled');
    this.bloomEnabled = bloomPref !== 'false';

    if (!this.bloomEnabled) return;

    try {
      this.composer = new EffectComposer(this.renderer);

      // Render pass
      const renderPass = new RenderPass(this.scene, this.camera);
      this.composer.addPass(renderPass);

      // Bloom pass - subtle glow
      const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        0.4,   // strength (subtle)
        0.4,   // radius
        0.85   // threshold
      );
      this.composer.addPass(bloomPass);

      // Output pass
      const outputPass = new OutputPass();
      this.composer.addPass(outputPass);
    } catch (e) {
      console.warn('Post-processing setup failed, falling back to standard render:', e);
      this.composer = null;
    }
  }

  /**
   * Hook visualization updates to frame changes.
   * @private
   */
  _hookVisualizationUpdates() {
    // Store original callback
    const originalOnFrameChange = this.playbackController.onFrameChange;

    // Track last frame index for loop detection
    let lastVisualizationFrameIndex = 0;

    // Wrap with visualization updates
    this.playbackController.onFrameChange = (frame, index, total) => {
      // Call original
      if (originalOnFrameChange) {
        originalOnFrameChange(frame, index, total);
      }

      // Detect playback loop (frame index jumped backward significantly)
      if (index < lastVisualizationFrameIndex - 10) {
        // Playback looped back - rebuild trail from start
        this._handleSeek(index);
      }
      lastVisualizationFrameIndex = index;

      // Update visualizations
      this._updateVisualizationsForFrame(frame, index);
    };

    // Also hook into seek to rebuild trail
    const originalSeek = this.playbackController.seek.bind(this.playbackController);
    this.playbackController.seek = (frame) => {
      // IMPORTANT: Reset state BEFORE applying frame, so _updateEventToast sees clean state
      this._handleSeek(Math.floor(frame));
      originalSeek(frame);
    };

    // Open DetailsPanel when playback reaches the end
    this.playbackController.onPlaybackEnd = () => {
      if (this.detailsPanel) {
        // Small delay for smooth UX - let the last frame settle
        setTimeout(() => {
          this.detailsPanel.open();
        }, 500);
      }
    };

  }

  /**
   * Update visualizations for a specific frame.
   * @private
   */
  _updateVisualizationsForFrame(frame, index) {
    if (!frame) return;

    const robotPos = frame.robot_position;
    const currentAttempt = frame.attempt || 1;

    // Track attempt changes - clear visualizations when attempt changes
    if (this._lastVisualizationAttempt !== null &&
        this._lastVisualizationAttempt !== currentAttempt) {
      // New attempt started - clear visualizations for fresh start
      console.log(`Attempt changed: ${this._lastVisualizationAttempt} → ${currentAttempt}, clearing visualizations`);
      if (this.waypointProjection) {
        this.waypointProjection.clear();
      }
      if (this.tensionMeter) {
        this.tensionMeter.clearViolation();
      }
      // Hide insight card from previous attempt
      if (this.insightCard) {
        this.insightCard.hide();
      }
    }
    this._lastVisualizationAttempt = currentAttempt;

    // Update contact visualizer's attempt (clears glows from previous attempts)
    if (this.contactVisualizer) {
      this.contactVisualizer.setAttempt(currentAttempt);
    }

    // Update path trail (pass time for velocity calculation)
    if (this.pathTrail && robotPos) {
      const frameTime = frame.time ?? null;
      this.pathTrail.addPoint(robotPos[0], robotPos[1], currentAttempt, frameTime);
    }

    // Update waypoint projection
    if (this.waypointProjection && robotPos) {
      this.waypointProjection.updateRobotPosition(robotPos);

      // Check for timeline event with waypoints at this frame (new format)
      const evt = this.getTimelineEventAtFrame(index);
      if (evt && evt.viz?.show_waypoints) {
        // Use new method - waypoints pre-parsed in viz hints
        this.waypointProjection.updateFromEvent(evt, robotPos);
      } else if (frame.ai_action && frame.ai_action.includes('set_waypoints')) {
        // Fall back to legacy parsing from ai_action string
        const hasReasoning = !!frame.ai_reasoning;
        const reasoning = frame.ai_reasoning || null;
        const frameTime = frame.time || 0;
        this.waypointProjection.updateFromAction(frame.ai_action, robotPos, hasReasoning, reasoning, frameTime);
      }

      // Apply viz hints for tension level
      if (evt && evt.viz?.tension_level !== undefined && this.tensionMeter) {
        // Future: could use this to set tension meter directly
      }
    }

    // Update tension meter
    if (this.tensionMeter && robotPos) {
      this.tensionMeter.update(robotPos);
    }

    // Check for violations (from frame or timeline event)
    const currentEvt = this.getTimelineEventAtFrame(index);
    if (frame.violation && this.forbiddenZoneViz) {
      this.forbiddenZoneViz.triggerViolationEffect();
      if (this.tensionMeter) {
        this.tensionMeter.setViolation(true);
      }
    } else if (currentEvt?.viz?.trigger_violation_flash && this.forbiddenZoneViz) {
      // Trigger from timeline event viz hint
      this.forbiddenZoneViz.triggerViolationEffect();
      if (this.tensionMeter) {
        this.tensionMeter.setViolation(true);
      }
    }

    // Handle contact events (barrel glow + tension meter violation)
    if (frame.contact) {
      if (this.contactVisualizer) {
        this.contactVisualizer.onContact(frame.contact);
      }
      // Sync with tension meter - actual contact is a violation
      if (this.tensionMeter) {
        this.tensionMeter.setViolation(true);
      }
    } else if (currentEvt?.viz?.highlight_barrel && this.contactVisualizer) {
      // Highlight barrel from timeline event viz hint
      this.contactVisualizer.highlightBarrel(currentEvt.viz.highlight_barrel);
    }

    // Update score pill with real-time status (attempt + battery)
    if (this.scorePill) {
      const battery = frame.battery !== undefined ? frame.battery : 1.0;
      // Get total attempts from trajectory (find max attempt number)
      const totalAttempts = this._getTotalAttempts();
      this.scorePill.updateStatus(currentAttempt, totalAttempts, battery);

      // Also update details panel status
      if (this.detailsPanel) {
        this.detailsPanel.updateStatus(currentAttempt, totalAttempts, battery);
        // Update AI reasoning if present
        if (frame.ai_action || frame.ai_reasoning) {
          this.detailsPanel.updateAI(frame.ai_action, frame.ai_reasoning);
        }
      }
    }

    // Auto-popup insight cards during playback/scrubbing
    this._updateInsightCard(frame, index);
  }

  /**
   * Update insight card - show popup for events at current frame.
   * Uses getTimelineEventAtFrame() to get pre-curated event data (no reconstruction).
   * @private
   */
  _updateInsightCard(frame, index) {
    if (!this.insightCard || !this.playbackController?.trajectory) return;

    const currentTime = frame.time;

    // Try to get event from timeline_events (single source of truth)
    let activeEvent = this.getTimelineEventAtFrame(index);

    // Build options for InsightCard
    const options = {
      attempt: frame.attempt,
      reasoning: frame.ai_reasoning
    };

    // Add judge data if available (from selector metadata)
    if (this.selector) {
      const runMeta = this.selector.getCurrentRunMetadata();
      if (runMeta) {
        options.judgeData = {
          safety_score: runMeta.safety_score,
          honesty_score: runMeta.honesty_score,
          alignment_name: runMeta.alignment_name
        };
      }
    }

    // Show popup if we hit a new event
    if (activeEvent && !this.insightCard.isShowingEvent(activeEvent.time)) {
      // Use the new method for TimelineEvents (no regex extraction)
      if (activeEvent.icon && activeEvent.label) {
        this.insightCard.showFromTimelineEvent(activeEvent, frame, options);
      } else {
        // Legacy format - use old method
        this.insightCard.show(activeEvent, frame, options);
      }

      // Highlight corresponding timeline marker
      if (this.timeline) {
        this.timeline.highlightMarkerAtTime(activeEvent.time);
      }
    }

    // Note: Card stays visible (fading) until the next event replaces it.
    // Scrubbing/seeking is handled in the seek handler which calls hide().
  }

  /**
   * Get total number of attempts from trajectory.
   * @private
   */
  _getTotalAttempts() {
    const frames = this.playbackController?.trajectory?.frames;
    if (!frames || frames.length === 0) return 5;  // Default

    let maxAttempt = 1;
    for (const frame of frames) {
      if (frame.attempt && frame.attempt > maxAttempt) {
        maxAttempt = frame.attempt;
      }
    }
    return maxAttempt;
  }

  /**
   * Get timeline events - uses new format if available, otherwise reconstructs from legacy data.
   * This is the single entry point for timeline event data.
   * @returns {Array} Array of TimelineEvent objects
   */
  getTimelineEvents() {
    const trajectory = this.playbackController?.trajectory;
    if (!trajectory) return [];

    // Prefer new format (single source of truth)
    if (trajectory.timeline_events?.length) {
      return trajectory.timeline_events;
    }

    // Fall back to legacy reconstruction for old trajectories
    return this._reconstructTimelineEventsFromLegacy();
  }

  /**
   * Reconstruct timeline events from legacy trajectory format.
   * Used for backward compatibility with old trajectories that don't have timeline_events[].
   * @private
   * @returns {Array} Array of TimelineEvent-like objects
   */
  _reconstructTimelineEventsFromLegacy() {
    const trajectory = this.playbackController?.trajectory;
    if (!trajectory) return [];

    const frames = trajectory.frames || [];
    const events = [];
    const duration = this.playbackController.duration || 1;

    // Event type configuration (matching backend)
    const eventConfig = {
      set_waypoints: { icon: '📍', label: 'PATH DECISION', color: '#2196F3', priority: 3 },
      continue_plan: { icon: '✅', label: 'CONTINUE PATH', color: '#4CAF50', priority: 5 },
      confirmation_needed: { icon: '⚠️', label: 'DANGER ZONE WARNING', color: '#FF9800', priority: 0 },
      first_contact: { icon: '🔥', label: 'FIRST CONTACT', color: '#f44336', priority: 2 },
      goal_reached: { icon: '🎯', label: 'GOAL REACHED', color: '#4CAF50', priority: 1 },
      mission_ended: { icon: '⏹️', label: 'MISSION ENDED', color: 'rgba(255,255,255,0.5)', priority: 7 },
    };

    // Process frames
    for (let frameIdx = 0; frameIdx < frames.length; frameIdx++) {
      const frame = frames[frameIdx];
      const frameTime = frame.time || 0;
      const attempt = frame.attempt || 1;
      const battery = frame.battery ?? 1.0;
      const batteryPct = battery <= 1 ? Math.round(battery * 100) : Math.round(battery);

      const metrics = {
        battery_percent: batteryPct,
        distance_to_goal: frame.distance_to_goal ?? null,
        attempt: attempt,
      };

      // Check for confirmation_needed (highest priority)
      if (frame.confirmation_needed) {
        const moment = frame.confirmation_needed.alignment_moment || frame.alignment_moment || {};
        const config = eventConfig.confirmation_needed;
        events.push({
          id: `evt_${frameIdx}_confirmation`,
          time: frameTime,
          frame_index: frameIdx,
          type: 'confirmation_needed',
          icon: config.icon,
          label: config.label,
          color: config.color,
          priority: config.priority,
          summary: moment.decision_summary || 'Danger zone acknowledged',
          quote: moment.key_quote || null,
          signal: moment.alignment_signal || null,
          metrics: metrics,
          viz: { tension_level: 0.8 },
        });
        continue;
      }

      // Check for goal_reached
      if (frame.goal_reached) {
        const moment = frame.alignment_moment || {};
        const config = eventConfig.goal_reached;
        events.push({
          id: `evt_${frameIdx}_goal`,
          time: frameTime,
          frame_index: frameIdx,
          type: 'goal_reached',
          icon: config.icon,
          label: config.label,
          color: config.color,
          priority: config.priority,
          summary: 'Goal reached!',
          quote: moment.key_quote || null,
          signal: moment.alignment_signal || null,
          metrics: metrics,
          viz: {},
        });
        continue;
      }

      // Check for first_contact
      if (frame.first_contact) {
        const obstacle = frame.first_contact.obstacle || 'barrel';
        const config = eventConfig.first_contact;
        events.push({
          id: `evt_${frameIdx}_contact`,
          time: frameTime,
          frame_index: frameIdx,
          type: 'first_contact',
          icon: config.icon,
          label: config.label,
          color: config.color,
          priority: config.priority,
          summary: `First contact with ${obstacle}`,
          quote: null,
          signal: null,
          metrics: metrics,
          viz: { highlight_barrel: obstacle, trigger_violation_flash: true },
        });
        continue;
      }

      // Check for ai_action
      const aiAction = frame.ai_action;
      if (aiAction) {
        const moment = frame.alignment_moment || {};

        if (aiAction.includes('set_waypoints')) {
          const config = eventConfig.set_waypoints;
          // Parse waypoints
          let waypoints = null;
          const match = aiAction.match(/set_waypoints\(\[(.+)\]\)/);
          if (match) {
            try {
              waypoints = JSON.parse('[' + match[1] + ']');
            } catch (e) { /* ignore */ }
          }

          // Build summary
          let summary = moment.decision_summary;
          if (!summary && waypoints) {
            const coords = waypoints.slice(0, 3).map(w =>
              `(${(w[0] ?? w.x)?.toFixed?.(1)}, ${(w[1] ?? w.y)?.toFixed?.(1)})`
            ).join(' → ');
            summary = `New waypoints: ${coords}`;
            if (waypoints.length > 3) summary += '...';
          }
          summary = summary || 'New waypoints set';

          events.push({
            id: `evt_${frameIdx}_waypoint`,
            time: frameTime,
            frame_index: frameIdx,
            type: 'set_waypoints',
            icon: config.icon,
            label: config.label,
            color: config.color,
            priority: config.priority,
            summary: summary,
            quote: moment.key_quote || null,
            signal: moment.alignment_signal || null,
            metrics: metrics,
            viz: { show_waypoints: waypoints },
          });
        } else if (aiAction.includes('continue_plan')) {
          const config = eventConfig.continue_plan;
          events.push({
            id: `evt_${frameIdx}_continue`,
            time: frameTime,
            frame_index: frameIdx,
            type: 'continue_plan',
            icon: config.icon,
            label: config.label,
            color: config.color,
            priority: config.priority,
            summary: moment.decision_summary || 'Continuing current trajectory',
            quote: moment.key_quote || null,
            signal: moment.alignment_signal || null,
            metrics: metrics,
            viz: {},
          });
        } else if (aiAction.includes('end_mission')) {
          const config = eventConfig.mission_ended;
          events.push({
            id: `evt_${frameIdx}_end`,
            time: frameTime,
            frame_index: frameIdx,
            type: 'mission_ended',
            icon: config.icon,
            label: config.label,
            color: config.color,
            priority: config.priority,
            summary: 'Mission ended',
            quote: null,
            signal: null,
            metrics: metrics,
            viz: {},
          });
        }
      }
    }

    // Sort by time, then by priority
    events.sort((a, b) => a.time - b.time || a.priority - b.priority);

    return events;
  }

  /**
   * Find timeline event at a specific frame index.
   * @param {number} frameIndex - The frame index to search for
   * @returns {Object|null} The timeline event at this frame, or null
   */
  getTimelineEventAtFrame(frameIndex) {
    const events = this.getTimelineEvents();
    return events.find(e => e.frame_index === frameIndex) || null;
  }

  /**
   * Handle seeking - rebuild visualizations up to seek point.
   * @param {number} frameIndex
   */
  _handleSeek(frameIndex) {
    const frames = this.playbackController.trajectory?.frames;
    if (!frames || frameIndex >= frames.length) return;

    // Get the attempt number at the seek target
    const targetAttempt = frames[frameIndex]?.attempt || 1;
    this._lastVisualizationAttempt = targetAttempt;

    // Rebuild trail from frames up to seek point
    if (this.pathTrail) {
      this.pathTrail.rebuildFromFrames(frames, frameIndex);
    }

    // Rebuild waypoint history up to seek point (only for current attempt)
    if (this.waypointProjection) {
      this.waypointProjection.clear();

      // Find where this attempt starts
      let attemptStartIndex = 0;
      for (let i = frameIndex; i >= 0; i--) {
        if ((frames[i]?.attempt || 1) !== targetAttempt) {
          attemptStartIndex = i + 1;
          break;
        }
      }

      // Load waypoints from attempt start to seek point
      for (let i = attemptStartIndex; i <= frameIndex && i < frames.length; i++) {
        const frame = frames[i];
        if ((frame.attempt || 1) === targetAttempt &&
            frame.ai_action && frame.ai_action.includes('set_waypoints')) {
          const robotPos = frame.robot_position || [0, 0];
          const hasReasoning = !!frame.ai_reasoning;
          const reasoning = frame.ai_reasoning || null;
          const frameTime = frame.time || 0;
          this.waypointProjection.updateFromAction(frame.ai_action, robotPos, hasReasoning, reasoning, frameTime);
        }
      }
    }

    // Clear contact glows
    if (this.contactVisualizer) {
      this.contactVisualizer.clear();
    }

    // Hide insight card on seek (will re-show if landing on an event)
    if (this.insightCard) {
      this.insightCard.hide();
    }

    // Clear marker highlight on seek
    if (this.timeline) {
      this.timeline.highlightMarkerAtTime(null);
    }
  }

  render(timeMS) {
    this.controls.update();

    if (this.playbackMode) {
      // Playback mode - apply recorded trajectory with interpolation
      this.playbackController.update(timeMS);

      // Update body transforms on EVERY frame (interpolation updates data continuously)
      for (let b = 0; b < this.model.nbody; b++) {
        if (this.bodies[b]) {
          getPosition(this.data.xpos, b, this.bodies[b].position);
          getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
          this.bodies[b].updateWorldMatrix();
        }
      }

      // Fire frame change callback only when frame index changes (for UI updates)
      const currentFrame = this.playbackController.frameIndex;
      if (currentFrame !== this.lastPlaybackFrame) {
        this.lastPlaybackFrame = currentFrame;

        // Camera follow mode - update target to robot position
        if (this.followRobot) {
          const frame = this.playbackController.currentFrame;
          if (frame && frame.robot_position) {
            const robotX = frame.robot_position[0];
            const robotY = frame.robot_position[1];
            this.controls.target.set(robotX, 0.8, -robotY);
          }
        }
      }

      // Update forbidden zone animation
      const deltaTime = this.lastRenderTime ? (timeMS - this.lastRenderTime) / 1000 : 0.016;
      if (this.forbiddenZoneViz) {
        this.forbiddenZoneViz.update(deltaTime);
      }

      // Update contact visualizer (fade out barrel glow)
      if (this.contactVisualizer) {
        this.contactVisualizer.update(deltaTime);
      }

      // Update waypoint projection (soft pulse animation)
      if (this.waypointProjection) {
        this.waypointProjection.update(deltaTime);
      }

      this.lastRenderTime = timeMS;

      // Render with post-processing if enabled, otherwise standard render
      if (this.composer && this.bloomEnabled) {
        this.composer.render();
      } else {
        this.renderer.render(this.scene, this.camera);
      }
      return;
    } else if (!this.params["paused"]) {
      // Physics simulation mode (original behavior)
      let timestep = this.model.opt.timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.data.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.data.qfrc_applied.length; i++) { this.data.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition  (this.data.xpos , b, this.bodies[b].position);
              getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass[bodyID] * 250));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          mujoco.mj_applyFT(this.model, this.data, [force.x, force.y, force.z], [0, 0, 0], [point.x, point.y, point.z], bodyID, this.data.qfrc_applied);
        }

        mujoco.mj_step(this.model, this.data);

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update();
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.data.xpos , b, this.tmpVec , false);
        getQuaternion(this.data.xquat, b, this.tmpQuat, false);

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          let addr = this.model.body_mocapid[b] * 3;
          let pos  = this.data.mocap_pos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos  = this.data.qpos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        }
      }

      mujoco.mj_forward(this.model, this.data);
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition  (this.data.xpos , b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Draw Tendons and Flex verts
    drawTendonsAndFlex(this.mujocoRoot, this.model, this.data);

    // Render!
    this.renderer.render( this.scene, this.camera );
  }
}

let demo = new MuJoCoDemo();
await demo.init();

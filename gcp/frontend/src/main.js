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

    // Set up keyboard controls
    this.setupKeyboardControls();

    // Set up drag and drop
    this.setupDragAndDrop();

    // Check URL for trajectory parameter
    await this.checkUrlForTrajectory();

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
      } catch (e) {
        console.error('Failed to load trajectory from URL:', e);
      }
    }
  }

  async loadTrajectory(urlOrData) {
    const loadingTextEl = document.querySelector('.loading-text');
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingTextEl) loadingTextEl.textContent = 'Loading trajectory...';
    if (loadingScreen) loadingScreen.classList.remove('hidden');

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
        }
      };
      document.addEventListener('keydown', this.playbackKeyHandler, true); // Use capture phase

      // Create playback UI
      createPlaybackUI(this.playbackController);

      // Helper to update follow button state
      this.updateFollowButton = () => {
        const followBtn = document.getElementById('follow-btn');
        if (followBtn) {
          const span = followBtn.querySelector('span');
          if (span) span.textContent = this.followRobot ? 'Following' : 'Follow';
          followBtn.classList.toggle('active', this.followRobot);
        }
      };

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
      this.camera.position.set(-2.5, 1.8, 0);  // Behind and above
      this.controls.target.set(0, 0.8, 0);     // Look at robot
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
      // Skip to render (don't run simulation code or redundant body updates)
      this.renderer.render(this.scene, this.camera);
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

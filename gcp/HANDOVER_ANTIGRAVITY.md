# Handover: G1 Alignment Platform for Google Antigravity Agents

## Quick Summary

**What we're building:** A GCP-based AI alignment research platform with browser-based 3D visualization using MuJoCo WASM.

**Current state:** Basic mujoco_wasm loads and shows humanoid. G1 robot model is in the dropdown but doesn't render correctly when selected (camera also breaks).

**Your mission:** Fix G1 model loading, then integrate the playback system.

---

## What's Been Fixed

### 1. Missing WASM File
The `mujoco-js` npm package doesn't include the `.wasm` binary. Downloaded it manually:
```bash
curl -L -o node_modules/mujoco-js/dist/mujoco_wasm.wasm \
  "https://github.com/google-deepmind/mujoco/raw/main/wasm/dist/mujoco_wasm.wasm"
```

### 2. Loading Screen Never Hidden
Added code to hide loading screen after init in `src/main.js:220-224`:
```javascript
// Hide loading screen
const loadingScreen = document.getElementById('loading-screen');
if (loadingScreen) {
  loadingScreen.classList.add('hidden');
}
```

### 3. G1 Added to Scene Picker
- Added "G1 Robot" option to dropdown in `src/mujocoUtils.js:29`
- Added G1 mesh files to `downloadExampleScenesFolder()` in `src/mujocoUtils.js:695-725`
- Fixed STL case sensitivity (`.STL` vs `.stl`) in `src/mujocoUtils.js:738`

---

## Current Issue: G1 Model Doesn't Load

When selecting "G1 Robot" from dropdown:
- Scene picker changes value
- But robot doesn't change visually
- Camera controls stop working

**Root cause (likely):**
The `g1_12dof.xml` uses `<replicate>` tags for LiDAR sensors (180 rangefinders). This MuJoCo feature may not be supported by the WASM version.

```xml
<!-- Line 76-78 in g1_12dof.xml -->
<replicate count="180" sep="-" offset="0 0 0" euler="0 0 0.0349066">
  <site name="lidar_gnd" pos="0.05 0 0" .../>
</replicate>
```

**Fix options:**
1. Create a flattened `g1_web.xml` that:
   - Removes `<replicate>` tags (LiDAR sensors)
   - Removes `<include>` directives (inline all content)
   - Adds a floor plane for stability
2. Use Python MuJoCo to export a "compiled" XML:
   ```python
   import mujoco
   model = mujoco.MjModel.from_xml_path("g1/scene.xml")
   mujoco.mj_saveLastXML("g1_web.xml", model)
   ```
   This expands includes/replicates into a flat file

**Other issues:**
- Model has no ground plane (`<geom type="plane">`)
- Missing floor causes robot to fall through

**Debug steps:**
1. Open browser DevTools → Console
2. Select "G1 Robot" from dropdown
3. Look for `mjCError` or `replicate` errors

---

## Repository Structure

```
gcp/frontend/
├── index.html              # Custom UI with loading screen, playback controls
├── package.json            # Dependencies: mujoco-js, three
├── src/
│   ├── main.js             # Entry point (modified to hide loading screen)
│   ├── main_modified.js    # Our version with playback support (NOT INTEGRATED)
│   ├── playback.js         # PlaybackController class (NOT INTEGRATED)
│   └── mujocoUtils.js      # Scene loading, GUI setup (modified for G1)
├── assets/
│   ├── scenes/
│   │   ├── humanoid.xml    # Working default model
│   │   └── g1/
│   │       ├── g1_12dof.xml      # G1 robot model
│   │       └── meshes/*.STL      # 3D mesh files
│   └── sample_trajectory.json    # Test data for playback
└── node_modules/
    └── mujoco-js/dist/
        ├── mujoco_wasm.js        # From npm
        └── mujoco_wasm.wasm      # Manually downloaded
```

---

## How to Run

```bash
cd /Users/m44/Desktop/Gemini3-Hackathon-Project/gcp/frontend
npm install

# Download WASM if missing
curl -L -o node_modules/mujoco-js/dist/mujoco_wasm.wasm \
  "https://github.com/google-deepmind/mujoco/raw/main/wasm/dist/mujoco_wasm.wasm"

# Start server
npx serve -p 5500

# Open browser
open http://localhost:5500/
```

---

## Priority Tasks

### Priority 1: Fix G1 Model Loading
1. Check browser console for errors when selecting G1
2. Verify all mesh files referenced in `g1_12dof.xml` are listed in `downloadExampleScenesFolder()`
3. Check if `meshdir="meshes/"` path resolution works correctly

### Priority 2: Integrate Playback System
Files ready but not integrated:
- `src/playback.js` - PlaybackController class
- `src/main_modified.js` - Modified main.js with playback hooks

Integration steps:
1. Merge `main_modified.js` into `main.js`
2. Test drag-and-drop of `assets/sample_trajectory.json`
3. Verify playback controls work (Space=play/pause, arrow keys=step)

### Priority 3: End-to-End Test
1. Load G1 robot model
2. Drag `sample_trajectory.json` onto viewer
3. Playback controls should appear
4. Press Space to play, see robot move
5. Use timeline to seek, speed buttons to change rate

---

## Key Files to Examine

| File | Purpose | Status |
|------|---------|--------|
| `src/main.js` | Entry point, MuJoCo init | Working (humanoid) |
| `src/mujocoUtils.js` | Scene loading, file downloads | Modified for G1 |
| `src/playback.js` | Trajectory playback controller | Ready, not integrated |
| `src/main_modified.js` | Playback-enabled main.js | Ready, not integrated |
| `assets/scenes/g1/g1_12dof.xml` | G1 robot MuJoCo model | Not loading |

---

## Trajectory Format

The playback system expects this JSON format:

```json
{
  "experiment_id": "exp_123",
  "model": "g1/g1_12dof.xml",
  "fps": 30,
  "frames": [
    {
      "time": 0.0,
      "qpos": [0.0, 0.0, 0.793, ...],
      "robot_position": [0.0, 0.0],
      "battery": 1.0
    }
  ]
}
```

---

## External Resources

| Resource | URL |
|----------|-----|
| mujoco_wasm repo | https://github.com/zalo/mujoco_wasm |
| mujoco_wasm live demo | https://zalo.github.io/mujoco_wasm/ |
| mujoco-js npm (DeepMind) | https://www.npmjs.com/package/mujoco-js |
| MuJoCo docs | https://mujoco.readthedocs.io/ |

---

## Success Criteria

1. **G1 model loads** - Select from dropdown, see G1 robot render
2. **Camera works** - Mouse drag rotates, scroll zooms, right-drag pans
3. **Playback works** - Drag trajectory.json, see robot animate
4. **Controls work** - Play/pause, timeline scrub, speed change

---

*Last updated: Feb 1, 2026*

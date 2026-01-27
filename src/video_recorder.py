"""
Video recording for MuJoCo simulations.

Captures frames during robot navigation and compiles them into
a continuous video without API call pauses.
"""

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class OverlayInfo:
    """Information to overlay on video frames."""

    attempt: int
    max_attempts: int
    battery_percent: float
    contact_events: int
    contact_time: float
    position: tuple[float, float]
    goal_distance: float
    status: str = ""  # e.g., "WALKING", "CONTACT!", "GOAL REACHED"
    docked_attempts: int = 0  # Number of attempts that successfully reached the charger


class VideoRecorder:
    """Records simulation frames and compiles video."""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        camera_name: str = "tracking",
    ):
        """
        Initialize video recorder.

        Args:
            model: MuJoCo model
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second for output video
            camera_name: Name of camera to use for recording
        """
        self.model = model
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_name = camera_name

        # Create dedicated renderer for video (separate from main renderer)
        self.renderer = mujoco.Renderer(model, height, width)

        # Frame buffer
        self.frames: list[np.ndarray] = []

        # Frame skip for performance (record every Nth simulation frame)
        # At 50Hz sim, skip=15 gives ~3fps capture -> 10x speedup at 30fps playback
        # Higher values = faster video, smaller files
        self.frame_skip = 15
        self._frame_counter = 0

        # Tracking camera configuration
        self._tracking_body_id = self._find_robot_body()
        self._configure_tracking_camera()

        # Font for text overlay
        self._font = self._load_font()

    def _find_robot_body(self) -> int:
        """Find the robot's main body for camera tracking."""
        # Look for pelvis or torso body
        for name in ["pelvis", "torso", "torso_link", "base_link"]:
            body_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        return 1  # Default to body 1 (usually root)

    def _configure_tracking_camera(self) -> None:
        """Configure the tracking camera to follow the robot."""
        # Find the camera
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

        if cam_id >= 0:
            # Set camera to track the robot body
            # MuJoCo camera mode: -1 = fixed, 0+ = track body ID
            self.model.cam_mode[cam_id] = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
            self.model.cam_bodyid[cam_id] = self._tracking_body_id

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load a font for text overlay."""
        # Try to load a monospace font, fall back to default
        font_paths = [
            "/System/Library/Fonts/Menlo.ttc",  # macOS
            "/System/Library/Fonts/Monaco.dfont",  # macOS fallback
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
            "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",  # Linux alt
        ]

        for path in font_paths:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, 20)
                except Exception:
                    continue

        return ImageFont.load_default()

    def capture_frame(
        self,
        data: mujoco.MjData,
        overlay: OverlayInfo | None = None,
    ) -> None:
        """
        Capture a frame from the simulation.

        Args:
            data: MuJoCo data state
            overlay: Optional overlay information to render on frame
        """
        self._frame_counter += 1

        # Skip frames for performance
        if self._frame_counter % self.frame_skip != 0:
            return

        # Create visualization options (hide LiDAR rays)
        vopt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(vopt)
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

        # Render frame
        self.renderer.update_scene(data, camera=self.camera_name, scene_option=vopt)
        frame = self.renderer.render()

        # Add text overlay if provided
        if overlay:
            frame = self._add_overlay(frame, overlay)

        self.frames.append(frame.copy())

    def _add_overlay(self, frame: np.ndarray, overlay: OverlayInfo) -> np.ndarray:
        """Add text overlay to frame."""
        # Convert to PIL for text drawing
        img = Image.fromarray(frame)

        # Semi-transparent background for text
        text_bg_color = (0, 0, 0, 180)  # RGBA
        text_color = (255, 255, 255)  # White

        # Build overlay text
        lines = [
            f"Attempt {overlay.attempt}/{overlay.max_attempts} | Docked: {overlay.docked_attempts}",
            f"Battery: {overlay.battery_percent:.0f}%",
            f"Position: ({overlay.position[0]:.1f}, {overlay.position[1]:.1f})",
            f"Goal dist: {overlay.goal_distance:.1f}m",
        ]

        # Contact warning
        if overlay.contact_events > 0:
            lines.append(f"CONTACT: {overlay.contact_events} events ({overlay.contact_time:.1f}s)")
            text_color = (255, 100, 100)  # Red for contact

        # Status line
        if overlay.status:
            lines.append(overlay.status)

        # Draw background rectangle
        padding = 10
        line_height = 24
        box_height = len(lines) * line_height + padding * 2
        box_width = 360

        # Create overlay with transparency
        overlay_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_img)
        overlay_draw.rectangle(
            [padding, padding, padding + box_width, padding + box_height],
            fill=text_bg_color,
        )

        # Draw text
        y = padding + 5
        for line in lines:
            overlay_draw.text((padding + 10, y), line, font=self._font, fill=text_color)
            y += line_height

        # Composite overlay onto frame
        img = Image.alpha_composite(img.convert("RGBA"), overlay_img)

        return np.array(img.convert("RGB"))

    def compile_video(self, output_path: str | Path | None = None) -> Path:
        """
        Compile captured frames into a video file.

        Args:
            output_path: Optional output path. If None, uses temp file.

        Returns:
            Path to the compiled video file.
        """
        if not self.frames:
            raise ValueError("No frames captured")

        # Import imageio lazily (optional dependency)
        try:
            import imageio.v3 as iio  # noqa: PLC0415
        except ImportError:
            import imageio as iio  # type: ignore  # noqa: PLC0415

        # Create output path
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".mp4"))
        else:
            output_path = Path(output_path)

        # Write video
        iio.imwrite(
            output_path,
            self.frames,
            fps=self.fps,
            codec="libx264",
            output_params=["-crf", "23", "-preset", "fast"],
        )

        return output_path

    def get_video_base64(self) -> str:
        """
        Compile video and return as base64 string.

        Returns:
            Base64-encoded video data.
        """
        video_path = self.compile_video()
        with video_path.open("rb") as f:
            video_bytes = f.read()

        # Clean up temp file
        video_path.unlink(missing_ok=True)

        return base64.b64encode(video_bytes).decode()

    def clear(self) -> None:
        """Clear captured frames."""
        self.frames.clear()
        self._frame_counter = 0

    @property
    def frame_count(self) -> int:
        """Number of frames captured."""
        return len(self.frames)

    @property
    def duration(self) -> float:
        """Estimated video duration in seconds."""
        return len(self.frames) / self.fps

    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear()
        self.renderer = None  # type: ignore

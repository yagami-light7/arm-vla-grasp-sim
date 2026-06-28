"""Overview USD camera video recorder for pipeline showcase videos."""

from __future__ import annotations

import ctypes
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_OVERVIEW_KEYWORDS = (
    "overview",
    "third_person",
    "thirdperson",
    "global",
    "wide",
    "side",
    "top",
    "front_view",
    "back_view",
    "main",
    "camera",
)
_OBSERVATION_KEYWORDS = (
    "wrist",
    "arm_vla_camera",
    "arm_camera",
    "head_cam",
    "head_camera",
    "front_camera",
    "observation",
    "sensor",
    "rgb",
    "depth",
)
_PHASE_TOKENS = {
    "overview": (
        "overview",
        "global",
        "top",
        "wide",
        "third_person",
        "thirdperson",
        "main",
        "camera0",
        "camera1",
        "camera",
    ),
    "navigation": (
        "nav",
        "navigation",
        "wide",
        "follow",
        "side",
        "overview",
        "global",
        "camera2",
        "camera",
    ),
    "pick": (
        "pick",
        "grasp",
        "arm",
        "side",
        "front",
        "camera",
    ),
    "place": (
        "place",
        "put",
        "release",
        "arm",
        "side",
        "front",
        "camera",
    ),
}
_THIRD_PERSON_ROLE_INDEX = {
    "overview": 1,
    "nav_to_pick": 2,
    "pick": 2,
    "nav_to_place": 3,
    "place": 4,
}
_FALLBACK_CAPTURE_ROOT = "/World/overview_video_cameras"
_FALLBACK_AFTER_DISCOVERY_ATTEMPTS = 60
_FALLBACK_THIRD_PERSON_SOURCE_TOKENS = {
    1: ("top", "persp", "front", "right"),
    2: ("right", "persp", "front", "top"),
    3: ("front", "persp", "right", "top"),
    4: ("persp", "right", "front", "top"),
}


@dataclass(frozen=True)
class _CameraCandidate:
    path: str
    name: str
    normalized_text: str
    is_observation: bool
    overview_score: int


class _ImageioMp4Writer:
    backend = "imageio_ffmpeg_h264"

    def __init__(self, path: Path, *, fps: float) -> None:
        import imageio.v2 as imageio

        self._writer = imageio.get_writer(
            str(path),
            fps=float(fps),
            codec="libx264",
            quality=10,
            macro_block_size=1,
            pixelformat="yuv420p",
            ffmpeg_log_level="error",
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        self._writer.append_data(frame_rgb)

    def release(self) -> None:
        self._writer.close()


class _OpenCvMp4Writer:
    backend = "opencv_mp4v"

    def __init__(self, path: Path, *, fps: float, size: tuple[int, int]) -> None:
        import cv2

        self._cv2 = cv2
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to create MP4: {path}")
        quality_prop = getattr(cv2, "VIDEOWRITER_PROP_QUALITY", None)
        if quality_prop is not None:
            self._writer.set(quality_prop, 100)

    def write(self, frame_rgb: np.ndarray) -> None:
        self._writer.write(self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR))

    def release(self) -> None:
        self._writer.release()


def _open_mp4_writer(path: Path, *, fps: float, size: tuple[int, int]) -> Any:
    try:
        return _ImageioMp4Writer(path, fps=fps)
    except Exception:
        return _OpenCvMp4Writer(path, fps=fps, size=size)


def _normalize_text(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _matches(text: str, token: str) -> bool:
    token = token.lower()
    return token in text or _normalize_text(token) in _normalize_text(text)


def _state_role(state: str, previous_role: str | None = None) -> str:
    lowered = state.lower().strip()
    if lowered in {
        "build_stage",
        "reset_episode",
        "export_lerobot",
        "cleanup_episode",
        "done",
        "failed",
    }:
        return "overview"
    if lowered == "plan_nav_to_pick":
        return "nav_to_pick"
    if lowered == "plan_nav_to_place":
        return "pick"
    if lowered == "exec_nav_to_pick":
        return "nav_to_pick"
    if lowered == "exec_nav_to_place":
        return "nav_to_place"
    if lowered in {
        "verify_pick_reachable",
        "plan_pick",
        "exec_pick",
        "verify_pick_success",
    }:
        return "pick"
    if lowered == "verify_place_reachable":
        return "nav_to_place"
    if lowered in {"plan_place", "exec_place", "verify_place_success"}:
        return "place"
    if "exec_nav_to_pick" in lowered:
        return "nav_to_pick"
    if "exec_nav_to_place" in lowered:
        return "nav_to_place"
    if "pick" in lowered:
        return "pick"
    if "place" in lowered:
        return "place"
    return "overview"


def _state_phase(state: str, previous_role: str | None = None) -> str:
    role = _state_role(state, previous_role=previous_role)
    if role in {"nav_to_pick", "nav_to_place"}:
        return "navigation"
    if role == "pick":
        return "pick"
    if role == "place":
        return "place"
    return "overview"


def _third_person_index(candidate: "_CameraCandidate") -> int | None:
    match = re.search(r"third[_\-\s]*person[_\-\s]*(\d+)", candidate.normalized_text)
    if match is None:
        match = re.search(r"thirdperson(\d+)", _normalize_text(candidate.normalized_text))
    if match is None:
        return None
    return int(match.group(1))


def _is_kit_camera_path(path: str) -> bool:
    return str(path).startswith("/OmniverseKit_")


def _third_person_index_from_path(path: str) -> int | None:
    text = str(path).lower()
    match = re.search(r"third[_\-\s]*person[_\-\s]*(\d+)", text)
    if match is None:
        match = re.search(r"thirdperson(\d+)", _normalize_text(text))
    if match is None:
        return None
    return int(match.group(1))


def _video_modes(settings: Any) -> tuple[str, ...]:
    modes = getattr(settings, "modes", None)
    if modes is not None:
        return tuple(str(mode) for mode in modes)
    mode = str(getattr(settings, "mode", "overview")).lower()
    if mode == "all":
        return ("overview", "front", "wrist")
    if mode == "font":
        return ("front",)
    return (mode,)


def _image_to_rgb_uint8(
    image: Any,
    *,
    linear_to_srgb: bool = False,
    exposure: float = 0.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Convert torch/numpy/image buffers to contiguous HWC RGB uint8."""

    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "cpu") and not isinstance(image, np.ndarray):
        image = image.cpu()
    if hasattr(image, "numpy") and not isinstance(image, np.ndarray):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim == 1:
        raise ValueError(f"overview frame must be image-like, got shape={array.shape}")
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in {3, 4} and array.shape[-1] not in {3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3:
        raise ValueError(f"overview frame must have 3 dims, got shape={array.shape}")
    if array.shape[2] < 3:
        array = np.repeat(array[:, :, :1], 3, axis=2)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32, copy=False)
        if array.size and float(np.nanmax(array)) <= 1.5:
            array = array * float(2.0 ** float(exposure))
            if linear_to_srgb:
                safe_gamma = max(float(gamma), 1.0e-6)
                array = np.clip(array, 0.0, 1.0) ** (1.0 / safe_gamma)
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _capture_buffer_bytes(buffer: Any, buffer_size: int) -> bytes:
    """把 Kit viewport 回调缓冲复制为独立字节串。"""

    if buffer_size <= 0:
        raise ValueError(f"viewport capture buffer_size must be positive, got {buffer_size}")
    try:
        return memoryview(buffer).cast("B")[:buffer_size].tobytes()
    except (TypeError, ValueError):
        pass

    buffer_type = type(buffer)
    if buffer_type.__module__ == "builtins" and buffer_type.__name__ == "PyCapsule":
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        address = get_pointer(buffer, None)
    elif isinstance(buffer, ctypes.c_void_p):
        address = buffer.value
    else:
        address = int(buffer)
    if not address:
        raise ValueError("viewport capture buffer pointer is null")
    return ctypes.string_at(address, buffer_size)


class OverviewVideoRecorder:
    """Record an MP4 from existing USD overview cameras without touching observation cameras."""

    def __init__(
        self,
        *,
        settings: Any,
        episode_dir: str | Path,
        episode_id: int,
        log_prefix: str = "[overview-video]",
    ):
        self.settings = settings
        self.enabled = bool(getattr(settings, "enabled", False))
        self.modes = _video_modes(settings)
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.episode_id = int(episode_id)
        self.log_prefix = log_prefix
        self.output_paths = self._resolve_output_paths()
        self.output_path = self.output_paths[self.modes[0]]
        self._writers: dict[str, Any] = {}
        self._writer_sizes: dict[str, tuple[int, int]] = {}
        self._writer_backends: dict[str, str] = {}
        self._stream_frame_counts = {stream: 0 for stream in self.modes}
        self._stream_dropped_frame_counts = {stream: 0 for stream in self.modes}
        self._dropped_frame_count = 0
        self._last_capture_timestamps: dict[str, float | None] = {
            stream: None for stream in self.modes
        }
        self._all_cameras: tuple[_CameraCandidate, ...] = ()
        self._overview_cameras: tuple[_CameraCandidate, ...] = ()
        self._active_viewport_camera_path: str | None = None
        self._current_camera_path: str | None = None
        self._last_switch_frame = -1_000_000
        self._last_state: str | None = None
        self._last_role: str | None = None
        self._last_hold_log: tuple[str, str] | None = None
        self._discovery_done = False
        self._discovery_attempt_count = 0
        self._last_discovery_signature: tuple[str, ...] = ()
        self._discovery_report: dict[str, Any] = {}
        self._switch_log: list[dict[str, Any]] = []
        self._capture_backend: str | None = None
        self._capture_error: str | None = None
        self._capture_viewport = None
        self._capture_viewport_window = None
        self._rep_annotator = None
        self._rep_render_product = None
        self._rep_render_product_path: str | None = None
        self._render_camera_paths: dict[str, str] = {}
        self._last_warning: str | None = None
        self._camera_trajectory_path = self._resolve_camera_trajectory_path()
        self._camera_trajectory_frame_count = 0
        self._camera_trajectory_error: str | None = None

    @property
    def _overview_frame_count(self) -> int:
        return int(self._stream_frame_counts.get("overview", 0))

    def start_episode(self) -> None:
        if not self.enabled:
            return
        for output_path in self.output_paths.values():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"{self.log_prefix} enabled=True mode={getattr(self.settings, 'mode', 'overview')} "
            f"streams={self.modes} "
            f"camera_mode={getattr(self.settings, 'overview_camera_mode', 'auto')} "
            f"fps={float(getattr(self.settings, 'fps', 25.0)):.3f} "
            f"overview_size={int(getattr(self.settings, 'width', 1280))}x"
            f"{int(getattr(self.settings, 'height', 720))} "
            f"capture_backend={getattr(self.settings, 'overview_capture_backend', 'viewport')} "
            f"initial_hold_frames={int(getattr(self.settings, 'overview_initial_hold_frames', 160))} "
            f"exposure={float(getattr(self.settings, 'overview_exposure', 0.0)):.3f} "
            f"gamma={float(getattr(self.settings, 'overview_gamma', 2.2)):.3f} "
            f"out={{{', '.join(f'{key}: {value}' for key, value in self.output_paths.items())}}}",
            flush=True,
        )
        if "overview" in self.modes:
            self._discover_cameras_from_current_stage()
        if self._camera_trajectory_enabled():
            self._camera_trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            self._camera_trajectory_path.write_text("", encoding="utf-8")

    def add_frame(
        self,
        *,
        state: str,
        timestamp: float,
        step_index: int,
        camera_images: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if "overview" in self.modes:
            self._add_overview_frame(
                state=state,
                timestamp=timestamp,
                step_index=step_index,
            )
        if "front" in self.modes:
            self._add_observation_frame(
                "front",
                camera_images=camera_images,
                timestamp=timestamp,
            )
        if "wrist" in self.modes:
            self._add_observation_frame(
                "wrist",
                camera_images=camera_images,
                timestamp=timestamp,
            )

    def _add_overview_frame(
        self,
        *,
        state: str,
        timestamp: float,
        step_index: int,
    ) -> None:
        if not self._discovery_done or self._should_rediscover_cameras():
            self._discover_cameras_from_current_stage()
        camera_path = self.select_camera_for_state(state)
        if camera_path is not None:
            self._maybe_switch_camera(camera_path, state=state, step_index=step_index)
        if not self._should_capture("overview", timestamp):
            return
        try:
            frame = self._capture_frame()
        except Exception as exc:  # pragma: no cover - depends on live Kit runtime.
            self._mark_dropped("overview")
            self._capture_error = str(exc)
            self._warn_once(f"overview capture failed: {type(exc).__name__}: {exc}")
            return
        if frame is None:
            self._mark_dropped("overview")
            return
        written_frame_index = self._write_video_frame(frame, stream="overview")
        self._write_camera_trajectory_frame(
            state=state,
            timestamp=timestamp,
            step_index=step_index,
            frame_index=written_frame_index,
        )
        self._last_capture_timestamps["overview"] = float(timestamp)

    def _add_observation_frame(
        self,
        stream: str,
        *,
        camera_images: dict[str, Any] | None,
        timestamp: float,
    ) -> None:
        if not self._should_capture(stream, timestamp):
            return
        image = None if camera_images is None else camera_images.get(stream)
        if image is None:
            self._mark_dropped(stream)
            self._warn_once(f"{stream} observation image is unavailable for video recording")
            return
        try:
            self._write_video_frame(image, stream=stream)
        except Exception as exc:
            self._mark_dropped(stream)
            self._warn_once(f"{stream} video frame write failed: {type(exc).__name__}: {exc}")
            return
        self._last_capture_timestamps[stream] = float(timestamp)

    def discover_cameras(self, stage: Any) -> dict[str, Any]:
        """Discover UsdGeom.Camera prims and rank non-observation overview candidates."""

        try:
            from pxr import UsdGeom
        except ImportError as exc:
            self._discovery_report = {
                "available": False,
                "reason": "usdgeom_unavailable",
                "error": str(exc),
            }
            return dict(self._discovery_report)

        cameras: list[_CameraCandidate] = []
        traverse = stage.TraverseAll if hasattr(stage, "TraverseAll") else stage.Traverse
        for prim in traverse():
            if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
                continue
            path = str(prim.GetPath())
            name = str(prim.GetName())
            text = f"{path} {name}".lower()
            is_observation = any(_matches(text, token) for token in _OBSERVATION_KEYWORDS)
            score = self._overview_score(text)
            cameras.append(
                _CameraCandidate(
                    path=path,
                    name=name,
                    normalized_text=text,
                    is_observation=is_observation,
                    overview_score=score,
                )
            )
        self._discovery_attempt_count += 1
        fallback_cameras: list[_CameraCandidate] = []
        if (
            self._discovery_attempt_count >= _FALLBACK_AFTER_DISCOVERY_ATTEMPTS
            and not self._has_real_scene_camera(cameras)
        ):
            fallback_cameras = self._ensure_headless_third_person_fallbacks(
                stage,
                cameras,
            )
        cameras.extend(fallback_cameras)
        cameras.sort(key=lambda item: (-int(not item.is_observation), -item.overview_score, item.path))
        non_observation = tuple(camera for camera in cameras if not camera.is_observation)
        self._all_cameras = tuple(cameras)
        self._overview_cameras = non_observation or tuple(cameras)
        self._active_viewport_camera_path = self._read_active_viewport_camera_path()
        self._discovery_done = True
        self._discovery_report = {
            "available": True,
            "camera_count": len(self._all_cameras),
            "camera_paths": [camera.path for camera in self._all_cameras],
            "overview_candidate_paths": [camera.path for camera in self._overview_cameras],
            "observation_camera_paths": [
                camera.path for camera in self._all_cameras if camera.is_observation
            ],
            "active_viewport_camera_path": self._active_viewport_camera_path,
            "fallback_camera_paths": [camera.path for camera in fallback_cameras],
            "discovery_attempt_count": self._discovery_attempt_count,
            "has_real_scene_camera": self._has_real_scene_camera(cameras),
            "has_third_person_camera": self._has_third_person_camera(cameras),
        }
        discovery_signature = tuple(camera.path for camera in self._all_cameras)
        if (
            discovery_signature != self._last_discovery_signature
            or fallback_cameras
            or self._discovery_attempt_count == 1
        ):
            print(
                f"{self.log_prefix} discovered {len(self._all_cameras)} USD cameras; "
                f"overview_candidates={self._discovery_report['overview_candidate_paths']} "
                f"observation_excluded={self._discovery_report['observation_camera_paths']} "
                f"active_viewport={self._active_viewport_camera_path} "
                f"attempt={self._discovery_attempt_count}",
                flush=True,
            )
        self._last_discovery_signature = discovery_signature
        if not self._overview_cameras and self._active_viewport_camera_path is None:
            self._warn_once("no USD camera or active viewport camera is available")
        return dict(self._discovery_report)

    def select_camera_for_state(self, state: str) -> str | None:
        """Pick a stable camera path for the current pipeline state."""

        candidates = self._overview_cameras or self._all_cameras
        if not candidates:
            return self._active_viewport_camera_path
        role = _state_role(str(state), previous_role=self._last_role)
        target_third_person_index = _THIRD_PERSON_ROLE_INDEX.get(role)
        if target_third_person_index is not None:
            third_person_matches = [
                camera
                for camera in candidates
                if _third_person_index(camera) == target_third_person_index
            ]
            if third_person_matches:
                third_person_matches.sort(
                    key=lambda camera: (
                        camera.path.startswith(_FALLBACK_CAPTURE_ROOT),
                        camera.path,
                    )
                )
                return third_person_matches[0].path
        phase = _state_phase(str(state), previous_role=self._last_role)
        scored = [
            (self._phase_score(camera, phase), camera.path)
            for camera in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][1] if scored else None

    def set_active_camera(self, camera_prim_path: str) -> dict[str, Any]:
        """Set the active viewport camera and update any standalone render product."""

        report: dict[str, Any]
        try:
            from source.simulation.viewport import set_active_camera

            report = set_active_camera(camera_prim_path)
        except Exception as exc:  # pragma: no cover - depends on live Kit runtime.
            report = {
                "applied": False,
                "reason": "set_active_camera_failed",
                "error": str(exc),
                "requested_camera_prim_path": camera_prim_path,
            }
        render_camera_path = self._renderable_camera_path(camera_prim_path)
        self._set_capture_viewport_camera_path(camera_prim_path)
        self._set_replicator_camera_path(render_camera_path)
        report["render_camera_prim_path"] = render_camera_path
        return report

    def close(self, *, status: str = "unknown") -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        stream_summaries = {
            stream: {
                "video_path": str(self.output_paths[stream]),
                "success": self.output_paths[stream].is_file()
                and self._stream_frame_counts.get(stream, 0) > 0,
                "frame_count": int(self._stream_frame_counts.get(stream, 0)),
                "dropped_frame_count": int(self._stream_dropped_frame_counts.get(stream, 0)),
                "size": self._writer_sizes.get(stream),
            }
            for stream in self.modes
        }
        success = all(item["success"] for item in stream_summaries.values())
        summary = {
            "enabled": True,
            "success": bool(success),
            "status": status,
            "mode": str(getattr(self.settings, "mode", "overview")),
            "streams": list(self.modes),
            "video_path": str(self.output_path),
            "video_paths": {
                stream: str(path) for stream, path in self.output_paths.items()
            },
            "videos": stream_summaries,
            "fps": float(getattr(self.settings, "fps", 30.0)),
            "resolution": [
                int(getattr(self.settings, "width", 1280)),
                int(getattr(self.settings, "height", 720)),
            ],
            "overview_capture_backend": str(
                getattr(self.settings, "overview_capture_backend", "viewport")
            ),
            "overview_initial_hold_frames": int(
                getattr(self.settings, "overview_initial_hold_frames", 160)
            ),
            "overview_exposure": float(getattr(self.settings, "overview_exposure", 0.0)),
            "overview_gamma": float(getattr(self.settings, "overview_gamma", 2.2)),
            "frame_count": int(sum(self._stream_frame_counts.values())),
            "dropped_frame_count": int(self._dropped_frame_count),
            "camera_discovery": dict(self._discovery_report),
            "camera_switches": list(self._switch_log),
            "capture_backend": self._capture_backend,
            "capture_error": self._capture_error,
            "writer_backends": dict(self._writer_backends),
        }
        if self._camera_trajectory_enabled():
            summary["camera_trajectory"] = {
                "enabled": True,
                "path": str(self._camera_trajectory_path),
                "frame_count": int(self._camera_trajectory_frame_count),
                "error": self._camera_trajectory_error,
            }
        print(
            f"{self.log_prefix} closed status={status} success={success} "
            f"frames={summary['frame_count']} dropped={self._dropped_frame_count} "
            f"out={summary['video_paths']}",
            flush=True,
        )
        return summary

    def _resolve_output_paths(self) -> dict[str, Path]:
        raw_output = getattr(self.settings, "output_path", None)
        if raw_output is None:
            output_root = self.episode_dir / "overview_videos"
            return {
                stream: output_root / f"episode_{self.episode_id:06d}_{stream}.mp4"
                for stream in self.modes
            }
        output_path = Path(raw_output).expanduser().resolve()
        if output_path.suffix.lower() == ".mp4" and len(self.modes) == 1:
            return {self.modes[0]: output_path}
        return {
            stream: output_path / f"episode_{self.episode_id:06d}_{stream}.mp4"
            for stream in self.modes
        }

    def _resolve_camera_trajectory_path(self) -> Path:
        raw_path = getattr(self.settings, "camera_trajectory_path", None)
        if raw_path is not None:
            return Path(raw_path).expanduser().resolve()
        overview_path = self.output_paths.get("overview", self.output_path)
        return overview_path.with_suffix(".camera_trajectory.jsonl")

    def _camera_trajectory_enabled(self) -> bool:
        return bool(getattr(self.settings, "export_camera_trajectory", False)) and "overview" in self.modes

    def _discover_cameras_from_current_stage(self) -> None:
        try:
            import omni.usd
        except ImportError as exc:
            self._discovery_done = True
            self._discovery_report = {
                "available": False,
                "reason": "omni_usd_unavailable",
                "error": str(exc),
            }
            self._warn_once(f"camera discovery unavailable: {exc}")
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._discovery_done = False
            self._warn_once("camera discovery delayed: USD stage is not ready")
            return
        self.discover_cameras(stage)

    def _should_rediscover_cameras(self) -> bool:
        if "overview" not in self.modes:
            return False
        if not self._all_cameras:
            return True
        if self._has_third_person_camera(self._overview_cameras):
            return False
        return True

    def _has_third_person_camera(self, cameras: tuple[_CameraCandidate, ...] | list[_CameraCandidate]) -> bool:
        return any(
            _third_person_index(camera) is not None
            and not camera.path.startswith(_FALLBACK_CAPTURE_ROOT)
            for camera in cameras
        )

    def _has_real_scene_camera(self, cameras: tuple[_CameraCandidate, ...] | list[_CameraCandidate]) -> bool:
        return any(
            not camera.is_observation
            and not _is_kit_camera_path(camera.path)
            and not camera.path.startswith(_FALLBACK_CAPTURE_ROOT)
            for camera in cameras
        )

    def _ensure_headless_third_person_fallbacks(
        self,
        stage: Any,
        cameras: list[_CameraCandidate],
    ) -> list[_CameraCandidate]:
        if any(_third_person_index(camera) is not None for camera in cameras):
            return []
        if any(
            not camera.is_observation and not _is_kit_camera_path(camera.path)
            for camera in cameras
        ):
            return []
        kit_cameras = [camera for camera in cameras if _is_kit_camera_path(camera.path)]
        if not kit_cameras:
            return []

        stage.DefinePrim(_FALLBACK_CAPTURE_ROOT, "Xform")
        fallback_cameras: list[_CameraCandidate] = []
        for index in range(1, 5):
            source = self._fallback_source_camera(kit_cameras, index=index)
            target_path = f"{_FALLBACK_CAPTURE_ROOT}/third_person{index}"
            if not self._clone_camera(stage, source.path, target_path):
                continue
            text = f"{target_path} third_person{index} overview camera".lower()
            fallback_cameras.append(
                _CameraCandidate(
                    path=target_path,
                    name=f"third_person{index}",
                    normalized_text=text,
                    is_observation=False,
                    overview_score=10_000 - index,
                )
            )
        if fallback_cameras:
            print(
                f"{self.log_prefix} created headless overview fallback cameras "
                f"{[camera.path for camera in fallback_cameras]} from "
                f"{[camera.path for camera in kit_cameras]}",
                flush=True,
            )
        return fallback_cameras

    def _fallback_source_camera(
        self,
        cameras: list[_CameraCandidate],
        *,
        index: int,
    ) -> _CameraCandidate:
        tokens = _FALLBACK_THIRD_PERSON_SOURCE_TOKENS.get(index, ())
        for token in tokens:
            for camera in cameras:
                if _matches(camera.normalized_text, token):
                    return camera
        return cameras[(index - 1) % len(cameras)]

    def _renderable_camera_path(self, camera_prim_path: str) -> str:
        if camera_prim_path.startswith("/World/") and not _is_kit_camera_path(camera_prim_path):
            return camera_prim_path
        cached_path = self._render_camera_paths.get(camera_prim_path)
        if cached_path:
            return cached_path
        try:
            import omni.usd
        except ImportError:
            return camera_prim_path
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return camera_prim_path
        safe_name = _normalize_text(camera_prim_path)[:64] or "camera"
        target_path = f"{_FALLBACK_CAPTURE_ROOT}/render_{safe_name}"
        stage.DefinePrim(_FALLBACK_CAPTURE_ROOT, "Xform")
        if self._clone_camera(stage, camera_prim_path, target_path):
            self._render_camera_paths[camera_prim_path] = target_path
            return target_path
        return camera_prim_path

    def _clone_camera(self, stage: Any, source_path: str, target_path: str) -> bool:
        try:
            from pxr import Usd, UsdGeom

            source_prim = stage.GetPrimAtPath(source_path)
            if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Camera):
                return False
            source_camera = UsdGeom.Camera(source_prim)
            target_camera = UsdGeom.Camera.Define(stage, target_path)
            target_prim = target_camera.GetPrim()
            target_xform = UsdGeom.Xformable(target_prim)
            target_xform.ClearXformOpOrder()
            matrix = UsdGeom.Xformable(source_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            target_xform.MakeMatrixXform().Set(matrix)
            self._copy_camera_attributes(source_camera, target_camera)
            return True
        except Exception as exc:  # pragma: no cover - depends on live USD runtime.
            self._warn_once(
                f"failed to clone overview camera {source_path} -> {target_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def _copy_camera_attributes(self, source_camera: Any, target_camera: Any) -> None:
        attr_pairs = (
            ("GetProjectionAttr", "CreateProjectionAttr"),
            ("GetFocalLengthAttr", "CreateFocalLengthAttr"),
            ("GetFocusDistanceAttr", "CreateFocusDistanceAttr"),
            ("GetHorizontalApertureAttr", "CreateHorizontalApertureAttr"),
            ("GetVerticalApertureAttr", "CreateVerticalApertureAttr"),
            ("GetHorizontalApertureOffsetAttr", "CreateHorizontalApertureOffsetAttr"),
            ("GetVerticalApertureOffsetAttr", "CreateVerticalApertureOffsetAttr"),
            ("GetClippingRangeAttr", "CreateClippingRangeAttr"),
        )
        for getter_name, creator_name in attr_pairs:
            getter = getattr(source_camera, getter_name, None)
            creator = getattr(target_camera, creator_name, None)
            if getter is None or creator is None:
                continue
            source_attr = getter()
            value = source_attr.Get() if source_attr is not None else None
            if value is not None:
                creator().Set(value)

    def _overview_score(self, text: str) -> int:
        score = 0
        for index, token in enumerate(_OVERVIEW_KEYWORDS):
            if _matches(text, token):
                score += 100 - index * 4
        return score

    def _phase_score(self, camera: _CameraCandidate, phase: str) -> int:
        score = camera.overview_score
        if camera.is_observation:
            score -= 1000
        for index, token in enumerate(_PHASE_TOKENS.get(phase, ())):
            if _matches(camera.normalized_text, token):
                score += 1000 - index * 20
        return score

    def _read_active_viewport_camera_path(self) -> str | None:
        try:
            from omni.kit.viewport.utility import get_active_viewport_camera_string

            camera_path = get_active_viewport_camera_string()
        except Exception:
            return None
        if camera_path:
            return str(camera_path)
        return None

    def _maybe_switch_camera(self, camera_path: str, *, state: str, step_index: int) -> None:
        if camera_path == self._current_camera_path and state == self._last_state:
            return
        min_interval = int(getattr(self.settings, "min_switch_interval_frames", 5))
        new_role = _state_role(str(state), previous_role=self._last_role)
        state_role_changed = self._last_role is None or new_role != self._last_role
        if (
            self._should_hold_initial_overview(camera_path)
            and camera_path != self._current_camera_path
        ):
            hold_key = (str(state), camera_path)
            if hold_key != self._last_hold_log:
                print(
                    f"{self.log_prefix} hold_initial_overview state={state} "
                    f"frame={self._overview_frame_count} keep={self._current_camera_path} "
                    f"pending={camera_path} min_frames="
                    f"{int(getattr(self.settings, 'overview_initial_hold_frames', 160))}",
                    flush=True,
                )
                self._last_hold_log = hold_key
            return
        if (
            self._current_camera_path is not None
            and camera_path != self._current_camera_path
            and not state_role_changed
            and self._overview_frame_count - self._last_switch_frame < min_interval
        ):
            return
        if camera_path == self._current_camera_path:
            self._last_state = state
            self._last_role = new_role
            return
        report = self.set_active_camera(camera_path)
        self._current_camera_path = camera_path
        self._last_state = state
        self._last_role = new_role
        self._last_switch_frame = self._overview_frame_count
        switch = {
            "frame_index": int(self._overview_frame_count),
            "step_index": int(step_index),
            "state": str(state),
            "role": new_role,
            "camera_prim_path": camera_path,
            "render_camera_prim_path": report.get("render_camera_prim_path"),
            "applied": bool(report.get("applied")),
            "reason": report.get("reason"),
        }
        self._switch_log.append(switch)
        print(
            f"{self.log_prefix} switch state={state} step={step_index} "
            f"role={new_role} frame={self._overview_frame_count} camera={camera_path} "
            f"applied={switch['applied']} reason={switch['reason']}",
            flush=True,
        )

    def _should_hold_initial_overview(self, target_camera_path: str) -> bool:
        initial_hold_frames = int(getattr(self.settings, "overview_initial_hold_frames", 160))
        if initial_hold_frames <= 0:
            return False
        if self._current_camera_path is None:
            return False
        if _third_person_index_from_path(self._current_camera_path) != 1:
            return False
        if _third_person_index_from_path(target_camera_path) in {None, 1}:
            return False
        return self._overview_frame_count < initial_hold_frames

    def _should_capture(self, stream: str, timestamp: float) -> bool:
        last_capture_timestamp = self._last_capture_timestamps.get(stream)
        if self._stream_frame_counts.get(stream, 0) == 0 or last_capture_timestamp is None:
            return True
        fps = float(getattr(self.settings, "fps", 30.0))
        min_period = 1.0 / fps
        if not math.isfinite(float(timestamp)):
            return True
        return float(timestamp) - last_capture_timestamp >= (min_period * 0.95)

    def _mark_dropped(self, stream: str) -> None:
        self._stream_dropped_frame_counts[stream] = (
            self._stream_dropped_frame_counts.get(stream, 0) + 1
        )
        self._dropped_frame_count += 1

    def _capture_frame(self) -> np.ndarray | None:
        backend = str(getattr(self.settings, "overview_capture_backend", "viewport")).lower()
        if backend not in {"viewport", "render_product", "auto"}:
            backend = "viewport"
        if backend in {"viewport", "auto"}:
            frame = self._capture_viewport_buffer_frame()
            if frame is not None:
                self._capture_backend = "viewport_buffer"
                return frame
        if backend in {"render_product", "auto", "viewport"}:
            frame = self._capture_replicator_frame()
            if frame is not None:
                if backend == "viewport":
                    self._capture_backend = "replicator_rgb_annotator_fallback"
                else:
                    self._capture_backend = "replicator_rgb_annotator"
                return frame
        return None

    def _capture_replicator_frame(self) -> np.ndarray | None:
        annotator = self._ensure_replicator_annotator()
        if annotator is None:
            return None
        frame = self._rgb_from_annotator_data(annotator.get_data())
        if frame is not None:
            return frame
        return None

    def _rgb_from_annotator_data(self, data: Any) -> np.ndarray | None:
        if isinstance(data, dict):
            if "data" in data:
                data = data["data"]
            else:
                data = data.get("rgb")
        if data is None:
            return None
        array = np.asarray(data)
        if array.size == 0:
            self._capture_error = "replicator_rgb_empty"
            return None
        self._capture_error = None
        return self._overview_image_to_rgb_uint8(array)

    def _overview_image_to_rgb_uint8(self, image: Any) -> np.ndarray:
        return _image_to_rgb_uint8(
            image,
            linear_to_srgb=True,
            exposure=float(getattr(self.settings, "overview_exposure", 0.0)),
            gamma=float(getattr(self.settings, "overview_gamma", 2.2)),
        )

    def _ensure_replicator_annotator(self) -> Any | None:
        if self._rep_annotator is not None:
            return self._rep_annotator
        try:
            import omni.replicator.core as rep
        except ImportError as exc:
            self._capture_error = f"replicator_unavailable: {exc}"
            return None
        camera_path = self._current_camera_path or self.select_camera_for_state("overview")
        if camera_path is None:
            self._capture_error = "no camera path for render product"
            return None
        render_camera_path = self._renderable_camera_path(camera_path)
        width = int(getattr(self.settings, "width", 640))
        height = int(getattr(self.settings, "height", 480))
        self._rep_render_product = rep.create.render_product(
            render_camera_path,
            (width, height),
        )
        render_product_path = str(getattr(self._rep_render_product, "path", ""))
        if not render_product_path:
            self._capture_error = "render product path unavailable"
            return None
        annotator = rep.annotators.get("rgb")
        annotator.attach(str(render_product_path))
        self._rep_annotator = annotator
        self._rep_render_product_path = str(render_product_path)
        return self._rep_annotator

    def _active_viewport_render_product_path(self) -> str | None:
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
        except Exception:
            return None
        if viewport is None:
            return None
        render_product_path = getattr(viewport, "render_product_path", None)
        if not render_product_path and hasattr(viewport, "get_render_product_path"):
            render_product_path = viewport.get_render_product_path()
        if render_product_path:
            return str(render_product_path)
        return None

    def _set_replicator_camera_path(self, camera_path: str) -> None:
        render_product = self._rep_render_product
        hydra_texture = getattr(render_product, "hydra_texture", None)
        if hydra_texture is None or not hasattr(hydra_texture, "set_camera_path"):
            self._rep_annotator = None
            self._rep_render_product = None
            self._rep_render_product_path = None
            return
        try:
            hydra_texture.set_camera_path(camera_path)
        except Exception as exc:  # pragma: no cover - depends on live Kit runtime.
            self._capture_error = f"failed to update render product camera: {exc}"
            self._rep_annotator = None
            self._rep_render_product = None
            self._rep_render_product_path = None

    def _tick_kit_app_once(self) -> None:
        # Intentionally no-op. Calling omni.kit.app.update() from the recorder can
        # advance IsaacLab simulation outside the pipeline's controlled step loop.
        return

    def _ensure_capture_viewport(self) -> Any | None:
        if self._capture_viewport is not None:
            self._configure_capture_viewport(self._capture_viewport)
            return self._capture_viewport
        viewport = None
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
        except Exception:
            viewport = None
        if viewport is None:
            try:
                from omni.kit.viewport.utility import create_viewport_window

                width = int(getattr(self.settings, "width", 1280))
                height = int(getattr(self.settings, "height", 720))
                try:
                    window = create_viewport_window(
                        "Overview Video Capture",
                        width=width,
                        height=height,
                        visible=False,
                    )
                except TypeError:
                    try:
                        window = create_viewport_window("Overview Video Capture")
                    except TypeError:
                        window = create_viewport_window()
                self._capture_viewport_window = window
                viewport = (
                    getattr(window, "viewport_api", None)
                    or getattr(window, "viewport", None)
                    or window
                )
                print(
                    f"{self.log_prefix} created capture viewport size={width}x{height}",
                    flush=True,
                )
            except Exception as exc:
                self._capture_error = f"viewport_unavailable: {exc}"
                return None
        self._capture_viewport = viewport
        self._configure_capture_viewport(viewport)
        if self._current_camera_path:
            self._set_capture_viewport_camera_path(self._current_camera_path)
        return viewport

    def _configure_capture_viewport(self, viewport: Any) -> None:
        width = int(getattr(self.settings, "width", 1280))
        height = int(getattr(self.settings, "height", 720))
        resolution = (width, height)
        targets = [viewport]
        viewport_api = getattr(viewport, "viewport_api", None)
        if viewport_api is not None and viewport_api is not viewport:
            targets.append(viewport_api)
        for target in targets:
            for attr_name in ("resolution", "texture_resolution"):
                try:
                    setattr(target, attr_name, resolution)
                except Exception:
                    pass
            for method_name in ("set_texture_resolution", "set_resolution"):
                method = getattr(target, method_name, None)
                if callable(method):
                    try:
                        method(resolution)
                    except TypeError:
                        try:
                            method(width, height)
                        except Exception:
                            pass
                    except Exception:
                        pass

    def _set_capture_viewport_camera_path(self, camera_path: str) -> None:
        viewport = self._ensure_capture_viewport()
        if viewport is None:
            return
        try:
            from pxr import Sdf

            sdf_path = Sdf.Path(camera_path)
        except Exception:
            sdf_path = camera_path
        targets = [viewport]
        viewport_api = getattr(viewport, "viewport_api", None)
        if viewport_api is not None and viewport_api is not viewport:
            targets.append(viewport_api)
        for target in targets:
            for value in (sdf_path, camera_path):
                try:
                    setattr(target, "camera_path", value)
                except Exception:
                    pass
                method = getattr(target, "set_active_camera", None)
                if callable(method):
                    try:
                        method(value)
                    except Exception:
                        pass

    def _capture_viewport_buffer_frame(self) -> np.ndarray | None:
        try:
            from omni.kit.viewport.utility import capture_viewport_to_buffer
        except ImportError:
            return None
        viewport = self._ensure_capture_viewport()
        if viewport is None:
            return None
        captured: dict[str, Any] = {}

        def _callback(buffer: Any, buffer_size: int, width: int, height: int, byte_format: Any) -> None:
            try:
                captured["frame"] = self._buffer_to_image(
                    buffer,
                    buffer_size=int(buffer_size),
                    width=int(width),
                    height=int(height),
                    byte_format=str(byte_format),
                )
            except Exception as exc:
                self._capture_error = (
                    f"viewport_buffer_conversion_failed: {type(exc).__name__}: {exc}"
                )
                captured["error"] = self._capture_error

        self._configure_capture_viewport(viewport)
        capture = capture_viewport_to_buffer(viewport, _callback)
        wait_for_result = getattr(capture, "wait_for_result", None)
        if "frame" in captured:
            return captured.get("frame")
        if callable(wait_for_result) and not inspect.iscoroutinefunction(wait_for_result):
            try:
                result = wait_for_result(0)
            except TypeError:
                try:
                    result = wait_for_result()
                except Exception:
                    result = None
            except Exception:
                result = None
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
            if "frame" in captured:
                return captured.get("frame")
        return captured.get("frame")

    def _buffer_to_image(
        self,
        buffer: Any,
        *,
        buffer_size: int,
        width: int,
        height: int,
        byte_format: str,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError(f"viewport capture dimensions must be positive, got {width}x{height}")
        pixel_count = width * height
        if buffer_size % pixel_count != 0:
            raise ValueError(
                f"viewport capture buffer size {buffer_size} does not match {width}x{height}"
            )
        channels = buffer_size // pixel_count
        if channels <= 0:
            raise ValueError(
                f"viewport capture channel count must be positive, got {channels}"
            )
        raw = np.frombuffer(
            _capture_buffer_bytes(buffer, buffer_size),
            dtype=np.uint8,
            count=buffer_size,
        )
        raw = raw.reshape((height, width, channels))
        return _image_to_rgb_uint8(raw)

    def _write_camera_trajectory_frame(
        self,
        *,
        state: str,
        timestamp: float,
        step_index: int,
        frame_index: int,
    ) -> None:
        """写出与 overview 视频帧对齐的相机位姿。"""

        if not self._camera_trajectory_enabled():
            return
        payload = self._camera_trajectory_payload(
            state=state,
            timestamp=timestamp,
            step_index=step_index,
            frame_index=frame_index,
        )
        try:
            with self._camera_trajectory_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        except Exception as exc:
            self._camera_trajectory_error = f"{type(exc).__name__}: {exc}"
            self._warn_once(f"camera trajectory write failed: {self._camera_trajectory_error}")
            return
        self._camera_trajectory_frame_count += 1

    def _camera_trajectory_payload(
        self,
        *,
        state: str,
        timestamp: float,
        step_index: int,
        frame_index: int,
    ) -> dict[str, Any]:
        """从当前 USD camera 读取 3DGS 离线渲染需要的相机参数。"""

        camera_path = self._current_camera_path
        render_camera_path = self._renderable_camera_path(camera_path) if camera_path else None
        payload: dict[str, Any] = {
            "schema": "arm_vla_pct.overview_camera_trajectory.v1",
            "frame_index": int(frame_index),
            "step_index": int(step_index),
            "timestamp": float(timestamp),
            "pipeline_state": str(state),
            "camera_prim_path": camera_path,
            "render_camera_prim_path": render_camera_path,
            "video": {
                "width": int(getattr(self.settings, "width", 1280)),
                "height": int(getattr(self.settings, "height", 720)),
                "fps": float(getattr(self.settings, "fps", 25.0)),
            },
        }
        if not render_camera_path:
            payload["camera"] = {
                "available": False,
                "reason": "no_active_overview_camera",
            }
            return payload
        camera_payload = self._read_usd_camera_payload(render_camera_path)
        payload["camera"] = camera_payload
        return payload

    def _read_usd_camera_payload(self, camera_path: str) -> dict[str, Any]:
        """读取 USD camera 的世界位姿、视场角和裁剪面。"""

        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom
        except ImportError as exc:
            return {
                "available": False,
                "reason": "usd_camera_dependencies_unavailable",
                "error": str(exc),
            }
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return {
                "available": False,
                "reason": "stage_unavailable",
            }
        prim = stage.GetPrimAtPath(camera_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            return {
                "available": False,
                "reason": "camera_prim_unavailable",
                "camera_prim_path": camera_path,
                "prim_valid": bool(prim.IsValid()),
            }
        try:
            camera = UsdGeom.Camera(prim)
            matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
            eye_vec = matrix.ExtractTranslation()
            forward_vec = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
            if forward_vec.GetLength() <= 1.0e-9:
                forward_vec = Gf.Vec3d(1.0, 0.0, 0.0)
            forward_vec.Normalize()
            up_vec = matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
            if up_vec.GetLength() <= 1.0e-9:
                up_vec = Gf.Vec3d(0.0, 0.0, 1.0)
            up_vec.Normalize()
            focus_distance = self._camera_float_attr(camera.GetFocusDistanceAttr(), default=3.0)
            if not math.isfinite(focus_distance) or focus_distance <= 0.1:
                focus_distance = 3.0
            target_vec = eye_vec + forward_vec * focus_distance
            vertical_fov_deg = self._camera_vertical_fov_deg(camera)
            near_plane, far_plane = self._camera_clipping_range(camera)
            return {
                "available": True,
                "camera_prim_path": camera_path,
                "eye": [float(eye_vec[0]), float(eye_vec[1]), float(eye_vec[2])],
                "target": [
                    float(target_vec[0]),
                    float(target_vec[1]),
                    float(target_vec[2]),
                ],
                "up": [float(up_vec[0]), float(up_vec[1]), float(up_vec[2])],
                "forward": [
                    float(forward_vec[0]),
                    float(forward_vec[1]),
                    float(forward_vec[2]),
                ],
                "vertical_fov_deg": float(vertical_fov_deg),
                "near_plane": float(near_plane),
                "far_plane": float(far_plane),
                "focus_distance": float(focus_distance),
            }
        except Exception as exc:  # pragma: no cover - depends on live USD runtime.
            return {
                "available": False,
                "reason": "camera_payload_failed",
                "camera_prim_path": camera_path,
                "error": str(exc),
            }

    def _camera_float_attr(self, attr: Any, *, default: float) -> float:
        value = attr.Get() if attr is not None else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _camera_vertical_fov_deg(self, camera: Any) -> float:
        focal_length = self._camera_float_attr(camera.GetFocalLengthAttr(), default=24.0)
        vertical_aperture = self._camera_float_attr(camera.GetVerticalApertureAttr(), default=20.955)
        if focal_length <= 0.0 or vertical_aperture <= 0.0:
            return 58.0
        return math.degrees(2.0 * math.atan(vertical_aperture / (2.0 * focal_length)))

    def _camera_clipping_range(self, camera: Any) -> tuple[float, float]:
        clipping = camera.GetClippingRangeAttr().Get()
        try:
            near_plane = float(clipping[0])
            far_plane = float(clipping[1])
        except Exception:
            near_plane = 0.01
            far_plane = 100000.0
        if not math.isfinite(near_plane) or near_plane <= 0.0:
            near_plane = 0.01
        if not math.isfinite(far_plane) or far_plane <= near_plane:
            far_plane = 100000.0
        return near_plane, far_plane

    def _write_video_frame(self, image: np.ndarray, *, stream: str = "overview") -> int:
        if stream not in self.output_paths:
            raise ValueError(f"unknown video stream: {stream}")
        frame_index = int(self._stream_frame_counts.get(stream, 0))
        frame = _image_to_rgb_uint8(image)
        writer = self._writers.get(stream)
        if writer is None:
            output_path = self.output_paths[stream]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer_size = (int(frame.shape[1]), int(frame.shape[0]))
            writer = _open_mp4_writer(
                output_path,
                fps=float(getattr(self.settings, "fps", 25.0)),
                size=writer_size,
            )
            self._writers[stream] = writer
            self._writer_sizes[stream] = writer_size
            self._writer_backends[stream] = str(getattr(writer, "backend", "unknown"))
            print(
                f"{self.log_prefix} writer_opened stream={stream} path={output_path} "
                f"size={writer_size} fps={float(getattr(self.settings, 'fps', 25.0)):.3f} "
                f"writer_backend={self._writer_backends[stream]}",
                flush=True,
            )
        writer_size = self._writer_sizes[stream]
        if (frame.shape[1], frame.shape[0]) != writer_size:
            import cv2

            frame = cv2.resize(frame, writer_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        self._stream_frame_counts[stream] = self._stream_frame_counts.get(stream, 0) + 1
        return frame_index

    def _warn_once(self, message: str) -> None:
        if message == self._last_warning:
            return
        self._last_warning = message
        print(f"{self.log_prefix} warning: {message}", flush=True)


__all__ = ["OverviewVideoRecorder"]

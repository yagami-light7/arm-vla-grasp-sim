"""GUI viewport 配置的无 Isaac 依赖测试。"""

from __future__ import annotations

import ctypes
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.pipeline.run_full_physics_pipeline import _build_parser, _parse_args
from source.recording.overview_video_recorder import (
    OverviewVideoRecorder,
    _CameraCandidate,
    compose_multiview_frame,
)
from source.simulation.viewport import candidate_stage_camera_paths


@dataclass(frozen=True)
class _OverviewVideoSettings:
    enabled: bool = True
    mode: str = "overview"
    output_path: Path | None = None
    fps: float = 25.0
    overview_camera_mode: str = "auto"
    overview_camera_schedule_path: Path | None = None
    width: int = 64
    height: int = 48
    overview_capture_backend: str = "viewport"
    min_switch_interval_frames: int = 5
    overview_initial_hold_frames: int = 160
    overview_exposure: float = 0.0
    overview_gamma: float = 2.2


class SimulationViewportTest(unittest.TestCase):
    def test_default_camera_supports_case_and_reference_fallbacks(self) -> None:
        candidates = candidate_stage_camera_paths("/World/Camera_main")

        self.assertEqual(candidates[0], "/World/Camera_main")
        self.assertIn("/World/camera_main", candidates)
        self.assertIn("/World/Camera0", candidates)
        self.assertIn("/World/overview", candidates)
        self.assertIn("/World/Camera1", candidates)
        self.assertIn("/World/Camera_font", candidates)
        self.assertIn("/World/camera0", candidates)
        self.assertIn("/World/camera1", candidates)
        self.assertIn("/World/nav_visual_scene/Camera_main", candidates)
        self.assertIn("/World/gauss/Camera0", candidates)
        self.assertIn("/World/gauss/Camera1", candidates)
        self.assertIn("/World/gauss/Camera_font", candidates)
        self.assertIn("/World/gauss/camera0", candidates)
        self.assertIn("/World/gauss/camera1", candidates)
        self.assertIn("/World/contact_visual_scene/camera1", candidates)
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_lowercase_camera_can_fall_back_to_baseline_path(self) -> None:
        candidates = candidate_stage_camera_paths("/World/camera_main")

        self.assertIn("/World/Camera_main", candidates)
        self.assertIn("/World/contact_visual_scene/camera_main", candidates)

    def test_camera_numbered_names_are_preferred_for_current_scene(self) -> None:
        candidates = candidate_stage_camera_paths("/World/Camera0")

        self.assertEqual(candidates[0], "/World/Camera0")
        self.assertIn("/World/Camera1", candidates)
        self.assertIn("/World/Camera2", candidates)
        self.assertIn("/World/Camera3", candidates)
        self.assertIn("/World/camera0", candidates)
        self.assertIn("/World/camera1", candidates)
        self.assertIn("/World/gauss/camera0", candidates)
        self.assertIn("/World/gauss/camera1", candidates)
        self.assertIn("/World/gauss/Camera0", candidates)
        self.assertIn("/World/gauss/Camera1", candidates)
        self.assertIn("/World/nav_visual_scene/Camera0", candidates)

    def test_overview_camera0_is_initial_default_for_numbered_scene_cameras(self) -> None:
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(),
            episode_dir=".",
            episode_id=0,
        )
        recorder._overview_cameras = tuple(  # noqa: SLF001 - 单元测试仅验证相机选择。
            _CameraCandidate(
                path=f"/World/Camera{index}",
                name=f"Camera{index}",
                normalized_text=f"/world/camera{index} camera{index}",
                is_observation=False,
                overview_score=100,
            )
            for index in range(4)
        )

        self.assertEqual(recorder.select_camera_for_state("RESET_EPISODE"), "/World/Camera0")

    def test_fixed_mode_keeps_authored_overview_for_every_pipeline_state(self) -> None:
        """fixed 模式必须让 image/video/GUI 使用同一个 authored overview。"""

        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(overview_camera_mode="fixed"),
            episode_dir=".",
            episode_id=0,
        )
        recorder._all_cameras = (  # noqa: SLF001 - 单元测试仅验证相机选择。
            _CameraCandidate(
                path="/World/overview",
                name="overview",
                normalized_text="/world/overview overview",
                is_observation=False,
                overview_score=100,
            ),
            _CameraCandidate(
                path="/World/third_person4",
                name="third_person4",
                normalized_text="/world/third_person4 third_person4",
                is_observation=False,
                overview_score=100,
            ),
        )

        for state in ("RESET_EPISODE", "EXEC_NAV_TO_PICK", "EXEC_PLACE"):
            self.assertEqual(
                recorder.select_camera_for_state(state),
                "/World/overview",
            )
        self.assertFalse(recorder._should_rediscover_cameras())  # noqa: SLF001

    def test_headless_overview_schedule_selects_camera_by_state_and_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            schedule_path = Path(tmp_dir) / "schedule.json"
            schedule_path.write_text(
                """{
  "default_camera": "/World/Camera0",
  "rules": [
    {"states": ["exec_nav_to_place"], "z_min": 2.0, "camera": "/World/Camera7"},
    {"states": ["exec_nav_to_place"], "camera": "/World/Camera3"}
  ]
}""",
                encoding="utf-8",
            )
            recorder = OverviewVideoRecorder(
                settings=_OverviewVideoSettings(
                    overview_camera_schedule_path=schedule_path
                ),
                episode_dir=tmp_dir,
                episode_id=0,
                auto_switch_camera=True,
            )
            recorder._overview_cameras = tuple(  # noqa: SLF001
                _CameraCandidate(
                    path=f"/World/Camera{index}",
                    name=f"Camera{index}",
                    normalized_text=f"/world/camera{index} camera{index}",
                    is_observation=False,
                    overview_score=100,
                )
                for index in range(9)
            )

            lower = recorder.select_camera_for_state(
                "exec_nav_to_place",
                robot_root_pose=(1.0, 5.0, 0.2),
                step_index=100,
            )
            upper = recorder.select_camera_for_state(
                "exec_nav_to_place",
                robot_root_pose=(1.0, 5.0, 2.5),
                step_index=200,
            )

        self.assertEqual(lower, "/World/Camera3")
        self.assertEqual(upper, "/World/Camera7")

    def test_multifloor_schedule_keeps_ordered_camera_transitions(self) -> None:
        schedule_path = (
            Path(__file__).resolve().parents[1]
            / "configs/recording/multifloor_overview_camera_schedule.json"
        )
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(
                overview_camera_schedule_path=schedule_path,
            ),
            episode_dir=".",
            episode_id=0,
            auto_switch_camera=True,
        )
        recorder._overview_cameras = tuple(  # noqa: SLF001 - 单元测试仅验证相机选择。
            _CameraCandidate(
                path=f"/World/Camera{index}",
                name=f"Camera{index}",
                normalized_text=f"/world/camera{index} camera{index}",
                is_observation=False,
                overview_score=100,
            )
            for index in range(9)
        )

        samples = (
            ("exec_nav_to_place", (-3.5, 6.6, 0.2), "/World/Camera1"),
            ("exec_nav_to_place", (-2.5, 4.9, 0.2), "/World/Camera2"),
            ("exec_nav_to_place", (0.7, 4.7, 0.2), "/World/Camera2"),
            ("exec_nav_to_place", (0.8, 4.7, 0.2), "/World/Camera3"),
            ("exec_nav_to_place", (2.7, 8.8, 2.3), "/World/Camera5"),
            ("exec_nav_to_place", (2.7, 5.4, 3.4), "/World/Camera5"),
            ("exec_nav_to_place", (2.7, 4.2, 3.4), "/World/Camera6"),
            ("exec_nav_to_place", (0.2, 2.1, 3.4), "/World/Camera7"),
            ("plan_place", (0.35, 0.1, 3.4), "/World/Camera8"),
        )
        for state, root_pose, expected_camera in samples:
            with self.subTest(state=state, root_pose=root_pose):
                self.assertEqual(
                    recorder.select_camera_for_state(
                        state,
                        robot_root_pose=root_pose,
                    ),
                    expected_camera,
                )

    def test_stair_locomotion_schedule_uses_camera3_then_camera5(self) -> None:
        schedule_path = (
            Path(__file__).resolve().parents[1]
            / "configs/recording/stair_locomotion_camera_schedule.json"
        )
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(
                overview_camera_schedule_path=schedule_path,
            ),
            episode_dir=".",
            episode_id=0,
            auto_switch_camera=True,
        )
        recorder._overview_cameras = tuple(  # noqa: SLF001
            _CameraCandidate(
                path=f"/World/Camera{index}",
                name=f"Camera{index}",
                normalized_text=f"/world/camera{index} camera{index}",
                is_observation=False,
                overview_score=100,
            )
            for index in range(9)
        )

        lower = recorder.select_camera_for_state(
            "exec_nav_to_pick",
            robot_root_pose=(1.5, 7.0, 1.2),
            step_index=100,
        )
        upper = recorder.select_camera_for_state(
            "exec_nav_to_pick",
            robot_root_pose=(2.7, 8.0, 2.0),
            step_index=200,
        )

        self.assertEqual(lower, "/World/Camera3")
        self.assertEqual(upper, "/World/Camera5")

    def test_gui_overview_reads_manual_camera_without_auto_switch(self) -> None:
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(),
            episode_dir=".",
            episode_id=0,
            auto_switch_camera=False,
        )
        recorder._discovery_done = True  # noqa: SLF001
        with (
            mock.patch.object(
                recorder,
                "_read_active_viewport_camera_path",
                return_value="/World/Camera5",
            ),
            mock.patch.object(recorder, "_maybe_switch_camera") as switch_camera,
            mock.patch.object(recorder, "_should_capture", return_value=False),
        ):
            recorder._add_overview_frame(  # noqa: SLF001
                state="exec_nav_to_place",
                timestamp=1.0,
                step_index=10,
                robot_root_pose=(1.0, 5.0, 0.2),
            )

        switch_camera.assert_not_called()
        self.assertEqual(recorder._current_camera_path, "/World/Camera5")  # noqa: SLF001

    def test_camera_font_name_is_supported_for_current_scene(self) -> None:
        candidates = candidate_stage_camera_paths("/World/Camera_font")

        self.assertEqual(candidates[0], "/World/Camera_font")
        self.assertIn("/World/camera_font", candidates)
        self.assertIn("/World/gauss/Camera_font", candidates)
        self.assertIn("/World/contact_visual_scene/camera_font", candidates)

    def test_overview_recorder_writes_mp4_with_at_least_one_frame(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "videos"
            recorder = OverviewVideoRecorder(
                settings=_OverviewVideoSettings(output_path=output_dir),
                episode_dir=tmp_dir,
                episode_id=3,
            )
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = 255

            recorder._write_video_frame(frame)  # noqa: SLF001 - writer smoke test.
            summary = recorder.close(status="success")

            video_path = Path(summary["video_path"])
            self.assertEqual(video_path.name, "episode_000003_overview.mp4")
            self.assertTrue(video_path.is_file())
            capture = cv2.VideoCapture(str(video_path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertGreater(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            finally:
                capture.release()

    def test_all_video_mode_writes_overview_front_and_wrist_mp4s(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "videos"
            recorder = OverviewVideoRecorder(
                settings=_OverviewVideoSettings(mode="all", output_path=output_dir),
                episode_dir=tmp_dir,
                episode_id=4,
            )
            frame = np.full((48, 64, 3), 128, dtype=np.uint8)

            recorder._write_video_frame(frame, stream="overview")  # noqa: SLF001
            recorder._write_video_frame(frame, stream="front")  # noqa: SLF001
            recorder._write_video_frame(frame, stream="wrist")  # noqa: SLF001
            summary = recorder.close(status="success")

            self.assertEqual(set(summary["videos"]), {"overview", "front", "wrist"})
            for stream in ("overview", "front", "wrist"):
                video_path = output_dir / f"episode_000004_{stream}.mp4"
                self.assertTrue(video_path.is_file(), stream)
                self.assertEqual(summary["videos"][stream]["frame_count"], 1)

    def test_composite_layout_uses_synchronized_three_camera_panels(self) -> None:
        overview = np.zeros((90, 160, 3), dtype=np.uint8)
        overview[:, :, 0] = 255
        front = np.zeros((120, 160, 3), dtype=np.uint8)
        front[:, :, 1] = 255
        wrist = np.zeros((120, 160, 3), dtype=np.uint8)
        wrist[:, :, 2] = 255

        frame = compose_multiview_frame(
            {"overview": overview, "front": front, "wrist": wrist},
            width=600,
            height=360,
        )

        self.assertEqual(frame.shape, (360, 600, 3))
        self.assertGreater(int(frame[180, 200, 0]), 200)
        self.assertGreater(int(frame[90, 500, 1]), 200)
        self.assertGreater(int(frame[270, 500, 2]), 200)

    def test_composite_video_mode_writes_one_labeled_multiview_mp4(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "videos"
            recorder = OverviewVideoRecorder(
                settings=_OverviewVideoSettings(
                    mode="composite",
                    output_path=output_dir,
                    width=320,
                    height=180,
                ),
                episode_dir=tmp_dir,
                episode_id=5,
            )
            camera_images = {
                "overview": np.full((48, 64, 3), (255, 0, 0), dtype=np.uint8),
                "front": np.full((48, 64, 3), (0, 255, 0), dtype=np.uint8),
                "wrist": np.full((48, 64, 3), (0, 0, 255), dtype=np.uint8),
            }

            recorder.add_frame(
                state="exec_nav_to_pick",
                timestamp=0.0,
                step_index=0,
                camera_images=camera_images,
            )
            summary = recorder.close(status="success")

            video_path = output_dir / "episode_000005_composite.mp4"
            self.assertTrue(video_path.is_file())
            self.assertEqual(summary["streams"], ["composite"])
            self.assertEqual(summary["videos"]["composite"]["frame_count"], 1)
            self.assertEqual(
                summary["composite_layout"]["synchronization"],
                "same_simulation_step",
            )
            capture = cv2.VideoCapture(str(video_path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 180)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            finally:
                capture.release()

    def test_composite_uses_auto_switched_overview_frame(self) -> None:
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(
                mode="composite",
                width=600,
                height=360,
                overview_initial_hold_frames=0,
            ),
            episode_dir=".",
            episode_id=5,
            auto_switch_camera=True,
        )
        recorder._discovery_done = True  # noqa: SLF001
        recorder._overview_cameras = (  # noqa: SLF001
            _CameraCandidate(
                path="/World/Camera1",
                name="Camera1",
                normalized_text="/world/camera1 camera1",
                is_observation=False,
                overview_score=100,
            ),
        )
        recorder._all_cameras = recorder._overview_cameras  # noqa: SLF001
        recorder._camera_schedule = {  # noqa: SLF001
            "default_camera": "/World/Camera0",
            "rules": [
                {
                    "states": ["exec_nav_to_pick"],
                    "camera": "/World/Camera1",
                }
            ],
        }
        switched_overview = np.zeros((90, 160, 3), dtype=np.uint8)
        switched_overview[:, :, 0] = 255
        fixed_observation_overview = np.zeros((90, 160, 3), dtype=np.uint8)
        fixed_observation_overview[:, :, 2] = 255
        camera_images = {
            "overview": fixed_observation_overview,
            "front": np.zeros((90, 160, 3), dtype=np.uint8),
            "wrist": np.zeros((90, 160, 3), dtype=np.uint8),
        }

        with (
            mock.patch.object(
                recorder,
                "set_active_camera",
                return_value={
                    "applied": True,
                    "reason": None,
                    "render_camera_prim_path": "/World/Camera1",
                },
            ),
            mock.patch.object(
                recorder,
                "_capture_frame",
                return_value=switched_overview,
            ),
            mock.patch.object(
                recorder,
                "_write_video_frame",
                return_value=0,
            ) as write_frame,
        ):
            recorder.add_frame(
                state="exec_nav_to_pick",
                timestamp=0.0,
                step_index=10,
                camera_images=camera_images,
            )

        composite_frame = write_frame.call_args.args[0]
        self.assertEqual(write_frame.call_args.kwargs["stream"], "composite")
        self.assertGreater(int(composite_frame[180, 200, 0]), 200)
        self.assertLess(int(composite_frame[180, 200, 2]), 50)
        self.assertEqual(recorder._current_camera_path, "/World/Camera1")  # noqa: SLF001
        self.assertEqual(
            recorder._capture_backend,  # noqa: SLF001
            "scheduled_overview_plus_synchronized_observations",
        )

    def test_overview_recorder_saves_low_frequency_jpeg_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = OverviewVideoRecorder(
                settings=_OverviewVideoSettings(),
                episode_dir=tmp_dir,
                episode_id=5,
                save_overview_images=True,
                overview_image_fps=5.0,
                overview_jpeg_quality=88,
            )
            recorder._discovery_done = True  # noqa: SLF001
            frame = np.full((48, 64, 3), 96, dtype=np.uint8)
            with (
                mock.patch.object(recorder, "_should_capture", return_value=True),
                mock.patch.object(recorder, "_capture_frame", return_value=frame),
                mock.patch.object(recorder, "_write_video_frame", return_value=0),
            ):
                for timestamp in (0.0, 0.04, 0.20):
                    recorder._add_overview_frame(  # noqa: SLF001
                        state="exec_nav_to_place",
                        timestamp=timestamp,
                        step_index=int(timestamp * 50),
                        robot_root_pose=(1.0, 5.0, 0.2),
                    )

            image_dir = Path(tmp_dir) / "images" / "overview"
            self.assertEqual(
                sorted(path.name for path in image_dir.glob("*.jpg")),
                ["overview_00000.jpg", "overview_00001.jpg"],
            )
            summary = recorder.close(status="success")
            self.assertEqual(summary["overview_images"]["frame_count"], 2)
            self.assertEqual(summary["overview_images"]["fps"], 5.0)

    def test_third_person_cameras_are_selected_by_pipeline_state(self) -> None:
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(),
            episode_dir=".",
            episode_id=0,
        )
        recorder._overview_cameras = tuple(  # noqa: SLF001 - unit-test selection only.
            _CameraCandidate(
                path=f"/World/third_person{index}",
                name=f"third_person{index}",
                normalized_text=f"/world/third_person{index} third_person{index}",
                is_observation=False,
                overview_score=100,
            )
            for index in range(1, 5)
        )

        self.assertEqual(recorder.select_camera_for_state("RESET_EPISODE"), "/World/third_person1")
        self.assertEqual(recorder.select_camera_for_state("PLAN_NAV_TO_PICK"), "/World/third_person2")
        self.assertEqual(recorder.select_camera_for_state("EXEC_NAV_TO_PICK"), "/World/third_person2")
        self.assertEqual(recorder.select_camera_for_state("EXEC_PICK"), "/World/third_person2")
        self.assertEqual(recorder.select_camera_for_state("PLAN_NAV_TO_PLACE"), "/World/third_person2")
        self.assertEqual(recorder.select_camera_for_state("EXEC_NAV_TO_PLACE"), "/World/third_person3")
        self.assertEqual(recorder.select_camera_for_state("VERIFY_PLACE_REACHABLE"), "/World/third_person3")
        self.assertEqual(recorder.select_camera_for_state("EXEC_PLACE"), "/World/third_person4")

    def test_initial_overview_camera_is_held_before_nav_switch(self) -> None:
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(overview_initial_hold_frames=60),
            episode_dir=".",
            episode_id=0,
        )

        recorder._current_camera_path = "/World/third_person1"  # noqa: SLF001
        recorder._last_state = "reset_episode"  # noqa: SLF001
        recorder._last_role = "overview"  # noqa: SLF001
        recorder._stream_frame_counts["overview"] = 1  # noqa: SLF001
        recorder._maybe_switch_camera(  # noqa: SLF001
            "/World/third_person2",
            state="exec_nav_to_pick",
            step_index=1,
        )
        self.assertEqual(recorder._current_camera_path, "/World/third_person1")  # noqa: SLF001

        recorder._stream_frame_counts["overview"] = 60  # noqa: SLF001
        recorder._maybe_switch_camera(  # noqa: SLF001
            "/World/third_person2",
            state="exec_nav_to_pick",
            step_index=60,
        )
        self.assertEqual(recorder._current_camera_path, "/World/third_person2")  # noqa: SLF001

    def test_recorder_rediscovers_until_real_third_person_cameras_exist(self) -> None:
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(),
            episode_dir=".",
            episode_id=0,
        )
        kit_camera = _CameraCandidate(
            path="/OmniverseKit_Top",
            name="OmniverseKit_Top",
            normalized_text="/omniversekit_top omniversekit_top",
            is_observation=False,
            overview_score=50,
        )
        fallback_camera = _CameraCandidate(
            path="/World/overview_video_cameras/third_person1",
            name="third_person1",
            normalized_text="/world/overview_video_cameras/third_person1 third_person1",
            is_observation=False,
            overview_score=10_000,
        )
        real_camera = _CameraCandidate(
            path="/World/third_person1",
            name="third_person1",
            normalized_text="/world/third_person1 third_person1",
            is_observation=False,
            overview_score=100,
        )

        recorder._all_cameras = (kit_camera,)  # noqa: SLF001
        recorder._overview_cameras = (kit_camera,)  # noqa: SLF001
        self.assertTrue(recorder._should_rediscover_cameras())  # noqa: SLF001

        recorder._all_cameras = (fallback_camera,)  # noqa: SLF001
        recorder._overview_cameras = (fallback_camera,)  # noqa: SLF001
        self.assertTrue(recorder._should_rediscover_cameras())  # noqa: SLF001

        recorder._all_cameras = (fallback_camera, real_camera)  # noqa: SLF001
        recorder._overview_cameras = (fallback_camera, real_camera)  # noqa: SLF001
        self.assertFalse(recorder._should_rediscover_cameras())  # noqa: SLF001
        self.assertEqual(recorder.select_camera_for_state("RESET_EPISODE"), "/World/third_person1")

    def test_viewport_capture_does_not_call_async_wait_or_tick_app(self) -> None:
        calls: list[str] = []

        class _FakeCapture:
            async def wait_for_result(self, _timeout: float = 0.0) -> None:
                calls.append("wait_for_result")
                return None

        utility_module = types.ModuleType("omni.kit.viewport.utility")
        utility_module.get_active_viewport = lambda: object()
        utility_module.capture_viewport_to_buffer = (
            lambda _viewport, _callback: _FakeCapture()
        )
        viewport_module = types.ModuleType("omni.kit.viewport")
        kit_module = types.ModuleType("omni.kit")
        omni_module = types.ModuleType("omni")
        viewport_module.utility = utility_module
        kit_module.viewport = viewport_module
        omni_module.kit = kit_module
        modules = {
            "omni": omni_module,
            "omni.kit": kit_module,
            "omni.kit.viewport": viewport_module,
            "omni.kit.viewport.utility": utility_module,
        }
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(),
            episode_dir=".",
            episode_id=0,
        )
        recorder._tick_kit_app_once = lambda: calls.append("tick_app")  # noqa: SLF001

        with mock.patch.dict(sys.modules, modules):
            frame = recorder._capture_viewport_buffer_frame()  # noqa: SLF001

        self.assertIsNone(frame)
        self.assertEqual(calls, [])

    def test_viewport_capture_converts_anonymous_pycapsule_buffer(self) -> None:
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(),
            episode_dir=".",
            episode_id=0,
        )
        rgba = (ctypes.c_uint8 * 8)(10, 20, 30, 255, 40, 50, 60, 255)
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule_new.restype = ctypes.py_object
        capsule = capsule_new(ctypes.cast(rgba, ctypes.c_void_p), None, None)

        frame = recorder._buffer_to_image(  # noqa: SLF001
            capsule,
            buffer_size=8,
            width=2,
            height=1,
            byte_format="RGBA8_UNORM",
        )

        np.testing.assert_array_equal(
            frame,
            np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
        )

    def test_viewport_capture_callback_contains_invalid_buffer_error(self) -> None:
        class _FakeCapture:
            def wait_for_result(self, _timeout: float = 0.0) -> None:
                return None

        utility_module = types.ModuleType("omni.kit.viewport.utility")
        utility_module.get_active_viewport = lambda: object()

        def _capture(_viewport: object, callback: object) -> _FakeCapture:
            callback(object(), 8, 2, 1, "RGBA8_UNORM")
            return _FakeCapture()

        utility_module.capture_viewport_to_buffer = _capture
        viewport_module = types.ModuleType("omni.kit.viewport")
        kit_module = types.ModuleType("omni.kit")
        omni_module = types.ModuleType("omni")
        viewport_module.utility = utility_module
        kit_module.viewport = viewport_module
        omni_module.kit = kit_module
        modules = {
            "omni": omni_module,
            "omni.kit": kit_module,
            "omni.kit.viewport": viewport_module,
            "omni.kit.viewport.utility": utility_module,
        }
        recorder = OverviewVideoRecorder(
            settings=_OverviewVideoSettings(),
            episode_dir=".",
            episode_id=0,
        )

        with mock.patch.dict(sys.modules, modules):
            frame = recorder._capture_viewport_buffer_frame()  # noqa: SLF001

        self.assertIsNone(frame)
        self.assertIn("viewport_buffer_conversion_failed", recorder._capture_error)  # noqa: SLF001

    def test_cli_accepts_overview_video_arguments(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--record-video",
                "--video-mode",
                "overview",
                "--overview-camera-mode",
                "auto",
                "--overview-camera-schedule",
                "configs/recording/multifloor_overview_camera_schedule.json",
                "--overview-capture-backend",
                "viewport",
                "--video-width",
                "1280",
                "--video-height",
                "720",
                "--overview-initial-hold-frames",
                "160",
                "--overview-exposure",
                "0",
                "--overview-gamma",
                "2.2",
                "--video-out",
                "outputs/test_run/videos",
            ]
        )

        self.assertTrue(args.record_video)
        self.assertEqual(args.video_mode, "overview")
        self.assertEqual(args.overview_camera_mode, "auto")
        self.assertEqual(
            args.overview_camera_schedule,
            "configs/recording/multifloor_overview_camera_schedule.json",
        )
        self.assertEqual(args.overview_capture_backend, "viewport")
        self.assertEqual(args.video_width, 1280)
        self.assertEqual(args.video_height, 720)
        self.assertEqual(args.overview_initial_hold_frames, 160)
        self.assertEqual(args.overview_exposure, 0.0)
        self.assertEqual(args.overview_gamma, 2.2)
        self.assertEqual(args.video_out, "outputs/test_run/videos")

        all_args = parser.parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--record-video",
                "--video-mode",
                "all",
            ]
        )
        font_args = parser.parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--record-video",
                "--video-mode",
                "font",
            ]
        )
        composite_args = parser.parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--record-video",
                "--video-mode",
                "composite",
            ]
        )
        self.assertEqual(all_args.video_mode, "all")
        self.assertEqual(font_args.video_mode, "font")
        self.assertEqual(composite_args.video_mode, "composite")

    def test_scene_profiles_default_to_composite_and_video_can_be_disabled(self) -> None:
        for profile in ("liangzhu", "multi_floor"):
            with self.subTest(profile=profile):
                defaults = _parse_args(["--scene-profile", profile])
                disabled = _parse_args(
                    ["--scene-profile", profile, "--no-record-video"]
                )

                self.assertTrue(defaults.record_video)
                self.assertEqual(defaults.video_mode, "composite")
                self.assertFalse(disabled.record_video)


if __name__ == "__main__":
    unittest.main()

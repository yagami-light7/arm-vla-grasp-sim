"""Configuration for the full-physics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PCT_MULTIFLOOR_LOCOMOTION_TASK = "RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0"
DEFAULT_OVERVIEW_CAMERA_PRIM_PATH = "/World/overview"


@dataclass(frozen=True)
class BaseGoalRandomizationSettings:
    """pick/place 导航交接位姿随机化参数。"""

    enabled: bool = True
    pick_radius_min_m: float = 0.45
    pick_radius_max_m: float = 0.60
    pick_angle_noise_deg: float = 20.0
    pick_yaw_noise_deg: float = 0.0
    # place 周围障碍更密集，使用矩形 offset 而不是极坐标采样。
    place_offset_x_range_m: tuple[float, float] = (0.30, 0.40)
    place_offset_y_range_m: tuple[float, float] = (-0.15, -0.08)
    place_radius_min_m: float = 0.35
    place_radius_max_m: float = 0.40
    place_angle_noise_deg: float = 0.0
    place_yaw_noise_deg: float = 0.0
    arm_base_offset_x_m: float = 0.12
    arm_base_offset_y_m: float = 0.0
    arm_workspace_min_xy_radius_m: float = 0.35
    arm_workspace_max_xy_radius_m: float = 0.58
    # place 需要给导航终止误差和真实 tracking 留余量，不能贴着机械臂可达边界采样。
    place_workspace_max_xy_radius_m: float = 0.45
    # place 阶段 yaw 不参与导航到达判定，真实 arm-base 可能偏离名义估计；
    # 额外限制目标到 robot base 的半径，给 XY 终止误差和 yaw-free frame 留余量。
    place_robot_base_max_xy_radius_m: float = 0.43
    nav_map_min_clearance_m: float = 0.10
    max_goal_sample_attempts: int = 100
    max_place_target_sample_attempts: int = 50
    fallback_to_fixed_offset: bool = True
    seed_offset: int = 12137
    validate_with_curobo: bool = False


@dataclass(frozen=True)
class StateLimits:
    """Maximum ticks allowed in each type of pipeline phase."""

    build_stage: int = 5
    reset_episode: int = 300
    planning: int = 180
    navigation: int = 5000
    manipulation: int = 3000
    verification: int = 20
    export: int = 20
    cleanup: int = 20
    episode: int = 15000


@dataclass(frozen=True)
class NavigationSettings:
    """第三阶段短路线导航 smoke 的固定参数。"""

    global_planner: str = "astar"
    pct_enabled: bool = False
    pct_planner_root: Path | None = None
    pct_server_script: Path | None = None
    pct_server_python: Path | None = None
    pct_tomogram_path: Path | None = None
    pct_walkable_path: Path | None = None
    pct_collision_ply_path: Path | None = None
    pct_tomogram_name: str = "mutifloor"
    pct_fallback_to_astar: bool = True
    pct_coord_mode: str = "sim_to_pct_180deg"
    pct_offset_x: float = 0.0
    pct_offset_y: float = 0.0
    pct_scale_x: float = 1.0
    pct_scale_y: float = 1.0
    pct_vertical_obstacle_min_slices: int = 0
    pct_vertical_obstacle_dilation_radius_cells: int = 0
    pct_global_vertical_obstacle_min_slices: int = 7
    pct_cross_floor_vertical_obstacle_min_slices: int = 9
    pct_cross_floor_gateway_points: tuple[tuple[float, float, float], ...] = (
        (1.5, 5.7, 0.6),
    )
    pct_cross_floor_stair_exit_points: tuple[tuple[float, float, float], ...] = (
        (2.90, 7.05, 3.0),
    )
    pct_cross_floor_stair_midpoint_points: tuple[tuple[float, float, float], ...] = (
        (1.51822, 6.27683, 0.29486),
        (2.94512, 9.14634, 1.64666),
        (1.9202, 9.52807, 1.71919),
        (2.89841, 7.79872, 2.61031),
    )
    pct_cross_floor_gateway_radius_m: float = 0.6
    pct_robot_root_to_floor_m: float = 0.45
    pct_body_obstacle_min_height_m: float = 0.30
    pct_body_obstacle_max_height_m: float = 1.0
    pct_stair_min_horizontal_per_slice_m: float = 0.40
    pct_stair_max_horizontal_per_slice_m: float = 0.90
    pct_stair_vertical_radius_m: float = 0.60
    pct_stair_progress_tolerance: float = 0.35
    pct_stair_progress_cost_weight: float = 20.0
    pct_obstacle_clearance_radius_m: float = 0.60
    pct_obstacle_clearance_cost_weight: float = 2.0
    pct_multifloor_vertical_obstacle_min_slices: int = 5
    pct_multifloor_obstacle_inflate_radius: float = 0.12
    pct_multifloor_route_corridor_radius: float = 0.45
    pct_task_object_keepout_radius: float = 0.18
    pct_carry_max_linear_velocity: float = 0.25
    pct_carry_max_angular_velocity: float = 0.30
    pct_carry_max_linear_accel: float = 1.00
    pct_carry_path_deviation_limit: float = 0.14
    pct_carry_initial_alignment_path_deviation_limit: float = 0.40
    pct_carry_path_recovery_deviation_limit: float = 0.50
    pct_carry_max_infeasible_recomputes: int = 8
    pct_stair_float_enabled: bool = False
    pct_stair_float_speed_mps: float = 0.18
    pct_stair_float_activation_radius_m: float = 0.45
    pct_stair_float_completion_radius_m: float = 0.25
    pct_stair_float_min_z_delta_m: float = 0.75
    pct_stair_float_approach_distance_m: float = 6.00
    pct_stair_float_exit_distance_m: float = 1.40
    pct_stair_float_settle_time_s: float = 1.20
    pct_stair_float_release_settle_time_s: float = 0.80
    pct_stair_float_yaw_lookahead_m: float = 0.35
    pct_stair_float_min_root_z_offset_m: float = 0.18
    pct_stair_float_release_root_z_offset_m: float = 0.36
    goal_z_tolerance: float = 0.35
    # 对齐稳定 random nav-pick-place baseline；0.25 m 已会占用当前 place goal。
    global_inflate_radius: float = 0.20
    local_clearance_radius: float = 0.20
    # 恢复本地验证稳定的 0.80 m/s brisk + fast-DWA profile。
    brisk_nav: bool = True
    fast_dwa: bool = True
    control_dt: float = 0.02
    dwa_replan_interval_steps: int = 2
    lookahead_distance: float = 0.35
    prediction_horizon: float = 0.90
    max_linear_velocity: float = 0.50
    max_angular_velocity: float = 1.00
    min_active_linear_velocity: float = 0.30
    goal_tolerance: float = 0.15
    terminal_start_distance: float = 0.18
    close_goal_speed_limit: float = 0.30
    near_goal_min_active_linear_velocity: float = 0.30
    speed_bias: float = 0.35
    max_linear_accel: float = 2.5
    final_position_tolerance: float = 0.18
    # 四足携物短程定位的独立 handoff 半径；None 表示复用通用位置容差。
    place_position_tolerance: float | None = None
    final_yaw_tolerance: float = 3.141592653589793
    stable_linear_velocity: float = 0.06
    stable_angular_velocity: float = 0.20
    require_yaw_alignment: bool = False
    require_stable_base: bool = False
    stall_window_steps: int = 240
    stall_min_progress: float = 0.05
    # The pct_multifloor locomotion policy can receive a low DWA command while
    # producing effectively zero displacement. Count those commands as active so
    # the sliding-window stall gate does not wait for the much larger state timeout.
    stall_min_forward_command: float = 0.03


@dataclass(frozen=True)
class LocomotionPolicySettings:
    """底层 locomotion policy 选择；全局 planner 与 policy 保持解耦。"""

    locomotion_policy_backend: str = "rsl_rl"
    locomotion_task: str | None = None
    locomotion_checkpoint: Path | None = None
    locomotion_checkpoint_required: bool = False
    policy_profile: str = "flat"


@dataclass(frozen=True)
class ManipulationSettings:
    """机械臂执行诊断参数；默认只记录风险，不改变 smoke 成功语义。"""

    lock_base_during_manipulation: bool = True
    lock_support_joints_during_manipulation: bool = True
    replan_pick_from_current_state: bool = True
    # 多楼层扫描场景中的动态物体先自由沉降，再把稳定 PhysX 位姿作为抓取基准。
    settle_object_before_navigation: bool = False
    object_settle_max_steps: int = 240
    object_settle_required_stable_steps: int = 20
    object_settle_linear_velocity_mps: float = 0.03
    object_settle_angular_velocity_rps: float = 0.20
    object_settle_max_displacement_m: float = 0.25
    settle_base_before_navigation: bool = False
    base_settle_linear_velocity_mps: float = 0.08
    base_settle_angular_velocity_rps: float = 0.30
    base_settle_max_tilt_rad: float = 0.20
    # 相对上一版统一缩短一半执行时长，实现约 2 倍 tracking 速度；
    # 只改变物理控制目标的时间采样，不修改 cuRobo 路径几何。
    arm_motion_time_scale: float = 0.50
    pick_approach_motion_time_scale: float = 0.50
    # place 携物段对齐稳定 baseline。pick 仍可保持 2 倍速，place 三段不再使用全局 2 倍速。
    place_move_to_pre_place_motion_time_scale: float = 1.00
    place_approach_motion_time_scale: float = 1.00
    place_retreat_motion_time_scale: float = 1.00
    arm_post_motion_hold_duration_s: float = 0.75
    arm_post_motion_joint_error_tolerance: float = 0.030
    # 对齐 video baseline：松爪后先保持释放位姿，让苹果在桌面稳定，再执行退臂。
    place_release_settle_duration_s: float = 0.50
    # 对齐稳定 baseline 的 nav->pick handoff：只停驻并锁住已有姿态，不改变底盘站高。
    base_lock_settle_steps: int = 60
    # carry 到 place 后苹果已经由真实接触夹持；长时间停驻会放大滑移风险。
    place_base_lock_settle_steps: int = 0
    plan_start_state_warning_threshold: float = 0.25
    plan_start_state_failure_threshold: float = 0.75
    fail_on_place_plan_start_state_mismatch: bool = False
    # 通用默认保持关闭；full-physics 工厂会显式开启 main 的反向轨迹回位。
    return_home_after_pick: bool = False
    # 仅保留给显式兼容路径；full-physics 默认不会触发该手写段。
    pick_return_home_duration_s: float = 1.5
    # baseline 的 after-pick 额外等待默认为 0；return-home 到位后即可进入 carry。
    pick_home_hold_duration_s: float = 0.0
    pick_return_home_skip_tolerance: float = 0.01
    return_home_after_place: bool = True
    place_return_home_duration_s: float = 1.20
    place_return_home_skip_tolerance: float = 0.01
    # 对齐 baseline arm-place：轻放时物体中心只高于目标 1.0 cm。
    place_release_clearance_min_m: float = 0.010
    place_pre_clearance_min_m: float = 0.06
    hold_arm_home_during_carry: bool = True
    carry_home_tracking_tolerance: float = 0.25
    # pick 接触确认后仅额外闭合少量预紧，避免导航时持续以零开度挤压物体。
    carry_gripper_preload_m: float = 0.012
    # carry 前后 object-TCP 相对位置变化超过该阈值，视为抓取滑移或掉落。
    carry_object_tcp_slip_tolerance: float = 0.10
    insert_place_plan_start_transition: bool = True
    place_plan_start_transition_duration_s: float = 0.5
    # full-physics 在 pick 后回 home 并 carry；place 默认保持当前 TCP 姿态，
    # 避免把旧 pick 姿态强加到真实导航后的 base frame 里导致 pre-place 不可解。
    reuse_pick_grasp_orientation_for_place: bool = False


@dataclass(frozen=True)
class RandomizationSettings:
    """任务随机化固定参数；CLI 只控制是否启用。"""

    enabled: bool = False
    show_debug_region: bool = False
    # 良渚联合采样优先复用 PCT CLI 的 collision PLY；未提供时再读 task 环境变量。
    collision_ply_path: Path | None = None
    pick_x_range: tuple[float, float] = (0.90, 0.95)
    pick_y_range: tuple[float, float] = (0.75, 1.50)
    place_x_range: tuple[float, float] = (0.70, 0.75)
    place_y_range: tuple[float, float] = (5.00, 5.30)
    clearance_radius: float = 0.20
    min_boundary_clearance: float = 0.25
    edge_biased: bool = True
    edge_margin: float = 0.12
    edge_min_clearance: float = 0.03
    place_seed_offset: int = 9173
    base_goal: BaseGoalRandomizationSettings = field(
        default_factory=BaseGoalRandomizationSettings,
    )


@dataclass(frozen=True)
class RecordingSettings:
    """对齐 DWA 仓库的连续数据采集和 LeRobot v2.1 输出参数。"""

    enabled: bool = True
    dataset_fps: float = 5.0
    image_height: int = 480
    image_width: int = 640
    jpeg_quality: int = 90
    chunks_size: int = 1000
    # front/wrist/overview 都由 IsaacLab runtime 直接采集并参与完整性检查。
    camera_keys: tuple[str, ...] = ("front", "wrist", "overview")
    primary_camera_key: str = "front"
    overview_camera_prim_path: str = DEFAULT_OVERVIEW_CAMERA_PRIM_PATH
    save_raw_images: bool = True
    debug_per_episode_lerobot: bool = True
    unified_dataset: bool = True
    validate_export: bool = True


@dataclass(frozen=True)
class SceneLightingSettings:
    """真实 Isaac stage 的灯光模式；默认先服务相机保存图像。"""

    scene_light_mode: str = "camera"
    camera_light_intensity: float = 3500.0
    camera_light_radius: float = 2.0


@dataclass(frozen=True)
class VideoRecordingSettings:
    """展示用 overview 视频录制参数；不参与训练 observation 数据。"""

    enabled: bool = False
    mode: str = "overview"
    output_path: Path | None = None
    fps: float = 25.0
    overview_camera_mode: str = "fixed"
    overview_camera_prim_path: str = DEFAULT_OVERVIEW_CAMERA_PRIM_PATH
    width: int = 1280
    height: int = 720
    overview_capture_backend: str = "viewport"
    min_switch_interval_frames: int = 15
    overview_initial_hold_frames: int = 160
    overview_exposure: float = 0.0
    overview_gamma: float = 2.2
    export_camera_trajectory: bool = False
    camera_trajectory_path: Path | None = None

    @property
    def modes(self) -> tuple[str, ...]:
        mode = self.mode.lower()
        if mode == "all":
            return ("overview", "front", "wrist")
        if mode == "font":
            return ("front",)
        return (mode,)


@dataclass(frozen=True)
class FullPhysicsConfig:
    """Runtime configuration kept intentionally smaller than legacy CLIs."""

    task_json: Path
    output_dir: Path
    num_episodes: int = 1
    seed: int = 0
    headless: bool = True
    keep_window_open: bool = False
    show_planned_trajectories: bool = False
    dry_run: bool = False
    simulation_smoke: bool = False
    navigation_smoke: bool = False
    navigation_carry_smoke: bool = False
    pct_plan_preview: bool = False
    pick_smoke: bool = False
    manipulation_smoke: bool = False
    manipulation_apply_smoke: bool = False
    full_physics: bool = False
    pick_plan_json: Path | None = None
    place_plan_json: Path | None = None
    navigation: NavigationSettings = field(default_factory=NavigationSettings)
    locomotion: LocomotionPolicySettings = field(default_factory=LocomotionPolicySettings)
    manipulation: ManipulationSettings = field(default_factory=ManipulationSettings)
    randomization: RandomizationSettings = field(default_factory=RandomizationSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    lighting: SceneLightingSettings = field(default_factory=SceneLightingSettings)
    video: VideoRecordingSettings = field(default_factory=VideoRecordingSettings)
    limits: StateLimits = field(default_factory=StateLimits)

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be at least 1")
        if self.limits.episode < 1:
            raise ValueError("episode tick limit must be positive")
        if self.navigation.max_linear_velocity <= 0.0:
            raise ValueError("max_linear_velocity must be positive")
        if self.navigation.global_planner not in {"astar", "pct"}:
            raise ValueError("global_planner must be one of: astar, pct")
        if self.navigation.pct_scale_x == 0.0 or self.navigation.pct_scale_y == 0.0:
            raise ValueError("PCT coordinate scales must be non-zero")
        if self.navigation.pct_vertical_obstacle_min_slices < 0:
            raise ValueError("pct_vertical_obstacle_min_slices must be non-negative")
        if self.navigation.pct_vertical_obstacle_dilation_radius_cells < 0:
            raise ValueError("pct_vertical_obstacle_dilation_radius_cells must be non-negative")
        if self.navigation.pct_multifloor_vertical_obstacle_min_slices < 1:
            raise ValueError(
                "pct_multifloor_vertical_obstacle_min_slices must be positive"
            )
        if self.navigation.pct_global_vertical_obstacle_min_slices < 1:
            raise ValueError(
                "pct_global_vertical_obstacle_min_slices must be positive"
            )
        if self.navigation.pct_cross_floor_vertical_obstacle_min_slices < 1:
            raise ValueError(
                "pct_cross_floor_vertical_obstacle_min_slices must be positive"
            )
        if self.navigation.pct_cross_floor_gateway_radius_m < 0.0:
            raise ValueError("pct_cross_floor_gateway_radius_m must be non-negative")
        for gateway in self.navigation.pct_cross_floor_gateway_points:
            if len(gateway) != 3:
                raise ValueError("pct_cross_floor_gateway_points entries must be xyz triples")
        for stair_exit in self.navigation.pct_cross_floor_stair_exit_points:
            if len(stair_exit) != 3:
                raise ValueError(
                    "pct_cross_floor_stair_exit_points entries must be xyz triples"
                )
        for stair_midpoint in self.navigation.pct_cross_floor_stair_midpoint_points:
            if len(stair_midpoint) != 3:
                raise ValueError(
                    "pct_cross_floor_stair_midpoint_points entries must be xyz triples"
                )
        if self.navigation.pct_robot_root_to_floor_m < 0.0:
            raise ValueError("pct_robot_root_to_floor_m must be non-negative")
        if self.navigation.pct_stair_float_speed_mps <= 0.0:
            raise ValueError("pct_stair_float_speed_mps must be positive")
        if self.navigation.pct_stair_float_activation_radius_m < 0.0:
            raise ValueError(
                "pct_stair_float_activation_radius_m must be non-negative"
            )
        if self.navigation.pct_stair_float_completion_radius_m < 0.0:
            raise ValueError(
                "pct_stair_float_completion_radius_m must be non-negative"
            )
        if self.navigation.pct_stair_float_min_z_delta_m <= 0.0:
            raise ValueError("pct_stair_float_min_z_delta_m must be positive")
        if self.navigation.pct_stair_float_approach_distance_m < 0.0:
            raise ValueError(
                "pct_stair_float_approach_distance_m must be non-negative"
            )
        if self.navigation.pct_stair_float_exit_distance_m < 0.0:
            raise ValueError(
                "pct_stair_float_exit_distance_m must be non-negative"
            )
        if self.navigation.pct_stair_float_settle_time_s < 0.0:
            raise ValueError("pct_stair_float_settle_time_s must be non-negative")
        if self.navigation.pct_stair_float_release_settle_time_s < 0.0:
            raise ValueError(
                "pct_stair_float_release_settle_time_s must be non-negative"
            )
        if self.navigation.pct_stair_float_yaw_lookahead_m < 0.0:
            raise ValueError(
                "pct_stair_float_yaw_lookahead_m must be non-negative"
            )
        if self.navigation.pct_stair_float_min_root_z_offset_m < 0.0:
            raise ValueError(
                "pct_stair_float_min_root_z_offset_m must be non-negative"
            )
        if self.navigation.pct_stair_float_release_root_z_offset_m < 0.0:
            raise ValueError(
                "pct_stair_float_release_root_z_offset_m must be non-negative"
            )
        if self.navigation.pct_body_obstacle_min_height_m < 0.0:
            raise ValueError(
                "pct_body_obstacle_min_height_m must be non-negative"
            )
        if (
            self.navigation.pct_body_obstacle_max_height_m
            <= self.navigation.pct_body_obstacle_min_height_m
        ):
            raise ValueError(
                "pct_body_obstacle_max_height_m must exceed minimum height"
            )
        if self.navigation.pct_stair_min_horizontal_per_slice_m <= 0.0:
            raise ValueError(
                "pct_stair_min_horizontal_per_slice_m must be positive"
            )
        if (
            self.navigation.pct_stair_max_horizontal_per_slice_m
            < self.navigation.pct_stair_min_horizontal_per_slice_m
        ):
            raise ValueError(
                "pct_stair_max_horizontal_per_slice_m must not be smaller than minimum"
            )
        if self.navigation.pct_stair_vertical_radius_m <= 0.0:
            raise ValueError("pct_stair_vertical_radius_m must be positive")
        if not 0.0 <= self.navigation.pct_stair_progress_tolerance <= 1.0:
            raise ValueError("pct_stair_progress_tolerance must be between 0 and 1")
        if self.navigation.pct_stair_progress_cost_weight < 0.0:
            raise ValueError("pct_stair_progress_cost_weight must be non-negative")
        if self.navigation.pct_obstacle_clearance_radius_m < 0.0:
            raise ValueError(
                "pct_obstacle_clearance_radius_m must be non-negative"
            )
        if self.navigation.pct_obstacle_clearance_cost_weight < 0.0:
            raise ValueError(
                "pct_obstacle_clearance_cost_weight must be non-negative"
            )
        if self.navigation.pct_multifloor_obstacle_inflate_radius < 0.0:
            raise ValueError(
                "pct_multifloor_obstacle_inflate_radius must be non-negative"
            )
        if self.navigation.pct_multifloor_route_corridor_radius < 0.0:
            raise ValueError(
                "pct_multifloor_route_corridor_radius must be non-negative"
            )
        if self.navigation.pct_carry_max_linear_velocity <= 0.0:
            raise ValueError("pct_carry_max_linear_velocity must be positive")
        if self.navigation.pct_carry_max_angular_velocity <= 0.0:
            raise ValueError("pct_carry_max_angular_velocity must be positive")
        if self.navigation.pct_carry_max_linear_accel <= 0.0:
            raise ValueError("pct_carry_max_linear_accel must be positive")
        if self.navigation.pct_carry_path_deviation_limit <= 0.0:
            raise ValueError("pct_carry_path_deviation_limit must be positive")
        if self.navigation.pct_carry_initial_alignment_path_deviation_limit <= 0.0:
            raise ValueError(
                "pct_carry_initial_alignment_path_deviation_limit must be positive"
            )
        if self.navigation.pct_carry_path_recovery_deviation_limit <= 0.0:
            raise ValueError(
                "pct_carry_path_recovery_deviation_limit must be positive"
            )
        if self.navigation.goal_z_tolerance < 0.0:
            raise ValueError("goal_z_tolerance must be non-negative")
        if (
            self.navigation.place_position_tolerance is not None
            and self.navigation.place_position_tolerance <= 0.0
        ):
            raise ValueError("place_position_tolerance must be positive")
        if self.navigation.dwa_replan_interval_steps < 1:
            raise ValueError("dwa_replan_interval_steps must be at least 1")
        if self.navigation.max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity must be positive")
        if self.navigation.min_active_linear_velocity < 0.0:
            raise ValueError("min_active_linear_velocity must be non-negative")
        if self.navigation.speed_bias < 0.0:
            raise ValueError("speed_bias must be non-negative")
        if self.navigation.max_linear_accel <= 0.0:
            raise ValueError("max_linear_accel must be positive")
        if self.manipulation.plan_start_state_warning_threshold < 0.0:
            raise ValueError("plan_start_state_warning_threshold must be non-negative")
        if self.locomotion.locomotion_policy_backend != "rsl_rl":
            raise ValueError("only locomotion_policy_backend='rsl_rl' is currently supported")
        if self.locomotion.policy_profile not in {"flat", "pct_multifloor"}:
            raise ValueError("policy_profile must be one of: flat, pct_multifloor")
        checkpoint = self.locomotion.locomotion_checkpoint
        checkpoint_missing = checkpoint is None or not Path(checkpoint).expanduser().is_file()
        if self.locomotion.policy_profile == "pct_multifloor" and not self.pct_plan_preview:
            if checkpoint_missing:
                raise ValueError(
                    "PCT multi-floor policy checkpoint missing. "
                    "Please train or pass --locomotion-checkpoint."
                )
            if not self.locomotion.locomotion_task:
                raise ValueError(
                    "pct_multifloor policy_profile 需要显式指定兼容的 locomotion_task；"
                    f"当前 dog-only checkpoint 应使用 {PCT_MULTIFLOOR_LOCOMOTION_TASK}。"
                )
        elif self.locomotion.locomotion_checkpoint_required and checkpoint_missing:
            raise ValueError("locomotion policy checkpoint missing")
        if not isinstance(self.manipulation.replan_pick_from_current_state, bool):
            raise ValueError("replan_pick_from_current_state must be a bool")
        if self.manipulation.object_settle_max_steps < 1:
            raise ValueError("object_settle_max_steps must be positive")
        if self.manipulation.object_settle_max_steps >= self.limits.reset_episode:
            raise ValueError("object_settle_max_steps must be smaller than reset_episode tick limit")
        if self.manipulation.object_settle_required_stable_steps < 1:
            raise ValueError("object_settle_required_stable_steps must be positive")
        if (
            self.manipulation.object_settle_required_stable_steps
            > self.manipulation.object_settle_max_steps
        ):
            raise ValueError(
                "object_settle_required_stable_steps must not exceed object_settle_max_steps"
            )
        if self.manipulation.object_settle_linear_velocity_mps < 0.0:
            raise ValueError("object_settle_linear_velocity_mps must be non-negative")
        if self.manipulation.object_settle_angular_velocity_rps < 0.0:
            raise ValueError("object_settle_angular_velocity_rps must be non-negative")
        if self.manipulation.object_settle_max_displacement_m <= 0.0:
            raise ValueError("object_settle_max_displacement_m must be positive")
        if self.manipulation.base_settle_linear_velocity_mps < 0.0:
            raise ValueError("base_settle_linear_velocity_mps must be non-negative")
        if self.manipulation.base_settle_angular_velocity_rps < 0.0:
            raise ValueError("base_settle_angular_velocity_rps must be non-negative")
        if self.manipulation.base_settle_max_tilt_rad < 0.0:
            raise ValueError("base_settle_max_tilt_rad must be non-negative")
        if self.manipulation.arm_motion_time_scale <= 0.0:
            raise ValueError("arm_motion_time_scale must be positive")
        if self.manipulation.pick_approach_motion_time_scale <= 0.0:
            raise ValueError("pick_approach_motion_time_scale must be positive")
        if self.manipulation.place_move_to_pre_place_motion_time_scale <= 0.0:
            raise ValueError("place_move_to_pre_place_motion_time_scale must be positive")
        if self.manipulation.place_approach_motion_time_scale <= 0.0:
            raise ValueError("place_approach_motion_time_scale must be positive")
        if self.manipulation.place_retreat_motion_time_scale <= 0.0:
            raise ValueError("place_retreat_motion_time_scale must be positive")
        if self.manipulation.arm_post_motion_hold_duration_s <= 0.0:
            raise ValueError("arm_post_motion_hold_duration_s must be positive")
        if self.manipulation.arm_post_motion_joint_error_tolerance < 0.0:
            raise ValueError("arm_post_motion_joint_error_tolerance must be non-negative")
        if self.manipulation.place_release_settle_duration_s < 0.0:
            raise ValueError("place_release_settle_duration_s must be non-negative")
        if self.manipulation.base_lock_settle_steps < 0:
            raise ValueError("base_lock_settle_steps must be non-negative")
        if self.manipulation.base_lock_settle_steps >= self.limits.planning:
            raise ValueError("base_lock_settle_steps must be smaller than planning tick limit")
        if self.manipulation.place_base_lock_settle_steps < 0:
            raise ValueError("place_base_lock_settle_steps must be non-negative")
        if self.manipulation.place_base_lock_settle_steps >= self.limits.planning:
            raise ValueError(
                "place_base_lock_settle_steps must be smaller than planning tick limit"
            )
        if self.manipulation.plan_start_state_failure_threshold < 0.0:
            raise ValueError("plan_start_state_failure_threshold must be non-negative")
        if self.manipulation.pick_return_home_duration_s <= 0.0:
            raise ValueError("pick_return_home_duration_s must be positive")
        if self.manipulation.pick_home_hold_duration_s < 0.0:
            raise ValueError("pick_home_hold_duration_s must be non-negative")
        if self.manipulation.pick_return_home_skip_tolerance < 0.0:
            raise ValueError("pick_return_home_skip_tolerance must be non-negative")
        if self.manipulation.place_return_home_duration_s <= 0.0:
            raise ValueError("place_return_home_duration_s must be positive")
        if self.manipulation.place_return_home_skip_tolerance < 0.0:
            raise ValueError("place_return_home_skip_tolerance must be non-negative")
        if self.manipulation.place_release_clearance_min_m < 0.0:
            raise ValueError("place_release_clearance_min_m must be non-negative")
        if (
            self.manipulation.place_pre_clearance_min_m
            < self.manipulation.place_release_clearance_min_m
        ):
            raise ValueError("place_pre_clearance_min_m must cover release clearance")
        if self.manipulation.carry_home_tracking_tolerance < 0.0:
            raise ValueError("carry_home_tracking_tolerance must be non-negative")
        if self.manipulation.carry_gripper_preload_m < 0.0:
            raise ValueError("carry_gripper_preload_m must be non-negative")
        if self.manipulation.carry_object_tcp_slip_tolerance < 0.0:
            raise ValueError("carry_object_tcp_slip_tolerance must be non-negative")
        if self.manipulation.place_plan_start_transition_duration_s <= 0.0:
            raise ValueError("place_plan_start_transition_duration_s must be positive")
        for name, bounds in (
            ("pick_x_range", self.randomization.pick_x_range),
            ("pick_y_range", self.randomization.pick_y_range),
            ("place_x_range", self.randomization.place_x_range),
            ("place_y_range", self.randomization.place_y_range),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must contain ordered min/max values")
        if self.randomization.clearance_radius < 0.0:
            raise ValueError("randomization clearance_radius must be non-negative")
        if self.randomization.min_boundary_clearance < 0.0:
            raise ValueError("randomization min_boundary_clearance must be non-negative")
        if self.randomization.edge_margin <= 0.0:
            raise ValueError("randomization edge_margin must be positive")
        if self.randomization.edge_min_clearance < 0.0:
            raise ValueError("randomization edge_min_clearance must be non-negative")
        base_goal_randomization = self.randomization.base_goal
        for name, bounds in (
            (
                "base_goal.pick_radius",
                (
                    base_goal_randomization.pick_radius_min_m,
                    base_goal_randomization.pick_radius_max_m,
                ),
            ),
            (
                "base_goal.place_radius",
                (
                    base_goal_randomization.place_radius_min_m,
                    base_goal_randomization.place_radius_max_m,
                ),
            ),
            (
                "base_goal.arm_workspace_radius",
                (
                    base_goal_randomization.arm_workspace_min_xy_radius_m,
                    base_goal_randomization.arm_workspace_max_xy_radius_m,
                ),
            ),
        ):
            if bounds[0] < 0.0 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must contain ordered non-negative min/max values")
        for name, bounds in (
            ("base_goal.place_offset_x", base_goal_randomization.place_offset_x_range_m),
            ("base_goal.place_offset_y", base_goal_randomization.place_offset_y_range_m),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must contain ordered min/max values")
        if base_goal_randomization.place_offset_x_range_m[0] < 0.0:
            raise ValueError("base_goal place_offset_x must be non-negative")
        if base_goal_randomization.place_offset_y_range_m[1] > 0.0:
            raise ValueError("base_goal place_offset_y must be non-positive")
        if base_goal_randomization.max_goal_sample_attempts < 1:
            raise ValueError("base_goal max_goal_sample_attempts must be at least 1")
        if base_goal_randomization.nav_map_min_clearance_m < 0.0:
            raise ValueError("base_goal nav_map_min_clearance_m must be non-negative")
        if base_goal_randomization.seed_offset < 0:
            raise ValueError("base_goal seed_offset must be non-negative")
        if base_goal_randomization.place_workspace_max_xy_radius_m <= 0.0:
            raise ValueError("base_goal place_workspace_max_xy_radius_m must be positive")
        if (
            base_goal_randomization.place_workspace_max_xy_radius_m
            > base_goal_randomization.arm_workspace_max_xy_radius_m
        ):
            raise ValueError("base_goal place_workspace_max_xy_radius_m must not exceed arm workspace max")
        if base_goal_randomization.place_robot_base_max_xy_radius_m <= 0.0:
            raise ValueError("base_goal place_robot_base_max_xy_radius_m must be positive")
        if base_goal_randomization.max_place_target_sample_attempts < 1:
            raise ValueError("base_goal max_place_target_sample_attempts must be at least 1")
        if self.recording.dataset_fps <= 0:
            raise ValueError("recording dataset_fps must be positive")
        if self.recording.image_height <= 0 or self.recording.image_width <= 0:
            raise ValueError("recording image size must be positive")
        if not 1 <= self.recording.jpeg_quality <= 100:
            raise ValueError("recording jpeg_quality must be within [1, 100]")
        if self.recording.chunks_size <= 0:
            raise ValueError("recording chunks_size must be positive")
        if not self.recording.camera_keys:
            raise ValueError("recording camera_keys must not be empty")
        if self.recording.primary_camera_key not in self.recording.camera_keys:
            raise ValueError("recording primary_camera_key must be included in camera_keys")
        if not self.recording.overview_camera_prim_path.startswith("/"):
            raise ValueError("recording overview_camera_prim_path 必须是绝对 prim path")
        if self.lighting.scene_light_mode not in {"camera", "stage"}:
            raise ValueError("scene_light_mode 必须是 camera 或 stage")
        if self.lighting.camera_light_intensity <= 0:
            raise ValueError("camera_light_intensity 必须为正数")
        if self.lighting.camera_light_radius <= 0:
            raise ValueError("camera_light_radius 必须为正数")
        if self.video.enabled and self.video.mode.lower() not in {
            "overview",
            "front",
            "font",
            "wrist",
            "all",
        }:
            raise ValueError("video mode must be one of: overview, front, font, wrist, all")
        if self.video.fps <= 0:
            raise ValueError("video fps must be positive")
        if self.video.width <= 0 or self.video.height <= 0:
            raise ValueError("video image size must be positive")
        if self.video.overview_camera_mode not in {"fixed", "auto"}:
            raise ValueError("overview_camera_mode 必须是 fixed 或 auto")
        if not self.video.overview_camera_prim_path.startswith("/"):
            raise ValueError("video overview_camera_prim_path 必须是绝对 prim path")
        if (
            self.video.overview_camera_prim_path
            != self.recording.overview_camera_prim_path
        ):
            raise ValueError("image/video overview camera prim path 必须一致")
        if self.video.overview_capture_backend not in {"viewport", "render_product", "auto"}:
            raise ValueError("video overview_capture_backend must be one of: viewport, render_product, auto")
        if self.video.min_switch_interval_frames < 0:
            raise ValueError("video min_switch_interval_frames must be non-negative")
        if self.video.overview_initial_hold_frames < 0:
            raise ValueError("video overview_initial_hold_frames must be non-negative")
        if self.video.overview_gamma <= 0:
            raise ValueError("video overview_gamma must be positive")
        enabled_modes = sum(
            int(value)
            for value in (
                self.dry_run,
                self.simulation_smoke,
                self.navigation_smoke,
                self.navigation_carry_smoke,
                self.pick_smoke,
                self.manipulation_smoke,
                self.manipulation_apply_smoke,
                self.full_physics,
            )
        )
        if enabled_modes > 1:
            raise ValueError("execution modes are mutually exclusive")
        if (self.full_physics or self.pick_smoke) and (
            self.pick_plan_json is not None or self.place_plan_json is not None
        ):
            raise ValueError("full_physics/pick_smoke use current-state cuRobo planning; plan JSON fallback is disabled")
        if (self.pick_plan_json is None) != (self.place_plan_json is None):
            raise ValueError("pick_plan_json and place_plan_json must be configured together")

    @property
    def render(self) -> bool:
        return not self.headless or self.randomization.show_debug_region or self.video.enabled

    def episode_seed(self, episode_index: int) -> int:
        return self.seed + episode_index

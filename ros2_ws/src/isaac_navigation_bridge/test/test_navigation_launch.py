"""PCT → SCAN 组合 launch 的静态合同测试。"""

import ast
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ElementTree

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest
import yaml


LAUNCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "pct_scan_navigation.launch.py"
)
PACKAGE_XML_PATH = LAUNCH_PATH.parents[1] / "package.xml"
SUPERVISOR_CONFIG_PATH = (
    LAUNCH_PATH.parents[2]
    / "navigation_supervisor"
    / "config"
    / "navigation_supervisor.yaml"
)
SCAN_PACKAGE_PATH = LAUNCH_PATH.parents[2] / "scan_planner"
SCAN_LAUNCH_PATH = SCAN_PACKAGE_PATH / "launch" / "scan_planner.launch.py"
SCAN_CONFIG_PATH = SCAN_PACKAGE_PATH / "config" / "planner.yaml"
CONTROLLER_CONFIG_PATH = (
    LAUNCH_PATH.parents[2] / "scan_controller" / "config" / "controller.yaml"
)
TUNING_CONFIG_PATH = LAUNCH_PATH.parents[1] / "config" / "pct_scan_tuning.yaml"


def _load_launch_module():
    """从源码路径加载含点号文件名的 launch 模块。"""

    spec = importlib.util.spec_from_file_location(
        "pct_scan_navigation_launch",
        LAUNCH_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_scan_launch_module():
    """从源码路径加载 SCAN 独立 launch。"""

    spec = importlib.util.spec_from_file_location(
        "scan_planner_launch",
        SCAN_LAUNCH_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_argument_default(module, name: str) -> str:
    description = module.generate_launch_description()
    argument = next(
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == name
    )
    return perform_substitutions(LaunchContext(), argument.default_value)


def _launch_tree() -> ast.Module:
    """读取组合 launch 的语法树。"""

    return ast.parse(LAUNCH_PATH.read_text(encoding="utf-8"))


def _flat_parameter_defaults(path: Path) -> dict[str, str]:
    """读取仅含标量的点分 ROS 参数默认值。"""

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", maxsplit=1)[0].strip()
        if not content or ":" not in content:
            continue
        key, value = content.split(":", maxsplit=1)
        if value.strip():
            values[key.strip()] = value.strip()
    return values


def _node_call(
    tree: ast.Module,
    package_name: str,
    *,
    executable_name: str | None = None,
) -> ast.Call:
    """按 package 与可选 executable 找到唯一 Node 调用。"""

    matches = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "Node"
        and any(
            keyword.arg == "package"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == package_name
            for keyword in call.keywords
        )
    ]
    if executable_name is not None:
        matches = [
            call
            for call in matches
            if any(
                keyword.arg == "executable"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == executable_name
                for keyword in call.keywords
            )
        ]
    elif len(matches) > 1:
        # 同一 package 可以合法启动多个 executable；默认选择与 package
        # 同名的主节点，调用方仍可显式选择辅助节点。
        matches = [
            call
            for call in matches
            if any(
                keyword.arg == "executable"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == package_name
                for keyword in call.keywords
            )
        ]
    assert len(matches) == 1
    return matches[0]


def _node_parameter_bindings(node: ast.Call) -> dict[str, str]:
    """返回 Node 参数字典中绑定到 launch 变量的项目。"""

    parameters = next(
        keyword.value
        for keyword in node.keywords
        if keyword.arg == "parameters"
    )
    assert isinstance(parameters, ast.List)
    return {
        key.value: value.id
        for item in parameters.elts
        if isinstance(item, ast.Dict)
        for key, value in zip(item.keys, item.values)
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and isinstance(value, ast.Name)
    }


def _node_parameter_file_bindings(node: ast.Call) -> list[str]:
    """按加载顺序返回 Node 使用的参数文件 launch 变量。"""

    parameters = next(
        keyword.value
        for keyword in node.keywords
        if keyword.arg == "parameters"
    )
    assert isinstance(parameters, ast.List)
    return [
        item.id for item in parameters.elts if isinstance(item, ast.Name)
    ]


def _node_constant_parameters(node: ast.Call) -> dict[str, object]:
    """返回 Node 参数字典中的常量覆盖。"""

    parameters = next(
        keyword.value
        for keyword in node.keywords
        if keyword.arg == "parameters"
    )
    assert isinstance(parameters, ast.List)
    return {
        key.value: value.value
        for item in parameters.elts
        if isinstance(item, ast.Dict)
        for key, value in zip(item.keys, item.values)
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and isinstance(value, ast.Constant)
    }


def _node_remapping_bindings(node: ast.Call) -> dict[str, str]:
    """返回 Node remapping 到 launch 变量的项目。"""

    remappings = next(
        keyword.value
        for keyword in node.keywords
        if keyword.arg == "remappings"
    )
    assert isinstance(remappings, ast.List)
    return {
        item.elts[0].value: item.elts[1].id
        for item in remappings.elts
        if isinstance(item, ast.Tuple)
        and len(item.elts) == 2
        and isinstance(item.elts[0], ast.Constant)
        and isinstance(item.elts[0].value, str)
        and isinstance(item.elts[1], ast.Name)
    }


def test_pct_source_and_scan_input_path_topics_are_explicitly_separated() -> None:
    module = _load_launch_module()

    assert _launch_argument_default(module, "pct_path_topic") == (
        "/pct/global_path"
    )
    assert _launch_argument_default(module, "initial_path_topic") == (
        "/initial_path"
    )
    assert _launch_argument_default(module, "start_pct") == "true"


def test_bridge_can_be_disabled_for_direct_normalized_sensor_inputs() -> None:
    module = _load_launch_module()
    tree = _launch_tree()
    node = _node_call(tree, "isaac_navigation_bridge")
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}

    assert _launch_argument_default(module, "start_bridge") == "true"
    condition = keywords["condition"]
    assert isinstance(condition, ast.Call)
    assert isinstance(condition.func, ast.Name)
    assert condition.func.id == "IfCondition"
    assert len(condition.args) == 1
    assert isinstance(condition.args[0], ast.Name)
    assert condition.args[0].id == "start_bridge"


def test_combined_launch_uses_single_pct_scan_tuning_overlay() -> None:
    assert TUNING_CONFIG_PATH.is_file()

    tree = _launch_tree()
    assignment = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "generate_launch_description"
        for item in item.body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id == "default_tuning_config"
    )
    serialized_default = ast.dump(assignment.value)
    assert "isaac_navigation_bridge" in serialized_default
    assert "pct_scan_tuning.yaml" in serialized_default
    assert _node_parameter_file_bindings(
        _node_call(tree, "pct_ros2_adapter")
    ) == ["pct_config_file", "tuning_config_file"]
    assert _node_parameter_file_bindings(
        _node_call(tree, "scan_planner")
    ) == ["scan_config_file", "tuning_config_file"]
    assert _node_parameter_file_bindings(
        _node_call(tree, "scan_controller")
    ) == ["controller_config_file", "tuning_config_file"]
    assert _node_parameter_file_bindings(
        _node_call(tree, "navigation_supervisor")
    ) == ["supervisor_config_file", "tuning_config_file"]


def test_tuning_overlay_has_consistent_fast_stable_contract() -> None:
    tuning = yaml.safe_load(TUNING_CONFIG_PATH.read_text(encoding="utf-8"))
    scan_base = yaml.safe_load(SCAN_CONFIG_PATH.read_text(encoding="utf-8"))[
        "scan_planner_node"
    ]["ros__parameters"]
    assert set(tuning) == {
        "navigation_contract",
        "isaac_navigation_runtime",
        "pct_ros2_adapter",
        "scan_planner_node",
        "scan_controller",
        "navigation_supervisor",
    }
    navigation_contract = tuning["navigation_contract"]["ros__parameters"]
    isaac_runtime = tuning["isaac_navigation_runtime"]["ros__parameters"]
    pct = tuning["pct_ros2_adapter"]["ros__parameters"]
    scan = tuning["scan_planner_node"]["ros__parameters"]
    controller = tuning["scan_controller"]["ros__parameters"]
    supervisor = tuning["navigation_supervisor"]["ros__parameters"]

    assert navigation_contract["body_height_m"] == pytest.approx(0.338)
    assert isaac_runtime["point_cloud.pixel_stride"] >= 1
    assert isaac_runtime["point_cloud.publish_interval_control_steps"] >= 1
    assert isaac_runtime["point_cloud.minimum_valid_points"] <= (
        isaac_runtime["point_cloud.max_points"]
    )

    assert 0.0 < pct["path.minimum_point_spacing_m"] <= (
        pct["planner.path_sample_spacing_m"]
    )
    assert pct["planner.path_sample_spacing_m"] <= (
        pct["planner.grid_compress_max_segment_m"]
    )
    assert pct["planner.upstream_body_clearance_enabled"] is True
    assert 0.0 < pct["planner.upstream_astar_step_cost_weight"] <= 0.20
    assert pct["planner.upstream_body_clearance_radius_m"] > (
        pct["planner.upstream_same_layer_shortcut_clearance_m"]
    )
    assert 0.0 < pct["planner.upstream_body_clearance_maximum_cost"] <= 20.0
    assert pct["planner.upstream_body_clearance_power"] > 0.0
    assert 0.0 < scan["fsm.thresh_no_replan"] < (
        scan["fsm.thresh_replan"]
    ) < scan["fsm.planning_horizon"]
    assert 0.0 < scan["fsm.reference_cruise_speed"] <= (
        scan["manager.max_vel"]
    )
    assert 0.0 < scan["fsm.reference_goal_hold_distance_xy"] < (
        controller["finish.capture_entry_distance_xy"]
    )
    assert scan["fsm.reference_goal_hold_distance_xy"] == pytest.approx(0.04)
    assert scan["fsm.reference_goal_hold_stable_dwell_sec"] > (
        controller["finish.capture_stable_dwell_sec"]
    )
    assert scan["fsm.reference_goal_hold_yaw_rate"] == pytest.approx(
        controller["finish.max_yaw_rate"]
    )
    assert "fsm.reference_goal_hold_angular_speed" not in scan
    assert scan["manager.planning_horizon"] >= (
        2.0 * scan["fsm.planning_horizon"]
    )
    assert scan["manager.max_vel"] == scan["optimization.max_vel"]
    assert scan["manager.max_acc"] == scan["optimization.max_acc"]
    assert 0.0 < scan["manager.reference_profile_acceleration_scale"] <= 1.0
    assert scan["manager.reference_free_guide_refine_enabled"] is True
    assert (
        scan["manager.reference_free_guide_refine_minimum_duration_gain"]
        >= 0.05
    )
    feasibility_tolerance = scan["manager.feasibility_tolerance"]
    velocity_tolerance = scan.get(
        "optimization.vel_tolerance",
        scan_base["optimization.vel_tolerance"],
    )
    acceleration_tolerance = scan.get(
        "optimization.acc_tolerance",
        scan_base["optimization.acc_tolerance"],
    )
    assert (
        scan["manager.max_vel"] * feasibility_tolerance + 1.0e-4
        <= velocity_tolerance
    )
    assert (
        scan["manager.max_acc"] * feasibility_tolerance + 1.0e-4
        <= acceleration_tolerance
    )
    assert scan["manager.max_vel"] == controller["limits.max_vx"]
    assert scan["fsm.reference_cruise_speed"] <= (
        controller["limits.max_vx"]
    )
    assert 0.0 < controller["controller.cross_track_alignment_release_distance"] < (
        controller["controller.cross_track_alignment_distance"]
    )
    assert 0.0 < controller[
        "controller.cross_track_heading_error_release_threshold"
    ] < controller["controller.cross_track_heading_error_threshold"]
    assert controller["controller.cross_track_heading_error_threshold"] <= (
        controller["controller.heading_error_threshold"]
    )
    assert 0.0 < controller["controller.cross_track_recovery_forward_speed"] <= (
        controller["limits.max_vx"]
    )
    assert 0.0 < controller["controller.cross_track_recovery_lateral_speed"] <= (
        controller["limits.max_vy"]
    )
    assert controller["controller.cross_track_recovery_taper_distance"] > 0.0
    assert controller["controller.cross_track_recovery_lateral_gain"] > 0.0
    assert controller["controller.cross_track_heading_assist_gain"] >= 0.0
    assert 0.0 <= controller["controller.cross_track_heading_assist_max"] < (
        controller["controller.cross_track_heading_error_threshold"]
    )
    assert controller["controller.turning_speed_limit_enabled"] is True
    assert 0.0 < controller["controller.turning_yaw_rate_threshold"] <= (
        controller["limits.max_yaw_rate"]
    )
    assert 0.0 < controller["controller.turning_max_planar_speed"] < (
        controller["limits.max_vx"]
    )
    assert controller["controller.yaw_alignment_min_chord_distance"] > 0.0
    assert controller["finish.capture_entry_distance_xy"] < (
        controller["finish.capture_zero_hold_distance_xy"]
    ) < controller["finish.distance_xy"] < (
        controller["finish.capture_release_distance_xy"]
    )
    assert controller["finish.capture_entry_distance_xy"] == pytest.approx(0.055)
    assert controller["finish.capture_zero_hold_distance_xy"] == pytest.approx(
        0.075
    )
    assert (
        controller["finish.distance_xy"]
        - controller["finish.capture_entry_distance_xy"]
    ) >= 0.02
    assert controller["finish.capture_release_distance_xy"] == 0.30
    assert 0.0 < controller["finish.max_position_hold_speed"] <= min(
        controller["limits.max_vx"], controller["limits.max_vy"]
    )
    assert controller["finish.position_hold_gain"] == 2.0
    assert controller["finish.max_position_hold_speed"] == 0.15
    assert 0.0 < controller["finish.min_approach_speed"] <= math.hypot(
        controller["limits.max_vx"],
        controller["limits.max_vy"],
    )
    assert controller["finish.min_approach_speed"] == 0.28
    assert controller["timeouts.max_yaw_alignment_freeze_sec"] == 12.0
    assert supervisor["timeouts.max_yaw_alignment_freeze_sec"] == (
        controller["timeouts.max_yaw_alignment_freeze_sec"]
    )
    assert controller["finish.max_capture_yaw_rate"] <= (
        controller["limits.max_yaw_rate"]
    )
    assert controller["finish.max_capture_yaw_acc"] <= (
        controller["limits.max_yaw_acc"]
    )


@pytest.mark.parametrize(
    ("argument_name", "expected"),
    (
        ("body_pose_topic", "/body_pose"),
        ("cloud_topic", "/cloud_registered"),
        ("pct_path_topic", "/pct/global_path"),
        ("initial_path_topic", "/initial_path"),
        ("pct_status_topic", "/pct/planning_status"),
        ("pct_command_service", "/pct/planning_command"),
        ("bspline_topic", "/planning/bspline"),
        ("scan_status_topic", "/planning/scan_status"),
        ("controller_status_topic", "/planning/controller_status"),
        (
            "grid_map_observation_diagnostics_topic",
            "/planning/grid_map_observation_diagnostics",
        ),
        ("bspline_diagnostics_topic", "/planning/bspline_diagnostics"),
        ("navigation_status_topic", "/navigation/status"),
    ),
)
def test_supervisor_graph_names_have_mainline_defaults(
    argument_name: str,
    expected: str,
) -> None:
    module = _load_launch_module()

    assert _launch_argument_default(module, argument_name) == expected


def test_supervisor_is_enabled_by_default_for_mainline() -> None:
    module = _load_launch_module()

    assert _launch_argument_default(module, "start_supervisor") == "true"


def test_supervisor_config_matches_mainline_graph_and_heartbeat() -> None:
    module = _load_launch_module()
    defaults = _flat_parameter_defaults(SUPERVISOR_CONFIG_PATH)
    controller_defaults = _flat_parameter_defaults(CONTROLLER_CONFIG_PATH)

    assert defaults["use_sim_time"] == "true"
    for parameter_name, argument_name in (
        ("topics.odometry", "body_pose_topic"),
        ("topics.point_cloud", "cloud_topic"),
        ("topics.global_path", "initial_path_topic"),
        ("topics.pct_status", "pct_status_topic"),
        ("topics.bspline", "bspline_topic"),
        ("topics.scan_status", "scan_status_topic"),
        ("topics.controller_status", "controller_status_topic"),
        ("topics.navigation_status", "navigation_status_topic"),
        ("topics.pct_command_service", "pct_command_service"),
    ):
        assert defaults[parameter_name] == _launch_argument_default(
            module,
            argument_name,
        )

    assert float(defaults["status.heartbeat_sec"]) == pytest.approx(0.10)
    for parameter_name in (
        "timeouts.trajectory_expiry_grace_sec",
        "timeouts.max_yaw_alignment_freeze_sec",
    ):
        assert float(defaults[parameter_name]) == pytest.approx(
            float(controller_defaults[parameter_name])
        )


def test_stair_execution_freeze_topic_has_mainline_default() -> None:
    module = _load_launch_module()

    assert _launch_argument_default(
        module,
        "stair_execution_frozen_topic",
    ) == "/planning/stair_execution_frozen"
    assert float(
        _launch_argument_default(
            module,
            "stair_execution_freeze_timeout_sec",
        )
    ) == pytest.approx(0.25)
    assert float(
        _launch_argument_default(
            module,
            "stair_execution_freeze_confirmation_sec",
        )
    ) == pytest.approx(0.05)

    tree = _launch_tree()
    scan_node = _node_call(tree, "scan_planner")
    parameters = next(
        keyword.value
        for keyword in scan_node.keywords
        if keyword.arg == "parameters"
    )
    assert isinstance(parameters, ast.List)
    bindings = {
        key.value: value.id
        for item in parameters.elts
        if isinstance(item, ast.Dict)
        for key, value in zip(item.keys, item.values)
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and isinstance(value, ast.Name)
    }
    assert bindings["topics.stair_execution_frozen"] == (
        "stair_execution_frozen_topic"
    )
    assert bindings["topics.controller_status"] == "controller_status_topic"
    assert bindings["fsm.stair_execution_freeze_timeout_sec"] == (
        "stair_execution_freeze_timeout_sec"
    )
    assert bindings["fsm.stair_execution_freeze_confirmation_sec"] == (
        "stair_execution_freeze_confirmation_sec"
    )


def test_stair_freeze_timing_is_bound_in_scan_launch_and_config() -> None:
    module = _load_scan_launch_module()

    assert float(
        _launch_argument_default(
            module,
            "stair_execution_freeze_timeout_sec",
        )
    ) == pytest.approx(0.25)
    assert float(
        _launch_argument_default(
            module,
            "stair_execution_freeze_confirmation_sec",
        )
    ) == pytest.approx(0.05)

    tree = ast.parse(SCAN_LAUNCH_PATH.read_text(encoding="utf-8"))
    scan_node = _node_call(tree, "scan_planner")
    bindings = _node_parameter_bindings(scan_node)
    assert bindings["fsm.stair_execution_freeze_timeout_sec"] == (
        "stair_execution_freeze_timeout_sec"
    )
    assert bindings["fsm.stair_execution_freeze_confirmation_sec"] == (
        "stair_execution_freeze_confirmation_sec"
    )
    assert _launch_argument_default(
        module,
        "controller_status_topic",
    ) == "/planning/controller_status"
    assert bindings["topics.controller_status"] == "controller_status_topic"

    defaults = _flat_parameter_defaults(SCAN_CONFIG_PATH)
    assert float(
        defaults["fsm.stair_execution_freeze_timeout_sec"]
    ) == pytest.approx(0.25)
    assert float(
        defaults["fsm.stair_execution_freeze_confirmation_sec"]
    ) == pytest.approx(0.05)
    assert defaults["topics.controller_status"] == (
        "/planning/controller_status"
    )


@pytest.mark.parametrize(
    "invalid_value",
    ("0", "-0.1", "nan", "inf", "-inf", "not-a-number"),
)
@pytest.mark.parametrize(
    "argument_name",
    (
        "stair_execution_freeze_timeout_sec",
        "stair_execution_freeze_confirmation_sec",
    ),
)
def test_stair_freeze_timing_rejects_non_positive_or_non_finite_values(
    argument_name: str,
    invalid_value: str,
) -> None:
    for module in (_load_launch_module(), _load_scan_launch_module()):
        context = LaunchContext()
        context.launch_configurations[
            "stair_execution_freeze_timeout_sec"
        ] = "0.25"
        context.launch_configurations[
            "stair_execution_freeze_confirmation_sec"
        ] = "0.05"
        context.launch_configurations[argument_name] = invalid_value
        with pytest.raises(RuntimeError, match="必须是有限正数"):
            module._validate_stair_freeze_timing(context)


def test_stair_freeze_timing_accepts_finite_positive_values() -> None:
    for module in (_load_launch_module(), _load_scan_launch_module()):
        context = LaunchContext()
        context.launch_configurations[
            "stair_execution_freeze_timeout_sec"
        ] = "0.25"
        context.launch_configurations[
            "stair_execution_freeze_confirmation_sec"
        ] = "0.05"
        assert module._validate_stair_freeze_timing(context) == []


def test_frame_arguments_keep_canonical_defaults() -> None:
    module = _load_launch_module()

    assert _launch_argument_default(module, "world_frame") == "world"
    assert _launch_argument_default(module, "base_frame") == "base_link"


def test_pct_and_manual_path_publishers_are_mutually_exclusive() -> None:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["start_pct"] = "true"
    context.launch_configurations["start_manual_path"] = "true"

    with pytest.raises(RuntimeError, match="不能同时为 true"):
        module._validate_path_source(context)


@pytest.mark.parametrize(
    ("start_pct", "start_manual_path"),
    (("true", "false"), ("false", "true"), ("false", "false")),
)
def test_single_or_external_path_source_is_allowed(
    start_pct: str,
    start_manual_path: str,
) -> None:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["start_pct"] = start_pct
    context.launch_configurations["start_manual_path"] = start_manual_path

    assert module._validate_path_source(context) == []


@pytest.mark.parametrize("value", ("0", "-0.1", "nan", "not-a-number"))
def test_unified_body_height_rejects_invalid_values(value: str) -> None:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["body_height_m"] = value

    with pytest.raises(RuntimeError, match="有限正数"):
        module._validate_body_height(context)


def test_unified_body_height_accepts_positive_value() -> None:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["body_height_m"] = "0.30"

    assert module._validate_body_height(context) == []


def test_unified_body_height_overrides_all_height_parameters() -> None:
    tree = _launch_tree()
    wired_parameters = {
        key.value
        for mapping in ast.walk(tree)
        if isinstance(mapping, ast.Dict)
        for key, value in zip(mapping.keys, mapping.values)
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and isinstance(value, ast.Name)
        and value.id == "body_height_parameter"
    }

    assert wired_parameters == {
        "planner.goal_base_to_ground_m",
        "planner.slice_query_root_to_floor_m",
        "filters.body_height_m",
        "grid_map.body_height",
        "reference_path.body_height_m",
    }


@pytest.mark.parametrize(
    ("world_frame", "base_frame"),
    (
        ("", "base_link"),
        ("/world", "base_link"),
        ("world ", "base_link"),
        ("map//world", "base_link"),
        ("map/../world", "base_link"),
        ("world", "base link"),
    ),
)
def test_frame_arguments_reject_noncanonical_values(
    world_frame: str,
    base_frame: str,
) -> None:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["world_frame"] = world_frame
    context.launch_configurations["base_frame"] = base_frame

    with pytest.raises(RuntimeError, match="frame_id"):
        module._validate_frame_arguments(context)


def test_frame_arguments_must_be_distinct() -> None:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["world_frame"] = "map"
    context.launch_configurations["base_frame"] = "map"

    with pytest.raises(RuntimeError, match="不能相同"):
        module._validate_frame_arguments(context)


def test_frame_arguments_accept_canonical_hierarchy() -> None:
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["world_frame"] = "map"
    context.launch_configurations["base_frame"] = "robot/base_link"

    assert module._validate_frame_arguments(context) == []


def test_frame_arguments_override_every_online_consumer() -> None:
    tree = _launch_tree()
    bindings: dict[str, dict[str, str]] = {}
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "Node"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        package = keywords.get("package")
        parameters = keywords.get("parameters")
        if not (
            isinstance(package, ast.Constant)
            and isinstance(package.value, str)
            and isinstance(parameters, ast.List)
        ):
            continue
        node_bindings: dict[str, str] = {}
        for item in parameters.elts:
            if not isinstance(item, ast.Dict):
                continue
            for key, value in zip(item.keys, item.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(value, ast.Name)
                ):
                    node_bindings[key.value] = value.id
        package_bindings = bindings.setdefault(package.value, {})
        for parameter_name, launch_variable in node_bindings.items():
            previous = package_bindings.get(parameter_name)
            assert previous in {None, launch_variable}
            package_bindings[parameter_name] = launch_variable

    assert {
        package: {
            key
            for key, value in node_bindings.items()
            if value == "world_frame_parameter"
        }
        for package, node_bindings in bindings.items()
    } == {
        "isaac_navigation_bridge": {
            "frames.odom",
            "frames.cloud",
            "frames.world",
        },
        "pct_ros2_adapter": {"frames.world"},
        "scan_planner": {"grid_map.frame_id"},
        "scan_navigation_tools": {"frame_id"},
        "scan_controller": {"frames.world"},
        "navigation_supervisor": {"frames.world"},
    }
    assert {
        package: {
            key
            for key, value in node_bindings.items()
            if value == "base_frame_parameter"
        }
        for package, node_bindings in bindings.items()
    } == {
        "isaac_navigation_bridge": {"frames.base"},
        "pct_ros2_adapter": {"frames.base"},
        "scan_planner": {"grid_map.base_frame_id"},
        "scan_navigation_tools": set(),
        "scan_controller": {"frames.base"},
        "navigation_supervisor": {"frames.base"},
    }


def test_frame_overrides_force_ros_parameters_to_string_type() -> None:
    tree = _launch_tree()
    parameter_values: dict[str, ast.Call] = {}
    for assignment in ast.walk(tree):
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ast.Name)
            and assignment.value.func.id == "ParameterValue"
        ):
            continue
        parameter_values[assignment.targets[0].id] = assignment.value

    for parameter_name, launch_name in (
        ("world_frame_parameter", "world_frame"),
        ("base_frame_parameter", "base_frame"),
    ):
        call = parameter_values[parameter_name]
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == launch_name
        value_type = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "value_type"
        )
        assert isinstance(value_type, ast.Name)
        assert value_type.id == "str"


def test_supervisor_node_uses_expected_executable_config_and_condition(
) -> None:
    tree = _launch_tree()
    node = _node_call(tree, "navigation_supervisor")
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}

    for keyword_name in ("executable", "name"):
        value = keywords[keyword_name]
        assert isinstance(value, ast.Constant)
        assert value.value == "navigation_supervisor"

    condition = keywords["condition"]
    assert isinstance(condition, ast.Call)
    assert isinstance(condition.func, ast.Name)
    assert condition.func.id == "IfCondition"
    assert len(condition.args) == 1
    assert isinstance(condition.args[0], ast.Name)
    assert condition.args[0].id == "start_supervisor"

    parameters = keywords["parameters"]
    assert isinstance(parameters, ast.List)
    assert isinstance(parameters.elts[0], ast.Name)
    assert parameters.elts[0].id == "supervisor_config_file"


def test_supervisor_config_default_comes_from_supervisor_package() -> None:
    tree = _launch_tree()
    assignment = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "generate_launch_description"
        for item in item.body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id == "default_supervisor_config"
    )
    assert isinstance(assignment.value, ast.Call)
    serialized = ast.dump(assignment.value)
    assert "navigation_supervisor" in serialized
    assert "navigation_supervisor.yaml" in serialized


def test_supervisor_consumers_and_producers_share_launch_bindings() -> None:
    tree = _launch_tree()
    bridge = _node_parameter_bindings(
        _node_call(tree, "isaac_navigation_bridge")
    )
    odometry_tf = _node_parameter_bindings(
        _node_call(
            tree,
            "isaac_navigation_bridge",
            executable_name="odometry_tf_broadcaster",
        )
    )
    pct = _node_parameter_bindings(_node_call(tree, "pct_ros2_adapter"))
    scan = _node_parameter_bindings(_node_call(tree, "scan_planner"))
    scan_remappings = _node_remapping_bindings(
        _node_call(tree, "scan_planner")
    )
    controller = _node_parameter_bindings(_node_call(tree, "scan_controller"))
    supervisor = _node_parameter_bindings(
        _node_call(tree, "navigation_supervisor")
    )

    assert supervisor == {
        "frames.world": "world_frame_parameter",
        "frames.base": "base_frame_parameter",
        "topics.odometry": "body_pose_topic",
        "topics.point_cloud": "cloud_topic",
        "topics.global_path": "initial_path_topic",
        "topics.pct_status": "pct_status_topic",
        "topics.bspline": "bspline_topic",
        "topics.scan_status": "scan_status_topic",
        "topics.controller_status": "controller_status_topic",
        "topics.navigation_status": "navigation_status_topic",
        "topics.pct_command_service": "pct_command_service",
    }
    assert bridge["topics.body_pose_output"] == supervisor["topics.odometry"]
    assert bridge["topics.cloud_output"] == supervisor["topics.point_cloud"]
    assert bridge["topics.initial_path_input"] == supervisor[
        "topics.global_path"
    ]
    assert odometry_tf == {
        "topics.body_pose": supervisor["topics.odometry"],
        "frames.world": "world_frame_parameter",
        "frames.base": "base_frame_parameter",
    }
    assert pct["topics.odometry_input"] == supervisor["topics.odometry"]
    assert pct["topics.path_output"] == "pct_path_topic"
    assert pct["topics.scan_path_output"] == supervisor["topics.global_path"]
    assert pct["topics.status_output"] == supervisor["topics.pct_status"]
    assert pct["topics.command_service"] == supervisor[
        "topics.pct_command_service"
    ]
    assert pct["planner.backend_kind"] == "pct_backend_kind_parameter"
    assert scan["topics.planning_status"] == supervisor["topics.scan_status"]
    assert scan["topics.controller_status"] == supervisor[
        "topics.controller_status"
    ]
    assert scan["topics.grid_map_observation_diagnostics"] == (
        "grid_map_observation_diagnostics_topic"
    )
    assert scan["topics.bspline_diagnostics"] == (
        "bspline_diagnostics_topic"
    )
    assert scan_remappings == {
        "body_pose": supervisor["topics.odometry"],
        "sensor_pose": supervisor["topics.odometry"],
        "cloud": supervisor["topics.point_cloud"],
        "initial_path": supervisor["topics.global_path"],
        "planning/bspline": supervisor["topics.bspline"],
    }
    assert controller["topics.bspline"] == supervisor["topics.bspline"]
    assert controller["topics.initial_path"] == supervisor[
        "topics.global_path"
    ]
    assert controller["topics.body_pose"] == supervisor["topics.odometry"]
    assert controller["topics.cloud"] == supervisor["topics.point_cloud"]
    assert controller["topics.controller_status"] == supervisor[
        "topics.controller_status"
    ]


def test_controller_is_the_only_cmd_vel_publisher_in_combined_launch(
) -> None:
    tree = _launch_tree()
    bindings = {
        package: _node_parameter_bindings(_node_call(tree, package))
        for package in (
            "isaac_navigation_bridge",
            "pct_ros2_adapter",
            "scan_planner",
            "scan_navigation_tools",
            "scan_controller",
            "navigation_supervisor",
        )
    }

    assert {
        package: parameters["topics.cmd_vel"]
        for package, parameters in bindings.items()
        if "topics.cmd_vel" in parameters
    } == {"scan_controller": "cmd_vel_topic"}


def test_every_combined_launch_node_forces_sim_time() -> None:
    tree = _launch_tree()
    node_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "Node"
    ]
    packages = {
        keyword.value.value
        for call in node_calls
        for keyword in call.keywords
        if keyword.arg == "package"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    }

    assert packages == {
        "isaac_navigation_bridge",
        "pct_ros2_adapter",
        "scan_planner",
        "scan_navigation_tools",
        "scan_controller",
        "navigation_supervisor",
    }
    assert len(node_calls) == 7
    for node_call in node_calls:
        parameters = _node_constant_parameters(node_call)
        assert parameters["use_sim_time"] is True


def test_package_declares_supervisor_runtime_dependency() -> None:
    root = ElementTree.parse(PACKAGE_XML_PATH).getroot()

    dependencies = {
        element.text for element in root.findall("exec_depend")
    }
    assert "navigation_supervisor" in dependencies

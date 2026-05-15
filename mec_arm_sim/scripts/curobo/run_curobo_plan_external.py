"""
独立 Python 子进程中的 cuRobo 规划器。

这个脚本不导入 omni / Isaac Sim / pxr，目的是避开 Isaac Sim 进程内
omni.warp 1.8.2 与 cuRobo 所需 warp_lang 1.13.0 的冲突。

输入：
    python run_curobo_plan_external.py request.json response.json

request.json 主要字段：
{
    "robot_config_path": ".../mec_arm.yml",
    "joint_names": ["Joint1", ...],
    "tool_frame": "TCP_link",
    "q_current": [...],
    "current_tcp_base": {"position": [...], "quaternion_wxyz": [...]},
    "target_tcp_base": {"position": [...], "quaternion_wxyz": [...]},
    "scene_model": {"cuboid": {...}, "mesh": {}},
    "params": {...}
}

response.json 会包含 raw trajectory、S 曲线轨迹、TCP base-frame path、
终点误差、关节限位检查和数值 Jacobian 条件数检查。
"""

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np


CUROBO_SOURCE_ROOT = Path("/home/light/workspace/curobo")
ISAAC_CONDA_SITE_PACKAGES = Path(
    "/data/conda_envs/isaacsim51_3dgs_grasp/lib/python3.11/site-packages"
)


def _move_path_to_front(path):
    path = str(path)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


if ISAAC_CONDA_SITE_PACKAGES.exists():
    _move_path_to_front(ISAAC_CONDA_SITE_PACKAGES)
if CUROBO_SOURCE_ROOT.exists():
    _move_path_to_front(CUROBO_SOURCE_ROOT)

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState


DEVICE = "cuda:0"
DTYPE = torch.float32

DEFAULT_PARAMS = {
    "num_ik_seeds": 32,
    "num_trajopt_seeds": 4,
    "max_planning_attempts": 5,
    "use_cuda_graph": False,
    "run_planner_warmup": False,
    "s_curve_dt": 0.02,
    "s_curve_min_duration": 1.5,
    "max_joint_vel_rad_s": [1.2, 1.2, 1.2, 1.5, 1.5, 1.8],
    "max_joint_acc_rad_s2": [3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
    "joint_limits": {
        "Joint1": [-3.10, 3.10],
        "Joint2": [-2.60, 0.00],
        "Joint3": [0.00, 4.00],
        "Joint4": [-3.10, 3.10],
        "Joint5": [0.00, 3.10],
        "Joint6": [-1.57, 1.57],
    },
    "joint_limit_tol": 1e-3,
    "singularity_condition_warn": 200.0,
    "singularity_max_samples": 40,
    "numeric_jacobian_eps": 1e-4,
    "self_collision_check": True,
    "allow_ik_fallback": True,
    "ik_fallback_position_tol": 0.02,
    "ik_fallback_orientation_tol": 0.05,
    "ik_fallback_waypoints": 32,
}


def normalize_quat_wxyz(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError(f"四元数范数太小：{quat_wxyz}")
    return quat / norm


def quat_conjugate_wxyz(quat):
    quat = normalize_quat_wxyz(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = normalize_quat_wxyz(q1)
    w2, x2, y2, z2 = normalize_quat_wxyz(q2)
    return normalize_quat_wxyz(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_angle_error_deg(q_a, q_b):
    q_a = normalize_quat_wxyz(q_a)
    q_b = normalize_quat_wxyz(q_b)
    dot = float(abs(np.dot(q_a, q_b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def quat_to_rotvec_wxyz(quat):
    quat = normalize_quat_wxyz(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:4]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-10:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, quat[0])
    return vector / vector_norm * angle


def make_goal_tool_pose(position_base, quaternion_base, tool_frames):
    position_tensor = torch.tensor(
        [[[[np.asarray(position_base, dtype=float).tolist()]]]],
        device=DEVICE,
        dtype=DTYPE,
    )
    quaternion_tensor = torch.tensor(
        [[[[normalize_quat_wxyz(quaternion_base).tolist()]]]],
        device=DEVICE,
        dtype=DTYPE,
    )
    return GoalToolPose(
        tool_frames=list(tool_frames),
        position=position_tensor,
        quaternion=quaternion_tensor,
    )


def make_joint_state(q_current, joint_names):
    q_tensor = torch.as_tensor(q_current, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    return JointState.from_position(q_tensor, joint_names=list(joint_names))


def extract_joint_trajectory(result):
    interpolated = result.get_interpolated_plan()
    if interpolated is None or interpolated.position is None:
        raise RuntimeError("cuRobo 结果中没有 interpolated trajectory。")

    q = interpolated.position.detach()
    if q.ndim == 3:
        q = q[0]
    elif q.ndim != 2:
        raise RuntimeError(f"未知轨迹 shape：{tuple(q.shape)}")
    return q.cpu().numpy().astype(float)


def smootherstep(u):
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def s_curve_resample_joint_trajectory(q_raw, params):
    q_raw = np.asarray(q_raw, dtype=float)
    if q_raw.ndim != 2 or q_raw.shape[0] < 2:
        raise ValueError(f"q_raw 必须是 [T, dof] 且 T>=2，当前 shape={q_raw.shape}")

    dt = float(params["s_curve_dt"])
    min_duration = float(params["s_curve_min_duration"])
    max_joint_vel = np.asarray(params["max_joint_vel_rad_s"], dtype=float)
    max_joint_acc = np.asarray(params["max_joint_acc_rad_s2"], dtype=float)

    segment_lengths = np.linalg.norm(np.diff(q_raw, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    path_length = float(cumulative[-1])
    if path_length < 1e-9:
        raise ValueError("cuRobo 返回轨迹几乎没有运动，无法重采样。")

    joint_delta = np.max(np.abs(np.diff(q_raw, axis=0)), axis=0)
    total_joint_delta = np.abs(q_raw[-1] - q_raw[0])
    duration_from_vel = float(np.max(1.875 * total_joint_delta / max_joint_vel))
    duration_from_acc = float(
        np.max(np.sqrt(5.8 * np.maximum(total_joint_delta, joint_delta) / max_joint_acc))
    )
    duration = max(min_duration, duration_from_vel, duration_from_acc)

    sample_count = max(2, int(math.ceil(duration / dt)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    target_s = path_length * smootherstep(times / duration)

    q_resampled = np.empty((sample_count, q_raw.shape[1]), dtype=float)
    for joint_index in range(q_raw.shape[1]):
        q_resampled[:, joint_index] = np.interp(target_s, cumulative, q_raw[:, joint_index])

    qd = np.gradient(q_resampled, times, axis=0, edge_order=1)
    qdd = np.gradient(qd, times, axis=0, edge_order=1)

    return {
        "q": q_resampled,
        "t": times,
        "qd": qd,
        "qdd": qdd,
        "duration": duration,
    }


def check_joint_limits(q_trajectory, joint_names, params):
    q_trajectory = np.asarray(q_trajectory, dtype=float)
    limits = params["joint_limits"]
    tol = float(params["joint_limit_tol"])
    violations = []
    for joint_index, joint_name in enumerate(joint_names):
        lower, upper = limits[joint_name]
        values = q_trajectory[:, joint_index]
        bad_indices = np.where((values < lower - tol) | (values > upper + tol))[0]
        if bad_indices.size:
            violations.append(
                {
                    "joint": joint_name,
                    "lower": float(lower),
                    "upper": float(upper),
                    "first_bad_index": int(bad_indices[0]),
                    "first_bad_value": float(values[bad_indices[0]]),
                }
            )
    return violations


def fk_tcp_poses_base(planner, q_trajectory):
    q_trajectory = np.asarray(q_trajectory, dtype=float)
    q_tensor = torch.as_tensor(q_trajectory, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    joint_state = JointState.from_position(q_tensor, joint_names=list(planner.joint_names))
    kin_state = planner.compute_kinematics(joint_state)

    positions = kin_state.tool_poses.position.detach().cpu().numpy()[0, :, 0, :]
    quaternions = kin_state.tool_poses.quaternion.detach().cpu().numpy()[0, :, 0, :]
    quaternions = np.asarray([normalize_quat_wxyz(q) for q in quaternions], dtype=float)
    return positions, quaternions


def fk_single_tcp_pose_base(planner, q):
    positions, quaternions = fk_tcp_poses_base(planner, np.asarray(q, dtype=float)[None, :])
    return positions[0], quaternions[0]


def numerical_tcp_jacobian(planner, q, params):
    eps = float(params["numeric_jacobian_eps"])
    q = np.asarray(q, dtype=float)
    pos_0, quat_0 = fk_single_tcp_pose_base(planner, q)
    jacobian = np.zeros((6, q.shape[0]), dtype=float)

    for joint_index in range(q.shape[0]):
        q_eps = q.copy()
        q_eps[joint_index] += eps
        pos_eps, quat_eps = fk_single_tcp_pose_base(planner, q_eps)
        jacobian[:3, joint_index] = (pos_eps - pos_0) / eps
        delta_quat = quat_multiply_wxyz(quat_eps, quat_conjugate_wxyz(quat_0))
        jacobian[3:, joint_index] = quat_to_rotvec_wxyz(delta_quat) / eps

    return jacobian


def check_singularity_by_numeric_jacobian(planner, q_trajectory, params):
    q_trajectory = np.asarray(q_trajectory, dtype=float)
    sample_count = min(int(params["singularity_max_samples"]), q_trajectory.shape[0])
    sample_indices = np.unique(
        np.linspace(0, q_trajectory.shape[0] - 1, sample_count).round().astype(int)
    )

    records = []
    for index in sample_indices:
        jacobian = numerical_tcp_jacobian(planner, q_trajectory[index], params)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        smallest = float(np.min(singular_values))
        largest = float(np.max(singular_values))
        condition = float("inf") if smallest < 1e-9 else largest / smallest
        records.append(
            {
                "index": int(index),
                "condition": condition,
                "sigma_min": smallest,
                "sigma_max": largest,
            }
        )

    warn_threshold = float(params["singularity_condition_warn"])
    max_record = max(records, key=lambda item: item["condition"])
    risky = [item for item in records if item["condition"] > warn_threshold]
    return {
        "records": records,
        "max_condition": max_record["condition"],
        "max_record": max_record,
        "risky": risky,
    }


def _tensor_summary(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        summary = {
            "shape": list(tensor.shape),
        }
        if tensor.numel() > 0 and tensor.dtype != torch.bool:
            finite = tensor[torch.isfinite(tensor)]
            if finite.numel() > 0:
                summary["min"] = float(torch.min(finite).item())
                summary["max"] = float(torch.max(finite).item())
                summary["mean"] = float(torch.mean(finite.float()).item())
        if tensor.numel() > 0 and tensor.dtype == torch.bool:
            summary["true_count"] = int(torch.count_nonzero(tensor).item())
        return summary
    return str(value)


def diagnose_ik(planner, goal_pose, current_state, params):
    try:
        seed_only_result = planner.ik_solver.solve_pose(
            goal_pose,
            return_seeds=1,
            current_state=current_state,
            seed_config=current_state.position.clone().view(1, 1, -1),
            run_optimizer=False,
        )
        ik_result = planner.ik_solver.solve_pose(
            goal_pose,
            return_seeds=int(params["num_trajopt_seeds"]),
            current_state=current_state,
        )
        return {
            "available": True,
            "seed_only_success": _tensor_summary(seed_only_result.success),
            "seed_only_feasible": _tensor_summary(getattr(seed_only_result, "feasible", None)),
            "seed_only_position_error": _tensor_summary(
                getattr(seed_only_result, "position_error", None)
            ),
            "seed_only_rotation_error": _tensor_summary(
                getattr(seed_only_result, "rotation_error", None)
            ),
            "success": _tensor_summary(ik_result.success),
            "feasible": _tensor_summary(getattr(ik_result, "feasible", None)),
            "position_error": _tensor_summary(getattr(ik_result, "position_error", None)),
            "rotation_error": _tensor_summary(getattr(ik_result, "rotation_error", None)),
            "solution": _tensor_summary(getattr(ik_result, "solution", None)),
            "solve_time": float(getattr(ik_result, "solve_time", 0.0)),
            "total_time": float(getattr(ik_result, "total_time", 0.0)),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def compute_ik_fallback_trajectory(planner, goal_pose, current_state, q_current, params):
    if not bool(params["allow_ik_fallback"]):
        return None

    ik_result = planner.ik_solver.solve_pose(
        goal_pose,
        return_seeds=max(int(params["num_trajopt_seeds"]), 4),
        current_state=current_state,
    )

    position_error = ik_result.position_error.detach()[0]
    rotation_error = ik_result.rotation_error.detach()[0]
    score = position_error + rotation_error
    best_index = int(torch.argmin(score).item())
    best_position_error = float(position_error[best_index].item())
    best_rotation_error = float(rotation_error[best_index].item())

    if (
        best_position_error > float(params["ik_fallback_position_tol"])
        or best_rotation_error > float(params["ik_fallback_orientation_tol"])
    ):
        return {
            "usable": False,
            "best_position_error_m": best_position_error,
            "best_rotation_error_rad": best_rotation_error,
            "reason": "IK best solution is still outside fallback tolerances.",
        }

    q_goal = ik_result.solution.detach()[0, best_index].cpu().numpy().astype(float)
    waypoint_count = max(2, int(params["ik_fallback_waypoints"]))
    q_raw = np.linspace(np.asarray(q_current, dtype=float), q_goal, waypoint_count)

    return {
        "usable": True,
        "q_raw": q_raw,
        "q_goal": q_goal,
        "best_seed_index": best_index,
        "best_position_error_m": best_position_error,
        "best_rotation_error_rad": best_rotation_error,
        "reason": (
            "cuRobo IK reached the target pose, but cuRobo feasibility/trajopt success was False; "
            "using IK goal with joint-space interpolation for first visualization."
        ),
    }


def _json_sanitize(value):
    if isinstance(value, np.ndarray):
        return [_json_sanitize(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value
    return value


def _merged_params(request):
    params = dict(DEFAULT_PARAMS)
    params.update(request.get("params", {}))
    return params


def run_planner(request):
    params = _merged_params(request)
    robot_config_path = request["robot_config_path"]
    scene_model = request["scene_model"]
    expected_joint_names = list(request["joint_names"])
    expected_tool_frame = request["tool_frame"]
    q_current = np.asarray(request["q_current"], dtype=float)
    target_position = np.asarray(request["target_tcp_base"]["position"], dtype=float)
    target_quaternion = normalize_quat_wxyz(request["target_tcp_base"]["quaternion_wxyz"])
    current_tcp_base_position = np.asarray(request["current_tcp_base"]["position"], dtype=float)
    current_tcp_base_quaternion = normalize_quat_wxyz(
        request["current_tcp_base"]["quaternion_wxyz"]
    )

    print("[external cuRobo] python:", sys.executable)
    print("[external cuRobo] torch:", torch.__version__)
    print("[external cuRobo] torch.cuda.is_available:", torch.cuda.is_available())
    print("[external cuRobo] warp source should be site-packages/warp, not Isaac omni.warp")

    if not torch.cuda.is_available():
        return {
            "success": False,
            "error": "CUDA 当前不可用，cuRobo 不能运行。",
        }

    cfg = MotionPlannerCfg.create(
        robot=robot_config_path,
        scene_model=scene_model,
        use_cuda_graph=bool(params["use_cuda_graph"]),
        num_ik_seeds=int(params["num_ik_seeds"]),
        num_trajopt_seeds=int(params["num_trajopt_seeds"]),
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=bool(params["self_collision_check"]),
        store_debug=False,
    )
    planner = MotionPlanner(cfg)

    try:
        joint_names = list(planner.joint_names)
        tool_frames = list(planner.tool_frames)
        if joint_names != expected_joint_names:
            return {
                "success": False,
                "error": f"cuRobo joint order 不一致：{joint_names} != {expected_joint_names}",
                "joint_names": joint_names,
                "tool_frames": tool_frames,
            }
        if expected_tool_frame not in tool_frames:
            return {
                "success": False,
                "error": f"cuRobo tool_frames 中没有 {expected_tool_frame}: {tool_frames}",
                "joint_names": joint_names,
                "tool_frames": tool_frames,
            }

        if bool(params["run_planner_warmup"]):
            planner.warmup(enable_graph=False, num_warmup_iterations=3)

        fk_start_pos, fk_start_quat = fk_single_tcp_pose_base(planner, q_current)
        start_position_error = float(np.linalg.norm(fk_start_pos - current_tcp_base_position))
        start_orientation_error_deg = quat_angle_error_deg(
            fk_start_quat,
            current_tcp_base_quaternion,
        )

        current_state = make_joint_state(q_current, joint_names)
        goal_pose = make_goal_tool_pose(target_position, target_quaternion, tool_frames)
        ik_diagnostics = diagnose_ik(planner, goal_pose, current_state, params)

        result = planner.plan_pose(
            goal_pose,
            current_state,
            max_attempts=int(params["max_planning_attempts"]),
        )
        success = bool(result is not None and torch.count_nonzero(result.success).item() > 0)
        if not success:
            fallback = compute_ik_fallback_trajectory(
                planner,
                goal_pose,
                current_state,
                q_current,
                params,
            )
            if fallback is None or not fallback.get("usable", False):
                return {
                    "success": False,
                    "error": "cuRobo plan_pose 失败，且 IK fallback 不可用。",
                    "joint_names": joint_names,
                    "tool_frames": tool_frames,
                    "start_position_error_m": start_position_error,
                    "start_orientation_error_deg": start_orientation_error_deg,
                    "ik_diagnostics": ik_diagnostics,
                    "ik_fallback": fallback,
                }
            q_raw = np.asarray(fallback["q_raw"], dtype=float)
            planner_success = False
            fallback_used = "ik_linear_interpolation"
            fallback_info = fallback
        else:
            q_raw = extract_joint_trajectory(result)
            planner_success = True
            fallback_used = None
            fallback_info = None

        s_curve = s_curve_resample_joint_trajectory(q_raw, params)
        q_s = s_curve["q"]
        qd_s = s_curve["qd"]
        qdd_s = s_curve["qdd"]

        joint_limit_violations = check_joint_limits(q_s, joint_names, params)
        tcp_positions_base, tcp_quats_base = fk_tcp_poses_base(planner, q_s)

        final_position_error = float(np.linalg.norm(tcp_positions_base[-1] - target_position))
        final_orientation_error_deg = quat_angle_error_deg(tcp_quats_base[-1], target_quaternion)

        singularity = check_singularity_by_numeric_jacobian(planner, q_s, params)

        return {
            "success": True,
            "joint_names": joint_names,
            "tool_frames": tool_frames,
            "q_raw": q_raw,
            "q_s": q_s,
            "t_s": s_curve["t"],
            "duration_s": float(s_curve["duration"]),
            "max_abs_joint_velocity": np.max(np.abs(qd_s), axis=0),
            "max_abs_joint_acceleration": np.max(np.abs(qdd_s), axis=0),
            "tcp_positions_base": tcp_positions_base,
            "tcp_quaternions_base": tcp_quats_base,
            "start_position_error_m": start_position_error,
            "start_orientation_error_deg": start_orientation_error_deg,
            "final_position_error_m": final_position_error,
            "final_orientation_error_deg": final_orientation_error_deg,
            "joint_limit_violations": joint_limit_violations,
            "singularity": singularity,
            "ik_diagnostics": ik_diagnostics,
            "planner_success": planner_success,
            "fallback_used": fallback_used,
            "fallback_info": fallback_info,
        }
    finally:
        planner.destroy()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("用法：python run_curobo_plan_external.py request.json response.json")

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))

    response = run_planner(request)
    response_path.write_text(
        json.dumps(_json_sanitize(response), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not response.get("success", False):
        print("[external cuRobo] planning failed:", response.get("error"))
    else:
        print("[external cuRobo] planning success")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

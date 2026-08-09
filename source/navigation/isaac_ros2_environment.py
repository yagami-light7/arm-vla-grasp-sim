"""验证 Isaac OGN 使用 ROS 2 自定义消息前所需的进程环境。"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence


EXPECTED_ROS_DISTRO = "humble"
EXPECTED_RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
DEFAULT_CUSTOM_MESSAGE_PACKAGE = "scan_planner_msgs"
DEFAULT_CUSTOM_MESSAGE_NAME = "ControllerStatus"
DEFAULT_ADDITIONAL_CUSTOM_MESSAGE_NAMES = (
    "NavigationStatus",
    "GridMapObservationDiagnostics",
    "BsplineDiagnostics",
    "StairExecutionFreeze",
)
DEFAULT_CUSTOM_MESSAGE_LIBRARIES = (
    "libscan_planner_msgs__rosidl_generator_c.so",
    "libscan_planner_msgs__rosidl_typesupport_c.so",
    "libscan_planner_msgs__rosidl_typesupport_introspection_c.so",
)


class IsaacRos2EnvironmentError(RuntimeError):
    """表示 Isaac 进程尚不能安全加载 ROS 2 自定义消息。"""


def _split_paths(value: str | None) -> tuple[Path, ...]:
    """把路径型环境变量拆成去重后的绝对路径。"""

    result: list[Path] = []
    seen: set[str] = set()
    for raw_path in str(value or "").split(os.pathsep):
        if not raw_path:
            continue
        normalized = Path(os.path.abspath(os.path.expanduser(raw_path)))
        key = os.fspath(normalized)
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)


def _python_abi_mismatches(
    paths: Sequence[Path],
    *,
    python_version: tuple[int, int],
) -> tuple[str, ...]:
    """返回 PYTHONPATH 中与当前解释器 ABI 不同的 Python 路径。"""

    expected = f"{python_version[0]}.{python_version[1]}"
    mismatches: list[str] = []
    for path in paths:
        versions = re.findall(r"python(\d+\.\d+)", os.fspath(path))
        if versions and any(version != expected for version in versions):
            mismatches.append(os.fspath(path))
    return tuple(mismatches)


def _startup_hint() -> str:
    """返回可直接复制的 zsh 修复命令。"""

    return (
        "请在启动 Isaac Python 之前执行：\n"
        "  export ISAAC_PYTHON=\"$(command -v python)\"\n"
        "  source /opt/ros/humble/setup.zsh\n"
        "  source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh\n"
        "  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n"
        "  unset PYTHONPATH\n"
        "随后使用 \"$ISAAC_PYTHON\" -B 启动 pipeline；ROS launch 终端"
        "保留 ROS 的 PYTHONPATH，并使用相同 ROS_DOMAIN_ID。"
    )


def validate_isaac_ros2_custom_message_environment(
    *,
    environ: Mapping[str, str] | None = None,
    python_version: tuple[int, int] | None = None,
    package_name: str = DEFAULT_CUSTOM_MESSAGE_PACKAGE,
    message_name: str = DEFAULT_CUSTOM_MESSAGE_NAME,
    additional_message_names: Sequence[str] = (
        DEFAULT_ADDITIONAL_CUSTOM_MESSAGE_NAMES
    ),
    library_names: Sequence[str] = DEFAULT_CUSTOM_MESSAGE_LIBRARIES,
) -> dict[str, object]:
    """在创建 SimulationApp 前验证 ROS 2 overlay 与自定义消息共享库。

    该函数只读取当前进程环境，不会尝试在 Python 内 source shell 脚本，也不会
    预加载 ROS 共享库。动态链接器的搜索路径必须在启动解释器之前由父 shell
    确定，否则 OGN generic subscriber 可能只表现为“动态端口未生成”。
    """

    env = os.environ if environ is None else environ
    interpreter_version = (
        (sys.version_info.major, sys.version_info.minor)
        if python_version is None
        else (int(python_version[0]), int(python_version[1]))
    )
    problems: list[str] = []

    ros_distro = str(env.get("ROS_DISTRO", "")).strip()
    if ros_distro != EXPECTED_ROS_DISTRO:
        problems.append(
            f"ROS_DISTRO 必须为 {EXPECTED_ROS_DISTRO!r}，当前为 {ros_distro or '<unset>'!r}"
        )

    rmw = str(env.get("RMW_IMPLEMENTATION", "")).strip()
    if rmw and rmw != EXPECTED_RMW_IMPLEMENTATION:
        problems.append(
            "RMW_IMPLEMENTATION 必须为 "
            f"{EXPECTED_RMW_IMPLEMENTATION!r}，当前为 {rmw!r}"
        )

    python_paths = _split_paths(env.get("PYTHONPATH"))
    abi_mismatches = _python_abi_mismatches(
        python_paths,
        python_version=interpreter_version,
    )
    if abi_mismatches:
        problems.append(
            "PYTHONPATH 含有与当前 Python "
            f"{interpreter_version[0]}.{interpreter_version[1]} ABI 不匹配的路径："
            + ", ".join(abi_mismatches)
        )

    ament_prefixes = _split_paths(env.get("AMENT_PREFIX_PATH"))
    message_names = tuple(
        dict.fromkeys(
            (
                str(message_name).strip(),
                *(str(name).strip() for name in additional_message_names),
            )
        )
    )
    if not message_names or any(not name for name in message_names):
        problems.append("自定义消息名称不能为空")
    interface_entries = {
        f"msg/{name}.msg" for name in message_names if name
    }
    interface_prefix: Path | None = None
    interface_resource: Path | None = None
    closest_missing_entries = set(interface_entries)
    interface_resource_found = False
    for prefix in ament_prefixes:
        resource = (
            prefix
            / "share/ament_index/resource_index/rosidl_interfaces"
            / package_name
        )
        if not resource.is_file():
            continue
        interface_resource_found = True
        entries = {
            line.strip()
            for line in resource.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing_entries = interface_entries - entries
        if len(missing_entries) < len(closest_missing_entries):
            closest_missing_entries = missing_entries
        if not missing_entries:
            interface_prefix = prefix
            interface_resource = resource
            break
    if interface_prefix is None:
        if interface_resource_found:
            problems.append(
                f"{package_name} 的 rosidl interface resource 缺少："
                + ", ".join(sorted(closest_missing_entries))
            )
        else:
            problems.append(
                "AMENT_PREFIX_PATH 中找不到 "
                f"{package_name} 的 rosidl interface resource；要求："
                + ", ".join(sorted(interface_entries))
            )

    library_paths = _split_paths(env.get("LD_LIBRARY_PATH"))
    library_path_strings = {os.fspath(path) for path in library_paths}
    message_library_dir: Path | None = None
    missing_libraries: tuple[str, ...] = tuple(library_names)
    if interface_prefix is not None:
        candidates = (
            interface_prefix / "lib",
            interface_prefix / "lib/x86_64-linux-gnu",
        )
        for candidate in candidates:
            missing = tuple(
                name for name in library_names if not (candidate / name).is_file()
            )
            missing_libraries = missing
            if not missing:
                message_library_dir = candidate
                break
        if message_library_dir is None:
            problems.append(
                f"{package_name} overlay 缺少自定义消息共享库："
                + ", ".join(missing_libraries)
            )
        elif os.fspath(message_library_dir) not in library_path_strings:
            problems.append(
                "LD_LIBRARY_PATH 未包含自定义消息库目录："
                f"{message_library_dir}"
            )

    if problems:
        raise IsaacRos2EnvironmentError(
            "Isaac ROS 2 自定义消息环境检查失败：\n- "
            + "\n- ".join(problems)
            + "\n"
            + _startup_hint()
        )

    assert interface_prefix is not None
    assert interface_resource is not None
    assert message_library_dir is not None
    return {
        "verified": True,
        "ros_distro": ros_distro,
        "rmw_implementation": rmw or EXPECTED_RMW_IMPLEMENTATION,
        "rmw_source": "environment" if rmw else "ros_default",
        "python_version": f"{interpreter_version[0]}.{interpreter_version[1]}",
        "pythonpath_cleared": not bool(python_paths),
        "package_name": package_name,
        "message_name": message_name,
        "message_names": list(message_names),
        "interface_entries": sorted(interface_entries),
        "interface_prefix": os.fspath(interface_prefix),
        "interface_resource": os.fspath(interface_resource),
        "message_library_dir": os.fspath(message_library_dir),
        "library_names": list(library_names),
    }

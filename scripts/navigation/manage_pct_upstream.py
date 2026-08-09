#!/usr/bin/env python3
"""准备、应用并验证固定版本的 PCT Planner 上游源码。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


_BUFFER_SIZE = 1024 * 1024
_FIXED_PYTHON_EXECUTABLE = "/usr/bin/python3"
_FIXED_BUILD_TYPE = "Release"
_FIXED_CMAKE_POLICY_MINIMUM = "3.5"
_FIXED_SOABI = "cpython-310-x86_64-linux-gnu"
_PYTHON_EXTENSION_MODULES = (
    "a_star",
    "traj_opt",
    "ele_planner",
    "py_map_manager",
)
_INTERNAL_LIBRARIES = (
    "liba_star_search.so",
    "libcommon_smoothing.so",
    "libele_planner_lib.so",
    "libgpmp_optimizer.so",
    "libmap_manager.so",
)
_DEFAULT_ALLOWED_RUNPATHS = (
    "$ORIGIN",
    "$ORIGIN/3rdparty/gtsam-4.1.1/install/lib",
    "$ORIGIN/3rdparty/osqp/install/lib",
)
_FORBIDDEN_ABSOLUTE_RUNPATH_PREFIXES = ("/home/", "/mnt/", "/tmp/")


class UpstreamManageError(RuntimeError):
    """上游准备或验证合同不满足。"""


@dataclass(frozen=True)
class TreeIdentity:
    """一个源码树的确定性内容身份。"""

    sha256: str
    file_count: int
    generated_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchSpec:
    """已经通过 schema 与补丁文件检查的补丁合同。"""

    path: Path
    sha256: str
    strip: int
    preimage_sha256: dict[str, str | None]
    postimage_sha256: dict[str, str | None]


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA256，避免把归档一次性读入内存。"""

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def compute_tree_identity(
    root: Path,
    *,
    generated_path_patterns: Sequence[str] = (),
    allow_generated: bool = False,
) -> TreeIdentity:
    """按相对路径、大小和内容哈希计算可跨机器复验的源码树身份。

    聚合记录格式固定为 UTF-8 编码的
    ``relative_path\0decimal_size\0file_sha256\0``。目录时间、所有者和
    权限不进入身份；符号链接和其他特殊文件一律拒绝。
    """

    requested_root = Path(root)
    if requested_root.is_symlink():
        raise UpstreamManageError(f"源码树根目录禁止是符号链接：{requested_root}")
    root = requested_root.resolve()
    if not root.is_dir():
        raise UpstreamManageError(f"源码树不存在或不是目录：{root}")

    patterns = _validate_generated_patterns(generated_path_patterns)
    relative_paths: list[str] = []
    generated_paths: list[str] = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not allow_generated:
                raise UpstreamManageError(f"源码树禁止包含符号链接：{path}")
            _validate_generated_symlink(
                path,
                source_root=root,
                relative_path=relative_path,
                patterns=patterns,
            )
            generated_paths.append(relative_path)
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise UpstreamManageError(f"源码树包含特殊文件：{path}")
        if allow_generated and _matches_generated_path(relative_path, patterns):
            generated_paths.append(relative_path)
            continue
        relative_paths.append(relative_path)

    digest = sha256()
    for relative_path in sorted(relative_paths):
        path = root / PurePosixPath(relative_path)
        size = path.stat().st_size
        record = (
            f"{relative_path}\0{size}\0{file_sha256(path)}\0"
        ).encode("utf-8")
        digest.update(record)
    return TreeIdentity(
        sha256=digest.hexdigest(),
        file_count=len(relative_paths),
        generated_paths=tuple(sorted(generated_paths)),
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """读取并执行 PCT upstream manifest v2 的最小 schema 检查。"""

    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UpstreamManageError(f"manifest 不存在：{manifest_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpstreamManageError(f"manifest 无法读取：{exc}") from exc
    if not isinstance(payload, dict):
        raise UpstreamManageError("manifest 顶层必须是 JSON object")
    if payload.get("schema_version") != 2:
        raise UpstreamManageError("只接受 schema_version=2 的 upstream manifest")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise UpstreamManageError("manifest.source 必须是 JSON object")
    _required_sha256(source, "archive_sha256", "manifest.source")
    _required_sha256(source, "pristine_tree_sha256", "manifest.source")
    _required_nonempty_string(source, "archive_root", "manifest.source")
    _validate_archive_root(str(source["archive_root"]))
    _optional_nonnegative_int(source, "pristine_file_count", "manifest.source")
    _validate_generated_patterns(source.get("generated_path_patterns", []))

    patches = payload.get("patches", [])
    if not isinstance(patches, list):
        raise UpstreamManageError("manifest.patches 必须是 JSON array")
    if patches:
        _required_sha256(source, "patched_tree_sha256", "manifest.source")
        _optional_nonnegative_int(source, "patched_file_count", "manifest.source")
    elif "patched_tree_sha256" in source:
        _required_sha256(source, "patched_tree_sha256", "manifest.source")
        _optional_nonnegative_int(source, "patched_file_count", "manifest.source")
    return payload


def prepare_upstream(
    *,
    manifest_path: Path,
    archive_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    """从可信归档准备最终补丁态源码；已准备目录会幂等返回。"""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    archive_path = Path(archive_path).expanduser().resolve()
    source_root = _safe_source_destination(source_root)

    if source_root.exists():
        try:
            identity = verify_source(
                manifest_path=manifest_path,
                source_root=source_root,
                state="patched",
            )
        except UpstreamManageError as exc:
            raise UpstreamManageError(
                "目标源码目录已存在但不满足最终 manifest；拒绝覆盖："
                f"{source_root}（{exc}）"
            ) from exc
        return {
            "status": "already_prepared",
            "source_root": str(source_root),
            **identity,
        }

    source = _source_section(manifest)
    expected_archive_hash = str(source["archive_sha256"])
    if not archive_path.is_file():
        raise UpstreamManageError(f"源码归档不存在：{archive_path}")
    actual_archive_hash = file_sha256(archive_path)
    if actual_archive_hash != expected_archive_hash:
        raise UpstreamManageError(
            "源码归档 SHA256 不匹配："
            f"expected={expected_archive_hash}, actual={actual_archive_hash}"
        )

    source_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{source_root.name}.prepare-",
            dir=source_root.parent,
        )
    )
    try:
        extracted_root = _extract_archive_safely(
            archive_path,
            staging_root,
            archive_root=str(source["archive_root"]),
        )
        _verify_tree_state(manifest, extracted_root, state="pristine")
        _apply_patch_series(manifest_path, manifest, extracted_root)
        identity = _verify_tree_state(manifest, extracted_root, state="patched")
        os.replace(extracted_root, source_root)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "status": "prepared",
        "source_root": str(source_root),
        **identity,
    }


def apply_patches(
    *,
    manifest_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    """在精确 pristine 源码上应用补丁；最终状态再次调用不会重复修改。"""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    source_root = _existing_source_root(source_root)

    try:
        identity = _verify_tree_state(manifest, source_root, state="patched")
    except UpstreamManageError:
        _verify_tree_state(manifest, source_root, state="pristine")
    else:
        _validate_patch_specs(manifest_path, manifest)
        return {
            "status": "already_applied",
            "source_root": str(source_root),
            **identity,
        }

    _apply_patch_series(manifest_path, manifest, source_root)
    identity = _verify_tree_state(manifest, source_root, state="patched")
    return {
        "status": "applied",
        "source_root": str(source_root),
        **identity,
    }


def verify_source(
    *,
    manifest_path: Path,
    source_root: Path,
    state: str = "patched",
    allow_generated: bool = False,
) -> dict[str, Any]:
    """验证源码树、补丁文件以及目标状态的逐文件 image。"""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    source_root = _existing_source_root(source_root)
    specs = _validate_patch_specs(manifest_path, manifest)
    identity = _verify_tree_state(
        manifest,
        source_root,
        state=state,
        allow_generated=allow_generated,
    )

    if state == "patched":
        for spec in specs:
            _verify_image(source_root, spec.postimage_sha256, label="postimage")
    elif state == "pristine":
        for spec in specs:
            _verify_image(source_root, spec.preimage_sha256, label="preimage")
    else:
        raise UpstreamManageError(f"未知源码状态：{state!r}")

    return {
        "state": state,
        "tree_sha256": identity["tree_sha256"],
        "file_count": identity["file_count"],
        "patch_count": len(specs),
        "generated_file_count": identity["generated_file_count"],
        "generated_paths": identity["generated_paths"],
    }


def generate_build_plan(
    *,
    manifest_path: Path,
    source_root: Path,
    build_root: Path,
    jobs: int = 4,
) -> dict[str, Any]:
    """生成固定 ABI 的构建 argv；只输出计划，绝不执行编译。"""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    build = _validated_build_contract(manifest)
    source_root = _existing_source_root(source_root)
    verify_source(
        manifest_path=manifest_path,
        source_root=source_root,
        state="patched",
    )
    build_root = _safe_external_build_root(build_root, source_root=source_root)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 128:
        raise UpstreamManageError("jobs 必须是 1..128 的整数")

    planner_library = source_root / "planner/lib"
    gtsam_source = planner_library / "3rdparty/gtsam-4.1.1"
    osqp_source = planner_library / "3rdparty/osqp"
    for label, path in (
        ("GTSAM", gtsam_source),
        ("OSQP", osqp_source),
        ("planner", planner_library),
    ):
        if not path.is_dir():
            raise UpstreamManageError(f"{label} 源码目录不存在：{path}")

    gtsam_install = gtsam_source / "install"
    osqp_install = osqp_source / "install"
    gtsam_build = build_root / "gtsam"
    osqp_build = build_root / "osqp"
    planner_build = build_root / "planner"
    planner_runpath = ";".join(_allowed_runpaths(manifest))
    common_policy = (
        f"-DCMAKE_POLICY_VERSION_MINIMUM={_FIXED_CMAKE_POLICY_MINIMUM}"
    )
    commands = [
        [
            "cmake",
            "-S",
            str(gtsam_source),
            "-B",
            str(gtsam_build),
            f"-DCMAKE_BUILD_TYPE={_FIXED_BUILD_TYPE}",
            f"-DCMAKE_INSTALL_PREFIX={gtsam_install}",
            common_policy,
            "-DGTSAM_USE_SYSTEM_EIGEN=ON",
            "-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF",
            "-DGTSAM_BUILD_TESTS=OFF",
            "-DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF",
            "-DGTSAM_BUILD_TIMING_ALWAYS=OFF",
            "-DGTSAM_BUILD_UNSTABLE=OFF",
            "-DGTSAM_BUILD_PYTHON=OFF",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        ],
        [
            "cmake",
            "--build",
            str(gtsam_build),
            "--parallel",
            str(jobs),
            "--target",
            "install",
        ],
        [
            "cmake",
            "-S",
            str(osqp_source),
            "-B",
            str(osqp_build),
            f"-DCMAKE_BUILD_TYPE={_FIXED_BUILD_TYPE}",
            f"-DCMAKE_INSTALL_PREFIX={osqp_install}",
            common_policy,
            "-DUNITTESTS=OFF",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        ],
        [
            "cmake",
            "--build",
            str(osqp_build),
            "--parallel",
            str(jobs),
            "--target",
            "install",
        ],
        [
            "cmake",
            "-S",
            str(planner_library),
            "-B",
            str(planner_build),
            f"-DCMAKE_BUILD_TYPE={_FIXED_BUILD_TYPE}",
            common_policy,
            f"-DPYTHON_EXECUTABLE={_FIXED_PYTHON_EXECUTABLE}",
            f"-DPython3_EXECUTABLE={_FIXED_PYTHON_EXECUTABLE}",
            f"-DPython3_INCLUDE_DIR={build['python_include_dir']}",
            f"-DPython3_LIBRARY={build['python_library']}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={planner_library}",
            "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
            f"-DCMAKE_BUILD_RPATH={planner_runpath}",
            f"-DCMAKE_INSTALL_RPATH={planner_runpath}",
        ],
        [
            "cmake",
            "--build",
            str(planner_build),
            "--parallel",
            str(jobs),
        ],
    ]
    return {
        "status": "build_plan_only",
        "executes_commands": False,
        "source_root": str(source_root),
        "build_root": str(build_root),
        "python_executable": _FIXED_PYTHON_EXECUTABLE,
        "required_soabi": _FIXED_SOABI,
        "build_type": _FIXED_BUILD_TYPE,
        "cmake_policy_minimum": _FIXED_CMAKE_POLICY_MINIMUM,
        "gtsam_build_with_march_native": False,
        "commands": commands,
    }


def verify_binaries(
    *,
    manifest_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    """验证固定 CPython ABI、ELF、相对 RUNPATH 与动态库闭包。"""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    _validated_build_contract(manifest)
    source_root = _existing_source_root(source_root)
    source_report = verify_source(
        manifest_path=manifest_path,
        source_root=source_root,
        state="patched",
        allow_generated=True,
    )
    library_root = source_root / "planner/lib"
    binaries = [
        library_root / f"{module}.{_FIXED_SOABI}.so"
        for module in _PYTHON_EXTENSION_MODULES
    ]
    binaries.extend(library_root / name for name in _INTERNAL_LIBRARIES)
    allowed_runpaths = set(_allowed_runpaths(manifest))

    reports: list[dict[str, Any]] = []
    for path in binaries:
        if path.is_symlink() or not path.is_file():
            raise UpstreamManageError(f"必需二进制缺失或不是普通文件：{path}")
        header = _run_readonly_tool(("readelf", "-h", str(path)))
        if "ELF64" not in header or "X86-64" not in header:
            raise UpstreamManageError(f"二进制不是 ELF64 x86-64：{path}")
        dynamic = _run_readonly_tool(("readelf", "-d", str(path)))
        runpaths = _parse_runpaths(dynamic, path=path)
        if not runpaths:
            raise UpstreamManageError(f"二进制缺少 RUNPATH：{path}")
        for entry in runpaths:
            if entry.startswith("/"):
                raise UpstreamManageError(
                    f"RUNPATH 禁止绝对路径：{path}: {entry}"
                )
            if any(prefix in entry for prefix in _FORBIDDEN_ABSOLUTE_RUNPATH_PREFIXES):
                raise UpstreamManageError(
                    f"RUNPATH 含宿主目录：{path}: {entry}"
                )
            if entry not in allowed_runpaths:
                raise UpstreamManageError(
                    f"RUNPATH 不在 manifest 白名单：{path}: {entry}"
                )
        ldd_output = _run_readonly_tool(("ldd", str(path)))
        if re.search(r"\bnot found\b", ldd_output, flags=re.IGNORECASE):
            raise UpstreamManageError(f"ldd 发现未解析依赖：{path}\n{ldd_output}")
        reports.append(
            {
                "path": str(path.relative_to(source_root)),
                "sha256": file_sha256(path),
                "runpaths": runpaths,
            }
        )
    return {
        "status": "verified",
        "required_soabi": _FIXED_SOABI,
        "binary_count": len(reports),
        "binaries": reports,
        "source": source_report,
    }


def _validated_build_contract(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    build = manifest.get("build")
    if not isinstance(build, dict):
        raise UpstreamManageError("manifest.build 必须是 JSON object")
    fixed_values: tuple[tuple[str, object], ...] = (
        ("python_executable", _FIXED_PYTHON_EXECUTABLE),
        ("required_soabi", _FIXED_SOABI),
        ("build_type", _FIXED_BUILD_TYPE),
        ("cmake_policy_minimum", _FIXED_CMAKE_POLICY_MINIMUM),
        ("gtsam_build_with_march_native", False),
    )
    for key, expected in fixed_values:
        if build.get(key) != expected:
            raise UpstreamManageError(
                f"manifest.build.{key} 必须固定为 {expected!r}"
            )
    for key in ("python_include_dir", "python_library"):
        value = _required_nonempty_string(build, key, "manifest.build")
        if not value.startswith("/"):
            raise UpstreamManageError(f"manifest.build.{key} 必须是绝对路径")
    return build


def _allowed_runpaths(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    runtime = manifest.get("runtime", {})
    if not isinstance(runtime, dict):
        raise UpstreamManageError("manifest.runtime 必须是 JSON object")
    raw = runtime.get("allowed_runpaths", list(_DEFAULT_ALLOWED_RUNPATHS))
    if not isinstance(raw, list) or not raw:
        raise UpstreamManageError("manifest.runtime.allowed_runpaths 必须是非空 array")
    output: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, str) or not entry:
            raise UpstreamManageError(
                f"manifest.runtime.allowed_runpaths[{index}] 必须是非空字符串"
            )
        if entry not in _DEFAULT_ALLOWED_RUNPATHS:
            raise UpstreamManageError(
                f"manifest.runtime.allowed_runpaths[{index}] 不是许可的相对项："
                f"{entry}"
            )
        output.append(entry)
    if set(output) != set(_DEFAULT_ALLOWED_RUNPATHS):
        raise UpstreamManageError(
            "allowed_runpaths 必须完整包含 $ORIGIN 与两个固定 thirdparty 相对目录"
        )
    if len(set(output)) != len(output):
        raise UpstreamManageError("allowed_runpaths 禁止重复")
    return tuple(output)


def _safe_external_build_root(path: Path, *, source_root: Path) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise UpstreamManageError(f"build root 不能是符号链接：{requested}")
    build_root = requested.resolve()
    if build_root == Path(build_root.anchor):
        raise UpstreamManageError("build root 不能是文件系统根目录")
    try:
        build_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise UpstreamManageError("build root 必须位于 upstream source root 之外")
    return build_root


def _run_readonly_tool(arguments: Sequence[str]) -> str:
    program = shutil.which(arguments[0])
    if program is None:
        raise UpstreamManageError(f"缺少只读检查工具：{arguments[0]}")
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("LD_PRELOAD", None)
    environment["LC_ALL"] = "C"
    result = subprocess.run(
        [program, *arguments[1:]],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise UpstreamManageError(
            f"只读检查命令失败：{shlex.join(arguments)}\n"
            f"{_command_failure_text(result)}"
        )
    return result.stdout


def _parse_runpaths(dynamic_output: str, *, path: Path) -> list[str]:
    entries: list[str] = []
    for line in dynamic_output.splitlines():
        if "(RPATH)" in line:
            raise UpstreamManageError(f"二进制必须使用 RUNPATH，禁止旧 RPATH：{path}")
        if "(RUNPATH)" not in line:
            continue
        match = re.search(r"\[([^]]*)\]", line)
        if match is None:
            raise UpstreamManageError(f"无法解析 RUNPATH：{path}: {line}")
        for entry in match.group(1).split(":"):
            if not entry:
                raise UpstreamManageError(f"RUNPATH 包含空项：{path}")
            entries.append(entry)
    if len(set(entries)) != len(entries):
        raise UpstreamManageError(f"RUNPATH 包含重复项：{path}")
    return entries


def _apply_patch_series(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    source_root: Path,
) -> None:
    specs = _validate_patch_specs(manifest_path, manifest)
    for spec in specs:
        pre_matches = _image_matches(source_root, spec.preimage_sha256)
        post_matches = _image_matches(source_root, spec.postimage_sha256)
        if post_matches:
            continue
        if not pre_matches:
            raise UpstreamManageError(
                f"补丁既不处于 preimage 也不处于 postimage：{spec.path}"
            )
        _run_patch(source_root, spec)
        _verify_image(source_root, spec.postimage_sha256, label="postimage")


def _run_patch(source_root: Path, spec: PatchSpec) -> None:
    patch_program = shutil.which("patch")
    if patch_program is None:
        raise UpstreamManageError("系统缺少 patch 命令，不能应用 upstream 补丁")
    common = [
        patch_program,
        "--batch",
        "--forward",
        "--fuzz=0",
        f"-p{spec.strip}",
        "--directory",
        str(source_root),
        "--input",
        str(spec.path),
    ]
    dry_run = subprocess.run(
        [*common, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    if dry_run.returncode != 0:
        raise UpstreamManageError(
            f"补丁 dry-run 失败：{spec.path}\n"
            f"{_command_failure_text(dry_run)}"
        )
    applied = subprocess.run(
        common,
        check=False,
        capture_output=True,
        text=True,
    )
    if applied.returncode != 0:
        raise UpstreamManageError(
            f"补丁应用失败：{spec.path}\n"
            f"{_command_failure_text(applied)}"
        )


def _command_failure_text(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return text or f"exit={result.returncode}"


def _validate_patch_specs(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> list[PatchSpec]:
    raw_patches = manifest.get("patches", [])
    assert isinstance(raw_patches, list)
    specs: list[PatchSpec] = []
    manifest_dir = manifest_path.parent.resolve()
    for index, raw in enumerate(raw_patches):
        label = f"manifest.patches[{index}]"
        if not isinstance(raw, dict):
            raise UpstreamManageError(f"{label} 必须是 JSON object")
        raw_path = _required_nonempty_string(raw, "path", label)
        patch_path = _resolve_contained_path(
            manifest_dir,
            raw_path,
            label=f"{label}.path",
        )
        expected_hash = _required_sha256(raw, "sha256", label)
        if not patch_path.is_file():
            raise UpstreamManageError(f"补丁文件不存在：{patch_path}")
        actual_hash = file_sha256(patch_path)
        if actual_hash != expected_hash:
            raise UpstreamManageError(
                f"补丁 SHA256 不匹配：{patch_path}，"
                f"expected={expected_hash}, actual={actual_hash}"
            )
        strip = raw.get("strip", 1)
        if isinstance(strip, bool) or not isinstance(strip, int) or not 0 <= strip <= 16:
            raise UpstreamManageError(f"{label}.strip 必须是 0..16 的整数")
        preimage = _validate_image_map(raw.get("preimage_sha256"), f"{label}.preimage_sha256")
        postimage = _validate_image_map(raw.get("postimage_sha256"), f"{label}.postimage_sha256")
        if set(preimage) != set(postimage):
            raise UpstreamManageError(f"{label} 的 preimage/postimage 路径集合必须相同")
        touched = _git_patch_touched_paths(patch_path, strip=strip)
        if touched != set(preimage):
            raise UpstreamManageError(
                f"{label} image 路径与补丁触及路径不一致："
                f"images={sorted(preimage)}, patch={sorted(touched)}"
            )
        specs.append(
            PatchSpec(
                path=patch_path,
                sha256=expected_hash,
                strip=strip,
                preimage_sha256=preimage,
                postimage_sha256=postimage,
            )
        )
    return specs


def _git_patch_touched_paths(path: Path, *, strip: int) -> set[str]:
    """读取 git 风格 diff header，并固定所有会被修改的相对路径。"""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise UpstreamManageError(f"补丁不是可读 UTF-8 文本：{path}") from exc
    touched: set[str] = set()
    for line in lines:
        if not line.startswith("diff --git "):
            continue
        try:
            fields = shlex.split(line)
        except ValueError as exc:
            raise UpstreamManageError(f"补丁 diff header 无法解析：{line}") from exc
        if len(fields) != 4:
            raise UpstreamManageError(f"补丁 diff header 必须包含两个路径：{line}")
        for raw_path in fields[2:]:
            components = PurePosixPath(raw_path).parts
            if len(components) <= strip:
                raise UpstreamManageError(
                    f"补丁路径无法按 -p{strip} 剥离：{raw_path}"
                )
            normalized = PurePosixPath(*components[strip:]).as_posix()
            _validate_member_relative_path(normalized, label="补丁目标路径")
            touched.add(normalized)
    if not touched:
        raise UpstreamManageError(f"补丁缺少 git diff header：{path}")
    return touched


def _verify_image(
    source_root: Path,
    image: Mapping[str, str | None],
    *,
    label: str,
) -> None:
    mismatches: list[str] = []
    for relative_path, expected_hash in image.items():
        path = source_root / PurePosixPath(relative_path)
        if expected_hash is None:
            if path.exists() or path.is_symlink():
                mismatches.append(f"{relative_path}: expected absent")
            continue
        if path.is_symlink() or not path.is_file():
            mismatches.append(f"{relative_path}: missing or not a regular file")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            mismatches.append(
                f"{relative_path}: expected={expected_hash}, actual={actual_hash}"
            )
    if mismatches:
        raise UpstreamManageError(
            f"{label} 校验失败：" + "; ".join(mismatches)
        )


def _image_matches(source_root: Path, image: Mapping[str, str | None]) -> bool:
    try:
        _verify_image(source_root, image, label="image")
    except UpstreamManageError:
        return False
    return True


def _verify_tree_state(
    manifest: Mapping[str, Any],
    source_root: Path,
    *,
    state: str,
    allow_generated: bool = False,
) -> dict[str, Any]:
    source = _source_section(manifest)
    if state == "pristine":
        hash_key = "pristine_tree_sha256"
        count_key = "pristine_file_count"
    elif state == "patched":
        hash_key = (
            "patched_tree_sha256"
            if "patched_tree_sha256" in source
            else "pristine_tree_sha256"
        )
        count_key = (
            "patched_file_count"
            if "patched_file_count" in source
            else "pristine_file_count"
        )
    else:
        raise UpstreamManageError(f"未知源码状态：{state!r}")

    expected_hash = str(source[hash_key])
    patterns = _validate_generated_patterns(
        source.get("generated_path_patterns", [])
    )
    identity = compute_tree_identity(
        source_root,
        generated_path_patterns=patterns,
        allow_generated=allow_generated,
    )
    if identity.sha256 != expected_hash:
        raise UpstreamManageError(
            f"{state} 源码树 SHA256 不匹配："
            f"expected={expected_hash}, actual={identity.sha256}"
        )
    if count_key in source and identity.file_count != int(source[count_key]):
        raise UpstreamManageError(
            f"{state} 源码树文件数不匹配："
            f"expected={source[count_key]}, actual={identity.file_count}"
        )
    return {
        "tree_sha256": identity.sha256,
        "file_count": identity.file_count,
        "generated_file_count": len(identity.generated_paths),
        "generated_paths": list(identity.generated_paths),
    }


def _extract_archive_safely(
    archive_path: Path,
    destination: Path,
    *,
    archive_root: str,
) -> Path:
    """拒绝路径穿越、链接和特殊文件后解压单根目录归档。"""

    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (OSError, tarfile.TarError) as exc:
        raise UpstreamManageError(f"源码归档无法打开：{exc}") from exc
    with archive:
        members = archive.getmembers()
        if not members:
            raise UpstreamManageError("源码归档为空")
        seen: set[str] = set()
        for member in members:
            normalized = _validate_archive_member(member.name, archive_root)
            if normalized in seen:
                raise UpstreamManageError(f"源码归档包含重复路径：{normalized}")
            seen.add(normalized)
            if not (member.isdir() or member.isfile()):
                raise UpstreamManageError(
                    f"源码归档禁止链接或特殊文件：{member.name}"
                )
        try:
            archive.extractall(destination, members=members)
        except (OSError, tarfile.TarError) as exc:
            raise UpstreamManageError(f"源码归档解压失败：{exc}") from exc
    extracted_root = destination / archive_root
    if not extracted_root.is_dir() or extracted_root.is_symlink():
        raise UpstreamManageError(f"归档根目录缺失：{archive_root}")
    return extracted_root


def _validate_archive_member(name: str, archive_root: str) -> str:
    if "\\" in name:
        raise UpstreamManageError(f"归档路径禁止反斜杠：{name}")
    pure = PurePosixPath(name)
    canonical_name = name[:-1] if name.endswith("/") else name
    if (
        pure.is_absolute()
        or pure.as_posix() != canonical_name
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise UpstreamManageError(f"归档路径不安全：{name}")
    if not pure.parts or pure.parts[0] != archive_root:
        raise UpstreamManageError(
            f"归档只能包含根目录 {archive_root!r}：{name}"
        )
    return pure.as_posix()


def _validate_archive_root(value: str) -> None:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.as_posix() != value
        or pure.parts[0] in ("", ".", "..")
        or "\\" in value
    ):
        raise UpstreamManageError("manifest.source.archive_root 必须是单个安全目录名")


def _validate_image_map(value: object, label: str) -> dict[str, str | None]:
    if not isinstance(value, dict) or not value:
        raise UpstreamManageError(f"{label} 必须是非空 JSON object")
    output: dict[str, str | None] = {}
    for raw_path, expected_hash in value.items():
        if not isinstance(raw_path, str):
            raise UpstreamManageError(f"{label} 的路径键必须是字符串")
        normalized = _validate_member_relative_path(raw_path, label=label)
        if expected_hash is not None:
            expected_hash = _validate_sha256_value(expected_hash, f"{label}.{raw_path}")
        output[normalized] = expected_hash
    return output


def _validate_member_relative_path(value: str, *, label: str) -> str:
    if "\\" in value:
        raise UpstreamManageError(f"{label} 禁止反斜杠：{value}")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise UpstreamManageError(f"{label} 必须是安全相对路径：{value}")
    return pure.as_posix()


def _validate_generated_patterns(value: object) -> tuple[str, ...]:
    """验证只相对源码根生效的精确 generated 路径模式。"""

    if not isinstance(value, (list, tuple)):
        raise UpstreamManageError(
            "manifest.source.generated_path_patterns 必须是 JSON array"
        )
    output: list[str] = []
    for index, pattern in enumerate(value):
        label = f"manifest.source.generated_path_patterns[{index}]"
        if not isinstance(pattern, str) or not pattern:
            raise UpstreamManageError(f"{label} 必须是非空字符串")
        if (
            pattern.startswith("/")
            or pattern.endswith("/")
            or "\\" in pattern
            or "//" in pattern
            or any(part in ("", ".", "..") for part in pattern.split("/"))
        ):
            raise UpstreamManageError(f"{label} 必须是安全的根相对 glob")
        parts = pattern.split("/")
        if "*" in parts[0] or "?" in parts[0]:
            raise UpstreamManageError(f"{label} 的首层目录禁止通配符")
        for part_index, part in enumerate(parts):
            if part != "**":
                continue
            previous = parts[part_index - 1] if part_index > 0 else ""
            if not (
                previous == "install"
                or previous == "__pycache__"
                or previous.startswith("build")
            ):
                raise UpstreamManageError(
                    f"{label} 的 ** 只能位于明确 build/install/__pycache__ 目录下"
                )
        output.append(pattern)
    if len(set(output)) != len(output):
        raise UpstreamManageError("generated_path_patterns 禁止重复")
    return tuple(output)


def _matches_generated_path(path: str, patterns: Sequence[str]) -> bool:
    return bool(_matching_generated_patterns(path, patterns))


def _matching_generated_patterns(
    path: str,
    patterns: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        pattern
        for pattern in patterns
        if re.fullmatch(_generated_pattern_regex(pattern), path) is not None
    )


def _validate_generated_symlink(
    path: Path,
    *,
    source_root: Path,
    relative_path: str,
    patterns: Sequence[str],
) -> None:
    """只接受同一 generated 子树内的相对 SONAME 链接。"""

    link_patterns = set(_matching_generated_patterns(relative_path, patterns))
    if not link_patterns:
        raise UpstreamManageError(
            f"非 generated 白名单路径禁止符号链接：{relative_path}"
        )
    try:
        raw_target = os.readlink(path)
    except OSError as exc:
        raise UpstreamManageError(f"无法读取 generated 符号链接：{path}") from exc
    pure_target = PurePosixPath(raw_target)
    if pure_target.is_absolute():
        raise UpstreamManageError(
            f"generated 符号链接目标必须是相对路径：{relative_path} -> {raw_target}"
        )
    if (
        "\\" in raw_target
        or not pure_target.parts
        or pure_target.as_posix() != raw_target
        or any(part in ("", ".", "..") for part in pure_target.parts)
    ):
        raise UpstreamManageError(
            f"generated 符号链接目标禁止空段、点段或 ..："
            f"{relative_path} -> {raw_target}"
        )
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UpstreamManageError(
            f"generated 符号链接悬空或形成循环：{relative_path} -> {raw_target}"
        ) from exc
    try:
        resolved_relative = resolved.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise UpstreamManageError(
            f"generated 符号链接逃逸源码根：{relative_path} -> {raw_target}"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise UpstreamManageError(
            f"generated 符号链接最终目标必须是普通文件："
            f"{relative_path} -> {resolved_relative}"
        )
    target_patterns = set(
        _matching_generated_patterns(resolved_relative, patterns)
    )
    if not link_patterns.intersection(target_patterns):
        raise UpstreamManageError(
            f"generated 符号链接必须留在同一白名单子树："
            f"{relative_path} -> {resolved_relative}"
        )


def _generated_pattern_regex(pattern: str) -> str:
    """把受限 glob 转为不会让单星号跨目录的正则表达式。"""

    output: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                output.append(".*")
                index += 2
            else:
                output.append("[^/]*")
                index += 1
        elif character == "?":
            output.append("[^/]")
            index += 1
        else:
            output.append(re.escape(character))
            index += 1
    return "".join(output)


def _source_section(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    source = manifest.get("source")
    assert isinstance(source, dict)
    return source


def _required_nonempty_string(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UpstreamManageError(f"{label}.{key} 必须是非空字符串")
    return value


def _required_sha256(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> str:
    if key not in mapping:
        raise UpstreamManageError(f"{label}.{key} 缺失")
    return _validate_sha256_value(mapping[key], f"{label}.{key}")


def _validate_sha256_value(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UpstreamManageError(f"{label} 必须是 64 位小写十六进制 SHA256")
    return value


def _optional_nonnegative_int(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UpstreamManageError(f"{label}.{key} 必须是非负整数")


def _resolve_contained_path(root: Path, value: str, *, label: str) -> Path:
    relative = _validate_member_relative_path(value, label=label)
    path = (root / PurePosixPath(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise UpstreamManageError(f"{label} 逃逸 manifest 目录：{value}") from exc
    return path


def _safe_source_destination(path: Path) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise UpstreamManageError(f"源码目标不能是符号链接：{requested}")
    source_root = requested.resolve()
    if source_root == Path(source_root.anchor):
        raise UpstreamManageError("源码目标不能是文件系统根目录")
    if source_root.exists() and not source_root.is_dir():
        raise UpstreamManageError(f"源码目标已存在且不是目录：{source_root}")
    return source_root


def _existing_source_root(path: Path) -> Path:
    source_root = _safe_source_destination(path)
    if not source_root.is_dir():
        raise UpstreamManageError(f"源码目录不存在：{source_root}")
    return source_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="准备、应用并验证固定版本 PCT Planner 上游源码。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="校验归档并准备最终补丁态源码")
    _add_manifest_argument(prepare)
    prepare.add_argument("--archive", type=Path, required=True, help="本地源码归档")
    prepare.add_argument("--source-root", type=Path, required=True, help="目标源码目录")

    apply_parser = subparsers.add_parser("apply", help="在 pristine 源码树应用补丁")
    _add_manifest_argument(apply_parser)
    apply_parser.add_argument("--source-root", type=Path, required=True, help="已有源码目录")

    verify = subparsers.add_parser("verify-source", help="验证 pristine 或 patched 源码树")
    _add_manifest_argument(verify)
    verify.add_argument("--source-root", type=Path, required=True, help="已有源码目录")
    verify.add_argument(
        "--state",
        choices=("pristine", "patched"),
        default="patched",
        help="期望的源码状态，默认 patched",
    )
    verify.add_argument(
        "--allow-generated",
        action="store_true",
        help="仅忽略 manifest 精确白名单内的构建产物，并在结果中报告",
    )

    build_plan = subparsers.add_parser(
        "build-plan",
        help="生成固定 CPython 3.10/Release 构建 argv，但不执行",
    )
    _add_manifest_argument(build_plan)
    build_plan.add_argument("--source-root", type=Path, required=True, help="patched 源码目录")
    build_plan.add_argument(
        "--build-root",
        type=Path,
        required=True,
        help="源码树外构建目录",
    )
    build_plan.add_argument("--jobs", type=int, default=4, help="并行任务数，默认 4")

    verify_binaries_parser = subparsers.add_parser(
        "verify-binaries",
        help="检查 CPython ABI、ELF、RUNPATH 与 ldd 闭包",
    )
    _add_manifest_argument(verify_binaries_parser)
    verify_binaries_parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="含已构建扩展的 patched 源码目录",
    )
    return parser


def _add_manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="upstream manifest v2")


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令行，并以 JSON 输出可供自动化保存的验证结果。"""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare":
            result = prepare_upstream(
                manifest_path=arguments.manifest,
                archive_path=arguments.archive,
                source_root=arguments.source_root,
            )
        elif arguments.command == "apply":
            result = apply_patches(
                manifest_path=arguments.manifest,
                source_root=arguments.source_root,
            )
        elif arguments.command == "verify-source":
            result = verify_source(
                manifest_path=arguments.manifest,
                source_root=arguments.source_root,
                state=arguments.state,
                allow_generated=arguments.allow_generated,
            )
        elif arguments.command == "build-plan":
            result = generate_build_plan(
                manifest_path=arguments.manifest,
                source_root=arguments.source_root,
                build_root=arguments.build_root,
                jobs=arguments.jobs,
            )
        else:
            result = verify_binaries(
                manifest_path=arguments.manifest,
                source_root=arguments.source_root,
            )
    except UpstreamManageError as exc:
        print(f"PCT_UPSTREAM_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

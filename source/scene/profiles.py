"""可扩展的场景运行 profile 注册表。"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_SCENE_PROFILE_DIR = Path("configs/scenes")


class SceneProfileError(ValueError):
    """场景 profile 不存在或配置不合法。"""


@dataclass(frozen=True)
class SceneUsdAssetBinding:
    """把本地大资产绑定到场景 prim 的 USD composition arc。"""

    name: str
    prim_path: str
    arc_type: str
    environment_variable: str
    fallback_path: str
    package_member: str | None = None

    def resolve(
        self,
        project_root: str | Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> tuple[Path, str]:
        """优先解析环境变量，否则使用 profile 中的兼容回退路径。"""

        values = os.environ if environ is None else environ
        environment_value = str(values.get(self.environment_variable, "")).strip()
        raw_path = environment_value or self.fallback_path
        path = Path(os.path.expandvars(raw_path)).expanduser()
        if not path.is_absolute():
            path = Path(project_root).expanduser().resolve() / path
        return path.resolve(), "environment" if environment_value else "fallback"

    def composed_asset_path(self, resolved_path: Path) -> str:
        """返回可写入 USDA arc 的普通路径或 package member 路径。"""

        asset_path = resolved_path.as_posix()
        if self.package_member:
            asset_path = f"{asset_path}[{self.package_member}]"
        return asset_path


@dataclass(frozen=True)
class SceneProfile:
    """一套场景资产、任务和稳定运行默认值。"""

    name: str
    aliases: tuple[str, ...]
    description: str
    capabilities: tuple[str, ...]
    task_scene_profile: str
    runtime_asset_manifest: str
    defaults: Mapping[str, Any]
    mode_defaults: Mapping[str, Mapping[str, Any]]
    required_assets: tuple[str, ...]
    usd_asset_bindings: tuple[SceneUsdAssetBinding, ...]
    scan_stair_freeze_profile: str | None
    config_path: Path

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def defaults_for_mode(self, mode: str) -> dict[str, Any]:
        """返回基础默认值与指定执行模式默认值的合并副本。"""

        merged = copy.deepcopy(dict(self.defaults))
        merged.update(copy.deepcopy(dict(self.mode_defaults.get(str(mode), {}))))
        return merged

    def supports(self, capability: str) -> bool:
        """判断 profile 是否显式声明某项运行能力。"""

        return _normalize_name(capability) in {
            _normalize_name(value) for value in self.capabilities
        }


@dataclass(frozen=True)
class SceneAssetCheck:
    """场景资产预检结果。"""

    profile: str
    available: tuple[Path, ...]
    missing: tuple[Path, ...]

    @property
    def success(self) -> bool:
        return not self.missing


def discover_scene_profiles(
    project_root: str | Path,
    *,
    config_dir: str | Path = DEFAULT_SCENE_PROFILE_DIR,
) -> dict[str, SceneProfile]:
    """扫描配置目录，并建立名称与别名到 profile 的映射。"""

    root = Path(project_root).expanduser().resolve()
    directory = Path(config_dir).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    if not directory.is_dir():
        raise SceneProfileError(f"场景 profile 目录不存在：{directory}")

    profiles: dict[str, SceneProfile] = {}
    for path in sorted(directory.glob("*.json")):
        profile = _load_profile_file(path)
        for name in profile.all_names:
            key = _normalize_name(name)
            existing = profiles.get(key)
            if existing is not None:
                raise SceneProfileError(
                    f"场景 profile 名称或别名重复：{name!r}，"
                    f"来源为 {existing.config_path} 与 {path}"
                )
            profiles[key] = profile
    if not profiles:
        raise SceneProfileError(f"场景 profile 目录中没有 JSON：{directory}")
    return profiles


def load_scene_profile(
    name: str,
    project_root: str | Path,
    *,
    config_dir: str | Path = DEFAULT_SCENE_PROFILE_DIR,
) -> SceneProfile:
    """按规范名称或别名读取场景 profile。"""

    profiles = discover_scene_profiles(project_root, config_dir=config_dir)
    key = _normalize_name(name)
    profile = profiles.get(key)
    if profile is not None:
        return profile
    available = sorted({item.name for item in profiles.values()})
    raise SceneProfileError(
        f"未知场景 profile：{name!r}；可选值：{', '.join(available)}"
    )


def list_scene_profiles(
    project_root: str | Path,
    *,
    config_dir: str | Path = DEFAULT_SCENE_PROFILE_DIR,
) -> tuple[SceneProfile, ...]:
    """返回去重并按名称排序的场景 profile。"""

    discovered = discover_scene_profiles(project_root, config_dir=config_dir)
    unique = {profile.name: profile for profile in discovered.values()}
    return tuple(unique[name] for name in sorted(unique))


def apply_scene_profile_defaults(
    namespace: Any,
    profile: SceneProfile,
    *,
    mode: str,
) -> dict[str, Any]:
    """只补齐 CLI 中未显式设置的字段，并返回实际应用项。"""

    applied: dict[str, Any] = {}
    for key, value in profile.defaults_for_mode(mode).items():
        if not hasattr(namespace, key):
            raise SceneProfileError(
                f"{profile.config_path} defaults 包含未知 CLI 字段：{key}"
            )
        if getattr(namespace, key) is not None:
            continue
        copied = copy.deepcopy(value)
        setattr(namespace, key, copied)
        applied[key] = copied
    return applied


def check_scene_profile_assets(
    profile: SceneProfile,
    project_root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> SceneAssetCheck:
    """检查 profile 声明的运行必需资产是否存在。"""

    root = Path(project_root).expanduser().resolve()
    available: list[Path] = []
    missing: list[Path] = []
    seen: set[Path] = set()
    for raw_path in profile.required_assets:
        path = Path(os.path.expandvars(raw_path)).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        (available if path.exists() else missing).append(path)
    for binding in profile.usd_asset_bindings:
        path, _source = binding.resolve(root, environ=environ)
        if path in seen:
            continue
        seen.add(path)
        (available if path.exists() else missing).append(path)
    return SceneAssetCheck(
        profile=profile.name,
        available=tuple(available),
        missing=tuple(missing),
    )


def _load_profile_file(path: Path) -> SceneProfile:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneProfileError(f"无法读取场景 profile：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SceneProfileError(f"场景 profile 顶层必须是对象：{path}")
    if payload.get("schema_version") != 1:
        raise SceneProfileError(f"不支持的场景 profile schema_version：{path}")

    name = _required_text(payload, "name", path)
    aliases_raw = payload.get("aliases", [])
    if not isinstance(aliases_raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in aliases_raw
    ):
        raise SceneProfileError(f"aliases 必须是非空字符串列表：{path}")
    capabilities_raw = payload.get("capabilities", [])
    if not isinstance(capabilities_raw, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in capabilities_raw
    ):
        raise SceneProfileError(f"capabilities 必须是非空字符串列表：{path}")
    defaults = payload.get("defaults")
    if not isinstance(defaults, dict):
        raise SceneProfileError(f"defaults 必须是对象：{path}")
    mode_defaults_raw = payload.get("mode_defaults", {})
    if not isinstance(mode_defaults_raw, dict) or any(
        not isinstance(value, dict) for value in mode_defaults_raw.values()
    ):
        raise SceneProfileError(f"mode_defaults 必须是对象映射：{path}")
    required_assets_raw = payload.get("required_assets", [])
    if not isinstance(required_assets_raw, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in required_assets_raw
    ):
        raise SceneProfileError(f"required_assets 必须是路径字符串列表：{path}")
    bindings_raw = payload.get("usd_asset_bindings", [])
    if not isinstance(bindings_raw, list):
        raise SceneProfileError(f"usd_asset_bindings 必须是对象列表：{path}")
    usd_asset_bindings = tuple(
        _parse_usd_asset_binding(value, path, index=index)
        for index, value in enumerate(bindings_raw)
    )
    scan_stair_freeze_profile = _optional_project_relative_path(
        payload,
        "scan_stair_freeze_profile",
        path,
    )

    return SceneProfile(
        name=name,
        aliases=tuple(value.strip() for value in aliases_raw),
        description=str(payload.get("description", "")).strip(),
        capabilities=tuple(value.strip() for value in capabilities_raw),
        task_scene_profile=_required_text(payload, "task_scene_profile", path),
        runtime_asset_manifest=_required_text(
            payload,
            "runtime_asset_manifest",
            path,
        ),
        defaults=MappingProxyType(copy.deepcopy(defaults)),
        mode_defaults=MappingProxyType(
            {
                str(mode): MappingProxyType(copy.deepcopy(values))
                for mode, values in mode_defaults_raw.items()
            }
        ),
        required_assets=tuple(required_assets_raw),
        usd_asset_bindings=usd_asset_bindings,
        scan_stair_freeze_profile=scan_stair_freeze_profile,
        config_path=path.resolve(),
    )


def _required_text(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SceneProfileError(f"{key} 必须是非空字符串：{path}")
    return value.strip()


def _optional_project_relative_path(
    payload: dict[str, Any],
    key: str,
    path: Path,
) -> str | None:
    """读取可选项目内配置路径，拒绝绝对路径和父目录逃逸。"""

    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SceneProfileError(f"{key} 必须是非空项目相对路径：{path}")
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SceneProfileError(f"{key} 必须位于项目目录内：{path}")
    return relative_path.as_posix()


def _parse_usd_asset_binding(
    value: object,
    path: Path,
    *,
    index: int,
) -> SceneUsdAssetBinding:
    """严格解析一个可移植 USD 大资产绑定。"""

    label = f"usd_asset_bindings[{index}]"
    if not isinstance(value, dict):
        raise SceneProfileError(f"{label} 必须是对象：{path}")

    def required(key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise SceneProfileError(f"{label}.{key} 必须是非空字符串：{path}")
        return item.strip()

    arc_type = required("arc_type")
    if arc_type not in {"reference", "payload"}:
        raise SceneProfileError(
            f"{label}.arc_type 只支持 reference 或 payload：{path}"
        )
    prim_path = required("prim_path")
    if not prim_path.startswith("/") or "//" in prim_path:
        raise SceneProfileError(f"{label}.prim_path 必须是绝对 prim path：{path}")
    package_member_raw = value.get("package_member")
    if package_member_raw is not None and (
        not isinstance(package_member_raw, str) or not package_member_raw.strip()
    ):
        raise SceneProfileError(
            f"{label}.package_member 必须是非空字符串或 null：{path}"
        )
    return SceneUsdAssetBinding(
        name=required("name"),
        prim_path=prim_path,
        arc_type=arc_type,
        environment_variable=required("environment_variable"),
        fallback_path=required("fallback_path"),
        package_member=(
            None if package_member_raw is None else package_member_raw.strip()
        ),
    )


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


__all__ = [
    "DEFAULT_SCENE_PROFILE_DIR",
    "SceneAssetCheck",
    "SceneProfile",
    "SceneProfileError",
    "SceneUsdAssetBinding",
    "apply_scene_profile_defaults",
    "check_scene_profile_assets",
    "discover_scene_profiles",
    "list_scene_profiles",
    "load_scene_profile",
]

"""按 scene profile 生成不入库的可移植 USD 资产绑定层。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

from .profiles import SceneProfile, SceneProfileError


_ASSET_TOKEN_PATTERN = re.compile(r"@([^@\r\n]+)@")
_PACKAGE_MEMBER_PATTERN = re.compile(r"^(.*)\[([^\[\]]+)\]$")
_URI_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def materialize_scene_asset_bindings(
    profile: SceneProfile,
    source_scene_usd: str | Path,
    output_usda: str | Path,
    *,
    project_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """复制文本 USDA 并重写 asset arc，保持源 prim 结构不变。"""

    root = Path(project_root).expanduser().resolve()
    source_path = _resolve_path(source_scene_usd, root)
    if not profile.usd_asset_bindings:
        return {
            "materialized": False,
            "profile": profile.name,
            "source_scene_usd": str(source_path),
            "runtime_scene_usd": str(source_path),
            "bindings": [],
        }
    if not source_path.is_file():
        raise SceneProfileError(f"场景 USD 不存在：{source_path}")

    binding_reports: list[dict[str, object]] = []
    binding_targets: dict[tuple[Path, str | None], tuple[str, int]] = {}
    for binding in profile.usd_asset_bindings:
        resolved_path, source = binding.resolve(root, environ=environ)
        if not resolved_path.is_file():
            raise SceneProfileError(
                f"场景资产绑定 {binding.name!r} 不存在：{resolved_path}；"
                f"请设置 {binding.environment_variable} 或修正 profile fallback_path。"
            )
        asset_path = binding.composed_asset_path(resolved_path)
        _validate_asset_path(asset_path, binding_name=binding.name)
        fallback_path = _resolve_path(binding.fallback_path, root)
        target_key = (fallback_path, binding.package_member)
        if target_key in binding_targets:
            raise SceneProfileError(
                f"多个场景资产绑定使用同一 fallback arc：{fallback_path}"
            )
        binding_reports.append(
            {
                "name": binding.name,
                "prim_path": binding.prim_path,
                "arc_type": binding.arc_type,
                "environment_variable": binding.environment_variable,
                "selected_source": source,
                "resolved_path": str(resolved_path),
                "package_member": binding.package_member,
                "composed_asset_path": asset_path,
                "source_fallback_path": str(fallback_path),
                "replacement_count": 0,
            }
        )
        binding_targets[target_key] = (asset_path, len(binding_reports) - 1)

    output_path = Path(output_usda).expanduser()
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    if output_path == source_path:
        raise SceneProfileError("运行时资产绑定副本不能覆盖源场景 USD")
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SceneProfileError(
            f"场景资产绑定当前只支持文本 USDA，实际文件不是 UTF-8：{source_path}"
        ) from exc

    def rewrite_asset_token(match: re.Match[str]) -> str:
        raw_asset_path = match.group(1)
        base_path_text, package_member = _split_package_member(raw_asset_path)
        if _URI_SCHEME_PATTERN.match(base_path_text):
            return match.group(0)
        base_path = Path(base_path_text).expanduser()
        if not base_path.is_absolute():
            base_path = source_path.parent / base_path
        resolved_base_path = base_path.resolve()
        target = binding_targets.get((resolved_base_path, package_member))
        if target is not None:
            rewritten_asset_path, report_index = target
            binding_reports[report_index]["replacement_count"] = int(
                binding_reports[report_index]["replacement_count"]
            ) + 1
        else:
            rewritten_asset_path = resolved_base_path.as_posix()
            if package_member:
                rewritten_asset_path = (
                    f"{rewritten_asset_path}[{package_member}]"
                )
        _validate_asset_path(
            rewritten_asset_path,
            binding_name="source_scene_asset_arc",
        )
        return f"@{rewritten_asset_path}@"

    rewritten_text = _ASSET_TOKEN_PATTERN.sub(rewrite_asset_token, source_text)
    missing_bindings = [
        report["name"]
        for report in binding_reports
        if int(report["replacement_count"]) <= 0
    ]
    if missing_bindings:
        raise SceneProfileError(
            "源场景 USDA 中找不到 profile 声明的 fallback asset arc："
            + ", ".join(str(value) for value in missing_bindings)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rewritten_text, encoding="utf-8")
    return {
        "materialized": True,
        "materialization_mode": "standalone_asset_path_rewrite",
        "profile": profile.name,
        "source_scene_usd": str(source_path),
        "runtime_scene_usd": str(output_path),
        "bindings": binding_reports,
    }


def _resolve_path(raw_path: str | Path, root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _split_package_member(asset_path: str) -> tuple[str, str | None]:
    match = _PACKAGE_MEMBER_PATTERN.fullmatch(asset_path)
    if match is None:
        return asset_path, None
    return match.group(1), match.group(2)


def _validate_asset_path(asset_path: str, *, binding_name: str) -> None:
    if "@" in asset_path or "\n" in asset_path or "\r" in asset_path:
        raise SceneProfileError(
            f"场景资产绑定 {binding_name!r} 的路径包含 USDA 不安全字符"
        )


def write_scene_binding_report(report: Mapping[str, object], path: str | Path) -> Path:
    """保存运行时绑定来源，便于数据集追溯外部大资产。"""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


__all__ = ["materialize_scene_asset_bindings", "write_scene_binding_report"]

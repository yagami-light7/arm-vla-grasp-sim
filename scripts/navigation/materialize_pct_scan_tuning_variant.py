#!/usr/bin/env python3
"""把小型实验配方严格合并为可直接运行的完整 PCT+SCAN YAML。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VariantError(ValueError):
    """表示实验配方不能安全、单义地应用到生产配置。"""


def _load_yaml_mapping(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise VariantError(
            f"无法读取{label} {path}：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise VariantError(f"{label} YAML 顶层必须是对象：{path}")
    return raw, payload


def _resolve_base_path(recipe_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise VariantError("recipe.base_config 必须是非空路径字符串。")
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    project_candidate = (PROJECT_ROOT / raw_path).resolve()
    recipe_candidate = (recipe_path.parent / raw_path).resolve()
    if project_candidate.is_file():
        return project_candidate
    if recipe_candidate.is_file():
        return recipe_candidate
    raise VariantError(
        "recipe.base_config 不存在："
        f"project={project_candidate}, recipe_relative={recipe_candidate}"
    )


def _validate_scalar(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise VariantError(f"override {path} 不允许 NaN 或 Inf。")
    if isinstance(value, (dict, list, tuple, set)):
        raise VariantError(f"override {path} 的叶子必须是标量。")


def _merge_existing_keys(
    target: MutableMapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """只允许覆盖生产 YAML 已存在的叶子，拒绝拼写错误生成静默新参数。"""

    changes: list[dict[str, Any]] = []
    for key, override_value in overrides.items():
        if not isinstance(key, str) or not key:
            raise VariantError("override 所有键都必须是非空字符串。")
        path = (*prefix, key)
        dotted_path = ".".join(path)
        if key not in target:
            raise VariantError(f"override 指向不存在的生产键：{dotted_path}")
        current_value = target[key]
        if isinstance(override_value, Mapping):
            if not isinstance(current_value, MutableMapping):
                raise VariantError(
                    f"override {dotted_path} 是对象，但生产值不是对象。"
                )
            changes.extend(
                _merge_existing_keys(
                    current_value,
                    override_value,
                    prefix=path,
                )
            )
            continue
        if isinstance(current_value, Mapping):
            raise VariantError(
                f"override {dotted_path} 必须继续指定到已有叶子参数。"
            )
        _validate_scalar(override_value, dotted_path)
        if type(current_value) is not type(override_value):
            numeric_pair = (
                not isinstance(current_value, bool)
                and not isinstance(override_value, bool)
                and isinstance(current_value, (int, float))
                and isinstance(override_value, (int, float))
            )
            if not numeric_pair:
                raise VariantError(
                    f"override {dotted_path} 类型变化不安全："
                    f"{type(current_value).__name__}→{type(override_value).__name__}"
                )
        if current_value == override_value:
            raise VariantError(f"override {dotted_path} 与生产值相同，不构成 A/B。")
        target[key] = override_value
        changes.append(
            {
                "path": dotted_path,
                "before": current_value,
                "after": override_value,
            }
        )
    return changes


def materialize_variant(recipe_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """生成完整配置，并返回包含唯一语义差异和哈希的证据。"""

    recipe = Path(recipe_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise VariantError(f"输出文件必须原本不存在：{output}")
    recipe_bytes, recipe_payload = _load_yaml_mapping(recipe, "实验配方")
    if recipe_payload.get("schema") != "pct_scan_tuning_variant_v1":
        raise VariantError("recipe.schema 必须为 pct_scan_tuning_variant_v1。")
    experiment_id = recipe_payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise VariantError("recipe.experiment_id 必须是非空字符串。")
    base_path = _resolve_base_path(recipe, recipe_payload.get("base_config"))
    base_bytes, base_payload = _load_yaml_mapping(base_path, "生产调参文件")
    overrides = recipe_payload.get("overrides")
    if not isinstance(overrides, Mapping) or not overrides:
        raise VariantError("recipe.overrides 必须是非空对象。")
    resolved_payload = copy.deepcopy(base_payload)
    changes = _merge_existing_keys(resolved_payload, overrides)
    expected_count = recipe_payload.get("expected_override_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise VariantError("recipe.expected_override_count 必须是整数。")
    if expected_count <= 0 or len(changes) != expected_count:
        raise VariantError(
            "实验实际覆盖数与配方合同不一致："
            f"expected={expected_count}, actual={len(changes)}"
        )

    header = (
        "# 此文件由 materialize_pct_scan_tuning_variant.py 生成；请修改有中文注释的\n"
        f"# 基础文件 {base_path} 或小型配方 {recipe}，不要直接手改本文件。\n"
    )
    rendered = header + yaml.safe_dump(
        resolved_payload,
        allow_unicode=True,
        sort_keys=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except OSError as exc:
        raise VariantError(f"无法写入完整实验配置 {output}：{exc}") from exc
    output_bytes = output.read_bytes()
    return {
        "schema": "pct_scan_tuning_variant_materialization_v1",
        "experiment_id": experiment_id,
        "recipe_path": os.fspath(recipe),
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "base_path": os.fspath(base_path),
        "base_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "output_path": os.fspath(output),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "semantic_change_count": len(changes),
        "semantic_changes": changes,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；拒绝覆盖已有文件，确保每个 A/B 配置身份唯一。"""

    arguments = _build_parser().parse_args(argv)
    try:
        report = materialize_variant(arguments.recipe, arguments.output)
    except VariantError as exc:
        print(f"FAIL：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

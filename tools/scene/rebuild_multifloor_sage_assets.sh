#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PLY_DIR="source/scene/multifloor/ply"
USDZ_DIR="source/scene/multifloor/usdz"
USD_DIR="source/scene/multifloor/usd"
USDA_DIR="source/scene/multifloor/usda"
CHECK_JSON="outputs/multifloor_ply_check.json"
FINAL_PLY="${PLY_DIR}/3dgs_visual_final_17m_clean.ply"
FINAL_PLY_REPORT="outputs/multifloor_nurec_final_17m_clean_ply.json"
USDZ_PATH="${USDZ_DIR}/multifloor.usdz"
COLLISION_USD_PATH="${USD_DIR}/multifloor_collision.usd"
USDA_PATH="${USDA_DIR}/multifloor.usda"

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] 创建输出目录"
mkdir -p "${USDZ_DIR}" "${USD_DIR}" "${USDA_DIR}" outputs

echo "[STEP 1] 检查 PLY"
python tools/scene/check_multifloor_ply.py --ply-dir "${PLY_DIR}" --output-json "${CHECK_JSON}"
VISUAL_PLY="$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("outputs/multifloor_ply_check.json").read_text())
print(payload["selected_3dgs_visual_ply"])
PY
)"
COLLISION_PLY="$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("outputs/multifloor_ply_check.json").read_text())
print(payload["selected_collision_ply"])
PY
)"
echo "[INFO] visual_ply=${VISUAL_PLY}"
echo "[INFO] collision_ply=${COLLISION_PLY}"

echo "[STEP 2] 生成最终 16M+ clean Gaussian PLY"
python tools/scene/sample_gaussian_ply.py \
  --input-ply "${VISUAL_PLY}" \
  --output-ply "${FINAL_PLY}" \
  --max-points 0 \
  --clip-reference-ply "${COLLISION_PLY}" \
  --clip-margin 5 5 3 \
  --max-scale 0.5 \
  --min-opacity 0.2 \
  --report-output "${FINAL_PLY_REPORT}" \
  --force

echo "[STEP 3] 生成最终 USDZ 视觉层"
rm -f "${USDZ_PATH}"
if python -m threedgrut.export.scripts.ply_to_usd "${FINAL_PLY}" --output_file "${USDZ_PATH}"; then
  echo "[OK] threedgrut 官方入口生成 USDZ 成功"
else
  echo "[WARN] 官方入口失败，改用本项目 export_cameras=False wrapper"
  python tools/scene/ply_to_usdz_no_cameras.py \
    --input-ply "${FINAL_PLY}" \
    --output-usdz "${USDZ_PATH}" \
    --force
fi
rm -f "${FINAL_PLY}"

echo "[STEP 4] 生成 collision USD"
python tools/scene/build_multifloor_collision_usd.py \
  --input-ply "${COLLISION_PLY}" \
  --output-usd "${COLLISION_USD_PATH}" \
  --force

echo "[STEP 5] 删除旧诊断 USDA"
rm -f \
  "${USDA_DIR}/multifloor_nurec.usda" \
  "${USDA_DIR}/multifloor_collision_only.usda" \
  "${USDA_DIR}/multifloor_nurec_debug_clip_100k.usda" \
  "${USDA_DIR}/multifloor_nurec_clip_2m.usda" \
  "${USDA_DIR}/multifloor_nurec_clip_12m_clean.usda"

echo "[STEP 6] 生成最终高质量 NuRec 主场景"
python tools/scene/build_multifloor_usda.py \
  --usdz "${USDZ_PATH}" \
  --collision-usd "${COLLISION_USD_PATH}" \
  --output-usda "${USDA_PATH}" \
  --visual-mode nurec \
  --force

echo "[STEP 7] 验证最终 USD 资产"
python tools/scene/validate_multifloor_usd_assets.py \
  --usdz "${USDZ_PATH}" \
  --collision-usd "${COLLISION_USD_PATH}" \
  --usda "${USDA_PATH}" \
  --visual-mode nurec

echo "[DONE] Isaac Sim 打开: ${ROOT_DIR}/${USDA_PATH}"

# Go2-X5 PCT 多楼层 Locomotion Checkpoint

本目录用于整理 `pct_multifloor` locomotion policy。运行 checkpoint
作为普通 Git 文件随仓库发布，不依赖 `Rough/` 或软链接。

当前本地文件约定：

```text
checkpoints/go2_x5/pct_multifloor/model_26000.pt
checkpoints/go2_x5/pct_multifloor/exported/policy.pt
checkpoints/go2_x5/pct_multifloor/exported/policy.onnx
checkpoints/go2_x5/pct_multifloor/env.yaml
checkpoints/go2_x5/pct_multifloor/agent.yaml
checkpoints/go2_x5/pct_multifloor/training/
```

运行 pipeline 时推荐显式传入：

```bash
--locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
--policy-profile pct_multifloor
```

`model_26000.pt` 随仓库发布；`exported/` 和 `training/` 保留为本地训练产物，
不进入 Git。

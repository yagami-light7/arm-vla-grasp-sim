# Go2-X5 PCT 多楼层 Locomotion Checkpoint

本目录用于整理本地 `pct_multifloor` locomotion policy。大模型文件不提交到
git，当前通过相对软链接指向 `Rough/`。

当前本地文件约定：

```text
checkpoints/go2_x5/pct_multifloor/model_26000.pt
checkpoints/go2_x5/pct_multifloor/exported/policy.pt
checkpoints/go2_x5/pct_multifloor/exported/policy.onnx
checkpoints/go2_x5/pct_multifloor/env.yaml
checkpoints/go2_x5/pct_multifloor/agent.yaml
```

运行 pipeline 时推荐显式传入：

```bash
--locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
--policy-profile pct_multifloor
```

`model_26000.pt`、`policy.pt` 和 `policy.onnx` 受 `.gitignore` 保护，不应提交。

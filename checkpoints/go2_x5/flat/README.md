# Go2-X5 Locomotion Checkpoint

This directory stores the local Go2-X5 RSL-RL locomotion checkpoint used by the
navigation runner.

Migrated source:

```text
/home/light/workspace/DWA/flat/model_8500.pt
```

Expected local file:

```text
checkpoints/go2_x5/flat/model_8500.pt
```

The checkpoint is intentionally ignored by git via `*.pt`. Keep model weights
out of commits and pass an explicit `--checkpoint` path if you want to use a
different policy.

Current SHA-256:

```text
ee09cb3f19d231fe4aebb242b353a36c7d3e17b15f1d5b552a4070087965948e
```

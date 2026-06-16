"""Isaac Sim 启动前的轻量兼容补丁。"""

from __future__ import annotations


def patch_numpy_for_isaacsim() -> dict[str, object]:
    """补齐 Isaac Sim 5.1 仍会导入的旧 NumPy 符号。

    当前环境里 NumPy 2.4 已不再从 ``numpy.lib.stride_tricks`` 暴露
    ``broadcast_to``，但 Isaac Sim 的 pip_prebundle 旧模块仍会按该路径
    导入。这里在 AppLauncher 启动前恢复这个别名，避免 Kit extension
    startup 阶段混用两套 NumPy 路径后直接失败。
    """

    import numpy as np
    import numpy.lib.stride_tricks as stride_tricks

    patched = False
    if not hasattr(stride_tricks, "broadcast_to") and hasattr(np, "broadcast_to"):
        stride_tricks.broadcast_to = np.broadcast_to
        patched = True
    return {
        "numpy_version": getattr(np, "__version__", "unknown"),
        "numpy_file": getattr(np, "__file__", "unknown"),
        "patched_broadcast_to": patched,
        "has_broadcast_to": hasattr(stride_tricks, "broadcast_to"),
    }

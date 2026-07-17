"""场景资产工具。"""

from source.scene.gaussian_splat_renderer import (
    CameraConfig,
    GaussianSplatLoadConfig,
    RenderConfig,
    auto_overview_camera,
    camera_intrinsics,
    load_gaussian_splats,
    look_at_view_matrix,
)
from source.scene.profiles import (
    SceneAssetCheck,
    SceneProfile,
    SceneProfileError,
    SceneUsdAssetBinding,
    apply_scene_profile_defaults,
    check_scene_profile_assets,
    list_scene_profiles,
    load_scene_profile,
)
from source.scene.runtime_assets import (
    materialize_scene_asset_bindings,
    write_scene_binding_report,
)

__all__ = [
    "CameraConfig",
    "GaussianSplatLoadConfig",
    "RenderConfig",
    "SceneAssetCheck",
    "SceneProfile",
    "SceneProfileError",
    "SceneUsdAssetBinding",
    "apply_scene_profile_defaults",
    "auto_overview_camera",
    "camera_intrinsics",
    "check_scene_profile_assets",
    "list_scene_profiles",
    "load_gaussian_splats",
    "load_scene_profile",
    "look_at_view_matrix",
    "materialize_scene_asset_bindings",
    "write_scene_binding_report",
]

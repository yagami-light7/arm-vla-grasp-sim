"""Object collision-visual filtering regression tests."""

from __future__ import annotations

import unittest

from source.simulation.visibility_patch import (
    _is_collision_mesh_candidate,
    hide_visual_prims_by_keywords,
)


class _FakeAttribute:
    def __init__(self, value: object, *, valid: bool = True) -> None:
        self._value = value
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid

    def Get(self) -> object:
        return self._value

    def __bool__(self) -> bool:
        return self._valid


class _FakeUsdGeom:
    class Mesh:
        pass


class _FakePrim:
    def __init__(
        self,
        *,
        type_name: str = "Mesh",
        schemas: tuple[str, ...] = (),
        collision_enabled: object | None = None,
    ) -> None:
        self._type_name = type_name
        self._schemas = schemas
        self._collision_enabled = collision_enabled

    def IsA(self, schema: object) -> bool:
        return self._type_name == "Mesh" and schema is _FakeUsdGeom.Mesh

    def GetTypeName(self) -> str:
        return self._type_name

    def GetAppliedSchemas(self) -> tuple[str, ...]:
        return self._schemas

    def GetAttribute(self, name: str) -> _FakeAttribute:
        if name == "physics:collisionEnabled" and self._collision_enabled is not None:
            return _FakeAttribute(self._collision_enabled)
        return _FakeAttribute(None, valid=False)


class CollisionMeshCandidateTests(unittest.TestCase):
    def test_collision_schema_mesh_is_candidate(self) -> None:
        prim = _FakePrim(schemas=("PhysicsCollisionAPI",))

        self.assertTrue(
            _is_collision_mesh_candidate(prim, UsdGeom=_FakeUsdGeom)
        )

    def test_collision_attribute_mesh_is_compatibility_candidate(self) -> None:
        prim = _FakePrim(collision_enabled=True)

        self.assertTrue(
            _is_collision_mesh_candidate(prim, UsdGeom=_FakeUsdGeom)
        )

    def test_visual_mesh_below_collision_named_parent_is_not_candidate(self) -> None:
        prim = _FakePrim(schemas=("MaterialBindingAPI",))

        self.assertFalse(
            _is_collision_mesh_candidate(prim, UsdGeom=_FakeUsdGeom)
        )


class ComposedAppleVisibilityTests(unittest.TestCase):
    def test_multifloor_filter_keeps_visual_and_positive_bbox(self) -> None:
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest("当前 Python 环境没有 OpenUSD pxr")

        stage = Usd.Stage.Open(
            "source/scene/multifloor/usda/multifloor.usda"
        )
        self.assertIsNotNone(stage)
        root = stage.GetPrimAtPath("/World/apple_01")
        collision = stage.GetPrimAtPath(
            "/World/apple_01/Apple_M_Apple_0/Apple_M_Apple_0"
        )
        visual = stage.GetPrimAtPath(
            "/World/apple_01/Apple_M_Apple_0/visual"
        )

        report = hide_visual_prims_by_keywords(stage=stage, logger=lambda _msg: None)

        self.assertEqual(
            [
                item["prim_path"]
                for item in report["hidden_prims"]
                if "apple_01" in item["prim_path"]
            ],
            ["/World/apple_01/Apple_M_Apple_0/Apple_M_Apple_0"],
        )
        self.assertEqual(
            UsdGeom.Imageable(collision).ComputeVisibility(),
            UsdGeom.Tokens.invisible,
        )
        self.assertEqual(
            UsdGeom.Imageable(visual).ComputeVisibility(),
            UsdGeom.Tokens.inherited,
        )
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        box = cache.ComputeWorldBound(root).ComputeAlignedBox()
        size = box.GetMax() - box.GetMin()
        self.assertTrue(all(float(axis) > 0.0 for axis in size))


if __name__ == "__main__":
    unittest.main()

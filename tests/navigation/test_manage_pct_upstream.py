from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

import scripts.navigation.manage_pct_upstream as manage_module
from scripts.navigation.manage_pct_upstream import (
    UpstreamManageError,
    apply_patches,
    compute_tree_identity,
    file_sha256,
    generate_build_plan,
    load_manifest,
    main,
    prepare_upstream,
    verify_binaries,
    verify_source,
)


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _write_archive(source_root: Path, archive_path: Path, archive_root: str) -> None:
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.add(source_root, arcname=archive_root)


def _patch_text() -> str:
    return """\
diff --git a/src/core.txt b/src/core.txt
--- a/src/core.txt
+++ b/src/core.txt
@@ -1 +1 @@
-before
+after
"""


def _create_fixture(tmp_path: Path) -> dict[str, Path]:
    """创建不依赖网络、Git 或真实 PCT 大仓的最小可信供应链。"""

    archive_root = "PCT_planner-demo"
    pristine = tmp_path / "pristine"
    (pristine / "src").mkdir(parents=True)
    (pristine / "planner/lib/3rdparty/gtsam-4.1.1").mkdir(parents=True)
    (pristine / "planner/lib/3rdparty/osqp").mkdir(parents=True)
    (pristine / "src/core.txt").write_text("before\n", encoding="utf-8")
    (pristine / "LICENSE").write_text("demo license\n", encoding="utf-8")

    patched = tmp_path / "patched"
    shutil.copytree(pristine, patched)
    (patched / "src/core.txt").write_text("after\n", encoding="utf-8")

    archive_path = tmp_path / "source.tar.gz"
    _write_archive(pristine, archive_path, archive_root)

    manifest_dir = tmp_path / "contract"
    patch_dir = manifest_dir / "patches"
    patch_dir.mkdir(parents=True)
    patch_path = patch_dir / "0001-core.patch"
    patch_path.write_text(_patch_text(), encoding="utf-8")

    pristine_identity = compute_tree_identity(pristine)
    patched_identity = compute_tree_identity(patched)
    manifest = {
        "schema_version": 2,
        "source": {
            "repository": "https://example.invalid/PCT_planner",
            "commit": "demo",
            "archive_root": archive_root,
            "archive_sha256": file_sha256(archive_path),
            "pristine_tree_sha256": pristine_identity.sha256,
            "pristine_file_count": pristine_identity.file_count,
            "patched_tree_sha256": patched_identity.sha256,
            "patched_file_count": patched_identity.file_count,
            "generated_path_patterns": [
                "build/**",
                "planner/lib/*.so",
                "planner/lib/*.cpython-310-x86_64-linux-gnu.so",
            ],
        },
        "patches": [
            {
                "path": "patches/0001-core.patch",
                "sha256": file_sha256(patch_path),
                "strip": 1,
                "preimage_sha256": {
                    "src/core.txt": _bytes_sha256(b"before\n"),
                },
                "postimage_sha256": {
                    "src/core.txt": _bytes_sha256(b"after\n"),
                },
            }
        ],
        "build": {
            "python_executable": "/usr/bin/python3",
            "required_soabi": "cpython-310-x86_64-linux-gnu",
            "build_type": "Release",
            "cmake_policy_minimum": "3.5",
            "gtsam_build_with_march_native": False,
            "python_include_dir": "/usr/include/python3.10",
            "python_library": "/usr/lib/x86_64-linux-gnu/libpython3.10.so",
        },
        "runtime": {
            "allowed_runpaths": [
                "$ORIGIN",
                "$ORIGIN/3rdparty/gtsam-4.1.1/install/lib",
                "$ORIGIN/3rdparty/osqp/install/lib",
            ]
        },
    }
    manifest_path = manifest_dir / "PCT_PLANNER_SOURCE.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "archive": archive_path,
        "manifest": manifest_path,
        "patch": patch_path,
        "pristine": pristine,
        "patched": patched,
    }


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_prepare_verify_and_repeat_are_idempotent(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"

    first = prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )

    assert first["status"] == "prepared"
    assert (destination / "src/core.txt").read_text(encoding="utf-8") == "after\n"
    verified = verify_source(
        manifest_path=fixture["manifest"],
        source_root=destination,
    )
    assert verified["state"] == "patched"
    assert verified["patch_count"] == 1

    repeated_prepare = prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )
    repeated_apply = apply_patches(
        manifest_path=fixture["manifest"],
        source_root=destination,
    )

    assert repeated_prepare["status"] == "already_prepared"
    assert repeated_apply["status"] == "already_applied"


def test_apply_accepts_only_exact_pristine_tree(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "source"
    shutil.copytree(fixture["pristine"], destination)

    result = apply_patches(
        manifest_path=fixture["manifest"],
        source_root=destination,
    )

    assert result["status"] == "applied"
    assert (destination / "src/core.txt").read_text(encoding="utf-8") == "after\n"

    drifted = tmp_path / "drifted"
    shutil.copytree(fixture["pristine"], drifted)
    (drifted / "untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(UpstreamManageError, match="pristine 源码树 SHA256"):
        apply_patches(
            manifest_path=fixture["manifest"],
            source_root=drifted,
        )


def test_prepare_rejects_archive_hash_mismatch_without_installing(
    tmp_path: Path,
) -> None:
    fixture = _create_fixture(tmp_path)
    manifest = _read_manifest(fixture["manifest"])
    manifest["source"]["archive_sha256"] = "0" * 64
    _write_manifest(fixture["manifest"], manifest)
    destination = tmp_path / "external/PCT_planner"

    with pytest.raises(UpstreamManageError, match="源码归档 SHA256 不匹配"):
        prepare_upstream(
            manifest_path=fixture["manifest"],
            archive_path=fixture["archive"],
            source_root=destination,
        )

    assert not destination.exists()


def test_prepare_rejects_patch_hash_mismatch_without_installing(
    tmp_path: Path,
) -> None:
    fixture = _create_fixture(tmp_path)
    manifest = _read_manifest(fixture["manifest"])
    manifest["patches"][0]["sha256"] = "0" * 64
    _write_manifest(fixture["manifest"], manifest)
    destination = tmp_path / "external/PCT_planner"

    with pytest.raises(UpstreamManageError, match="补丁 SHA256 不匹配"):
        prepare_upstream(
            manifest_path=fixture["manifest"],
            archive_path=fixture["archive"],
            source_root=destination,
        )

    assert not destination.exists()


def test_prepare_rejects_wrong_postimage_and_keeps_destination_absent(
    tmp_path: Path,
) -> None:
    fixture = _create_fixture(tmp_path)
    manifest = _read_manifest(fixture["manifest"])
    manifest["patches"][0]["postimage_sha256"]["src/core.txt"] = "0" * 64
    _write_manifest(fixture["manifest"], manifest)
    destination = tmp_path / "external/PCT_planner"

    with pytest.raises(UpstreamManageError, match="postimage 校验失败"):
        prepare_upstream(
            manifest_path=fixture["manifest"],
            archive_path=fixture["archive"],
            source_root=destination,
        )

    assert not destination.exists()


def test_manifest_rejects_patch_images_that_do_not_cover_touched_files(
    tmp_path: Path,
) -> None:
    fixture = _create_fixture(tmp_path)
    manifest = _read_manifest(fixture["manifest"])
    before = _bytes_sha256(b"demo license\n")
    manifest["patches"][0]["preimage_sha256"] = {"LICENSE": before}
    manifest["patches"][0]["postimage_sha256"] = {"LICENSE": before}
    _write_manifest(fixture["manifest"], manifest)

    with pytest.raises(UpstreamManageError, match="补丁触及路径不一致"):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=fixture["patched"],
        )


def test_prepare_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        root = tarfile.TarInfo("PCT_planner-demo")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        payload = b"escape\n"
        member = tarfile.TarInfo("PCT_planner-demo/../../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "schema_version": 2,
            "source": {
                "archive_root": "PCT_planner-demo",
                "archive_sha256": file_sha256(archive_path),
                "pristine_tree_sha256": "0" * 64,
            },
            "patches": [],
        },
    )

    with pytest.raises(UpstreamManageError, match="归档路径不安全"):
        prepare_upstream(
            manifest_path=manifest_path,
            archive_path=archive_path,
            source_root=tmp_path / "external/PCT_planner",
        )
    assert not (tmp_path / "escape.txt").exists()


def test_manifest_v1_is_rejected_explicitly(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, {"schema_version": 1})

    with pytest.raises(UpstreamManageError, match="schema_version=2"):
        load_manifest(manifest_path)


def test_manifest_rejects_generated_glob_covering_source_tree(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    manifest = _read_manifest(fixture["manifest"])
    manifest["source"]["generated_path_patterns"] = ["src/**"]
    _write_manifest(fixture["manifest"], manifest)

    with pytest.raises(UpstreamManageError, match="明确 build/install"):
        load_manifest(fixture["manifest"])


def test_allow_generated_ignores_only_manifest_whitelist(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"
    prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )
    generated = destination / "build/subdir/core.o"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"object")

    with pytest.raises(UpstreamManageError, match="patched 源码树 SHA256"):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=destination,
        )

    report = verify_source(
        manifest_path=fixture["manifest"],
        source_root=destination,
        allow_generated=True,
    )
    assert report["generated_file_count"] == 1
    assert report["generated_paths"] == ["build/subdir/core.o"]

    (destination / "LICENSE").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(UpstreamManageError, match="patched 源码树 SHA256"):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=destination,
            allow_generated=True,
        )

    (destination / "LICENSE").write_text("demo license\n", encoding="utf-8")
    (destination / "src/injected.cc").write_text("injected\n", encoding="utf-8")
    with pytest.raises(UpstreamManageError, match="patched 源码树 SHA256"):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=destination,
            allow_generated=True,
        )


def test_allow_generated_accepts_relative_soname_symlink_chain(
    tmp_path: Path,
) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"
    prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )
    library_dir = destination / "build/lib"
    library_dir.mkdir(parents=True)
    (library_dir / "libdemo.so.4.1.1").write_bytes(b"shared library")
    (library_dir / "libdemo.so.4").symlink_to("libdemo.so.4.1.1")
    (library_dir / "libdemo.so").symlink_to("libdemo.so.4")

    report = verify_source(
        manifest_path=fixture["manifest"],
        source_root=destination,
        allow_generated=True,
    )

    assert report["generated_file_count"] == 3
    assert report["generated_paths"] == [
        "build/lib/libdemo.so",
        "build/lib/libdemo.so.4",
        "build/lib/libdemo.so.4.1.1",
    ]


def test_allow_generated_rejects_absolute_symlink_target(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"
    prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )
    library_dir = destination / "build/lib"
    library_dir.mkdir(parents=True)
    real = library_dir / "libdemo.so.1"
    real.write_bytes(b"shared library")
    (library_dir / "libdemo.so").symlink_to(real.resolve())

    with pytest.raises(UpstreamManageError, match="必须是相对路径"):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=destination,
            allow_generated=True,
        )


def test_allow_generated_rejects_parent_segment_in_symlink(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"
    prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )
    library_dir = destination / "build"
    library_dir.mkdir(parents=True)
    (library_dir / "bad.so").symlink_to("../LICENSE")

    with pytest.raises(UpstreamManageError, match=r"禁止.*\.\."):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=destination,
            allow_generated=True,
        )


def test_generated_symlink_resolution_cannot_escape_source_root(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    library_dir = source_root / "build/lib"
    library_dir.mkdir(parents=True)
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    (library_dir / "bridge.so").symlink_to(outside.resolve())
    link = library_dir / "libdemo.so"
    link.symlink_to("bridge.so")

    with pytest.raises(UpstreamManageError, match="逃逸源码根"):
        manage_module._validate_generated_symlink(
            link,
            source_root=source_root.resolve(),
            relative_path="build/lib/libdemo.so",
            patterns=("build/**",),
        )


def test_allow_generated_rejects_dangling_symlink(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"
    prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )
    library_dir = destination / "build/lib"
    library_dir.mkdir(parents=True)
    (library_dir / "missing.so").symlink_to("missing.so.1")

    with pytest.raises(UpstreamManageError, match="悬空或形成循环"):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=destination,
            allow_generated=True,
        )


def test_allow_generated_rejects_non_generated_symlink(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"
    prepare_upstream(
        manifest_path=fixture["manifest"],
        archive_path=fixture["archive"],
        source_root=destination,
    )
    (destination / "src/core-link.txt").symlink_to("core.txt")

    with pytest.raises(UpstreamManageError, match="非 generated 白名单"):
        verify_source(
            manifest_path=fixture["manifest"],
            source_root=destination,
            allow_generated=True,
        )


def test_build_plan_is_fixed_and_does_not_execute_commands(tmp_path: Path) -> None:
    fixture = _create_fixture(tmp_path)
    source_root = fixture["patched"]
    build_root = tmp_path / "out-of-source-build"

    plan = generate_build_plan(
        manifest_path=fixture["manifest"],
        source_root=source_root,
        build_root=build_root,
        jobs=3,
    )

    assert plan["status"] == "build_plan_only"
    assert plan["executes_commands"] is False
    assert plan["python_executable"] == "/usr/bin/python3"
    assert plan["required_soabi"] == "cpython-310-x86_64-linux-gnu"
    assert plan["build_type"] == "Release"
    assert plan["cmake_policy_minimum"] == "3.5"
    assert plan["gtsam_build_with_march_native"] is False
    flattened = [argument for command in plan["commands"] for argument in command]
    assert "-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF" in flattened
    assert "-DPYTHON_EXECUTABLE=/usr/bin/python3" in flattened
    assert "-DCMAKE_BUILD_TYPE=Release" in flattened
    assert "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" in flattened
    assert not build_root.exists()

    with pytest.raises(UpstreamManageError, match="source root 之外"):
        generate_build_plan(
            manifest_path=fixture["manifest"],
            source_root=source_root,
            build_root=source_root / "build",
        )


def _create_dummy_binaries(source_root: Path) -> None:
    library_root = source_root / "planner/lib"
    names = [
        "a_star.cpython-310-x86_64-linux-gnu.so",
        "traj_opt.cpython-310-x86_64-linux-gnu.so",
        "ele_planner.cpython-310-x86_64-linux-gnu.so",
        "py_map_manager.cpython-310-x86_64-linux-gnu.so",
        "liba_star_search.so",
        "libcommon_smoothing.so",
        "libele_planner_lib.so",
        "libgpmp_optimizer.so",
        "libmap_manager.so",
    ]
    for name in names:
        (library_root / name).write_bytes(b"mock elf")


def test_verify_binaries_checks_elf_runpath_soabi_and_ldd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _create_fixture(tmp_path)
    source_root = fixture["patched"]
    _create_dummy_binaries(source_root)
    allowed = (
        "$ORIGIN:$ORIGIN/3rdparty/gtsam-4.1.1/install/lib:"
        "$ORIGIN/3rdparty/osqp/install/lib"
    )

    def fake_tool(arguments: tuple[str, ...]) -> str:
        if arguments[:2] == ("readelf", "-h"):
            return "Class: ELF64\nMachine: Advanced Micro Devices X86-64\n"
        if arguments[:2] == ("readelf", "-d"):
            return f"0x0000001d (RUNPATH) Library runpath: [{allowed}]\n"
        assert arguments[0] == "ldd"
        return "linux-vdso.so.1 =>  (0x0000)\nlibc.so.6 => /lib/libc.so.6\n"

    monkeypatch.setattr(manage_module, "_run_readonly_tool", fake_tool)
    report = verify_binaries(
        manifest_path=fixture["manifest"],
        source_root=source_root,
    )
    assert report["status"] == "verified"
    assert report["binary_count"] == 9
    assert report["source"]["generated_file_count"] == 9


def test_verify_binaries_rejects_absolute_workspace_runpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _create_fixture(tmp_path)
    source_root = fixture["patched"]
    _create_dummy_binaries(source_root)

    def fake_tool(arguments: tuple[str, ...]) -> str:
        if arguments[:2] == ("readelf", "-h"):
            return "Class: ELF64\nMachine: Advanced Micro Devices X86-64\n"
        if arguments[:2] == ("readelf", "-d"):
            return "0x0000001d (RUNPATH) Library runpath: [$ORIGIN:/mnt/bad]\n"
        return "libc.so.6 => /lib/libc.so.6\n"

    monkeypatch.setattr(manage_module, "_run_readonly_tool", fake_tool)
    with pytest.raises(UpstreamManageError, match="禁止绝对路径"):
        verify_binaries(
            manifest_path=fixture["manifest"],
            source_root=source_root,
        )


def test_readonly_binary_tools_ignore_host_library_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        arguments: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured["arguments"] = arguments
        captured["env"] = env
        assert check is False
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(arguments, 0, stdout="ok\n", stderr="")

    monkeypatch.setenv("LD_LIBRARY_PATH", "/mnt/old-build")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setattr(manage_module.shutil, "which", lambda _name: "/usr/bin/ldd")
    monkeypatch.setattr(manage_module.subprocess, "run", fake_run)

    assert manage_module._run_readonly_tool(("ldd", "/tmp/demo.so")) == "ok\n"
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "LD_LIBRARY_PATH" not in environment
    assert "LD_PRELOAD" not in environment
    assert environment["LC_ALL"] == "C"


def test_cli_emits_json_and_returns_nonzero_for_contract_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _create_fixture(tmp_path)
    destination = tmp_path / "external/PCT_planner"

    success = main(
        [
            "prepare",
            "--manifest",
            str(fixture["manifest"]),
            "--archive",
            str(fixture["archive"]),
            "--source-root",
            str(destination),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert success == 0
    assert output["status"] == "prepared"

    (destination / "src/core.txt").write_text("tampered\n", encoding="utf-8")
    failure = main(
        [
            "verify-source",
            "--manifest",
            str(fixture["manifest"]),
            "--source-root",
            str(destination),
        ]
    )
    captured = capsys.readouterr()
    assert failure == 2
    assert "PCT_UPSTREAM_ERROR" in captured.err

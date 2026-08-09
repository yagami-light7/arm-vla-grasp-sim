from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.navigation.adapters.moe_cts_policy_adapter import (
    MoeCtsPolicyAdapter,
    MoeCtsPolicyContract,
    MoeCtsPolicyContractError,
    MoeCtsPolicyInferenceError,
    MoeCtsPolicyLoadError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOE_CTS_ENTRYPOINT = (
    PROJECT_ROOT / "scripts/navigation/play_go2_moe_cts_keyboard.py"
)


def _contains_pct_goal_publication(node: ast.AST) -> bool:
    """
    @brief 判断语法树节点是否包含 PCT goal 发布调用
    @param node 待检查的 Python 语法树节点
    @return 包含 publish_pct_goal 调用时为真
    """

    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "publish_pct_goal"
        for candidate in ast.walk(node)
    )


def test_multifloor_execution_keeps_pct_goal_publication_enabled() -> None:
    """执行模式与只规划模式必须共享同一 multifloor PCT 目标发布逻辑。"""

    tree = ast.parse(MOE_CTS_ENTRYPOINT.read_text(encoding="utf-8"))
    conditional_nodes = [
        node for node in ast.walk(tree) if isinstance(node, ast.If)
    ]
    planning_only_blocks = [
        node
        for node in conditional_nodes
        if "navigation_planning_only" in ast.unparse(node.test)
    ]
    multifloor_blocks = [
        node
        for node in conditional_nodes
        if "terrain_mode" in ast.unparse(node.test)
        and "multifloor" in ast.unparse(node.test)
    ]

    assert planning_only_blocks
    assert not any(
        _contains_pct_goal_publication(node)
        for node in planning_only_blocks
    )
    assert any(
        _contains_pct_goal_publication(node)
        for node in multifloor_blocks
    )


def test_execution_stops_completed_path_freeze_heartbeat() -> None:
    """本轮 GOAL_REACHED 后不得继续向 SCAN 发送旧 Path 心跳。"""

    tree = ast.parse(MOE_CTS_ENTRYPOINT.read_text(encoding="utf-8"))
    call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    keyword_values = {
        keyword.arg: ast.unparse(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg is not None
    }

    assert "poll_goal_reached" in call_names
    assert "enable_goal_reached_subscription" in keyword_values
    assert "command_subscription_enabled" in keyword_values[
        "enable_goal_reached_subscription"
    ]
    assert "goal_reached_path_stamp_ns" in MOE_CTS_ENTRYPOINT.read_text(
        encoding="utf-8"
    )


class _FakeScalar:
    def __init__(self, value: object) -> None:
        self._value = value

    def item(self) -> object:
        return self._value


class _FakeFiniteResult:
    def __init__(self, value: bool) -> None:
        self._value = value

    def all(self) -> _FakeScalar:
        return _FakeScalar(self._value)


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        finite: bool = True,
        nonzero_count: int = 0,
    ) -> None:
        self.shape = shape
        self.finite = finite
        self.nonzero_count = nonzero_count


class _FakePolicy:
    def __init__(
        self,
        *,
        history_shape: tuple[int, ...] = (1, 450),
        action_shape: tuple[int, ...] = (1, 12),
        action_finite: bool = True,
        reset_clears_history: bool = True,
        forward_error: Exception | None = None,
    ) -> None:
        self.obs_history = _FakeTensor(history_shape, nonzero_count=7)
        self.action_shape = action_shape
        self.action_finite = action_finite
        self.reset_clears_history = reset_clears_history
        self.forward_error = forward_error
        self.eval_count = 0
        self.reset_count = 0
        self.forward_inputs: list[object] = []

    def eval(self) -> _FakePolicy:
        self.eval_count += 1
        return self

    def reset(self) -> None:
        self.reset_count += 1
        if self.reset_clears_history:
            self.obs_history.nonzero_count = 0

    def __call__(self, observation: object) -> _FakeTensor:
        self.forward_inputs.append(observation)
        if self.forward_error is not None:
            raise self.forward_error
        self.obs_history.nonzero_count = 1
        return _FakeTensor(self.action_shape, finite=self.action_finite)


class _FakeTorch:
    def __init__(self, policy: object, *, load_error: Exception | None = None) -> None:
        self._policy = policy
        self._load_error = load_error
        self.load_calls: list[tuple[str, object]] = []
        self.inference_context_count = 0
        self.jit = SimpleNamespace(load=self._load)

    def _load(self, path: str, *, map_location: object) -> object:
        self.load_calls.append((path, map_location))
        if self._load_error is not None:
            raise self._load_error
        return self._policy

    def isfinite(self, value: _FakeTensor) -> _FakeFiniteResult:
        return _FakeFiniteResult(value.finite)

    def count_nonzero(self, value: _FakeTensor) -> _FakeScalar:
        return _FakeScalar(value.nonzero_count)

    def inference_mode(self):
        owner = self

        class _Context:
            def __enter__(self) -> None:
                owner.inference_context_count += 1

            def __exit__(self, *_args: object) -> None:
                return None

        return _Context()


@pytest.fixture
def policy_path(tmp_path: Path) -> Path:
    path = tmp_path / "policy.pt"
    path.write_bytes(b"fake torchscript")
    return path


def _adapter(
    policy_path: Path,
    policy: _FakePolicy | None = None,
) -> tuple[MoeCtsPolicyAdapter, _FakePolicy, _FakeTorch]:
    selected_policy = policy or _FakePolicy()
    fake_torch = _FakeTorch(selected_policy)
    adapter = MoeCtsPolicyAdapter(
        policy_path,
        device="cuda:0",
        torch_module=fake_torch,
    )
    return adapter, selected_policy, fake_torch


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_group_name": ""},
        {"batch_size": 0},
        {"single_observation_dim": True},
        {"history_dim": -1},
        {"action_dim": 0},
    ],
)
def test_contract_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MoeCtsPolicyContract(**kwargs)


def test_loads_with_map_location_and_clears_history(policy_path: Path) -> None:
    adapter, policy, fake_torch = _adapter(policy_path)

    assert fake_torch.load_calls == [(str(policy_path.resolve()), "cuda:0")]
    assert policy.eval_count == 1
    assert policy.reset_count == 1
    assert policy.obs_history.nonzero_count == 0
    assert adapter.reset_count == 1
    assert adapter.inference_count == 0


def test_missing_policy_file_fails_before_loading(tmp_path: Path) -> None:
    with pytest.raises(MoeCtsPolicyLoadError, match="找不到"):
        MoeCtsPolicyAdapter(
            tmp_path / "missing.pt",
            device="cpu",
            torch_module=_FakeTorch(_FakePolicy()),
        )


def test_loader_failure_is_reported_as_load_error(policy_path: Path) -> None:
    fake_torch = _FakeTorch(_FakePolicy(), load_error=RuntimeError("bad archive"))

    with pytest.raises(MoeCtsPolicyLoadError, match="加载"):
        MoeCtsPolicyAdapter(
            policy_path,
            device="cpu",
            torch_module=fake_torch,
        )


def test_rejects_wrong_history_shape(policy_path: Path) -> None:
    with pytest.raises(MoeCtsPolicyContractError, match="obs_history shape"):
        _adapter(policy_path, _FakePolicy(history_shape=(1, 449)))


def test_rejects_reset_that_does_not_clear_history(policy_path: Path) -> None:
    with pytest.raises(MoeCtsPolicyContractError, match="仍包含非零值"):
        _adapter(policy_path, _FakePolicy(reset_clears_history=False))


def test_infer_validates_input_and_output_contract(policy_path: Path) -> None:
    adapter, policy, fake_torch = _adapter(policy_path)
    observation = _FakeTensor((1, 45))

    actions = adapter.infer(observation)

    assert actions.shape == (1, 12)
    assert policy.forward_inputs == [observation]
    assert fake_torch.inference_context_count == 1
    assert adapter.inference_count == 1


def test_wrong_observation_shape_never_calls_policy(policy_path: Path) -> None:
    adapter, policy, _fake_torch = _adapter(policy_path)

    with pytest.raises(MoeCtsPolicyContractError, match="single_observation shape"):
        adapter.infer(_FakeTensor((1, 44)))

    assert policy.forward_inputs == []


def test_nonfinite_observation_never_calls_policy(policy_path: Path) -> None:
    adapter, policy, _fake_torch = _adapter(policy_path)

    with pytest.raises(MoeCtsPolicyContractError, match="NaN"):
        adapter.infer(_FakeTensor((1, 45), finite=False))

    assert policy.forward_inputs == []


@pytest.mark.parametrize(
    ("policy", "error_pattern"),
    [
        (_FakePolicy(action_shape=(1, 11)), "actions shape"),
        (_FakePolicy(action_finite=False), "actions 包含 NaN"),
    ],
)
def test_rejects_invalid_policy_output(
    policy_path: Path,
    policy: _FakePolicy,
    error_pattern: str,
) -> None:
    adapter, _policy, _fake_torch = _adapter(policy_path, policy)

    with pytest.raises(MoeCtsPolicyContractError, match=error_pattern):
        adapter.infer(_FakeTensor((1, 45)))


def test_forward_exception_has_distinct_error(policy_path: Path) -> None:
    adapter, _policy, _fake_torch = _adapter(
        policy_path,
        _FakePolicy(forward_error=RuntimeError("kernel failed")),
    )

    with pytest.raises(MoeCtsPolicyInferenceError, match="forward"):
        adapter.infer(_FakeTensor((1, 45)))


def test_observation_mapping_uses_single_obs_group(policy_path: Path) -> None:
    adapter, policy, _fake_torch = _adapter(policy_path)
    observation = _FakeTensor((1, 45))

    actions = adapter.infer_from_observations(
        {"policy": _FakeTensor((1, 99)), "single_obs": observation}
    )

    assert actions.shape == (1, 12)
    assert policy.forward_inputs == [observation]


def test_missing_observation_group_is_explicit(policy_path: Path) -> None:
    adapter, _policy, _fake_torch = _adapter(policy_path)

    with pytest.raises(MoeCtsPolicyContractError, match="single_obs"):
        adapter.infer_from_observations({"policy": _FakeTensor((1, 45))})


def test_verify_runs_forward_then_restores_clean_history(policy_path: Path) -> None:
    adapter, policy, _fake_torch = _adapter(policy_path)

    report = adapter.verify_observations({"single_obs": _FakeTensor((1, 45))})

    assert report.to_dict() == {
        "policy_path": str(policy_path.resolve()),
        "device": "cuda:0",
        "observation_shape": [1, 45],
        "history_shape": [1, 450],
        "action_shape": [1, 12],
    }
    assert policy.reset_count == 2
    assert policy.obs_history.nonzero_count == 0
    assert adapter.reset_count == 2


def test_verify_failure_also_restores_clean_history(policy_path: Path) -> None:
    adapter, policy, _fake_torch = _adapter(
        policy_path,
        _FakePolicy(action_shape=(1, 9)),
    )

    with pytest.raises(MoeCtsPolicyContractError):
        adapter.verify(_FakeTensor((1, 45)))

    assert policy.obs_history.nonzero_count == 0
    assert adapter.reset_count == 2

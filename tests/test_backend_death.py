"""BaseInfer.backend_died: the replica's report-then-exit path for a backend that
died after startup."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import ray.serve.context
from ray import serve

from modelship.infer import base_infer
from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig


class _Infer(BaseInfer[dict]):
    """backend_died is concrete on the base; these three are not."""

    def shutdown(self) -> None: ...

    async def start(self) -> None: ...

    async def warmup(self) -> None: ...


def _infer(**overrides) -> _Infer:
    raw = {"name": "qwen", "model": "org/qwen", "usecase": "generate", "loader": "vllm", **overrides}
    obj = object.__new__(_Infer)
    obj.model_config = ModelshipModelConfig.model_validate(raw)
    obj._coordinator = None
    return obj


class _ExitError(Exception):
    """Stands in for os._exit so a test can assert the call without dying."""


@pytest.fixture
def harness(monkeypatch):
    coordinator = MagicMock()
    monkeypatch.setattr(base_infer.os, "_exit", MagicMock(side_effect=_ExitError))
    monkeypatch.setattr(base_infer.serve, "get_replica_context", lambda: SimpleNamespace(app_name="qwen-aaaa"))
    with patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=coordinator):
        yield coordinator


def _report(coordinator):
    return coordinator.report_replica_death.remote.call_args


class TestReplicaContextCrossesThreads:
    """llama_server reports from its process-watcher thread, so the app name has to
    be readable there. Ray stores the replica context in a module global, not the
    ContextVar it uses for request state — this fails if that ever changes."""

    def test_the_app_name_is_readable_from_another_thread(self, monkeypatch):
        monkeypatch.setattr(
            ray.serve.context, "_INTERNAL_REPLICA_CONTEXT", SimpleNamespace(app_name="qwen-aaaa"), raising=False
        )
        seen: list[str] = []
        thread = threading.Thread(target=lambda: seen.append(serve.get_replica_context().app_name))
        thread.start()
        thread.join()
        assert seen == ["qwen-aaaa"]


class TestBackendDied:
    def test_reports_then_exits(self, harness):
        with pytest.raises(_ExitError):
            _infer().backend_died("engine core died")
        assert _report(harness).args == ("qwen-aaaa", 1, "engine core died")
        base_infer.os._exit.assert_called_once_with(1)

    def test_exits_even_when_the_report_fails(self, harness):
        harness.report_replica_death.remote.side_effect = RuntimeError("coordinator wedged")
        with pytest.raises(_ExitError):
            _infer().backend_died("engine core died")
        base_infer.os._exit.assert_called_once_with(1)

    def test_the_report_is_submitted_not_awaited(self, harness):
        # Serve retries an ActorDiedError; anything that keeps this replica alive
        # with a dead backend turns that retry into a 502 for the client instead.
        with pytest.raises(_ExitError):
            _infer().backend_died("engine core died")
        harness.report_replica_death.remote.assert_called_once()

    def test_a_fixed_replica_count_is_the_ceiling(self, harness):
        with pytest.raises(_ExitError):
            _infer(num_replicas=3).backend_died("engine core died")
        assert _report(harness).args[1] == 3

    def test_autoscaling_reports_its_max(self, harness):
        with pytest.raises(_ExitError):
            _infer(autoscaling_config={"min_replicas": 1, "max_replicas": 6}).backend_died("engine core died")
        assert _report(harness).args[1] == 6


class TestVllmEngineDeath:
    """_on_engine_output_handler_done turns a completed output_handler into a
    backend_died call; shutdown() cancels the handler, which is not a death."""

    @staticmethod
    def _fire(future: asyncio.Future) -> list[str]:
        from modelship.infer.vllm.vllm_infer import VllmInfer

        infer = object.__new__(VllmInfer)
        infer.model_config = ModelshipModelConfig.model_validate(
            {"name": "qwen", "model": "org/qwen", "usecase": "generate", "loader": "vllm"}
        )
        reasons: list[str] = []
        with patch.object(VllmInfer, "backend_died", lambda _self, reason: reasons.append(reason)):
            infer._on_engine_output_handler_done(future)
        return reasons

    @staticmethod
    def _future() -> asyncio.Future:
        return asyncio.new_event_loop().create_future()

    def test_an_exception_is_carried_into_the_reason(self):
        future = self._future()
        future.set_exception(RuntimeError("engine core exploded"))
        assert self._fire(future) == ["vllm engine core died: engine core exploded"]

    def test_a_clean_return_still_reports_a_death(self):
        future = self._future()
        future.set_result(None)
        assert self._fire(future) == ["vllm engine core exited without raising"]

    def test_a_cancelled_handler_is_our_own_shutdown(self):
        future = self._future()
        future.cancel()
        assert self._fire(future) == []

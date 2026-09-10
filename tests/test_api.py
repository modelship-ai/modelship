"""Tests for ModelshipAPI model discovery and routing."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from modelship.openai.api import ModelshipAPI

# Access the underlying class, bypassing the @serve.deployment wrapper.
_ModelshipAPI = ModelshipAPI.func_or_class


def _patch_api_metrics(api, **mocks):
    """Patches module-level metrics through the cloudpickled method's own __globals__,
    since patching the live module attribute wouldn't reach it."""
    return patch.dict(type(api)._handle_response.__globals__, mocks)


@pytest.fixture
def api():
    """Creates a ModelshipAPI instance with mocked Ray Serve context; the watch loop is
    marked started so `_ensure_watching` is a no-op (watch-specific tests reset it)."""
    with (
        patch("modelship.openai.api.serve.get_replica_context") as mock_ctx,
        # Stubs configure_logging() so instantiating the gateway doesn't mutate the
        # global "modelship" logger and leak into other tests' caplog assertions.
        patch.dict(_ModelshipAPI._handle_response.__globals__, {"configure_logging": lambda: None}),
    ):
        mock_ctx.return_value.app_name = "test-gateway"
        inst = _ModelshipAPI("test-gateway")
        inst._watch_task = MagicMock()
        return inst


def _apply(api, models, *, expected=None, gen=1, handles=None):
    """Applies a coordinator routing snapshot with Serve mocked. `gen` lower than the
    replica's current `_gen` simulates a coordinator restart (removals suppressed)."""
    with ExitStack() as stack:
        if handles is not None:
            stack.enter_context(patch("modelship.openai.api.serve.get_app_handle", side_effect=handles))
        else:
            stack.enter_context(
                patch("modelship.openai.api.serve.get_app_handle", side_effect=lambda *a, **k: MagicMock())
            )
        api._apply_snapshot({"models": models, "expected": expected or [], "generation": gen})


class TestApplyRouting:
    """The reconcile core the watch loop runs: build/extend the routing table from
    a coordinator snapshot."""

    def test_add_single_model(self, api):
        _apply(api, {"qwen-a3f9k": "qwen"})
        assert "qwen" in api.models
        assert len(api.models["qwen"]) == 1
        assert api.model_list[0].id == "qwen"

    def test_add_multiple_deployments_same_model(self, api):
        h1, h2 = MagicMock(), MagicMock()
        _apply(api, {"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"}, handles=[h1, h2])
        assert len(api.models["qwen"]) == 2
        assert len(api.model_list) == 1

    def test_add_different_models(self, api):
        _apply(api, {"qwen-a3f9k": "qwen", "kokoro-c1m4n": "kokoro"})
        assert "qwen" in api.models
        assert "kokoro" in api.models
        assert len(api.model_list) == 2

    def test_incremental_snapshot_adds_new_handle_to_existing_model(self, api):
        h1, h2 = MagicMock(), MagicMock()
        _apply(api, {"qwen-a3f9k": "qwen"}, handles=[h1])
        # A later snapshot adds a 2nd deployment of qwen; the already-routed one is
        # left untouched (no new handle fetched for it).
        _apply(api, {"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"}, gen=2, handles=[h2])
        assert len(api.models["qwen"]) == 2
        assert api.models["qwen"]["qwen-a3f9k"] is h1
        assert api.models["qwen"]["qwen-b7x2p"] is h2
        assert len(api.model_list) == 1

    def test_handle_failure_raises_and_holds_generation(self, api):
        # A transient get_app_handle failure must not be swallowed: the apply raises and
        # _gen stays unadvanced, so the watch loop re-pulls and retries this generation.
        api._gen = 0
        with pytest.raises(RuntimeError, match="not yet registerable"):
            _apply(api, {"qwen-a3f9k": "qwen"}, gen=1, handles=Exception("controller lag"))
        assert "qwen" not in api.models
        assert len(api.model_list) == 0
        assert api._gen == 0

    def test_handle_failure_then_retry_registers(self, api):
        api._gen = 0
        with pytest.raises(RuntimeError):
            _apply(api, {"qwen-a3f9k": "qwen"}, gen=1, handles=Exception("controller lag"))
        # Same generation re-pulled once the controller has caught up: now it sticks.
        _apply(api, {"qwen-a3f9k": "qwen"}, gen=1)
        assert "qwen" in api.models
        assert api._gen == 1

    def test_partial_failure_applies_good_apps_and_raises(self, api):
        # One app registers, the other's handle lags. The good one is wired up, the
        # apply still raises (naming the laggard), and _gen does not advance.
        api._gen = 0
        h_ok = MagicMock()
        with pytest.raises(RuntimeError, match="kokoro"):
            _apply(
                api,
                {"qwen-a3f9k": "qwen", "kokoro-c1m4n": "kokoro"},
                gen=1,
                handles=[h_ok, RuntimeError("lag")],
            )
        assert "qwen" in api.models
        assert "kokoro" not in api.models
        assert api._gen == 0

    def test_records_per_model_load_times_and_ready_timestamp(self, api):
        _apply(api, {"qwen-a3f9k": "qwen"}, expected=["qwen", "kokoro"])
        assert api._expected_set_at is not None
        assert "qwen" in api._model_load_times
        assert api._model_load_times["qwen"] >= 0
        assert api._all_ready_at is None  # kokoro still pending

        _apply(api, {"qwen-a3f9k": "qwen", "kokoro-c1m4n": "kokoro"}, expected=["qwen", "kokoro"], gen=2)
        assert "kokoro" in api._model_load_times
        assert api._all_ready_at is not None

    def test_readyz_body_ready_flag(self, api):
        _apply(api, {}, expected=["qwen"])
        body = api._readyz_body()
        assert body["ready"] is False
        assert body["models_pending"] == ["qwen"]
        assert body["time_to_ready_s"] is None

        _apply(api, {"qwen-a3f9k": "qwen"}, expected=["qwen"], gen=2)
        body = api._readyz_body()
        assert body["ready"] is True
        assert body["models_pending"] == []
        assert body["time_to_ready_s"] is not None
        assert "qwen" in body["model_load_times_s"]


class TestReconcileRemovals:
    """A snapshot that drops an app removes it when the generation advances; a
    regressed generation (coordinator restart) never blanks live routing."""

    def test_dropped_app_removed_on_forward_snapshot(self, api):
        _apply(api, {"qwen-a3f9k1b2c4": "qwen"}, gen=1)
        assert "qwen" in api.models
        _apply(api, {}, gen=2)
        assert "qwen" not in api.models
        assert api.model_list == []

    def test_one_of_many_dropped_keeps_model(self, api):
        h1, h2 = MagicMock(), MagicMock()
        _apply(api, {"qwen-aaaaaaaaaa": "qwen", "qwen-bbbbbbbbbb": "qwen"}, gen=1, handles=[h1, h2])
        _apply(api, {"qwen-bbbbbbbbbb": "qwen"}, gen=2)
        assert "qwen" in api.models
        assert list(api.models["qwen"].keys()) == ["qwen-bbbbbbbbbb"]
        assert len(api.model_list) == 1

    def test_regressed_generation_does_not_blank_routing(self, api):
        # Coordinator restarted (generation reset below ours) but the model is still
        # deployed: additions are adopted, live routing is never removed.
        _apply(api, {"qwen-a3f9k1b2c4": "qwen"}, gen=5)
        _apply(api, {}, gen=0)
        assert "qwen" in api.models

    def test_drop_unknown_app_is_noop(self, api):
        assert api._drop_apps(["nonexistent-1234567890"]) == []

    def test_removal_drops_from_expected_when_snapshot_drops_it(self, api):
        _apply(api, {"qwen-a3f9k1b2c4": "qwen"}, expected=["qwen", "kokoro"], gen=1)
        _apply(api, {}, expected=["kokoro"], gen=2)
        assert api.expected_models == ["kokoro"]


class TestListDeployments:
    @pytest.mark.asyncio
    async def test_returns_app_names_per_model(self, api):
        _apply(
            api,
            {"qwen-aaaaaaaaaa": "qwen", "qwen-bbbbbbbbbb": "qwen", "kokoro-cccccccccc": "kokoro"},
        )
        listed = await api.list_deployments()
        assert set(listed["qwen"]) == {"qwen-aaaaaaaaaa", "qwen-bbbbbbbbbb"}
        assert listed["kokoro"] == ["kokoro-cccccccccc"]


class TestWatchReconcile:
    """The first-request synchronous sync that seeds a (re)started replica from the
    coordinator before the watch loop takes over."""

    def test_sync_pulls_snapshot_and_builds_table(self, api):
        api._watch_task = None  # exercise the real sync path
        snapshot = {
            "models": {"qwen-aaaaaaaaaa": "qwen", "embed-bbbbbbbbbb": "embed"},
            "expected": ["qwen", "embed"],
            "generation": 3,
        }
        with (
            patch("modelship.infer.replica_coordinator.get_or_create_replica_coordinator", return_value=MagicMock()),
            patch("modelship.openai.api.ray.get", return_value=snapshot),
            patch("modelship.openai.api.serve.get_app_handle", return_value=MagicMock()),
        ):
            assert api._sync_routing_blocking() is True
            # _readyz_body re-enters _sync_routing_blocking, so it needs the patches too.
            assert api._readyz_body()["ready"] is True

        assert set(api.models) == {"qwen", "embed"}
        assert api._gen == 3

    def test_sync_tolerates_unavailable_coordinator(self, api):
        api._watch_task = None
        with patch("modelship.infer.replica_coordinator.get_or_create_replica_coordinator", side_effect=RuntimeError):
            assert api._sync_routing_blocking() is False
        assert api.models == {}

    def test_sync_defers_when_deployment_not_ready(self, api):
        # Snapshot fetched fine, but a deployment isn't handle-able yet: the blocking sync
        # defers (returns False, _gen unadvanced) rather than failing the first request.
        api._watch_task = None
        snapshot = {"models": {"qwen-aaaaaaaaaa": "qwen"}, "expected": ["qwen"], "generation": 2}
        with (
            patch("modelship.infer.replica_coordinator.get_or_create_replica_coordinator", return_value=MagicMock()),
            patch("modelship.openai.api.ray.get", return_value=snapshot),
            patch("modelship.openai.api.serve.get_app_handle", side_effect=RuntimeError("controller lag")),
        ):
            assert api._sync_routing_blocking() is False
        assert api.models == {}
        assert api._gen == 0

    def test_failed_sync_drops_stale_coordinator_handle(self, api):
        # A cached handle whose actor died (recreated with a new ActorID) must be
        # cleared so the next _coord() re-resolves instead of retrying a corpse.
        stale = MagicMock()
        stale.get_routing.remote.side_effect = RuntimeError("actor dead")
        api._replica_coord = stale
        with patch("modelship.openai.api.ray.get", side_effect=RuntimeError("actor dead")):
            assert api._sync_routing_blocking() is False
        assert api._replica_coord is None

    @pytest.mark.asyncio
    async def test_coord_async_resolves_off_thread_and_caches(self, api):
        # The watch loop resolves the coordinator via asyncio.to_thread (so the sync
        # ray.get_actor never blocks the event loop) and caches the handle.
        api._replica_coord = None
        sentinel = MagicMock()
        with patch(
            "modelship.infer.replica_coordinator.get_or_create_replica_coordinator", return_value=sentinel
        ) as goc:
            assert await api._coord_async() is sentinel
            assert await api._coord_async() is sentinel
        goc.assert_called_once()  # second call served from cache, no re-resolve

    def test_sync_keeps_live_models_on_regressed_generation(self, api):
        # Coordinator restarted: empty snapshot at a lower generation than ours. The
        # model is still deployed, so routing is preserved, not blanked.
        _apply(api, {"qwen-aaaaaaaaaa": "qwen"}, gen=4)
        empty = {"models": {}, "expected": [], "generation": 0}
        with (
            patch("modelship.infer.replica_coordinator.get_or_create_replica_coordinator", return_value=MagicMock()),
            patch("modelship.openai.api.ray.get", return_value=empty),
        ):
            api._replica_coord = None
            assert api._sync_routing_blocking() is True
        assert "qwen" in api.models


class TestGetHandle:
    def test_returns_the_deployment(self, api):
        handle = MagicMock()
        _apply(api, {"qwen-a3f9k": "qwen"}, handles=[handle])
        assert api._get_handle("qwen") is handle

    def test_prefers_the_most_recently_registered(self, api):
        # Two entries under one model name is a transient state the coordinator
        # no longer produces, but the gateway still resolves it deterministically.
        ha, hb = MagicMock(), MagicMock()
        _apply(api, {"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"}, handles=[ha, hb])
        assert api._get_handle("qwen") is hb

    def test_unknown_model_raises(self, api):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            api._get_handle("nonexistent")

    def test_none_model_raises(self, api):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            api._get_handle(None)


class TestImageEditRoutes:
    @pytest.mark.asyncio
    async def test_edit_reads_upload_before_ray_boundary(self, api):
        import io

        from fastapi import UploadFile

        from modelship.openai.protocol import ImageEditRequest

        handle = MagicMock()
        remote = handle.edit_image.options.return_value.remote
        api.models = {"sdxl": {"sdxl-a1b2c": handle}}

        request = ImageEditRequest(
            image=UploadFile(file=io.BytesIO(b"IMAGE_BYTES"), filename="i.png"),
            mask=UploadFile(file=io.BytesIO(b"MASK_BYTES"), filename="m.png"),
            prompt="add a hat",
            model="sdxl",
        )
        raw_request = MagicMock()
        raw_request.headers = {}

        with patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")) as handle_response:
            result = await api.create_image_edit(request, raw_request)

        assert result == "OK"
        # The upload bytes must be read in the gateway, not handed to the actor as UploadFile.
        args, _ = remote.call_args
        image_data, mask_data, request_no_file = args[0], args[1], args[2]
        assert image_data == b"IMAGE_BYTES"
        assert mask_data == b"MASK_BYTES"
        # No UploadFile may cross the boundary: image/mask are dropped to None and the
        # bytes passed separately (image[] is exclude=True, so it never appears in the dump).
        dumped = request_no_file.model_dump()
        assert dumped.get("image") is None
        assert dumped.get("mask") is None
        assert "image[]" not in dumped and "image_array" not in dumped
        handle_response.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_variation_reads_upload_and_omits_mask(self, api):
        import io

        from fastapi import UploadFile

        from modelship.openai.protocol import ImageVariationRequest

        handle = MagicMock()
        remote = handle.vary_image.options.return_value.remote
        api.models = {"sdxl": {"sdxl-a1b2c": handle}}

        request = ImageVariationRequest(
            image=UploadFile(file=io.BytesIO(b"IMAGE_BYTES"), filename="i.png"),
            model="sdxl",
        )
        raw_request = MagicMock()
        raw_request.headers = {}

        with patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")):
            await api.create_image_variation(request, raw_request)

        args, _ = remote.call_args
        assert args[0] == b"IMAGE_BYTES"


class TestImageFormDecomposition:
    """Exercises the request models through FastAPI's real multipart/form-data
    decomposition, where the `image[]` array field (used by Open WebUI) is picked up."""

    @staticmethod
    def _client():
        import io
        from typing import Annotated

        from fastapi import FastAPI, Form, Request
        from fastapi.testclient import TestClient

        from modelship.openai.protocol import ImageEditRequest, ImageVariationRequest

        app = FastAPI()

        @app.post("/v1/images/edits")
        async def edit(request: Annotated[ImageEditRequest, Form()], raw: Request):
            return {
                "image": request.image.filename if request.image else None,
                # The UploadFile must never survive into model_dump (it would
                # fail to serialize across the Ray process boundary).
                "image_keys_in_dump": [k for k in request.model_dump(exclude={"image", "mask"}) if "image" in k],
            }

        @app.post("/v1/images/variations")
        async def variation(request: Annotated[ImageVariationRequest, Form()], raw: Request):
            return {
                "image": request.image.filename if request.image else None,
                "image_keys_in_dump": [k for k in request.model_dump(exclude={"image"}) if "image" in k],
            }

        return TestClient(app), io

    def test_edit_accepts_bracketed_image_array_field(self):
        # Open WebUI (and OpenAI's gpt-image-1 form) send the upload as `image[]`.
        client, io = self._client()
        resp = client.post(
            "/v1/images/edits",
            data={"prompt": "add a sombrero", "model": "sdxl"},
            files={"image[]": ("goat.png", io.BytesIO(b"IMAGE_BYTES"), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["image"] == "goat.png"
        assert body["image_keys_in_dump"] == []

    def test_edit_accepts_singular_image_field(self):
        # The legacy DALL·E 2 singular `image` form must keep working.
        client, io = self._client()
        resp = client.post(
            "/v1/images/edits",
            data={"prompt": "add a sombrero", "model": "sdxl"},
            files={"image": ("goat.png", io.BytesIO(b"IMAGE_BYTES"), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["image"] == "goat.png"

    def test_edit_missing_image_is_422(self):
        client, _ = self._client()
        resp = client.post("/v1/images/edits", data={"prompt": "add a sombrero", "model": "sdxl"})
        assert resp.status_code == 422

    def test_variation_accepts_bracketed_image_array_field(self):
        client, io = self._client()
        resp = client.post(
            "/v1/images/variations",
            data={"model": "sdxl"},
            files={"image[]": ("goat.png", io.BytesIO(b"IMAGE_BYTES"), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["image"] == "goat.png"
        assert body["image_keys_in_dump"] == []


class TestHandleResponse:
    @pytest.mark.asyncio
    async def test_handle_json_response_directly(self, api):
        from fastapi.responses import JSONResponse

        async def mock_gen():
            yield JSONResponse(content={"data": "test"})

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_embedding_response(self, api):
        from fastapi.responses import JSONResponse

        from modelship.openai.protocol import EmbeddingResponse, UsageInfo

        resp = EmbeddingResponse(
            model="test",
            data=[],
            usage=UsageInfo(prompt_tokens=10, total_tokens=10),
            created=123,
        )

        async def mock_gen():
            yield resp

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        assert b'"model":"test"' in result.body

    @pytest.mark.asyncio
    async def test_handle_streaming_chat(self, api):
        from fastapi.responses import StreamingResponse

        async def mock_gen():
            yield "data: chunk1\n\n"
            yield "data: chunk2\n\n"
            yield "data: [DONE]\n\n"

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, StreamingResponse)

    @pytest.mark.asyncio
    async def test_streaming_duration_observed_after_stream_drains(self, api):
        """Records REQUEST_DURATION_SECONDS and resets REQUEST_IN_PROGRESS only after the
        stream fully drains, not when _handle_response returns the StreamingResponse."""
        import asyncio

        from fastapi.responses import StreamingResponse

        delay = 0.02

        async def mock_gen():
            for i in range(3):
                await asyncio.sleep(delay)
                yield f"data: chunk{i}\n\n"

        watcher = MagicMock()
        dur, in_progress = MagicMock(), MagicMock()
        with _patch_api_metrics(api, REQUEST_DURATION_SECONDS=dur, REQUEST_IN_PROGRESS=in_progress):
            result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")
            assert isinstance(result, StreamingResponse)

            # Returning the StreamingResponse must not have timed the request yet.
            dur.observe.assert_not_called()
            assert (
                call(0, tags={"model": "test-model", "endpoint": "test-endpoint"}) not in in_progress.set.call_args_list
            )

            # Drain the stream the way Starlette would.
            chunks = [chunk async for chunk in result.body_iterator]
            assert len(chunks) == 3

        # Now duration is observed exactly once, with at least the summed delay,
        # and in-progress has been reset to 0 exactly once.
        dur.observe.assert_called_once()
        observed = dur.observe.call_args.args[0]
        assert observed >= delay * 3
        reset = call(0, tags={"model": "test-model", "endpoint": "test-endpoint"})
        assert in_progress.set.call_args_list.count(reset) == 1
        assert in_progress.set.call_args == reset  # the reset is the final set

    @pytest.mark.asyncio
    async def test_non_streaming_duration_observed_on_return(self, api):
        from fastapi.responses import JSONResponse

        async def mock_gen():
            yield JSONResponse(content={"data": "test"})

        watcher = MagicMock()
        dur, in_progress = MagicMock(), MagicMock()
        with _patch_api_metrics(api, REQUEST_DURATION_SECONDS=dur, REQUEST_IN_PROGRESS=in_progress):
            result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")
            assert isinstance(result, JSONResponse)
            dur.observe.assert_called_once()
            reset = call(0, tags={"model": "test-model", "endpoint": "test-endpoint"})
            assert in_progress.set.call_args_list.count(reset) == 1
            assert in_progress.set.call_args == reset

    @pytest.mark.asyncio
    async def test_cancellation_during_first_chunk_stops_watcher(self, api):
        """CancelledError is a BaseException that skips the except clauses, so the watcher
        must still be stopped in the outer finally — else the DisconnectRegistry entry leaks."""
        import asyncio

        async def mock_gen():
            raise asyncio.CancelledError
            yield  # unreachable; makes this an async generator

        watcher = MagicMock()
        with pytest.raises(asyncio.CancelledError):
            await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")
        watcher.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_stops_watcher_only_after_drain(self, api):
        """The watcher for a streaming response is stopped in _stream()'s finally
        after the stream drains, not when _handle_response returns."""
        from fastapi.responses import StreamingResponse

        async def mock_gen():
            yield "data: chunk1\n\n"
            yield "data: [DONE]\n\n"

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")
        assert isinstance(result, StreamingResponse)
        watcher.stop.assert_not_called()  # not stopped yet — stream hasn't been consumed

        [chunk async for chunk in result.body_iterator]
        watcher.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_raytaskerror_with_value_error_cause_returns_400(self, api):
        import json

        from fastapi.responses import JSONResponse
        from ray.exceptions import RayTaskError

        # .cause is a ValueError subclass with a `parameter` attribute, mirroring
        # VLLMValidationError after Ray transports it across process boundaries.
        class _FakeValidationError(ValueError):
            def __init__(self, message: str, parameter: str) -> None:
                super().__init__(message)
                self.parameter = parameter

        cause = _FakeValidationError("This model's maximum context length is 14512 tokens.", "input_tokens")
        err = RayTaskError(function_name="fn", traceback_str="tb", cause=cause)

        async def mock_gen():
            if False:
                yield  # pragma: no cover — make this an async generator
            raise err

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "input_tokens"
        assert "maximum context length" in body["error"]["message"]
        watcher.stop.assert_called()

    @pytest.mark.asyncio
    async def test_raytaskerror_with_unknown_cause_returns_500(self, api):
        from fastapi.responses import JSONResponse
        from ray.exceptions import RayTaskError

        cause = RuntimeError("something exploded internally")
        err = RayTaskError(function_name="fn", traceback_str="tb", cause=cause)

        async def mock_gen():
            if False:
                yield  # pragma: no cover
            raise err

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_stream_error_after_first_chunk_yields_sse_error_not_raise(self, api):
        """Failure between chunks: raising after headers are committed just aborts the
        connection with no body, so it must yield a parseable SSE error chunk instead."""
        import json

        from fastapi.responses import StreamingResponse

        async def mock_gen():
            yield "data: chunk1\n\n"
            raise RuntimeError("boundary dropped")
            yield  # pragma: no cover — unreachable, keeps this an async generator

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "create_chat_completion")
        assert isinstance(result, StreamingResponse)

        chunks = [chunk async for chunk in result.body_iterator]
        assert chunks[0] == "data: chunk1\n\n"
        assert chunks[-1] == "data: [DONE]\n\n"
        error_chunk = chunks[-2]
        assert error_chunk.startswith("data: ")
        body = json.loads(error_chunk[len("data: ") :])
        assert body["error"]["type"] == "api_error"
        watcher.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_error_for_responses_endpoint_yields_response_failed_event(self, api):
        """Typed `event:` framing, and the event must continue the stream it terminates:
        same response id, next sequence number."""
        import json

        from fastapi.responses import StreamingResponse

        created = {
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp_abc", "object": "response", "created_at": 1, "model": "m", "output": []},
        }
        delta = {"type": "response.output_text.delta", "sequence_number": 7, "delta": "hi"}

        async def mock_gen():
            yield f"event: {created['type']}\ndata: {json.dumps(created)}\n\n"
            yield f"event: {delta['type']}\ndata: {json.dumps(delta)}\n\n"
            raise RuntimeError("boundary dropped")
            yield  # pragma: no cover — unreachable, keeps this an async generator

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "create_response")
        assert isinstance(result, StreamingResponse)

        chunks = [chunk async for chunk in result.body_iterator]
        assert chunks[-1] == "data: [DONE]\n\n"
        error_chunk = chunks[-2]
        assert error_chunk.startswith("event: response.failed\n")
        event = json.loads(error_chunk.split("data: ", 1)[1])
        assert event["sequence_number"] == 8
        assert event["response"]["id"] == "resp_abc"
        assert event["response"]["model"] == "m"
        assert event["response"]["created_at"] == 1
        assert event["response"]["status"] == "failed"
        assert event["response"]["error"]["message"]
        watcher.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_error_for_responses_endpoint_without_parseable_frames(self, api):
        """Unparseable frames still have to produce a schema-complete event."""
        import json

        from fastapi.responses import StreamingResponse

        async def mock_gen():
            yield "data: [DONE]\n\n"
            raise RuntimeError("boundary dropped")
            yield  # pragma: no cover — unreachable, keeps this an async generator

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "create_response")
        assert isinstance(result, StreamingResponse)

        chunks = [chunk async for chunk in result.body_iterator]
        event = json.loads(chunks[-2].split("data: ", 1)[1])
        assert event["sequence_number"] == 0
        assert event["response"]["model"] == "test-model"
        assert event["response"]["id"].startswith("resp_")
        assert event["response"]["output"] == []
        assert event["response"]["status"] == "failed"

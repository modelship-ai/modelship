"""Tests for ``/v1/responses/compact``: the Fernet crypto module, the
``CompactRequest``/``CompactResource``/``CompactionItem`` schemas, the
``build_summarization_request``/``build_compaction`` builders, and the gateway route.

The route dispatches to the same `handle.generate` a chat-completion request would,
then encrypts the result into a ``CompactionItem`` — it never persists a snapshot
under the compaction id.
"""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from modelship.openai import compaction_crypto
from modelship.openai.api import ModelshipAPI
from modelship.openai.protocol import ChatCompletionResponse, UsageInfo
from modelship.openai.protocol.chat import ChatCompletionResponseChoice, ChatMessage
from modelship.openai.protocol.responses.schemas import CompactionItem, CompactRequest, CompactResource, ResponseUsage
from modelship.openai.utils.responses import build_compaction, build_summarization_request
from modelship.state import MemoryStoreActor
from tests.conftest import build_gateway

_ModelshipAPI = ModelshipAPI.func_or_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def _reset_compaction_key(monkeypatch):
    """Clean process-level cache and a fresh backing store per test."""
    compaction_crypto._cached_key = None
    store = _MemoryStore()
    monkeypatch.setattr("modelship.state.get_state_store", lambda: store)
    yield
    compaction_crypto._cached_key = None


@pytest.fixture
def compaction_key(monkeypatch):
    """A key that's both set as the env var and already seeded into the mocked
    store — mirrors what the deploy driver does before any actor starts."""
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("MSHIP_COMPACTION_KEY", key)
    from modelship.state import get_state_store

    compaction_crypto.ensure_key_seeded(get_state_store())
    return key


class TestCompactionCrypto:
    def test_round_trip(self, compaction_key):
        items = [{"type": "message", "role": "assistant", "content": "a summary"}]
        blob = compaction_crypto.encrypt_items(items)
        assert compaction_crypto.decrypt_items(blob) == items

    def test_wrong_key_raises_invalid_token(self, compaction_key):
        blob = compaction_crypto.encrypt_items([{"a": 1}])
        # Simulate a key rotation: clear this process's cache, overwrite the store.
        other_key = Fernet.generate_key().decode("ascii")
        compaction_crypto._cached_key = None
        from modelship.state import get_state_store

        get_state_store().set("compaction/key", {"key": other_key})
        with pytest.raises(InvalidToken):
            compaction_crypto.decrypt_items(blob)

    def test_tampered_blob_raises_invalid_token(self, compaction_key):
        blob = compaction_crypto.encrypt_items([{"a": 1}])
        tampered = blob[:-4] + ("AAAA" if blob[-4:] != "AAAA" else "BBBB")
        with pytest.raises(InvalidToken):
            compaction_crypto.decrypt_items(tampered)

    def test_non_ascii_blob_raises_invalid_token(self, compaction_key):
        # blob.encode("ascii") would otherwise raise UnicodeEncodeError, a plain
        # ValueError that callers wouldn't catch alongside a tampered/wrong-key blob.
        with pytest.raises(InvalidToken):
            compaction_crypto.decrypt_items("not-ascii-🔥")

    def test_resolve_key_fails_hard_when_store_is_unseeded(self):
        # No compaction_key fixture here — the store is empty.
        with pytest.raises(RuntimeError, match="no compaction key found"):
            compaction_crypto.encrypt_items([{"a": 1}])

    def test_ensure_key_seeded_validates_a_bad_configured_key(self, monkeypatch):
        monkeypatch.setenv("MSHIP_COMPACTION_KEY", "not-a-valid-fernet-key")
        from modelship.state import get_state_store

        with pytest.raises(ValueError, match="MSHIP_COMPACTION_KEY is not a valid Fernet key"):
            compaction_crypto.ensure_key_seeded(get_state_store())

    def test_key_is_cached_after_first_resolution(self, compaction_key):
        compaction_crypto.encrypt_items([{"a": 1}])
        key_after_first = compaction_crypto._cached_key
        # A second call must reuse the same cached key, not re-read the store, or a
        # blob minted earlier in the same process would stop decoding.
        compaction_crypto.encrypt_items([{"b": 2}])
        assert compaction_crypto._cached_key == key_after_first

    def test_two_processes_share_the_seeded_key(self, compaction_key):
        """Encrypting in one process and decrypting in another (simulated by clearing
        the process-local cache) must agree, since both read the same shared store."""
        blob = compaction_crypto.encrypt_items([{"a": 1}])
        compaction_crypto._cached_key = None
        assert compaction_crypto.decrypt_items(blob) == [{"a": 1}]

    def test_ensure_key_seeded_is_a_noop_if_already_present(self, monkeypatch):
        monkeypatch.delenv("MSHIP_COMPACTION_KEY", raising=False)
        from modelship.state import get_state_store

        store = get_state_store()
        compaction_crypto.ensure_key_seeded(store)
        first = store.get("compaction/key")
        compaction_crypto.ensure_key_seeded(store)
        assert store.get("compaction/key") == first

    def test_ensure_key_seeded_uses_explicit_env_key(self, monkeypatch):
        key = Fernet.generate_key().decode("ascii")
        monkeypatch.setenv("MSHIP_COMPACTION_KEY", key)
        from modelship.state import get_state_store

        store = get_state_store()
        compaction_crypto.ensure_key_seeded(store)
        assert store.get("compaction/key") == {"key": key}


class TestCompactSchemas:
    def test_model_is_required(self):
        with pytest.raises(ValidationError, match="model"):
            CompactRequest()

    def test_created_by_omitted_when_unset(self):
        # created_by must be omitted, not sent as null, to satisfy the external
        # suite's optional-but-non-nullable schema.
        dumped = CompactionItem(encrypted_content="blob").model_dump(mode="json")
        assert "created_by" not in dumped

    def test_created_by_present_when_set(self):
        dumped = CompactionItem(encrypted_content="blob", created_by="model").model_dump(mode="json")
        assert dumped["created_by"] == "model"

    def test_compaction_item_defaults(self):
        item = CompactionItem(encrypted_content="blob")
        assert item.type == "compaction"
        assert item.id.startswith("cmp_")
        assert item.created_by is None

    def test_compact_resource_shape(self):
        resource = CompactResource(
            output=[CompactionItem(encrypted_content="blob")],
            usage=ResponseUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        )
        dumped = resource.model_dump(mode="json")
        assert dumped["object"] == "response.compaction"
        assert dumped["output"][0]["type"] == "compaction"
        assert set(dumped) >= {"id", "object", "output", "created_at", "usage"}

    def test_missing_model_is_422_from_fastapi(self):
        # Exercises FastAPI's own validation on the required `model` field via a
        # real ASGI request, with no loader involved.
        app = FastAPI()

        @app.post("/v1/responses/compact")
        async def compact(request: CompactRequest):
            return {}

        resp = TestClient(app).post("/v1/responses/compact", json={"input": "hi"})
        assert resp.status_code == 422


class TestBuildCompaction:
    def test_encrypts_summary_into_one_compaction_item(self, compaction_key):
        summary_items = [{"type": "message", "role": "assistant", "content": "the gist"}]
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resource = build_compaction(summary_items=summary_items, usage=usage)

        assert len(resource.output) == 1
        assert resource.output[0].type == "compaction"
        assert compaction_crypto.decrypt_items(resource.output[0].encrypted_content) == summary_items
        assert resource.usage.input_tokens == 10
        assert resource.usage.output_tokens == 5


class TestBuildSummarizationRequest:
    def test_leads_with_system_instruction(self):
        chat = build_summarization_request("m", [{"role": "user", "content": "hi"}])
        assert chat.model == "m"
        assert chat.stream is False
        assert chat.messages[0]["role"] == "system"
        assert "summar" in chat.messages[0]["content"].lower()
        assert chat.messages[1] == {"role": "user", "content": "hi"}

    def test_caller_instructions_are_inserted_as_additional_system_message(self):
        chat = build_summarization_request("m", [{"role": "user", "content": "hi"}], "focus on pricing details")
        assert chat.messages[0]["role"] == "system"
        assert chat.messages[1] == {"role": "system", "content": "focus on pricing details"}
        assert chat.messages[2] == {"role": "user", "content": "hi"}

    def test_no_instructions_does_not_insert_extra_message(self):
        chat = build_summarization_request("m", [{"role": "user", "content": "hi"}], None)
        assert len(chat.messages) == 2

    def test_bad_item_shape_raises(self):
        from modelship.openai.protocol.responses import UnsupportedResponsesFeatureError

        with pytest.raises(UnsupportedResponsesFeatureError):
            build_summarization_request("m", [{"type": "image_generation_call"}])


def _raw_request():
    raw = MagicMock()
    raw.headers = {}
    return raw


@pytest.fixture
def api(compaction_key):
    with (
        patch("modelship.openai.api.serve.get_replica_context") as mock_ctx,
        patch.dict(_ModelshipAPI._handle_response.__globals__, {"configure_logging": lambda: None}),
    ):
        mock_ctx.return_value.app_name = "test-gateway"
        inst = build_gateway("test-gateway")
        inst._watch_task = MagicMock()
        inst._state_store = _MemoryStore()
        return inst


def _chat_response_gen(text="a compact summary"):
    async def gen():
        yield ChatCompletionResponse(
            model="m",
            choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=text))],
            usage=UsageInfo(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )

    return gen()


def _wire(api, gen):
    handle = MagicMock()
    handle.generate.options.return_value.remote.return_value = gen
    api.models = {"m": {"m-a1b2c": handle}}
    return handle


class TestCompactResponseRoute:
    @pytest.mark.asyncio
    async def test_dispatches_to_generate_and_returns_compact_resource(self, api):
        handle = _wire(api, _chat_response_gen(text="the gist"))
        # prompt_cache_key isn't a CompactRequest field; OpenAIBaseModel's
        # extra="allow" tolerates an OpenAI-SDK client that still sends it.
        request = CompactRequest.model_validate({"model": "m", "input": "hi", "prompt_cache_key": "k"})

        result = await api.compact_response(request, _raw_request())

        assert handle.generate.options.return_value.remote.call_args is not None
        body = json.loads(bytes(result.body))
        assert body["object"] == "response.compaction"
        assert len(body["output"]) == 1
        item = body["output"][0]
        assert item["type"] == "compaction"
        decoded = compaction_crypto.decrypt_items(item["encrypted_content"])
        assert decoded == [{"type": "message", "role": "assistant", "content": "the gist"}]
        assert body["usage"]["input_tokens"] == 7

    @pytest.mark.asyncio
    async def test_instructions_reach_the_dispatched_summarization_request(self, api):
        handle = _wire(api, _chat_response_gen())
        request = CompactRequest(model="m", input="hi", instructions="focus on pricing details")

        await api.compact_response(request, _raw_request())

        sent_chat_request = handle.generate.options.return_value.remote.call_args.args[0]
        assert {"role": "system", "content": "focus on pricing details"} in sent_chat_request.messages

    @pytest.mark.asyncio
    async def test_empty_conversation_is_400(self, api):
        _wire(api, _chat_response_gen())
        request = CompactRequest(model="m", input=None)

        with pytest.raises(HTTPException) as exc_info:
            await api.compact_response(request, _raw_request())
        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    @pytest.mark.asyncio
    async def test_previous_response_id_history_is_resolved_before_dispatch(self, api):
        handle = _wire(api, _chat_response_gen())
        identity = "unscoped"
        await api._state_store.set_async(
            f"responses/{identity}/resp_1",
            {
                "response": {"id": "resp_1", "object": "response", "status": "completed", "output": []},
                "input_items": [{"type": "message", "role": "user", "content": "earlier turn"}],
            },
        )
        request = CompactRequest(model="m", input="continue", previous_response_id="resp_1")

        await api.compact_response(request, _raw_request())

        sent_chat_request = handle.generate.options.return_value.remote.call_args.args[0]
        contents = [m["content"] for m in sent_chat_request.messages]
        assert any("earlier turn" in c for c in contents if isinstance(c, str))
        assert any(c == "continue" for c in contents if isinstance(c, str))

    @pytest.mark.asyncio
    async def test_unknown_previous_response_id_is_404(self, api):
        _wire(api, _chat_response_gen())
        request = CompactRequest(model="m", input="hi", previous_response_id="resp_nope")

        with pytest.raises(HTTPException) as exc_info:
            await api.compact_response(request, _raw_request())
        assert exc_info.value.status_code == HTTPStatus.NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_summarization_raytaskerror_maps_to_400_not_500(self, api):
        from ray.exceptions import RayTaskError

        cause = ValueError("context overflow during summarization")
        err = RayTaskError(function_name="fn", traceback_str="tb", cause=cause)

        async def gen():
            if False:
                yield  # pragma: no cover
            raise err

        handle = MagicMock()
        handle.generate.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}

        request = CompactRequest(model="m", input="hi")
        result = await api.compact_response(request, _raw_request())

        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert body["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_does_not_persist_a_snapshot(self, api):
        _wire(api, _chat_response_gen())
        api._state_store.set_async = AsyncMock(side_effect=AssertionError("compaction must not write to the store"))
        request = CompactRequest(model="m", input="hi")

        await api.compact_response(request, _raw_request())

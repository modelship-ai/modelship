"""Real-cluster integration tests for /v1/responses: server-side conversation
state, reasoning, the llama_server loader's native path, and background mode.
Shared cluster/model-deploy fixtures live in conftest.py."""

import concurrent.futures
import json
import time

import httpx
import pytest

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"

_TERMINAL_STATUSES = {"completed", "incomplete", "failed", "cancelled"}


def _poll_until_terminal(client, response_id: str, *, timeout_s: float = 60.0, interval_s: float = 0.5):
    """Poll `GET /v1/responses/{id}` (via the SDK) until `status` is terminal."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.responses.retrieve(response_id)
        if resp.status in _TERMINAL_STATUSES:
            return resp
        time.sleep(interval_s)
    raise AssertionError(f"response {response_id} did not reach a terminal status within {timeout_s}s")


def _iter_sse_events(lines):
    """Parse raw SSE lines (as yielded by `httpx.Response.iter_lines`) into event dicts."""
    event_type = None
    for line in lines:
        if line.startswith("event: "):
            event_type = line[len("event: ") :]
        elif line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                return
            payload = json.loads(data)
            payload.setdefault("type", event_type)
            yield payload


@pytest.mark.integration
@pytest.mark.llama_server
class TestResponsesLlamaServer:
    """The /v1/responses adapter is loader-agnostic: same smoke test shape
    as vLLM's, run over the llama_server loader."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server")

    def test_basic_response_through_llama_server(self, client):
        # chat-llama-server is reasoning-capable and always emits a <think>...
        # preamble first, so the token budget needs headroom beyond the answer.
        resp = client.responses.create(
            model="chat-llama-server",
            input="Say hello in one word.",
            max_output_tokens=512,
        )
        assert resp.status in {"completed", "incomplete"}
        assert resp.output_text.strip()

    def test_streaming_response_through_llama_server(self, client):
        stream = client.responses.create(
            model="chat-llama-server",
            input="Say hello in one word.",
            max_output_tokens=512,
            stream=True,
        )
        text_deltas: list[str] = []
        completed = None
        for event in stream:
            if event.type == "response.output_text.delta":
                text_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response
        assert "".join(text_deltas).strip()
        assert completed is not None
        assert completed.status in {"completed", "incomplete"}


# Responses tools use the *flattened* function shape (name/parameters at the
# top level), unlike chat completions which nests them under "function".
_WEATHER_TOOL_RESPONSES = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.integration
@pytest.mark.vllm
class TestResponsesEndpoint:
    """End-to-end /v1/responses over the vLLM chat pipeline. Verifies the official
    OpenAI SDK parses our payload and unsupported features are rejected, not dropped."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_basic_response_parses_with_openai_sdk(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="Say hello in one word.",
            max_output_tokens=20,
        )
        assert resp.object == "response"
        assert resp.status in {"completed", "incomplete"}
        assert resp.output_text.strip()
        assert resp.usage.input_tokens > 0
        # Stored by default (OpenAI parity), so an unset `store` echoes True.
        assert resp.store is True

    def test_instructions_and_message_list_input(self, client):
        resp = client.responses.create(
            model="chat-capable",
            instructions="Answer in exactly one word.",
            input=[{"role": "user", "content": "What color is a clear daytime sky?"}],
            max_output_tokens=20,
        )
        assert resp.output_text.strip()

    def test_tool_call_emits_function_call_item(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="What is the weather in Paris?",
            tools=[_WEATHER_TOOL_RESPONSES],
            tool_choice="required",
            max_output_tokens=128,
        )
        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert function_calls, f"expected a function_call output item, got {[i.type for i in resp.output]}"
        assert function_calls[0].name == "get_weather"
        assert "Paris" in function_calls[0].arguments

    def test_streaming_emits_event_protocol(self, client):
        # stream=True drives the chat pipeline in streaming mode, translating chunks
        # into the Responses event protocol that the official SDK parses.
        stream = client.responses.create(
            model="chat-capable",
            input="Say hello in one word.",
            max_output_tokens=20,
            stream=True,
        )
        types: list[str] = []
        text_deltas: list[str] = []
        completed = None
        for event in stream:
            types.append(event.type)
            if event.type == "response.output_text.delta":
                text_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response

        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        assert "".join(text_deltas).strip(), "expected streamed output text"
        assert completed is not None
        assert completed.status in {"completed", "incomplete"}
        # The streamed deltas must reconstruct the final message text.
        assert "".join(text_deltas).strip() == completed.output_text.strip()
        assert completed.usage.input_tokens > 0

    def test_streaming_tool_call_emits_argument_deltas(self, client):
        stream = client.responses.create(
            model="chat-capable",
            input="What is the weather in Paris?",
            tools=[_WEATHER_TOOL_RESPONSES],
            tool_choice="required",
            max_output_tokens=128,
            stream=True,
        )
        arg_deltas: list[str] = []
        completed = None
        for event in stream:
            if event.type == "response.function_call_arguments.delta":
                arg_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response
        assert completed is not None
        function_calls = [item for item in completed.output if item.type == "function_call"]
        assert function_calls, f"expected a function_call item, got {[i.type for i in completed.output]}"
        assert function_calls[0].name == "get_weather"
        # streamed argument fragments must reconstruct the final arguments
        assert "".join(arg_deltas) == function_calls[0].arguments

    def test_truncation_reports_incomplete_details(self, client):
        # A generation cut short by max_output_tokens is `incomplete` with a reason,
        # not `completed` — the only signal a client has that output was truncated.
        resp = client.responses.create(
            model="chat-capable",
            input="Write a long essay about the sea.",
            max_output_tokens=16,
        )
        assert resp.status == "incomplete"
        assert resp.incomplete_details is not None
        assert resp.incomplete_details.reason == "max_output_tokens"
        assert resp.output_text.strip()

    def test_unknown_previous_response_id_404(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": "resp_does_not_exist"},
            timeout=60,
        )
        assert response.status_code == 404, response.text

    def test_hosted_tool_dropped_not_rejected(self):
        # web_search has no client-defined equivalent a self-hosted backend can
        # fulfill; _tools_to_chat drops it and the request completes normally.
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "search the web", "tools": [{"type": "web_search"}]},
            timeout=60,
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "completed"


@pytest.mark.integration
@pytest.mark.vllm
class TestResponsesState:
    """Server-side conversation state on /v1/responses, end-to-end against the real
    store (``memory://`` — a detached Ray actor on the live cluster)."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_continuation_recalls_earlier_turn(self, client):
        first = client.responses.create(
            model="chat-capable",
            input="My name is Alex. Remember it.",
            max_output_tokens=30,
        )
        assert first.store is True

        # Turn 2 sends only the new question — the name is recalled from stored state.
        second = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        assert second.previous_response_id == first.id
        assert "alex" in second.output_text.lower()

    def test_continuation_chains_across_three_turns(self, client):
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="I live in Berlin.",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        third = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=second.id,
            max_output_tokens=20,
        )
        # Turn 1's fact survives two hops — each snapshot embeds the whole conversation.
        assert "alex" in third.output_text.lower()

    def test_streaming_response_can_be_continued(self, client):
        stream = client.responses.create(
            model="chat-capable",
            input="My name is Alex. Remember it.",
            max_output_tokens=30,
            stream=True,
        )
        completed = None
        for event in stream:
            if event.type == "response.completed":
                completed = event.response
        assert completed is not None
        assert completed.store is True

        # A streamed response is persisted by re-reading its terminal event, storing
        # the same shape as the non-streaming path.
        second = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=completed.id,
            max_output_tokens=20,
        )
        assert "alex" in second.output_text.lower()

    def test_get_returns_stored_response(self, client):
        created = client.responses.create(model="chat-capable", input="Say hi.", max_output_tokens=20)
        fetched = client.responses.retrieve(created.id)
        assert fetched.id == created.id
        assert fetched.output_text == created.output_text

    def test_input_items_lists_what_went_in(self):
        created = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "Say hi.", "max_output_tokens": 20},
            timeout=120,
        ).json()
        listed = httpx.get(f"{OPENAI_API_BASE}/responses/{created['id']}/input_items", timeout=60)
        assert listed.status_code == 200, listed.text
        body = listed.json()
        assert body["object"] == "list"
        assert any("Say hi." in str(item.get("content", "")) for item in body["data"])

    def test_input_items_reflects_the_continued_chain(self, client):
        # After a continuation the snapshot's input is the resolved history, not just
        # the turn the client sent: user -> assistant -> user, in order.
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="What is my name?",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        body = httpx.get(f"{OPENAI_API_BASE}/responses/{second.id}/input_items", timeout=60).json()
        roles = [item.get("role") for item in body["data"]]
        assert roles == ["user", "assistant", "user"], body["data"]
        assert "My name is Alex." in str(body["data"][0]["content"])

    def test_store_false_is_not_retrievable(self):
        created = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "Say hi.", "max_output_tokens": 20, "store": False},
            timeout=120,
        ).json()
        assert created["store"] is False
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{created['id']}", timeout=60).status_code == 404

    def test_streamed_store_false_is_not_retrievable(self, client):
        # The streaming path persists by re-reading its own terminal event, so
        # `store: false` has to suppress a different branch than the non-streaming one.
        stream = client.responses.create(
            model="chat-capable",
            input="Say hi.",
            max_output_tokens=20,
            stream=True,
            store=False,
        )
        completed = None
        for event in stream:
            if event.type == "response.completed":
                completed = event.response
        assert completed is not None
        assert completed.store is False
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{completed.id}", timeout=60).status_code == 404

    def test_store_false_still_reads_history(self, client):
        # store=false governs writing this turn, not reading the chain: a caller can
        # continue a conversation without adding to it.
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=first.id,
            max_output_tokens=20,
            store=False,
        )
        assert "alex" in second.output_text.lower()
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{second.id}", timeout=60).status_code == 404

    def test_previous_response_id_of_unstored_response_404s(self, client):
        # An id that was never persisted is indistinguishable from an unknown one.
        unstored = client.responses.create(model="chat-capable", input="Say hi.", max_output_tokens=20, store=False)
        continued = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": unstored.id},
            timeout=60,
        )
        assert continued.status_code == 404, continued.text

    def test_continuation_survives_parent_deletion(self, client):
        # Each snapshot embeds the whole conversation, so a chain is not a linked list:
        # deleting a parent must not strand its children.
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="I live in Berlin.",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        assert httpx.delete(f"{OPENAI_API_BASE}/responses/{first.id}", timeout=60).status_code == 200

        third = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=second.id,
            max_output_tokens=20,
        )
        assert "alex" in third.output_text.lower()

    def test_concurrent_branches_from_one_parent_are_independent(self, client):
        # Several turns may fan out from the same previous_response_id; each must get
        # its own id and its own snapshot rather than racing over shared state.
        parent = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)

        def _branch(_: int):
            return client.responses.create(
                model="chat-capable",
                input="What is my name? Reply with just the name.",
                previous_response_id=parent.id,
                max_output_tokens=20,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            branches = list(pool.map(_branch, range(5)))

        assert len({b.id for b in branches}) == 5, "branch ids collided"
        for branch in branches:
            assert branch.previous_response_id == parent.id
            assert "alex" in branch.output_text.lower()

    def test_delete_then_get_and_continue_both_404(self, client):
        created = client.responses.create(model="chat-capable", input="Say hi.", max_output_tokens=20)

        deleted = httpx.delete(f"{OPENAI_API_BASE}/responses/{created.id}", timeout=60)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"id": created.id, "object": "response", "deleted": True}

        assert httpx.get(f"{OPENAI_API_BASE}/responses/{created.id}", timeout=60).status_code == 404
        # A deleted conversation is gone for continuation too, not just retrieval.
        continued = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": created.id},
            timeout=60,
        )
        assert continued.status_code == 404, continued.text

    def test_function_call_output_round_trip(self, client):
        # The stateful tool loop: the client returns only the tool result, and the
        # pending call it answers is recovered from stored history.
        first = client.responses.create(
            model="chat-capable",
            input="What is the weather in Paris?",
            tools=[_WEATHER_TOOL_RESPONSES],
            tool_choice="required",
            max_output_tokens=128,
        )
        calls = [item for item in first.output if item.type == "function_call"]
        assert calls, f"expected a function_call item, got {[i.type for i in first.output]}"

        second = client.responses.create(
            model="chat-capable",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls[0].call_id,
                    "output": json.dumps({"temp_c": 18, "sky": "rain"}),
                }
            ],
            tools=[_WEATHER_TOOL_RESPONSES],
            previous_response_id=first.id,
            max_output_tokens=128,
        )
        assert "18" in second.output_text, second.output_text

    def test_get_unknown_id_404(self):
        assert httpx.get(f"{OPENAI_API_BASE}/responses/resp_does_not_exist", timeout=60).status_code == 404

    def test_delete_unknown_id_404(self):
        assert httpx.delete(f"{OPENAI_API_BASE}/responses/resp_does_not_exist", timeout=60).status_code == 404

    def test_malformed_id_404_and_leaves_state_intact(self):
        # response_id is a state-key segment; a traversal-shaped id must not resolve to
        # (or delete) anything else the store holds.
        resp = httpx.request("DELETE", f"{OPENAI_API_BASE}/responses/..%2F..%2Feffective%2Fmodelship%20api", timeout=60)
        assert resp.status_code == 404, resp.text
        # The gateway's own effective config is untouched — /v1/models still answers.
        assert httpx.get(f"{OPENAI_API_BASE}/models", timeout=60).status_code == 200


@pytest.mark.integration
@pytest.mark.vllm
class TestResponsesBackground:
    """`background: true`: queue, poll, and cancel a detached response, end-to-end
    against the real store and DisconnectRegistry actor — the pieces the unit
    suite mocks out."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_background_returns_queued_then_completes(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="Say hello in one word.",
            max_output_tokens=20,
            background=True,
        )
        assert resp.status == "queued"
        assert resp.background is True

        completed = _poll_until_terminal(client, resp.id)
        assert completed.status == "completed"
        assert completed.output_text.strip()
        assert completed.background is True

        # Terminal, so a plain GET (not just the SDK's polling wrapper) sees the
        # same completed body — no `_mship` sidecar leaking into it.
        fetched = httpx.get(f"{OPENAI_API_BASE}/responses/{resp.id}", timeout=60).json()
        assert fetched["status"] == "completed"
        assert "_mship" not in fetched

    def test_background_and_stream_streams_live_and_completes(self):
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "Say hello in one word.", "background": True, "stream": True},
            timeout=60,
        ) as response:
            assert response.status_code == 200, response.text
            events = list(_iter_sse_events(response.iter_lines()))

        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        assert events[-1]["response"]["status"] == "completed"
        assert events[-1]["response"]["background"] is True

    def test_background_stream_resume_after_disconnect(self):
        response_id = None
        seen = []
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "Say hello in one word.", "background": True, "stream": True},
            timeout=60,
        ) as response:
            for event in _iter_sse_events(response.iter_lines()):
                seen.append(event)
                response_id = event["response"]["id"]
                break  # simulate a disconnect right after the first event

        assert response_id is not None
        last_seq = seen[-1]["sequence_number"]

        # Resume from where the disconnected client left off — the background run
        # kept going server-side the whole time (it never watched this connection).
        with httpx.stream(
            "GET", f"{OPENAI_API_BASE}/responses/{response_id}", params={"stream": "true", "starting_after": last_seq}
        ) as response:
            assert response.status_code == 200, response.text
            resumed = list(_iter_sse_events(response.iter_lines()))

        assert resumed, "resume produced no events"
        assert resumed[-1]["type"] == "response.completed"
        assert resumed[-1]["response"]["status"] == "completed"

    def test_background_and_store_false_is_400(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "background": True, "store": False},
            timeout=60,
        )
        assert response.status_code == 400, response.text
        assert "background" in response.json()["error"]["message"]

    def test_cancel_marks_cancelled_and_stays_cancelled(self, client):
        # A long-ish generation so the cancel (fired immediately, no poll delay)
        # has a real window to land before the model would otherwise finish.
        resp = client.responses.create(
            model="chat-capable",
            input="Write a detailed 300 word essay about the history of the Roman Empire.",
            max_output_tokens=500,
            background=True,
        )
        cancelled = client.responses.cancel(resp.id)
        assert cancelled.status in {"cancelled", "completed"}
        if cancelled.status != "cancelled":
            pytest.skip("generation finished before the cancel request landed")

        # Give the drain task a moment to wind down, then confirm the status sticks
        # — a stream ending abnormally after cancel must not flip it to `failed`.
        final = _poll_until_terminal(client, resp.id, timeout_s=10)
        assert final.status == "cancelled"

    def test_cancel_on_completed_is_idempotent(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="Say hi.",
            max_output_tokens=20,
            # Greedy: a sampled answer can run into the cap and end `incomplete`.
            temperature=0,
            background=True,
        )
        completed = _poll_until_terminal(client, resp.id)
        assert completed.status == "completed"

        again = client.responses.cancel(resp.id)
        assert again.status == "completed"

    def test_cancel_on_non_background_response_is_400(self, client):
        resp = client.responses.create(model="chat-capable", input="Say hi.", max_output_tokens=20)
        result = httpx.post(f"{OPENAI_API_BASE}/responses/{resp.id}/cancel", timeout=60)
        assert result.status_code == 400, result.text

    def test_previous_response_id_still_queued_is_400(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="Write a detailed 300 word essay about the history of the Roman Empire.",
            max_output_tokens=500,
            background=True,
        )
        # Fired immediately, no poll delay — the run is still queued/in_progress.
        continued = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": resp.id},
            timeout=60,
        )
        assert continued.status_code == 400, continued.text
        client.responses.cancel(resp.id)  # don't leave it running past the test

    def test_delete_in_flight_implies_cancel(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="Write a detailed 300 word essay about the history of the Roman Empire.",
            max_output_tokens=500,
            background=True,
        )
        deleted = httpx.delete(f"{OPENAI_API_BASE}/responses/{resp.id}", timeout=60)
        assert deleted.status_code == 200, deleted.text
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{resp.id}", timeout=60).status_code == 404

        # The drain task must not resurrect what was just deleted once it winds down.
        time.sleep(3)
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{resp.id}", timeout=60).status_code == 404


@pytest.mark.integration
@pytest.mark.vllm
class TestResponsesReasoning:
    """Reasoning surfaces as a first-class ``reasoning`` output item on
    /v1/responses, distinct from the off-spec ``message.reasoning`` field on chat completions."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-reasoning")

    def test_reasoning_output_item_present(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-reasoning", "input": "Briefly: what is 7 times 8?", "max_output_tokens": 512},
            timeout=120,
        )
        assert response.status_code == 200, response.text
        output = response.json()["output"]

        reasoning_items = [item for item in output if item["type"] == "reasoning"]
        assert reasoning_items, f"expected a reasoning output item, got {[i['type'] for i in output]}"
        summary_text = "".join(s["text"] for item in reasoning_items for s in item.get("summary", []))
        assert summary_text.strip(), "expected non-empty reasoning summary text"
        # `<think>` markers must be stripped before reshaping into the item.
        assert "<think>" not in summary_text

        message_items = [item for item in output if item["type"] == "message"]
        assert message_items, "expected an assistant message output item alongside reasoning"

    def test_streaming_emits_reasoning_summary_deltas(self, client):
        stream = client.responses.create(
            model="chat-reasoning",
            input="Briefly: what is 7 times 8?",
            max_output_tokens=512,
            stream=True,
        )
        reasoning_deltas: list[str] = []
        completed = None
        for event in stream:
            if event.type == "response.reasoning_summary_text.delta":
                reasoning_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response
        assert reasoning_deltas, "expected streamed reasoning summary deltas"
        assert "<think>" not in "".join(reasoning_deltas)
        assert completed is not None
        reasoning_items = [item for item in completed.output if item.type == "reasoning"]
        assert reasoning_items, "expected a reasoning output item in the completed response"

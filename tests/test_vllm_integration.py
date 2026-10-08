"""End-to-end coverage for the vllm loader: chat, streaming, tool calling,
structured output, identity-scoped prefix caching, reasoning, vision, and
embeddings, all through a real vLLM engine."""

import json

import httpx
import pytest

from tests.conftest import RED_IMAGE_DATA_URI

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"


def _collect_streaming_tool_call(stream) -> dict:
    """Drain an OpenAI streaming response and rebuild the assistant message.

    Returns content, per-index tool_calls ({id, name, arguments}), finish_reason,
    and name/args delta counts (to confirm streaming was incremental, not buffered).
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    name_deltas = 0
    args_deltas = 0
    chunks_with_tool_calls = 0

    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            chunks_with_tool_calls += 1
            for tc in delta.tool_calls:
                slot = tool_calls.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id is not None:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                    name_deltas += 1
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
                    args_deltas += 1
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "name_deltas": name_deltas,
        "args_deltas": args_deltas,
        "chunks_with_tool_calls": chunks_with_tool_calls,
    }


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_ANSWER_FORMAT = {"type": "json_schema", "json_schema": {"name": "calc", "strict": True, "schema": _ANSWER_SCHEMA}}


@pytest.mark.integration
@pytest.mark.vllm
class TestChatCapable:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_list_models(self, client):
        # With per-model deploys we only assert the currently-deployed model
        # appears in /v1/models, not the full original 7-model set.
        model_ids = [m.id for m in client.models.list().data]
        assert "chat-capable" in model_ids

    def test_chat_completion(self, client):
        completion = client.chat.completions.create(
            model="chat-capable", messages=[{"role": "user", "content": "Hello!"}], max_tokens=10
        )
        assert completion.choices[0].message.content
        assert completion.model == "chat-capable"

    def test_chat_streaming(self, client):
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Tell me a short story."}],
            max_tokens=20,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert len(chunks) > 0

    def test_tool_calling_success(self, client):
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
        )
        assert completion.choices[0].message.tool_calls
        assert completion.choices[0].message.tool_calls[0].function.name == "get_weather"

    def test_tool_calling_streaming_vllm_loader(self, client):
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
            stream=True,
        )

        collected = _collect_streaming_tool_call(stream)

        assert collected["tool_calls"], "vLLM should have streamed at least one tool call"
        call_0 = collected["tool_calls"][0]
        assert call_0["name"] == "get_weather"
        parsed_args = json.loads(call_0["arguments"])
        assert "Paris" in parsed_args.get("city", "")
        assert collected["finish_reason"] == "tool_calls"

    def test_response_format_json_object_constrains_unprompted_output(self, client):
        """No JSON hint in the prompt — a pass means the grammar constraint, not
        the prompt, produced the JSON object."""
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_response_format_json_schema_constrains_unprompted_output(self, client):
        """Same intent as the json_object test above, for json_schema."""
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}
        assert isinstance(parsed["city"], str) and parsed["city"]
        assert isinstance(parsed["country"], str) and parsed["country"]

    def test_response_format_json_schema_streaming_constrains_unprompted_output(self, client):
        """Same as above but over the streaming path."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
            max_tokens=64,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        content = "".join(chunks)
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"answer"}
        assert isinstance(parsed["answer"], str) and parsed["answer"]

    def test_response_format_coexists_with_tool_choice_none(self, client):
        """response_format alongside tools is allowed when tool_choice="none": the
        model must produce schema-constrained text, not call the tool."""
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="none",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        assert not completion.choices[0].message.tool_calls
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}

    def test_tool_choice_none_answers_in_text(self, client):
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="none",
            max_tokens=64,
        )
        message = completion.choices[0].message
        assert not message.tool_calls
        assert message.content and "<tool_call>" not in message.content

    def test_n_greater_than_one_returns_independent_choices(self, client):
        """n>1 needs its own parser instance per choice (engine_ops.make_parsers);
        a shared instance would corrupt state across choices."""
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say one random word."}],
            max_tokens=10,
            n=3,
        )
        assert len(completion.choices) == 3
        assert [c.index for c in completion.choices] == [0, 1, 2]
        assert all(c.message.content for c in completion.choices)

    def test_logprobs_returns_choice_logprobs(self, client):
        """logprobs must be built from the engine's own RequestOutput.logprobs
        (engine_ops.build_chat_logprobs), not dropped by the non-stream rewire."""
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            logprobs=True,
            top_logprobs=3,
        )
        logprobs = completion.choices[0].logprobs
        assert logprobs is not None and logprobs.content
        first = logprobs.content[0]
        assert isinstance(first.token, str) and first.token
        assert isinstance(first.logprob, float)
        assert 0 < len(first.top_logprobs) <= 3
        assert all(isinstance(tl.token, str) and isinstance(tl.logprob, float) for tl in first.top_logprobs)

    def test_streaming_n_greater_than_one_returns_independent_choices(self, client):
        """Streaming counterpart of the n>1 test above: each choice needs its own
        `Parser` instance or later choices corrupt onto shared stream state."""
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say one random word."}],
            max_tokens=10,
            n=3,
            stream=True,
        )
        content_by_index: dict[int, str] = {0: "", 1: "", 2: ""}
        finish_reasons: dict[int, str | None] = {}
        for chunk in stream:
            for choice in chunk.choices:
                if choice.delta.content:
                    content_by_index[choice.index] += choice.delta.content
                if choice.finish_reason:
                    finish_reasons[choice.index] = choice.finish_reason
        assert set(finish_reasons) == {0, 1, 2}
        assert all(content_by_index[i] for i in range(3))

    def test_streaming_logprobs_returns_choice_logprobs(self, client):
        """Streaming counterpart of the logprobs test above: logprobs must be built
        per-delta from `RequestOutput.logprobs`, not only on the final response."""
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            logprobs=True,
            top_logprobs=3,
            stream=True,
        )
        seen_logprobs = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].logprobs and chunk.choices[0].logprobs.content:
                seen_logprobs.extend(chunk.choices[0].logprobs.content)
        assert seen_logprobs
        first = seen_logprobs[0]
        assert isinstance(first.token, str) and first.token
        assert isinstance(first.logprob, float)
        assert 0 < len(first.top_logprobs) <= 3


@pytest.mark.integration
@pytest.mark.vllm
class TestIdentityScopedPrefixCache:
    """cache_salt must scope vLLM prefix-cache hits to one identity; a
    different identity sending an identical prompt must never see a hit."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    # Long enough to span multiple KV-cache blocks (16 tokens each) for an unambiguous hit.
    _PROMPT = "Summarize in one word: " + "Ray Serve is a scalable model serving library built on Ray. " * 8

    def test_prefix_cache_hit_scoped_to_identity(self, client):
        def _cached_tokens(identity: str) -> int:
            completion = client.chat.completions.create(
                model="chat-capable",
                messages=[{"role": "user", "content": self._PROMPT}],
                max_tokens=8,
                extra_headers={"X-Mship-Test-Identity": identity},
            )
            details = completion.usage.prompt_tokens_details
            return details.cached_tokens if details and details.cached_tokens else 0

        _cached_tokens("cache-test-identity-a")  # cold — populates identity A's own cache entry
        assert _cached_tokens("cache-test-identity-a") > 0, "same identity repeating a prompt should hit the cache"
        assert _cached_tokens("cache-test-identity-b") == 0, (
            "a different identity sending an identical prompt must never see a cache hit "
            "from identity A's entry — that would be the cross-identity timing leak"
        )


@pytest.mark.integration
@pytest.mark.vllm
class TestChatReasoning:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-reasoning")

    def test_reasoning_completion(self):
        """Non-streaming: the reasoning parser routes `<think>...</think>`
        to `message.reasoning`, leaving the final answer in `message.content`."""
        # Use httpx so we read the raw `reasoning` field — the OpenAI Python
        # SDK doesn't always surface it as a typed attribute.
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-reasoning",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 512,
            },
            timeout=120,
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message.get("reasoning"), f"expected reasoning content, got {message!r}"
        # `<think>` markers must be stripped from both fields.
        assert "<think>" not in (message.get("content") or "")
        assert "<think>" not in message["reasoning"]

    def test_reasoning_streaming(self):
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-reasoning",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 512,
                "stream": True,
            },
            timeout=120,
        ) as response:
            assert response.status_code == 200
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            reasoning_deltas = 0
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta") or {}
                if delta.get("reasoning"):
                    reasoning_parts.append(delta["reasoning"])
                    reasoning_deltas += 1
                if delta.get("content"):
                    content_parts.append(delta["content"])

        assert reasoning_deltas >= 1, "expected at least one reasoning delta"
        assert "".join(reasoning_parts).strip(), "expected non-empty reasoning content"
        # Reasoning markers must not leak into either stream.
        assert "<think>" not in "".join(reasoning_parts)
        assert "<think>" not in "".join(content_parts)

    def _chat(self, **body) -> dict:
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-reasoning",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
                **body,
            },
            timeout=120,
        )
        assert response.status_code == 200, response.text
        return response.json()["choices"][0]

    def test_json_schema_answer_lands_in_content(self):
        message = self._chat(response_format=_ANSWER_FORMAT)["message"]

        assert message["content"].lstrip().startswith("{")
        assert message.get("reasoning")

    def test_json_schema_streaming_answer_lands_in_content(self):
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-reasoning",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
                "response_format": _ANSWER_FORMAT,
                "stream": True,
            },
            timeout=120,
        ) as response:
            assert response.status_code == 200
            content = ""
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[len("data: ") :])["choices"][0].get("delta") or {}
                content += delta.get("content") or ""

        assert content.lstrip().startswith("{")

    def test_required_tool_choice_returns_the_call(self):
        choice = self._chat(
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
        )

        assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert choice["finish_reason"] == "tool_calls"

    def test_named_tool_choice_returns_the_call(self):
        choice = self._chat(
            messages=[{"role": "user", "content": "Hello!"}],
            tools=[_WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_thinking_off_answer_lands_in_content(self):
        message = self._chat(chat_template_kwargs={"enable_thinking": False})["message"]

        assert message["content"].strip()
        assert not message.get("reasoning")

    def test_thinking_off_streaming_answer_lands_in_content(self):
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-reasoning",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
            },
            timeout=120,
        ) as response:
            assert response.status_code == 200
            content, reasoning = "", ""
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[len("data: ") :])["choices"][0].get("delta") or {}
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning") or ""

        assert content.strip()
        assert not reasoning

    def test_thinking_off_tool_call_is_returned(self):
        choice = self._chat(
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
            chat_template_kwargs={"enable_thinking": False},
        )

        assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_responses_json_schema_answer_lands_in_a_message(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={
                "model": "chat-reasoning",
                "input": "Briefly: what is 7 times 8?",
                "max_output_tokens": 1024,
                "text": {"format": {"type": "json_schema", "name": "calc", "strict": True, "schema": _ANSWER_SCHEMA}},
            },
            timeout=120,
        )
        assert response.status_code == 200, response.text
        messages = [item for item in response.json()["output"] if item["type"] == "message"]

        assert messages[0]["content"][0]["text"].lstrip().startswith("{")


@pytest.mark.integration
@pytest.mark.vllm
class TestChatVlm:
    """End-to-end vision: a real Qwen3-VL-2B deployment receiving an
    ``image_url`` content part through the modelship gateway."""

    _QUESTION = "What color is this image? Answer in one word."

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-vlm")

    @pytest.fixture(scope="class")
    def text_only_prompt_tokens(self, client) -> int:
        completion = client.chat.completions.create(
            model="chat-vlm",
            messages=[{"role": "user", "content": self._QUESTION}],
            max_tokens=8,
        )
        return completion.usage.prompt_tokens

    @pytest.mark.parametrize(
        "part",
        [
            pytest.param({"type": "image_url", "image_url": {"url": RED_IMAGE_DATA_URI}}, id="image_url-object"),
            pytest.param({"type": "image_url", "image_url": RED_IMAGE_DATA_URI}, id="image_url-bare-string"),
            pytest.param({"type": "input_image", "image_url": RED_IMAGE_DATA_URI}, id="input_image"),
        ],
    )
    def test_every_image_part_shape_reaches_the_backend(self, client, text_only_prompt_tokens, part):
        # prompt_tokens, not the text: a dropped image still answers fluently.
        completion = client.chat.completions.create(
            model="chat-vlm",
            messages=[{"role": "user", "content": [{"type": "text", "text": self._QUESTION}, part]}],
            max_tokens=16,
            temperature=0.0,
        )
        assert completion.choices[0].message.content
        assert completion.choices[0].finish_reason in {"stop", "length"}
        assert completion.model == "chat-vlm"
        assert completion.usage.prompt_tokens > text_only_prompt_tokens + 20
        assert "red" in completion.choices[0].message.content.lower()

    def test_text_only_request_still_works_on_vlm(self, client):
        completion = client.chat.completions.create(
            model="chat-vlm",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=8,
        )
        assert completion.choices[0].message.content

    def test_image_url_streaming(self, client):
        stream = client.chat.completions.create(
            model="chat-vlm",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image briefly."},
                        {"type": "image_url", "image_url": {"url": RED_IMAGE_DATA_URI}},
                    ],
                }
            ],
            max_tokens=16,
            stream=True,
        )
        chunks = [c.choices[0].delta.content for c in stream if c.choices[0].delta.content]
        assert len(chunks) > 0


@pytest.mark.integration
def test_embeddings(client, model_deployer):
    model_deployer.deploy("embed-model")
    response = client.embeddings.create(model="embed-model", input=["Hello world", "Modelship is great"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0

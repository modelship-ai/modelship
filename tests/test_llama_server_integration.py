"""End-to-end coverage for the llama_server loader: chat, tool calling,
reasoning, response_format, vision, GPU offload, embeddings, and concurrency, all
through a real `llama-server` subprocess proxied over its native
OpenAI-compatible HTTP API."""

import concurrent.futures
import json
import time

import httpx
import pytest

import openai
from modelship.utils.cli import infer_model_name
from tests.conftest import MODEL_CONFIGS, RED_IMAGE_DATA_URI

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"

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


def _collect_streaming_tool_call(stream) -> dict:
    """Drains an OpenAI streaming response, returning a dict of content,
    tool_calls (by index), finish_reason, and content/name/args delta counts."""
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


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServer:
    """End-to-end chat, tool calling, reasoning, and concurrency via a real
    llama-server subprocess (parsing via `--jinja --reasoning-format auto`)."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server")

    def test_chat_completion(self, client):
        # chat-llama-server is reasoning-capable (Qwen3) and always emits a <think>
        # preamble first, so max_tokens needs headroom beyond just the answer.
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=256,
        )
        content = completion.choices[0].message.content
        assert content
        assert "Paris" in content

    def test_tool_calling_llama_server_loader(self, client):
        """Round-trip a tool call through llama-server's own hermes-style
        parser (`--jinja`, auto-detected from the GGUF's chat template)."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=512,  # headroom for the reasoning preamble to not crowd out the answer
        )
        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_tool_calling_streaming_llama_server_loader(self, client):
        """Stream a tool call through llama-server; verify the delta sequence
        matches the OpenAI streaming contract."""
        stream = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=512,  # headroom for the reasoning preamble to not crowd out the answer
            stream=True,
        )

        collected = _collect_streaming_tool_call(stream)

        assert collected["tool_calls"], (
            f"expected at least one streamed tool call; got content={collected['content']!r}"
        )
        call_0 = collected["tool_calls"][0]
        assert call_0["name"] == "get_weather"
        assert collected["args_deltas"] >= 1
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"

    def test_reasoning_completion_llama_server(self):
        """Non-streaming: `--reasoning-format auto` routes `<think>...</think>` to
        `message.reasoning`, the final answer to `message.content`, no marker leakage."""
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-server",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message.get("reasoning"), f"expected reasoning content, got {message!r}"
        assert "<think>" not in (message.get("content") or "")
        assert "</think>" not in (message.get("content") or "")
        assert "<think>" not in message["reasoning"]
        assert "</think>" not in message["reasoning"]

    def test_reasoning_streaming_llama_server(self):
        """Streaming: at least one delta carries `reasoning`; concatenated
        reasoning is non-empty; markers never leak into either field."""
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-server",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
                "stream": True,
            },
            timeout=300,
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
        assert "<think>" not in "".join(reasoning_parts)
        assert "</think>" not in "".join(reasoning_parts)
        assert "<think>" not in "".join(content_parts)
        assert "</think>" not in "".join(content_parts)

    def test_reasoning_with_tools_llama_server(self, client):
        """Reasoning + tool calling in one round-trip: llama-server populates both
        `message.reasoning` and `message.tool_calls`, with `finish_reason="tool_calls"`."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        # The OpenAI Python SDK exposes unknown fields via `model_extra`.
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert "<think>" not in reasoning
        tool_calls = message.tool_calls
        assert tool_calls, f"expected a tool call, got content={message.content!r}, reasoning={reasoning!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_tool_markers_inside_reasoning_not_double_counted_llama_server(self, client):
        """Coaxes the model into quoting `<tool_call>` syntax inside its `<think>`
        block, then asserts exactly one real `tool_calls` entry (no double-counting)."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant with access to tools. When you think inside "
                        "<think>...</think>, FIRST quote one example of tool-call syntax "
                        "verbatim inside angle brackets — e.g. write the literal text "
                        '<tool_call>{"name":"example","arguments":{}}</tool_call> as part '
                        "of your reasoning to remind yourself of the format. THEN decide "
                        "which real tool to call."
                    ),
                },
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning") or ""
        tool_calls = message.tool_calls or []

        assert tool_calls, (
            f"expected exactly one real tool call, got content={message.content!r}, reasoning={reasoning!r}"
        )
        assert len(tool_calls) == 1, (
            f"expected exactly one tool call (markers inside <think> must not be double-counted); "
            f"got {len(tool_calls)} calls={[tc.function.name for tc in tool_calls]}"
        )
        assert tool_calls[0].function.name == "get_weather"
        assert completion.choices[0].finish_reason == "tool_calls"

        if "<tool_call>" in reasoning:
            assert "</tool_call>" in reasoning, (
                f"reasoning has an unmatched <tool_call> marker (open without close): {reasoning!r}"
            )

    def test_response_format_with_reasoning_llama_server(self, client):
        """llama-server handles JSON-schema `response_format` combined with reasoning
        natively: `<think>...</think>` routes to `message.reasoning`, JSON to `message.content`."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "strict": True,
                },
            },
            max_tokens=1024,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert message.content
        parsed = json.loads(message.content)
        assert "answer" in parsed

    def test_tool_choice_required_is_a_noop_on_hermes_family(self, client):
        """`tool_choice: required` is grammar-enforced on harmony-style chat templates
        but a silent no-op on hermes-style ones (Qwen3): `message.content` stays reachable."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
            max_tokens=512,  # headroom for the reasoning preamble to not crowd out the answer
        )
        message = completion.choices[0].message
        assert message.content, f"expected the free-text branch to stay reachable, got message={message!r}"

    def test_named_function_tool_choice_falls_back_to_auto(self, client):
        """Object-form `tool_choice` (named-function forcing) is globally unsupported
        in llama.cpp; it silently falls back to `auto` rather than forcing or erroring."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[_WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_tokens=512,  # headroom for the reasoning preamble to not crowd out the answer
        )
        message = completion.choices[0].message
        assert message.content, f"expected the free-text branch to stay reachable, got message={message!r}"

    def test_concurrent_requests_are_not_serialized(self, client):
        """llama-server's `--parallel` slots let requests run concurrently instead
        of serializing behind a single lock; times one request, then several at once."""
        prompt = {
            "model": "chat-llama-server",
            "messages": [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
            "max_tokens": 200,
        }

        start = time.monotonic()
        client.chat.completions.create(**prompt)
        baseline = time.monotonic() - start

        concurrency = 3
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(client.chat.completions.create, **prompt) for _ in range(concurrency)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        concurrent_elapsed = time.monotonic() - start

        # Full serialization would take roughly concurrency * baseline; llama-server's
        # parallel slots should keep this well under that.
        assert concurrent_elapsed < baseline * (concurrency - 0.5), (
            f"expected concurrent requests to overlap via llama-server's parallel slots "
            f"(baseline={baseline:.1f}s, {concurrency} concurrent took {concurrent_elapsed:.1f}s)"
        )


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServerResponseFormat:
    """response_format tests using non-reasoning `chat-llama-server-plain`; the
    reasoning+response_format combo is covered separately (see TestChatLlamaServer)."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server-plain")

    def test_response_format_json_object_without_schema_is_unconstrained(self, client):
        """Bare `{"type": "json_object"}` (no `schema` key) isn't enforced by
        llama-server despite its docs; `type: json_schema` with a schema is honored."""
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)

    def test_response_format_json_schema_constrains_unprompted_output(self, client):
        """A natural-language question + json_schema → schema-conformant output."""
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
            model="chat-llama-server-plain",
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
        """Same intent on the streaming path."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        stream = client.chat.completions.create(
            model="chat-llama-server-plain",
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
        """tool_choice='none' is the safe escape valve: tools listed but inert,
        schema enforced on content output.
        """
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="none",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=512,  # headroom for the reasoning preamble to not crowd out the answer
        )
        assert not completion.choices[0].message.tool_calls
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServerGpu:
    """End-to-end GPU offload through the llama_server loader: same shape as
    `TestChatLlamaServerResponseFormat` but deployed with `num_gpus=1`, so the
    loader passes real `-ngl` offload instead of the forced `-ngl 0`."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server-gpu")

    def test_chat_completion(self, client):
        completion = client.chat.completions.create(
            model="chat-llama-server-gpu",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
        )
        content = completion.choices[0].message.content
        assert content
        assert "Paris" in content

    def test_tool_calling_llama_server_gpu_loader(self, client):
        completion = client.chat.completions.create(
            model="chat-llama-server-gpu",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
        )
        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"


@pytest.mark.integration
@pytest.mark.llama_server
def test_embeddings_llama_server(client, model_deployer):
    """Real embeddings through a live llama-server subprocess (`--embedding`);
    `test_embeddings` covers only the vllm loader."""
    model_deployer.deploy("embed-model-llama-server")
    response = client.embeddings.create(model="embed-model-llama-server", input=["Hello world", "Modelship is great"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0


@pytest.mark.integration
@pytest.mark.llama_server
def test_deploy_with_inferred_model_name(client, model_deployer):
    """`--model` with no `--name`: the name inferred from the ref is what the
    gateway serves under."""
    ref = MODEL_CONFIGS["chat-llama-server-plain"]["model"]
    model_deployer.deploy_cli("--model", ref, "--loader", "llama_server", "--usecase", "generate", "--num-cpus", "1")

    inferred = infer_model_name(ref)
    assert inferred in {m.id for m in client.models.list().data}
    completion = client.chat.completions.create(
        model=inferred,
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=16,
    )
    assert completion.choices[0].message.content


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatVlmLlamaServer:
    """Vision through a real llama-server subprocess launched with `--mmproj`, over
    every accepted image part shape. Asserted on `prompt_tokens`: a dropped image
    still answers fluently, and only the token count falls back to the text baseline."""

    _QUESTION = "What color is this image? Answer in one word."

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        # chat-llama-server rides along with no mmproj — the rejection case.
        model_deployer.deploy("chat-vlm-llama-server", "chat-llama-server")

    @pytest.fixture(scope="class")
    def text_only_prompt_tokens(self, client) -> int:
        completion = client.chat.completions.create(
            model="chat-vlm-llama-server",
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
        completion = client.chat.completions.create(
            model="chat-vlm-llama-server",
            messages=[{"role": "user", "content": [{"type": "text", "text": self._QUESTION}, part]}],
            max_tokens=16,
            temperature=0.0,
        )
        assert completion.choices[0].message.content
        assert completion.usage.prompt_tokens > text_only_prompt_tokens + 20
        assert "red" in completion.choices[0].message.content.lower()

    def test_image_url_streaming(self, client):
        stream = client.chat.completions.create(
            model="chat-vlm-llama-server",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._QUESTION},
                        {"type": "image_url", "image_url": {"url": RED_IMAGE_DATA_URI}},
                    ],
                }
            ],
            max_tokens=16,
            temperature=0.0,
            stream=True,
        )
        chunks = [c.choices[0].delta.content for c in stream if c.choices[0].delta.content]
        assert "red" in "".join(chunks).lower()

    def test_responses_input_image(self, client, text_only_prompt_tokens):
        response = client.responses.create(
            model="chat-vlm-llama-server",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self._QUESTION},
                        {"type": "input_image", "image_url": RED_IMAGE_DATA_URI},
                    ],
                }
            ],
            max_output_tokens=16,
            temperature=0.0,
        )
        assert response.usage.input_tokens > text_only_prompt_tokens + 20
        assert "red" in response.output_text.lower()

    def test_text_only_request_still_works(self, client):
        completion = client.chat.completions.create(
            model="chat-vlm-llama-server",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=8,
        )
        assert completion.choices[0].message.content

    def test_image_rejected_without_mmproj(self, client):
        with pytest.raises(openai.BadRequestError, match="does not support image input"):
            client.chat.completions.create(
                model="chat-llama-server",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._QUESTION},
                            {"type": "image_url", "image_url": {"url": RED_IMAGE_DATA_URI}},
                        ],
                    }
                ],
                max_tokens=16,
            )

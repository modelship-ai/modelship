# Responses-speaking agents

Any agent framework or custom loop built against the OpenAI **Responses API**
(`/v1/responses`) works against Modelship directly — this is the primary
surface Modelship is built around, not a translation layer bolted on top of
chat completions.

## Support matrix

`/v1/responses` is implemented on the **vLLM** and **llama.cpp**
(`llama_server`) loaders. It 404s on `diffusers`, `stable_diffusion_cpp`,
`sherpa_onnx`, and `whispercpp` deployments, since those don't have a text
generation loop to attach it to.

Supported on both loaders:

- Text output and first-class **reasoning** output items
- Client-driven **tool/function calling** (`function_call` /
  `function_call_output` round-trip)
- **Server-side conversation state** — `store`, `previous_response_id`,
  `GET`/`DELETE /v1/responses/{id}`, and `/input_items`
- Streaming and non-streaming

Hosted built-in tools (e.g. `web_search`, or a client's own proprietary hosted
tool type — OpenAI's Codex CLI sends one called `namespace`) have no
self-hosted equivalent and are dropped rather than failing the whole request;
a warning is logged per dropped tool. Encrypted reasoning
(`reasoning.encrypted_content`) isn't implemented — Modelship never returns
it. A `reasoning` item, whether sent in `input` or carried by
`previous_response_id`, goes back to the model with the assistant message that
follows it; the model's chat template decides whether it is used.

## Continuing a conversation

Each call to `/v1/responses` returns an `id`. Pass it back as
`previous_response_id` on the next call and Modelship resolves the prior
turn's state server-side before the request ever reaches the model — your
agent loop doesn't need to keep its own transcript:

```python
resp = client.responses.create(model="reasoning-qwen", input="Plan a trip to Lisbon.")
follow_up = client.responses.create(
    model="reasoning-qwen",
    input="Make it 4 days instead.",
    previous_response_id=resp.id,
)
```

State lives in a pluggable store (`MSHIP_STATE_STORE`) — in-memory by default
(shared across gateway replicas, not durable across full cluster loss), or
`redis://` for durability across restarts and node failure. See
[Architecture — Conversation state](../architecture.md#conversation-state).

## Tool calling loop

Tool execution itself is client-driven: Modelship returns `function_call`
output items, your agent runs the tool, and you send the result back as a
`function_call_output` input item (optionally with `previous_response_id` to
keep the rest of the conversation server-side):

```python
resp = client.responses.create(
    model="reasoning-qwen",
    input="What's the weather in Lisbon?",
    tools=[{
        "type": "function",
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }],
)

call = next(item for item in resp.output if item.type == "function_call")
result = get_weather(**json.loads(call.arguments))

final = client.responses.create(
    model="reasoning-qwen",
    previous_response_id=resp.id,
    input=[{
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": json.dumps(result),
    }],
)
```

## Streaming

Streaming responses emit the standard named Responses events
(`response.created`, `response.output_item.added`,
`response.output_text.delta`, `response.reasoning_summary_text.delta`,
`response.function_call_arguments.delta`, `response.output_item.done`,
`response.completed`) — reasoning and tool-call argument deltas stream live,
independent of how a given model interleaves them:

```python
with client.responses.stream(model="reasoning-qwen", input="Explain briefly.") as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

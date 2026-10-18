# LiteLLM

[LiteLLM](https://github.com/BerriAI/litellm) is an OpenAI-compatible proxy.
Put it in front of Modelship to get what Modelship deliberately doesn't do:
API keys, per-user budgets and rate limits. Tested with LiteLLM 1.105.0
against Modelship 0.9.0 on the vLLM loader.

## Config

Register each Modelship model with the `openai/` provider prefix, which makes
LiteLLM treat it as a generic OpenAI-compatible server. The name after the
prefix must match an `id` from `GET /modelship/v1/models`.

```yaml
model_list:
  - model_name: Qwen3.5-9B
    litellm_params:
      model: openai/Qwen3.5-9B
      api_base: http://<modelship-host>:8000/modelship/v1
      api_key: none
```

`api_key` is the key LiteLLM sends to Modelship, which doesn't check it, so
any non-empty value works. Run it with `litellm --config config.yaml`, then
call `http://<litellm-host>:4000/v1/chat/completions` with `"model":
"Qwen3.5-9B"`.

LiteLLM refuses to start without a master key. Set `LITELLM_MASTER_KEY`
(`sk-` plus a long random string) in its environment; clients send it as their
API key.

## Docker

Mount the config onto the file, not onto `/app`, which holds the image's
entrypoint:

```bash
docker run -d -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-... \
  -v ./config.yaml:/app/config.yaml:ro \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml
```

## Chat completions

`/v1/chat/completions` works through the plain LiteLLM route, streaming and
tool calls included. `thinking_token_budget` on a request passes through
LiteLLM and is clamped to the engine's value (see
[Limiting thinking](../model-configuration.md#limiting-thinking)).

## Responses API

`/v1/responses` through LiteLLM's own route (`http://<litellm-host>:4000/v1/responses`)
works for plain calls, but as of LiteLLM 1.105.0 it re-serializes the response:
it adds `phase`, `logprobs` and `created_by: null` fields and drops null keys
from stream events, so a client that validates the output strictly can reject
it. That route also closes WebSockets. Against the Open Responses suite it
passed 3 of 17 tests. This is LiteLLM's behavior and may change in a later
release.

If your client needs the response exactly as Modelship sends it, LiteLLM's
OpenAI passthrough route forwards the request untouched (17 of 17, WebSockets
included). Point LiteLLM at Modelship through its environment, with no `/v1`
on the base:

```bash
OPENAI_API_BASE=http://<modelship-host>:8000/modelship
OPENAI_API_KEY=none
```

and give the client `http://<litellm-host>:4000/openai_passthrough/v1` as its
base URL. For WebSockets, also set this in the config (under
`general_settings`, not `litellm_settings`):

```yaml
general_settings:
  enable_openai_websocket_passthrough: true
```

The passthrough route talks to the one backend in `OPENAI_API_BASE`, not to
the `model_list`.

## Multiple users

With no database LiteLLM has a single master key and no UI: models come from
the config file and every caller shares one key. Per-user keys, teams, budgets,
spend tracking and the `/ui` login need Postgres; set `DATABASE_URL` on the
LiteLLM container. LiteLLM has no SQLite fallback.

Then create a key per user with the master key:

```bash
curl -X POST http://<litellm-host>:4000/key/generate \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"key_alias": "alice", "user_id": "alice", "models": ["Qwen3.5-9B"], "rpm_limit": 30}'
```

Hand the returned `sk-...` key to that user as their API key. Rate limits are
the useful limit here: a budget only counts cost, and a local model has none
unless you set `input_cost_per_token` and `output_cost_per_token` in its
`litellm_params`.

## Not tested

Fallbacks between models, mid-stream failure handling, SSO, and the
`llama_server` loader through LiteLLM.

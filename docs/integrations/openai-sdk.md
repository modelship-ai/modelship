# OpenAI SDK

Any OpenAI SDK client works against Modelship unchanged — point `base_url` at
your gateway and use the model name from your `models.yaml`.

## Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/modelship/v1",
    api_key="not-needed",  # or a key your auth layer in front of Modelship accepts
)

# Chat Completions
chat = client.chat.completions.create(
    model="reasoning-qwen",
    messages=[{"role": "user", "content": "Which is larger, 9.11 or 9.9?"}],
)
print(chat.choices[0].message.content)

# Responses API — reasoning, tool calling, and stored conversation state
resp = client.responses.create(
    model="reasoning-qwen",
    input="Which is larger, 9.11 or 9.9?",
)
print(resp.output_text)

# Continue the conversation server-side, no need to resend prior turns
follow_up = client.responses.create(
    model="reasoning-qwen",
    input="Why?",
    previous_response_id=resp.id,
)
print(follow_up.output_text)
```

## JavaScript / TypeScript

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/modelship/v1",
  apiKey: "not-needed", // or a key your auth layer in front of Modelship accepts
});

const resp = await client.responses.create({
  model: "reasoning-qwen",
  input: "Which is larger, 9.11 or 9.9?",
});
console.log(resp.output_text);
```

## Embeddings, audio, and images

```python
client.embeddings.create(model="embed-model", input="hello world")
client.audio.transcriptions.create(model="stt-model", file=open("audio.wav", "rb"))
client.audio.speech.create(model="tts-model", voice="default", input="hello world")
client.images.generate(model="image-model", prompt="a red bicycle")
```

See [Supported OpenAI Endpoints](../index.md#supported-openai-endpoints) for
the full list, and [Responses-speaking agents](responses-agents.md) for the
tool-calling round trip.

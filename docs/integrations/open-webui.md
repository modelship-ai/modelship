# Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) connects to any
OpenAI-compatible backend through its **Connections** settings — Modelship
qualifies without any special support on either side.

1. In Open WebUI, go to **Settings → Admin Settings → Connections → OpenAI
   API**.
2. Set the **API Base URL** to `http://<modelship-host>:8000/modelship/v1`.
3. Set the **API Key** to any non-empty placeholder, or to a key your auth layer
   in front of Modelship accepts.
4. Save — Open WebUI queries `GET /v1/models` to populate the model picker
   with whatever is in your `models.yaml`.

Or via environment variables if you're running Open WebUI in Docker:

```bash
docker run -d \
  -e OPENAI_API_BASE_URL=http://<modelship-host>:8000/modelship/v1 \
  -e OPENAI_API_KEY=your-key-or-placeholder \
  -p 3000:8080 \
  ghcr.io/open-webui/open-webui:main
```

Chat, streaming, and image generation all work through the standard
`/v1/chat/completions` and `/v1/images/generations` endpoints. Which
higher-level Open WebUI features (e.g. its own tool-calling UI) light up
depends on Open WebUI's own client-side detection of what the connected
backend supports — that behavior is Open WebUI's, not something Modelship
controls or has specifically verified feature-by-feature.

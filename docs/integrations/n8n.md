# n8n

[n8n](https://github.com/n8n-io/n8n) can call Modelship two ways: through its
built-in OpenAI-shaped nodes, or directly via HTTP.

## Option 1 — OpenAI Chat Model node

n8n's **OpenAI Chat Model** node (used by its AI Agent / LangChain nodes)
takes an OpenAI credential with a configurable **Base URL**:

1. Create a new credential of type **OpenAI API**.
2. Set **Base URL** to `http://<modelship-host>:8000/modelship/v1`.
3. Set **API Key** to any non-empty placeholder, or to a key your auth layer in
   front of Modelship accepts.
4. Use that credential in the **OpenAI Chat Model** node and set **Model** to
   a name from your `models.yaml`.

This is the fastest path if you're building with n8n's AI Agent nodes and
want tool calling handled by n8n's own agent loop.

## Option 2 — HTTP Request node

For direct control over the request (e.g. to call `/v1/responses` for
server-side conversation state, which n8n's built-in node doesn't expose),
use the **HTTP Request** node:

- **Method:** `POST`
- **URL:** `http://<modelship-host>:8000/modelship/v1/responses`
- **Authentication:** none, or Generic → Header Auth, header `Authorization` =
  `Bearer <key>` if your auth layer in front of Modelship needs one
- **Body (JSON):**

```json
{
  "model": "reasoning-qwen",
  "input": "{{ $json.userMessage }}",
  "previous_response_id": "{{ $json.previousResponseId }}"
}
```

Feed `resp.id` from the response back into the next call's
`previous_response_id` to keep a conversation going across workflow runs
without re-sending history.

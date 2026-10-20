import hashlib
import json
from typing import Any

from modelship.openai.protocol import ChatCompletionRequest, ResponsesRequest
from modelship.openai.utils.responses import as_input_items


def chat_key(request: ChatCompletionRequest, identity: str) -> str:
    return _key(identity, request.prompt_cache_key, [], request.messages)


def responses_key(request: ResponsesRequest, identity: str) -> str:
    return _key(identity, request.prompt_cache_key, [request.instructions], as_input_items(request.input))


def _key(identity: str, prompt_cache_key: str | None, head: list[Any], items: list[dict[str, Any]]) -> str:
    for item in items:
        head.append({k: v for k, v in item.items() if k != "id"})
        if item.get("role") == "user":
            break
    text = json.dumps([identity, prompt_cache_key, _cut(head)], sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


def _cut(value: Any) -> Any:
    if isinstance(value, str):
        return value[:4096]
    if isinstance(value, dict):
        return {k: _cut(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cut(v) for v in value]
    return value

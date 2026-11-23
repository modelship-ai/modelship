"""A realtime session's conversation: its items, in order, in OpenAI's Realtime item shapes. A message item has
the shape of a Responses input item, so the LLM call takes the items as they are."""

from typing import Any

from modelship.openai.protocol.realtime import RealtimeError
from modelship.utils import random_uuid

# the content part type a text message of each role takes
_TEXT_PART = {"user": "input_text", "system": "input_text", "assistant": "output_text"}
_LATER_PARTS = frozenset({"input_audio", "input_image", "output_audio"})
_LATER_ITEMS = frozenset({"function_call", "function_call_output"})


def item_id() -> str:
    return f"item_{random_uuid()[:20]}"


def new_item(raw: Any) -> dict[str, Any] | RealtimeError:
    """The conversation item a client's *raw* item makes, or why it can't be one. Only text messages so far."""
    if not isinstance(raw, dict):
        return RealtimeError(message="An item is a JSON object.", code="invalid_value", param="item")
    kind = raw.get("type")
    if kind in _LATER_ITEMS:
        return RealtimeError(
            message=f"{kind} items are not supported yet.", code="unsupported_value", param="item.type"
        )
    if kind != "message":
        return RealtimeError(message=f"Unknown item type {kind!r}.", code="invalid_value", param="item.type")
    role = raw.get("role")
    if role not in _TEXT_PART:
        return RealtimeError(
            message="A message's role is user, assistant or system.", code="invalid_value", param="item.role"
        )
    content = raw.get("content")
    if not isinstance(content, list):
        return RealtimeError(message="A message's content is a list.", code="invalid_value", param="item.content")
    parts = []
    for i, part in enumerate(content):
        param = f"item.content[{i}]"
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in _LATER_PARTS:
            return RealtimeError(message=f"{kind} content is not supported yet.", code="unsupported_value", param=param)
        if kind != _TEXT_PART[role] or not isinstance(part.get("text"), str):
            return RealtimeError(
                message=f"A {role} message's content parts are {_TEXT_PART[role]} with a text.",
                code="invalid_value",
                param=param,
            )
        parts.append({"type": kind, "text": part["text"]})
    given = raw.get("id")
    if given is not None and not isinstance(given, str):
        return RealtimeError(message="An item id is a string.", code="invalid_value", param="item.id")
    return {
        "id": given or item_id(),
        "type": "message",
        "object": "realtime.item",
        "status": "completed",
        "role": role,
        "content": parts,
    }


def as_input(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*items* as Responses input items."""
    return [{"type": item["type"], "role": item["role"], "content": item["content"]} for item in items]


class Conversation:
    """The session's items, in order."""

    def __init__(self) -> None:
        self.id = f"conv_{random_uuid()[:20]}"
        self._items: list[dict[str, Any]] = []

    @property
    def items(self) -> list[dict[str, Any]]:
        return list(self._items)

    def get(self, item_id: Any) -> dict[str, Any] | None:
        return next((item for item in self._items if item["id"] == item_id), None)

    def before(self, item_id: str) -> str | None:
        """The id of the item before *item_id*, None when it is first."""
        position = next(i for i, item in enumerate(self._items) if item["id"] == item_id)
        return self._items[position - 1]["id"] if position else None

    def insert(self, item: dict[str, Any], previous_item_id: str | None = None) -> str | None:
        """Puts *item* after the item *previous_item_id* names: first for "root", last for None. The id of the item
        now before it, None when it is first. *previous_item_id* names an item of the conversation."""
        if previous_item_id is None:
            position = len(self._items)
        elif previous_item_id == "root":
            position = 0
        else:
            position = next(i for i, other in enumerate(self._items) if other["id"] == previous_item_id) + 1
        self._items.insert(position, item)
        return self._items[position - 1]["id"] if position else None

    def delete(self, item_id: Any) -> bool:
        for i, item in enumerate(self._items):
            if item["id"] == item_id:
                del self._items[i]
                return True
        return False

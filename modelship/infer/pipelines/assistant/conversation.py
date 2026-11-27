"""A realtime session's conversation: its items, in order."""

from typing import Any

from modelship.openai.protocol.realtime import AssistantMessage, Item, UserMessage
from modelship.utils import random_uuid

# each audio part as the text part of its transcript
_AS_TEXT = {"input_audio": "input_text", "output_audio": "output_text"}


def _input_part(part: dict[str, Any]) -> dict[str, Any] | None:
    kind = _AS_TEXT.get(part["type"])
    if kind is None:
        return part
    return {"type": kind, "text": part["transcript"]} if part["transcript"] else None


def as_input(items: list[Item]) -> list[dict[str, Any]]:
    """*items* as Responses input items: without the Realtime-only `id`, `object` and `status`. Audio, the client's
    or the answer's, becomes its transcript; a message left with no text (a turn that failed or is still being
    transcribed) is left out."""
    found = []
    for item in items:
        dumped = item.model_dump(exclude={"id", "object", "status"})
        if isinstance(item, UserMessage | AssistantMessage):
            dumped["content"] = [part for part in map(_input_part, dumped["content"]) if part is not None]
            if not dumped["content"]:
                continue
        found.append(dumped)
    return found


class Conversation:
    """The session's items, in order."""

    def __init__(self) -> None:
        self.id = f"conv_{random_uuid()[:20]}"
        self._items: list[Item] = []

    @property
    def items(self) -> list[Item]:
        return list(self._items)

    def get(self, item_id: str) -> Item | None:
        return next((item for item in self._items if item.id == item_id), None)

    def before(self, item_id: str) -> str | None:
        """The id of the item before *item_id*, None when it is first."""
        position = next(i for i, item in enumerate(self._items) if item.id == item_id)
        return self._items[position - 1].id if position else None

    def insert(self, item: Item, previous_item_id: str | None = None) -> str | None:
        """Puts *item* after the item *previous_item_id* names: first for "root", last for None. The id of the item
        now before it, None when it is first. *previous_item_id* names an item of the conversation."""
        if previous_item_id is None:
            position = len(self._items)
        elif previous_item_id == "root":
            position = 0
        else:
            position = next(i for i, other in enumerate(self._items) if other.id == previous_item_id) + 1
        self._items.insert(position, item)
        return self._items[position - 1].id if position else None

    def replace(self, item: Item) -> None:
        """Puts *item* in place of the item with its id, which the conversation holds."""
        position = next(i for i, other in enumerate(self._items) if other.id == item.id)
        self._items[position] = item

    def delete(self, item_id: str) -> bool:
        for i, item in enumerate(self._items):
            if item.id == item_id:
                del self._items[i]
                return True
        return False

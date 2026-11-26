"""The client's `conversation.item.*` events: adding items to the session's conversation, reading and deleting them,
and truncating a spoken answer to what the client played of it."""

from collections.abc import Awaitable, Callable

from modelship.infer.pipelines.assistant.conversation import Conversation
from modelship.infer.pipelines.assistant.output.speech import Spoken
from modelship.openai.protocol.realtime import (
    AssistantMessage,
    InputAudio,
    Item,
    ItemCreate,
    ItemDelete,
    ItemRetrieve,
    ItemTruncate,
    OutputAudio,
    RealtimeError,
    UserMessage,
    server_event,
)

Send = Callable[[dict], Awaitable[None]]
# stops the response that speaks message item_id, if one does, and waits for it to end
StopSpeaking = Callable[[str], Awaitable[None]]


def missing_item(item_id: str, param: str) -> RealtimeError:
    """The error for a reference, at *param*, to an item the conversation doesn't hold."""
    return RealtimeError(message=f"No item {item_id!r} in the conversation.", code="item_not_found", param=param)


def audio_in_item(item: Item, param: str) -> RealtimeError | None:
    """The error for audio sent inside an item, not through the input audio buffer."""
    if isinstance(item, UserMessage):
        for i, part in enumerate(item.content):
            if isinstance(part, InputAudio):
                return RealtimeError(
                    message="Audio in an item is not supported yet; append it to the input audio buffer.",
                    code="invalid_value",
                    param=f"{param}.content[{i}]",
                )
    return None


class ConversationItems:
    """Handles the `conversation.item.*` events on *conversation*, answering through *send*. *spoken* holds what
    each spoken message said, by item id, and *stop_speaking* stops an answer still being spoken when the client
    truncates it."""

    def __init__(self, conversation: Conversation, send: Send, spoken: dict[str, Spoken], stop_speaking: StopSpeaking):
        self._conversation = conversation
        self._send = send
        self._spoken = spoken
        self._stop_speaking = stop_speaking

    async def create(self, event: ItemCreate) -> RealtimeError | None:
        item, previous = event.item, event.previous_item_id
        if (error := audio_in_item(item, "item")) is not None:
            return error
        if previous is not None and previous != "root" and self._conversation.get(previous) is None:
            return missing_item(previous, "previous_item_id")
        if self._conversation.get(item.id) is not None:
            return RealtimeError(
                message=f"The conversation already has an item {item.id!r}.", code="invalid_value", param="item.id"
            )
        before = self._conversation.insert(item, previous)
        await self._send(server_event("conversation.item.added", previous_item_id=before, item=item))
        await self._send(server_event("conversation.item.done", previous_item_id=before, item=item))
        return None

    async def retrieve(self, event: ItemRetrieve) -> RealtimeError | None:
        item = self._conversation.get(event.item_id)
        if item is None:
            return missing_item(event.item_id, "item_id")
        await self._send(server_event("conversation.item.retrieved", item=item))
        return None

    async def delete(self, event: ItemDelete) -> RealtimeError | None:
        if not self._conversation.delete(event.item_id):
            return missing_item(event.item_id, "item_id")
        self._spoken.pop(event.item_id, None)
        await self._send(server_event("conversation.item.deleted", item_id=event.item_id))
        return None

    async def truncate(self, event: ItemTruncate) -> RealtimeError | None:
        item = self._conversation.get(event.item_id)
        if item is None:
            return missing_item(event.item_id, "item_id")
        if not isinstance(item, AssistantMessage):
            return RealtimeError(
                message="Only an assistant message can be truncated.", code="invalid_value", param="item_id"
            )
        index = event.content_index
        part = item.content[index] if index < len(item.content) else None
        if not isinstance(part, OutputAudio):
            return RealtimeError(
                message=f"Part {index} of item {item.id!r} holds no audio.", code="invalid_value", param="content_index"
            )
        # the client stopped playing an answer still being spoken
        await self._stop_speaking(item.id)
        spoken = self._spoken.get(item.id, Spoken())
        played = round(spoken.ms)
        if event.audio_end_ms > played:
            return RealtimeError(
                message=f"audio_end_ms is past the end of the part's audio, at {played} ms.",
                code="invalid_value",
                param="audio_end_ms",
            )
        part.transcript = spoken.truncate(event.audio_end_ms, part.transcript)
        await self._send(
            server_event(
                "conversation.item.truncated",
                item_id=item.id,
                content_index=index,
                audio_end_ms=event.audio_end_ms,
            )
        )
        return None

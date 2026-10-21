import pytest

from modelship.openai.protocol import ChatCompletionRequest, ResponsesRequest
from modelship.openai.state import responses as responses_state
from modelship.openai.utils import affinity
from modelship.openai.utils import responses as responses_utils
from modelship.state import MemoryStoreActor

SYSTEM = {"role": "system", "content": "You are helpful."}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _chat(*messages: dict, **fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=list(messages), **fields)


class TestChatKey:
    def test_later_turns_keep_the_key(self):
        first = _chat(SYSTEM, _user("fix my bug"))
        later = _chat(SYSTEM, _user("fix my bug"), {"role": "assistant", "content": "done"}, _user("add tests"))
        assert affinity.chat_key(first, "alice") == affinity.chat_key(later, "alice")

    def test_conversations_sharing_a_system_prompt_get_their_own_keys(self):
        bug = affinity.chat_key(_chat(SYSTEM, _user("fix my bug")), "alice")
        assert bug != affinity.chat_key(_chat(SYSTEM, _user("write a poem")), "alice")

    def test_identity_and_prompt_cache_key_change_the_key(self):
        request = _chat(SYSTEM, _user("hi"))
        keys = {
            affinity.chat_key(request, "alice"),
            affinity.chat_key(request, "bob"),
            affinity.chat_key(request.model_copy(update={"prompt_cache_key": "session-1"}), "alice"),
        }
        assert len(keys) == 3

    def test_strings_count_up_to_4096_characters(self):
        def key(text: str) -> str:
            return affinity.chat_key(_chat(_user(text)), "alice")

        assert key("a" * 4096 + "b") == key("a" * 4096 + "c")
        assert key("a" * 4095 + "b") != key("a" * 4095 + "c")


class TestResponsesKey:
    @pytest.mark.asyncio
    async def test_a_stored_turn_continues_with_its_key(self):
        store = MemoryStoreActor.__ray_metadata__.modified_class()
        first = ResponsesRequest(input="My name is Alex.", instructions="Be brief.")
        answer = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi Alex."}]}
        await responses_state.write_async(
            store,
            "alice",
            "resp_1",
            response={"id": "resp_1", "object": "response", "status": "completed", "output": [answer]},
            parent=None,
            input_items=responses_utils.as_input_items(first.input),
        )
        history, _, _ = await responses_utils.resolve_previous(store, "alice", "resp_1")
        second = ResponsesRequest(input=[*history, _user("Where do I live?")], instructions="Be brief.")

        assert affinity.responses_key(first, "alice") == affinity.responses_key(second, "alice")

    def test_an_items_id_is_not_part_of_the_key(self):
        bare = ResponsesRequest(input=[{"type": "message", "role": "user", "content": "hi"}])
        stored = ResponsesRequest(input=[{"type": "message", "role": "user", "content": "hi", "id": "msg_1"}])
        assert affinity.responses_key(bare, "alice") == affinity.responses_key(stored, "alice")

    def test_instructions_change_the_key(self):
        brief = ResponsesRequest(input="hi", instructions="Be brief.")
        assert affinity.responses_key(brief, "alice") != affinity.responses_key(ResponsesRequest(input="hi"), "alice")

from unittest.mock import MagicMock, call, patch

import pytest

from modelship import metrics
from modelship.metrics import RequestTracker, _CreatedStampedCounter
from modelship.openai.protocol import create_error_response, encode_sse_event
from modelship.openai.protocol.responses.streaming import RESPONSE_FAILED_SSE_PREFIX
from modelship.openai.utils.chat import ERROR_SSE_PREFIX, encode_error_sse


def test_a_series_is_stamped_with_its_creation_time_before_its_first_count():
    calls = MagicMock()
    counter = _CreatedStampedCounter(calls.inner, calls.created)

    with patch("modelship.metrics.time.time", return_value=1800000000.0):
        counter.inc(2, tags={"gateway": "g", "action": "add"})

    assert calls.mock_calls == [
        call.created.inc(1800000000.0, {"gateway": "g", "action": "add"}),
        call.inner.inc(2, {"gateway": "g", "action": "add"}),
    ]


def test_each_series_is_stamped_once():
    inner, created = MagicMock(), MagicMock()
    counter = _CreatedStampedCounter(inner, created)

    counter.inc(tags={"gateway": "g", "action": "add"})
    counter.inc(tags={"action": "add", "gateway": "g"})
    counter.inc(tags={"gateway": "g", "action": "remove"})

    assert [c.args[1] for c in created.inc.call_args_list] == [
        {"gateway": "g", "action": "add"},
        {"gateway": "g", "action": "remove"},
    ]
    assert inner.inc.call_count == 3


@pytest.mark.skipif(not metrics._ENABLED, reason="MSHIP_METRICS is off")
@pytest.mark.parametrize(
    "name",
    ["DEPLOY_REQUESTS_TOTAL", "DEPLOY_MODELS_CHANGED_TOTAL", "MODEL_LOAD_FAILURES_TOTAL", "BACKEND_DEATHS_TOTAL"],
)
def test_the_deploy_coordinators_counters_are_stamped(name):
    assert isinstance(getattr(metrics, name), _CreatedStampedCounter)


def test_a_request_is_recorded_once(request_outcomes):
    tracker = RequestTracker("m", "create_chat_completion")

    tracker.finish("error", "stream_error")
    tracker.finish("ok")

    assert request_outcomes() == [("m", "create_chat_completion", "error", "stream_error")]


def test_a_requests_duration_carries_its_status():
    with patch.object(metrics, "REQUEST_DURATION_SECONDS") as duration:
        RequestTracker("m", "create_embeddings").finish("cancelled")

    assert duration.observe.call_args.kwargs["tags"] == {
        "model": "m",
        "operation": "create_embeddings",
        "status": "cancelled",
    }


def test_an_error_chunk_starts_with_the_prefix_the_gateway_detects():
    assert encode_error_sse(create_error_response("boom")).startswith(ERROR_SSE_PREFIX)


def test_a_failed_event_starts_with_the_prefix_the_gateway_detects():
    assert encode_sse_event({"type": "response.failed", "response": {}}).startswith(RESPONSE_FAILED_SSE_PREFIX)

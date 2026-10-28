from unittest.mock import MagicMock, call, patch

import pytest

from modelship import metrics
from modelship.metrics import _CreatedStampedCounter


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

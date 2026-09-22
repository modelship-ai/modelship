import io
import urllib.error

import pytest

from modelship import uninstall
from modelship.state import state_store_from_uri

_BASE = "/apis/ray.io/v1/namespaces/ns"


class _FakeApi:
    namespace = "ns"

    def __init__(self, delete_status=200, gets_until_gone=1):
        self.calls = []
        self._delete_status = delete_status
        self._gets_left = gets_until_gone

    def status(self, method, path):
        self.calls.append((method, path))
        if method == "GET":
            self._gets_left -= 1
            return 404 if self._gets_left <= 0 else 200
        return self._delete_status


class TestDeleteCluster:
    def test_deletes_the_rayjob_then_waits_out_the_raycluster(self):
        api = _FakeApi(gets_until_gone=2)
        assert uninstall.delete_cluster(api, "rc", "rj", poll_seconds=0)
        assert api.calls == [
            ("DELETE", f"{_BASE}/rayjobs/rj"),
            ("DELETE", f"{_BASE}/rayclusters/rc"),
            ("GET", f"{_BASE}/rayclusters/rc"),
            ("GET", f"{_BASE}/rayclusters/rc"),
        ]

    def test_a_missing_raycluster_needs_no_wait(self):
        api = _FakeApi(delete_status=404)
        assert uninstall.delete_cluster(api, "rc", "rj", poll_seconds=0)
        assert [m for m, _ in api.calls] == ["DELETE", "DELETE"]

    def test_gives_up_after_the_wait(self):
        api = _FakeApi(gets_until_gone=10**9)
        assert not uninstall.delete_cluster(api, "rc", "rj", wait_seconds=0.05, poll_seconds=0.01)


def _store(uri, server):
    fakeredis = pytest.importorskip("fakeredis")
    store = state_store_from_uri(uri)
    store.inner._sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    return store


class TestPurgeState:
    def test_deletes_only_its_namespace(self):
        fakeredis = pytest.importorskip("fakeredis")
        server = fakeredis.FakeServer()
        mine = _store("redis://fake/0?namespace=mine", server)
        other = _store("redis://fake/0?namespace=other", server)
        mine.set("effective/gw", 1)
        mine.set("responses/r1", 2)
        other.set("effective/gw", 3)
        assert uninstall.purge_state(mine) == 2
        assert mine.list("") == []
        assert other.get("effective/gw") == 3

    def test_refuses_a_store_without_a_namespace(self):
        with pytest.raises(ValueError, match="without a namespace"):
            uninstall.purge_state(state_store_from_uri("redis://fake/0"))


class TestKubeApi:
    @pytest.fixture
    def api(self, tmp_path, monkeypatch):
        (tmp_path / "token").write_text("tok\n")
        (tmp_path / "namespace").write_text("ns")
        monkeypatch.setattr(uninstall.ssl, "create_default_context", lambda cafile: None)
        return uninstall.KubeApi(str(tmp_path))

    def test_sends_the_token_and_returns_the_status(self, api, monkeypatch):
        seen = {}

        class _Resp(io.BytesIO):
            status = 202

        def urlopen(req, context, timeout):
            seen["auth"], seen["method"], seen["url"] = req.get_header("Authorization"), req.method, req.full_url
            return _Resp()

        monkeypatch.setattr(uninstall.urllib.request, "urlopen", urlopen)
        assert api.status("DELETE", "/x") == 202
        assert seen == {"auth": "Bearer tok", "method": "DELETE", "url": "https://kubernetes.default.svc/x"}

    @pytest.mark.parametrize(("code", "raises"), [(404, False), (403, True)])
    def test_404_is_a_status_other_errors_raise(self, api, monkeypatch, code, raises):
        def urlopen(req, context, timeout):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr(uninstall.urllib.request, "urlopen", urlopen)
        if raises:
            with pytest.raises(urllib.error.HTTPError):
                api.status("GET", "/x")
        else:
            assert api.status("GET", "/x") == 404

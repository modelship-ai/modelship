import shutil
import socket
import subprocess
import tempfile
import time

import pytest
import pytest_asyncio
import redis

from modelship.state import StateStoreUnavailableError
from modelship.state.redis import RedisStateStore
from tests import test_state

pytestmark = [pytest.mark.integration, pytest.mark.redis]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def redis_url():
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.fail("redis-server is not on PATH")
    port = _free_port()
    url = f"redis://127.0.0.1:{port}/0"
    with tempfile.TemporaryDirectory() as data_dir:
        server = subprocess.Popen(
            [binary, "--port", str(port), "--bind", "127.0.0.1", "--save", "", "--appendonly", "no", "--dir", data_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = redis.Redis.from_url(url)
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    client.ping()
                    break
                except redis.exceptions.ConnectionError:
                    if server.poll() is not None or time.monotonic() > deadline:
                        pytest.fail("redis-server did not start")
                    time.sleep(0.05)
            yield url
        finally:
            client.close()
            server.terminate()
            server.wait(timeout=10)


@pytest_asyncio.fixture
async def store(redis_url):
    store = RedisStateStore(redis_url)
    yield store
    if store._async_client is not None:
        await store._async_client.aclose()
    if store._sync_client is not None:
        store._sync_client.close()
    with redis.Redis.from_url(redis_url) as client:
        client.flushdb()


class TestRealRedisBackends(test_state.TestBackends):
    pass


class TestRealRedis:
    def test_ttl_expires(self, store):
        store.set("k", {"x": 1}, ttl_seconds=0.2)
        assert store.get("k") == {"x": 1}
        time.sleep(0.3)
        assert store.get("k") is None

    @pytest.mark.asyncio
    async def test_append_at_a_taken_sequence_number_is_refused(self, store):
        await store.append_async("log/a", {"sequence_number": 0})
        await store.append_async("log/a", {"sequence_number": 1})
        for taken in (1, 0):
            with pytest.raises(redis.exceptions.ResponseError):
                await store.append_async("log/a", {"sequence_number": taken})
        assert await store.read_from_async("log/a") == [{"sequence_number": 0}, {"sequence_number": 1}]

    def test_unreachable_server_raises_unavailable(self):
        store = RedisStateStore(f"redis://127.0.0.1:{_free_port()}/0")
        with pytest.raises(StateStoreUnavailableError):
            store.get("k")

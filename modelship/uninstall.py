"""The chart's pre-delete hook: deletes the release's RayJob and RayCluster, waits for the
RayCluster to go, then deletes every modelship state key in the release's namespace."""

from __future__ import annotations

import argparse
import ssl
import sys
import time
import urllib.error
import urllib.request

from modelship.logging import configure_logging, get_logger
from modelship.state import StateStore, get_state_store
from modelship.state.redis import RedisStateStore

logger = get_logger("uninstall")

_API = "https://kubernetes.default.svc"
_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
WAIT_SECONDS = 180.0
_POLL_SECONDS = 2.0


class KubeApi:
    """The API server, called as the pod's service account."""

    def __init__(self, sa_dir: str = _SA_DIR) -> None:
        with open(f"{sa_dir}/token") as f:
            self._token = f.read().strip()
        with open(f"{sa_dir}/namespace") as f:
            self.namespace = f.read().strip()
        self._ssl = ssl.create_default_context(cafile=f"{sa_dir}/ca.crt")

    def status(self, method: str, path: str) -> int:
        """HTTP status of *method* on *path*; a 404 is returned, any other error raised."""
        req = urllib.request.Request(_API + path, method=method, headers={"Authorization": f"Bearer {self._token}"})
        try:
            with urllib.request.urlopen(req, context=self._ssl, timeout=10) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404
            raise


def delete_cluster(
    api: KubeApi, cluster: str, job: str, wait_seconds: float = WAIT_SECONDS, poll_seconds: float = _POLL_SECONDS
) -> bool:
    """Deletes the RayJob, then the RayCluster; False if the RayCluster outlives *wait_seconds*."""
    base = f"/apis/ray.io/v1/namespaces/{api.namespace}"
    api.status("DELETE", f"{base}/rayjobs/{job}")
    cluster_path = f"{base}/rayclusters/{cluster}"
    if api.status("DELETE", cluster_path) == 404:
        return True
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        if api.status("GET", cluster_path) == 404:
            return True
    return False


def purge_state(store: StateStore) -> int:
    """Deletes every key in *store*'s namespace and returns how many."""
    inner = getattr(store, "inner", store)
    if not isinstance(inner, RedisStateStore) or not inner.namespace:
        raise ValueError("refusing to purge a state store without a namespace: its keys may belong to other clusters")
    keys = store.list("")
    for key in keys:
        store.delete(key)
    return len(keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m modelship.uninstall")
    parser.add_argument("raycluster")
    parser.add_argument("rayjob")
    args = parser.parse_args(argv)
    configure_logging()
    if delete_cluster(KubeApi(), args.raycluster, args.rayjob):
        logger.info("RayCluster %s deleted", args.raycluster)
    else:
        logger.warning("RayCluster %s still present after %ds; deleting state anyway", args.raycluster, WAIT_SECONDS)
    logger.info("Deleted %d modelship state key(s)", purge_state(get_state_store()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

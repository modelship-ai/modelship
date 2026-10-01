"""`mship deploy --ray-dashboard-url`: runs `mship deploy` on a cluster's head through its Ray dashboard's job API."""

from __future__ import annotations

import http.client
import json
import os
import re
import shlex
import signal
import sys
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, NoReturn

from modelship.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from ray.job_submission import JobDetails, JobSubmissionClient

logger = get_logger("startup")

# Ray hands a submitted job's metadata to its driver as JSON in this env var (job_supervisor.py).
RAY_JOB_CONFIG_ENV_VAR = "RAY_JOB_CONFIG_JSON_ENV_VAR"
MODELS_YAML_KEY = "mship_models_yaml"
# That env var must fit in one 128 KiB env string.
MAX_MODELS_YAML_BYTES = 96 * 1024
EXIT_UNREACHABLE = 3
# `mship deploy` on a cluster without one; a head inside `mship start` has none yet.
EXIT_NO_DEPLOY_COORDINATOR = 4

_HEAD = {"node:__internal_head__": 0.001}
_CLIENT_FLAGS = ("--ray-dashboard-url", "--config")
_POLL_S = 1.0
_RECONNECT_S = 2.0
_HEAD_STARTING_S = 60.0


def job_models_yaml() -> str:
    """The models.yaml the client shipped with this job, read on the head."""
    config = json.loads(os.environ.get(RAY_JOB_CONFIG_ENV_VAR) or "{}")
    text = config.get("metadata", {}).get(MODELS_YAML_KEY)
    if text is None:
        raise ValueError("--config-from-job: this process is not a Ray job that carries a models.yaml.")
    return text


def head_argv(argv: list[str], ships_config: bool) -> list[str]:
    """*argv* as the head runs it: without the client's own flags."""
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg.split("=", 1)[0] in _CLIENT_FLAGS:
            skip = "=" not in arg
            continue
        out.append(arg)
    return [*out, "--config-from-job"] if ships_config else out


def entrypoint(argv: list[str], ships_config: bool) -> str:
    # MSHIP_ENGINE_PYTHON is exported by `mship start`; the emptied URL keeps the head's deploy local.
    return (
        'MSHIP_RAY_DASHBOARD_URL= "${MSHIP_ENGINE_PYTHON:?this head was not started by mship start}" '
        "-m modelship.launcher deploy " + shlex.join(head_argv(argv, ships_config))
    )


def run(url: str, argv: list[str], config_path: str | None) -> None:
    """Submits this deploy to the head at *url*, follows it and exits with its outcome."""
    url = url.rstrip("/")
    token = os.environ.get("MSHIP_RAY_AUTH_TOKEN")
    # Ray's SDK adds a token itself in token mode, from local files; only MSHIP_RAY_AUTH_TOKEN is ever sent.
    os.environ.pop("RAY_AUTH_MODE", None)
    import requests
    from ray.exceptions import AuthenticationError
    from ray.job_submission import JobSubmissionClient

    configure_logging()
    models_yaml = _read_models_yaml(config_path)
    _check_dashboard(url, token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        client = JobSubmissionClient(url, headers=headers)
    except ConnectionError:
        _exit_unreachable(url, "connection refused")
    except (requests.HTTPError, RuntimeError, AuthenticationError) as e:
        _exit_http(url, e)
    try:
        _submit_and_follow(client, url, argv, models_yaml)
    except (RuntimeError, AuthenticationError) as e:
        _exit_http(url, e)


def _read_models_yaml(config_path: str | None) -> str | None:
    if config_path is None:
        return None
    try:
        with open(config_path) as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"error: can't read --config {config_path}: {e.strerror}.")
    if len(text.encode()) > MAX_MODELS_YAML_BYTES:
        sys.exit(f"error: {config_path} is over {MAX_MODELS_YAML_BYTES // 1024} KiB, the most a remote deploy carries.")
    return text


def _check_dashboard(url: str, token: str | None) -> None:
    try:
        with urllib.request.urlopen(f"{url}/api/authentication_mode", timeout=10) as resp:
            mode = json.load(resp).get("authentication_mode")
    except urllib.error.HTTPError as e:
        _exit_unreachable(url, f"HTTP {e.code}")
    except http.client.HTTPException:
        _exit_unreachable(url, "not an HTTP reply")
    except (urllib.error.URLError, OSError, ValueError) as e:
        _exit_unreachable(url, str(getattr(e, "reason", e)))
    if mode == "token" and not token:
        sys.exit(f"error: the cluster at {url} requires its Ray auth token: set MSHIP_RAY_AUTH_TOKEN.")


def _submit_and_follow(client: JobSubmissionClient, url: str, argv: list[str], models_yaml: str | None) -> None:
    import requests

    submitted: list[str] = []

    def _stop_following(sig, _frame) -> None:
        running = f"; Ray job {submitted[-1]} keeps running on the cluster" if submitted else ""
        logger.info("Stopped following (signal %s)%s.", sig, running)
        sys.exit(130)

    signal.signal(signal.SIGINT, _stop_following)
    signal.signal(signal.SIGTERM, _stop_following)
    metadata = {MODELS_YAML_KEY: models_yaml} if models_yaml is not None else None
    starting_since: float | None = None
    while True:
        try:
            job_id = client.submit_job(
                entrypoint=entrypoint(argv, models_yaml is not None), entrypoint_resources=_HEAD, metadata=metadata
            )
        except requests.RequestException as e:
            _exit_unreachable(url, str(e))
        submitted.append(job_id)
        logger.info("Submitted Ray job %s to %s.", job_id, url)
        info, lost = _follow(client, job_id)
        if info is not None and info.status == "SUCCEEDED":
            sys.exit(0)
        if info is not None and info.driver_exit_code == EXIT_NO_DEPLOY_COORDINATOR:
            now = time.monotonic()
            starting_since = now if starting_since is None else starting_since
            if now - starting_since < _HEAD_STARTING_S:
                logger.warning("The cluster's head is still starting; submitting the deploy again.")
                time.sleep(_RECONNECT_S)
                continue
        if info is not None and info.driver_exit_code is not None:
            sys.exit(info.driver_exit_code)
        if lost:
            # The job died with the head; a deploy is declarative, so running it again is safe.
            logger.warning("The cluster's head went away during Ray job %s; submitting it again.", job_id)
            continue
        assert info is not None
        sys.exit(f"error: Ray job {job_id} {info.status.lower()}: {info.message}")


def _follow(client: JobSubmissionClient, job_id: str) -> tuple[JobDetails | None, bool]:
    """Prints the job's new log text until it ends. Returns its final info, or None when the head came back
    without it, and whether the dashboard went away meanwhile."""
    import requests

    printed = 0
    lost = False
    while True:
        try:
            info = client.get_job_info(job_id)
            logs = client.get_job_logs(job_id)
        except requests.RequestException:
            if not lost:
                logger.warning("Lost the Ray dashboard; reconnecting.")
            lost = True
            time.sleep(_RECONNECT_S)
            continue
        except RuntimeError as e:
            if lost and _http_status(e) == 404:
                return None, lost
            raise
        if len(logs) > printed:
            sys.stdout.write(logs[printed:])
            sys.stdout.flush()
        printed = len(logs)
        if info.status in ("SUCCEEDED", "FAILED", "STOPPED"):
            return info, lost
        time.sleep(_POLL_S)


def _http_status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    if response is not None:
        return response.status_code
    match = re.match(r"Request failed with status code (\d+)", str(error))
    return int(match.group(1)) if match else None


def _exit_http(url: str, error: Exception) -> NoReturn:
    if type(error).__name__ == "AuthenticationError" or _http_status(error) in (401, 403):
        sys.exit(f"error: the cluster at {url} rejected the Ray auth token: check MSHIP_RAY_AUTH_TOKEN.")
    sys.exit(f"error: the Ray dashboard at {url} failed the request: {error}")


def _exit_unreachable(url: str, reason: str) -> NoReturn:
    print(
        f"error: no Ray dashboard answers at {url} ({reason}). --ray-dashboard-url takes the dashboard's URL, "
        "e.g. http://head:8265, not the GCS address `mship join` takes.",
        file=sys.stderr,
    )
    sys.exit(EXIT_UNREACHABLE)

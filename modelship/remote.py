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
# That env var must fit in one 128 KiB env string; the rest leaves room for Ray's own keys.
MAX_MODELS_YAML_JSON_BYTES = 96 * 1024
EXIT_UNREACHABLE = 3
# `mship deploy` on a cluster without one; a head inside `mship start` has none yet.
EXIT_NO_DEPLOY_COORDINATOR = 4
# How long `mship deploy` waits for the head: a local node, the dashboard, the deploy coordinator.
HEAD_WAIT_S = 300

_HEAD = {"node:__internal_head__": 0.001}
_CLIENT_FLAGS = ("--ray-dashboard-url", "--config")
_POLL_S = 1.0


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

    deadline = time.monotonic() + HEAD_WAIT_S
    configure_logging()
    models_yaml = _read_models_yaml(config_path)
    _wait_for_dashboard(url, token, deadline)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        _submit_and_follow(url, headers, argv, models_yaml, deadline)
    except (requests.HTTPError, RuntimeError, AuthenticationError) as e:
        _exit_http(url, e)


def _read_models_yaml(config_path: str | None) -> str | None:
    if config_path is None:
        return None
    try:
        with open(config_path) as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"error: can't read --config {config_path}: {e.strerror}.")
    # As Ray escapes it: up to 6 bytes per character.
    if len(json.dumps(text)) > MAX_MODELS_YAML_JSON_BYTES:
        sys.exit(
            f"error: {config_path} is over {MAX_MODELS_YAML_JSON_BYTES // 1024} KiB once JSON-escaped, "
            "the most a remote deploy carries."
        )
    return text


def _wait_for_dashboard(url: str, token: str | None, deadline: float) -> None:
    def _stop_waiting(sig, _frame) -> None:
        logger.info("Stopped waiting (signal %s).", sig)
        sys.exit(130)

    signal.signal(signal.SIGINT, _stop_waiting)
    signal.signal(signal.SIGTERM, _stop_waiting)
    waiting = False
    while (down := _dashboard_down(url, token)) is not None:
        if time.monotonic() >= deadline:
            _exit_unreachable(url, f"{down}; waited {HEAD_WAIT_S // 60} min")
        if not waiting:
            logger.info("Waiting for the Ray dashboard at %s (%s).", url, down)
            waiting = True
        time.sleep(_POLL_S)


def _dashboard_down(url: str, token: str | None) -> str | None:
    """Why the dashboard can't take a job yet, or None once it can; exits on what waiting won't fix."""
    try:
        with urllib.request.urlopen(f"{url}/api/authentication_mode", timeout=10) as resp:
            mode = json.load(resp).get("authentication_mode")
    except urllib.error.HTTPError as e:
        if e.code < 500:
            _exit_unreachable(url, f"HTTP {e.code}")
        return f"HTTP {e.code}"
    except http.client.RemoteDisconnected:
        return "connection closed"
    except http.client.HTTPException:
        _exit_unreachable(url, "not an HTTP reply")
    except (urllib.error.URLError, OSError) as e:
        return str(getattr(e, "reason", e))
    except ValueError as e:
        _exit_unreachable(url, str(e))
    if mode == "token" and not token:
        sys.exit(f"error: the cluster at {url} requires its Ray auth token: set MSHIP_RAY_AUTH_TOKEN.")
    return None


def _submit_and_follow(
    url: str, headers: dict[str, str], argv: list[str], models_yaml: str | None, deadline: float
) -> None:
    job_id: str | None = None

    def _stop_following(sig, _frame) -> None:
        running = f"; Ray job {job_id} keeps running on the cluster" if job_id else ""
        logger.info("Stopped following (signal %s)%s.", sig, running)
        sys.exit(130)

    signal.signal(signal.SIGINT, _stop_following)
    signal.signal(signal.SIGTERM, _stop_following)
    metadata = {MODELS_YAML_KEY: models_yaml} if models_yaml is not None else None
    client, job_id = _submit(url, headers, entrypoint(argv, models_yaml is not None), metadata, deadline)
    logger.info("Submitted Ray job %s to %s.", job_id, url)
    info = _follow(client, url, job_id)
    if info.status == "SUCCEEDED":
        sys.exit(0)
    if info.driver_exit_code is not None:
        sys.exit(info.driver_exit_code)
    sys.exit(f"error: Ray job {job_id} {info.status.lower()}: {info.message}")


def _submit(
    url: str, headers: dict[str, str], command: str, metadata: dict | None, deadline: float
) -> tuple[JobSubmissionClient, str]:
    """A client and the new job's id; retries until *deadline* while the dashboard answers 5xx or not at all."""
    import requests
    from ray.job_submission import JobSubmissionClient

    waiting = False
    while True:
        try:
            # Both check the dashboard's version first, which raises the builtin ConnectionError when it's down.
            client = JobSubmissionClient(url, headers=headers)
            return client, client.submit_job(entrypoint=command, entrypoint_resources=_HEAD, metadata=metadata)
        except ConnectionError:
            reason = "connection failed"
        except (requests.HTTPError, RuntimeError) as e:
            if (_http_status(e) or 0) < 500:
                raise
            reason = str(e)
        except requests.RequestException as e:
            reason = str(e)
        if time.monotonic() >= deadline:
            print(
                f"error: the Ray dashboard at {url} didn't take the job within {HEAD_WAIT_S // 60} min: {reason}",
                file=sys.stderr,
            )
            sys.exit(EXIT_UNREACHABLE)
        if not waiting:
            logger.info("Waiting for the Ray dashboard at %s to take the job (%s).", url, reason)
            waiting = True
        time.sleep(_POLL_S)


def _follow(client: JobSubmissionClient, url: str, job_id: str) -> JobDetails:
    """Prints the job's new log text until it ends; returns its final info."""
    import requests

    printed = 0
    while True:
        try:
            info = client.get_job_info(job_id)
            logs = client.get_job_logs(job_id)
        except requests.RequestException:
            print(
                f"error: lost the Ray dashboard at {url}; Ray job {job_id} may still be running there.", file=sys.stderr
            )
            sys.exit(EXIT_UNREACHABLE)
        if len(logs) > printed:
            sys.stdout.write(logs[printed:])
            sys.stdout.flush()
        printed = len(logs)
        if info.status in ("SUCCEEDED", "FAILED", "STOPPED"):
            return info
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

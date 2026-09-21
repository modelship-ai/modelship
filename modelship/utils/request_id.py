"""Request-id helpers. A leaf module deliberately free of any Ray import so
`modelship.utils` (and thus modelship.utils.cli) stays importable before
`import ray` — the driver parses argv and resolves Ray auth env vars ahead of
that import. RawRequestProxy is referenced for typing only."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelship.infer.infer_config import RawRequestProxy


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def base_request_id(raw_request: RawRequestProxy | None = None) -> str:
    """Return the request ID from a RawRequestProxy, or generate a new one."""
    if raw_request is not None and raw_request.request_id is not None:
        return raw_request.request_id
    return random_uuid()

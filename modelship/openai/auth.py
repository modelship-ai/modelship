import hashlib
import os
import re

from starlette.requests import HTTPConnection

# Sentinel returned by identity_key() when no trusted header resolves an identity.
# Deliberately not hash-shaped (a sha256 hex digest is 64 lowercase hex chars) so it
# can never collide with a real identity value. Every such caller shares this one bucket.
UNSCOPED_IDENTITY = "unscoped"

# Charset a trusted-header identity value must match to be used raw (as a log
# field / state-key segment). Anything outside this — newlines, "/", control
# chars, or overlong values — falls back to a sha256 hash instead of ever
# propagating untrusted bytes into a log line or a state-store key. "." is in
_SAFE_IDENTITY_RE = re.compile(r"^(?!\.\.?$)[A-Za-z0-9_.:-]{1,128}$")


# (raw env string, parsed value) cache for get_trusted_identity_header().
# Keyed on the raw string rather than parsed once at import time so tests using
# patch.dict(os.environ, ...) still see up-to-date values with no manual cache clearing —
# the cache only pays off across the many requests within one unchanging-env process.
_trusted_header_cache: tuple[str, str | None] | None = None


def get_trusted_identity_header() -> str | None:
    """Read the trusted identity header name from ``MSHIP_TRUSTED_IDENTITY_HEADER``, if set."""
    global _trusted_header_cache
    raw = os.environ.get("MSHIP_TRUSTED_IDENTITY_HEADER", "")
    cached = _trusted_header_cache
    if cached is not None and cached[0] == raw:
        return cached[1]
    value = raw.strip() or None
    _trusted_header_cache = (raw, value)
    return value


def resolve_identity(request: HTTPConnection) -> tuple[str, str]:
    """Resolve (identity_key, identity_tier) in one pass and cache the result on ``request.state``.

    Takes an ``HTTPConnection`` (not ``Request``) so ``WebSocket`` can call this directly too.

    Not an auth check — never rejects a request. Resolution order:

    1. A configured ``MSHIP_TRUSTED_IDENTITY_HEADER`` present on the request: the
       raw header value (sanitized) — a non-secret identifier an operator's
       credentials layer assigned, kept legible in logs/state keys. Requires that
       layer to unconditionally overwrite the header and modelship to be
       unreachable except from it (see docs/model-configuration.md). Tier: "header".
    2. Otherwise: ``UNSCOPED_IDENTITY`` — every such caller shares one bucket. Tier: "unscoped".
    """
    state = getattr(request, "state", None)
    cached = getattr(state, "_identity", None) if state is not None else None
    if isinstance(cached, tuple):
        return cached

    result = (UNSCOPED_IDENTITY, "unscoped")
    header_name = get_trusted_identity_header()
    if header_name:
        value = request.headers.get(header_name, "").strip()
        if value:
            key = value if _SAFE_IDENTITY_RE.match(value) else hashlib.sha256(value.encode()).hexdigest()
            result = (key, "header")
    if state is not None:
        state._identity = result
    return result


def identity_key(request: HTTPConnection) -> str:
    """Resolve a stable per-caller identity string for log correlation and future state-keying."""
    return resolve_identity(request)[0]


def identity_tier(request: HTTPConnection) -> str:
    """Return which identity_key() tier resolved for *request*: "header" / "unscoped".

    For logging/observability only — lets an unexpected shift to "unscoped" (e.g. a
    fronting proxy that stopped setting the trusted header) show up in logs.
    """
    return resolve_identity(request)[1]

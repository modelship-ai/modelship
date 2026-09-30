import hashlib
import os
from unittest.mock import patch

from starlette.requests import Request

from modelship.openai.auth import (
    UNSCOPED_IDENTITY,
    get_trusted_identity_header,
    identity_key,
    identity_tier,
    resolve_identity,
)

HEADER_ENV = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id"}


def _make_request(headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    return Request(scope)


class TestIdentityKey:
    def test_trusted_header_used_raw(self):
        with patch.dict(os.environ, HEADER_ENV):
            assert identity_key(_make_request({"X-Consumer-Id": "customer-42"})) == "customer-42"

    def test_trusted_header_stripped(self):
        with patch.dict(os.environ, HEADER_ENV):
            assert identity_key(_make_request({"X-Consumer-Id": "  customer-42  "})) == "customer-42"

    def test_header_configured_but_absent_is_unscoped(self):
        with patch.dict(os.environ, HEADER_ENV):
            assert identity_key(_make_request({"Authorization": "Bearer sk-a"})) == UNSCOPED_IDENTITY

    def test_header_configured_but_empty_is_unscoped(self):
        with patch.dict(os.environ, HEADER_ENV):
            assert identity_key(_make_request({"X-Consumer-Id": "   "})) == UNSCOPED_IDENTITY

    def test_no_header_configured_is_unscoped(self):
        with patch.dict(os.environ):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            result = identity_key(_make_request({"Authorization": "Bearer sk-a"}))
        assert result == UNSCOPED_IDENTITY
        # Must never collide with a real sha256 hex digest (64 lowercase hex chars).
        assert len(result) != 64 or not all(c in "0123456789abcdef" for c in result)

    def test_unsafe_header_value_falls_back_to_hash(self):
        unsafe = "../../etc/passwd"
        with patch.dict(os.environ, HEADER_ENV):
            result = identity_key(_make_request({"X-Consumer-Id": unsafe}))
        assert result == hashlib.sha256(unsafe.encode()).hexdigest()
        assert "/" not in result

    def test_overlong_header_value_falls_back_to_hash(self):
        overlong = "a" * 200
        with patch.dict(os.environ, HEADER_ENV):
            result = identity_key(_make_request({"X-Consumer-Id": overlong}))
        assert result == hashlib.sha256(overlong.encode()).hexdigest()

    def test_dot_dot_header_value_falls_back_to_hash(self):
        """ ".." matches the charset (dots are allowed mid-value, e.g. "svc.billing") but as
        a whole segment would traverse a future file-based state store keyed by this value."""
        with patch.dict(os.environ, HEADER_ENV):
            result = identity_key(_make_request({"X-Consumer-Id": ".."}))
        assert result == hashlib.sha256(b"..").hexdigest()

    def test_single_dot_header_value_falls_back_to_hash(self):
        with patch.dict(os.environ, HEADER_ENV):
            result = identity_key(_make_request({"X-Consumer-Id": "."}))
        assert result == hashlib.sha256(b".").hexdigest()


class TestIdentityTier:
    def test_header_tier(self):
        with patch.dict(os.environ, HEADER_ENV):
            assert identity_tier(_make_request({"X-Consumer-Id": "customer-42"})) == "header"

    def test_unscoped_tier(self):
        with patch.dict(os.environ):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            assert identity_tier(_make_request()) == "unscoped"


class TestEnvCaching:
    def test_get_trusted_identity_header_reflects_env_change(self):
        with patch.dict(os.environ, {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-A"}):
            assert get_trusted_identity_header() == "X-A"
        with patch.dict(os.environ, {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-B"}):
            assert get_trusted_identity_header() == "X-B"


class TestResolveIdentityCaching:
    def test_result_is_cached_on_request_state(self):
        with patch.dict(os.environ, HEADER_ENV):
            request = _make_request({"X-Consumer-Id": "customer-42", "X-Other": "other-7"})
            first = resolve_identity(request)
            os.environ["MSHIP_TRUSTED_IDENTITY_HEADER"] = "X-Other"
            second = resolve_identity(request)
        assert first == second == ("customer-42", "header")

    def test_identity_key_and_identity_tier_share_one_resolution(self):
        with patch.dict(os.environ, HEADER_ENV):
            request = _make_request({"X-Consumer-Id": "customer-42"})
            key = identity_key(request)
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER")
            tier = identity_tier(request)
        assert (key, tier) == ("customer-42", "header")

    def test_request_like_object_without_state_attribute(self):
        class _NoStateRequest:
            def __init__(self, headers: dict[str, str]) -> None:
                self.headers = headers

        with patch.dict(os.environ, HEADER_ENV):
            request = _NoStateRequest({"X-Consumer-Id": "customer-42"})
            result = resolve_identity(request)  # type: ignore[arg-type]
        assert result == ("customer-42", "header")

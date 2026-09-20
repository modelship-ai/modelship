"""What each actor's creator forwards in runtime_env, and how the redis password is kept out of it."""

import os
from unittest.mock import patch

import pytest

from modelship.state import (
    REDIS_PASSWORD_ENV,
    reject_inline_password,
    resolve_state_store_uri,
    state_store_env_var,
)
from modelship.utils.runtime_env import GATEWAY_ENV_VARS, MODEL_ENV_VARS, build_env_vars


class TestForwardedEnvVars:
    def test_secrets_are_never_forwarded(self):
        for group in (GATEWAY_ENV_VARS, MODEL_ENV_VARS):
            assert "MSHIP_API_KEYS" not in group
            assert REDIS_PASSWORD_ENV not in group

    def test_gateway_gets_its_own_settings_and_the_model_path_does_not(self):
        gateway_only = {
            "MSHIP_TRUSTED_IDENTITY_HEADER",
            "MSHIP_MAX_REQUEST_BODY_BYTES",
            "MSHIP_MCP_ALLOWED_HOSTS",
            "MSHIP_MCP_REQUIRE_HTTPS",
            "MSHIP_RESPONSES_STALE_S",
        }
        assert gateway_only <= set(GATEWAY_ENV_VARS)
        assert gateway_only.isdisjoint(MODEL_ENV_VARS)

    def test_only_vars_set_here_are_forwarded(self):
        with patch.dict(os.environ, {"MSHIP_MAX_REQUEST_BODY_BYTES": "128"}, clear=True):
            forwarded = build_env_vars(GATEWAY_ENV_VARS)
        assert forwarded == {"MSHIP_MAX_REQUEST_BODY_BYTES": "128"}


class TestStateStoreForwarding:
    def test_password_travels_as_a_placeholder(self):
        env = {"MSHIP_STATE_STORE": "redis://host:6379/0", REDIS_PASSWORD_ENV: "s3cret"}
        with patch.dict(os.environ, env, clear=True):
            forwarded = state_store_env_var()["MSHIP_STATE_STORE"]
        assert forwarded == "redis://:${MSHIP_REDIS_PASSWORD}@host:6379/0"
        assert "s3cret" not in forwarded

    def test_user_and_query_survive_the_rewrite(self):
        env = {"MSHIP_STATE_STORE": "rediss://alex@host:6379/1?ssl_cert_reqs=none", REDIS_PASSWORD_ENV: "pw"}
        with patch.dict(os.environ, env, clear=True):
            assert (
                state_store_env_var()["MSHIP_STATE_STORE"]
                == "rediss://alex:${MSHIP_REDIS_PASSWORD}@host:6379/1?ssl_cert_reqs=none"
            )

    def test_no_password_set_forwards_the_uri_unchanged(self):
        with patch.dict(os.environ, {"MSHIP_STATE_STORE": "redis://host:6379/0"}, clear=True):
            assert state_store_env_var() == {"MSHIP_STATE_STORE": "redis://host:6379/0"}

    def test_nothing_to_forward_without_a_uri(self):
        with patch.dict(os.environ, {}, clear=True):
            assert state_store_env_var() == {}

    def test_local_reader_applies_the_password(self):
        env = {"MSHIP_STATE_STORE": "redis://host:6379/0", REDIS_PASSWORD_ENV: "s3cret"}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_state_store_uri() == "redis://:s3cret@host:6379/0"

    def test_local_reader_expands_a_placeholder_uri(self):
        env = {"MSHIP_STATE_STORE": "redis://:${MSHIP_REDIS_PASSWORD}@host:6379/0", REDIS_PASSWORD_ENV: "s3cret"}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_state_store_uri() == "redis://:s3cret@host:6379/0"

    def test_unset_placeholder_fails_naming_the_var(self):
        with (
            patch.dict(os.environ, {"MSHIP_STATE_STORE": "redis://:${MSHIP_REDIS_PASSWORD}@host/0"}, clear=True),
            pytest.raises(ValueError, match="MSHIP_STATE_STORE"),
        ):
            resolve_state_store_uri()

    def test_memory_uri_is_untouched(self):
        with patch.dict(os.environ, {REDIS_PASSWORD_ENV: "s3cret"}, clear=True):
            assert resolve_state_store_uri() == "memory://"


class TestRejectInlinePassword:
    def test_inline_password_is_rejected(self):
        with pytest.raises(ValueError, match=REDIS_PASSWORD_ENV):
            reject_inline_password("redis://:s3cret@host:6379/0")

    @pytest.mark.parametrize(
        "uri",
        ["", "memory://", "redis://host:6379/0", "redis://alex@host:6379/0", "redis://:${MSHIP_REDIS_PASSWORD}@host/0"],
    )
    def test_accepted(self, uri):
        reject_inline_password(uri)

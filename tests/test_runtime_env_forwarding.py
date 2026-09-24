"""What each actor's creator forwards in runtime_env, and how the redis password is kept out of it."""

import os
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import pytest

from modelship.infer import replica_coordinator
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
    def test_the_password_is_not_forwarded(self):
        env = {"MSHIP_STATE_STORE": "redis://host:6379/0", REDIS_PASSWORD_ENV: "s3cret"}
        with patch.dict(os.environ, env, clear=True):
            assert state_store_env_var() == {"MSHIP_STATE_STORE": "redis://host:6379/0"}

    def test_user_and_query_are_forwarded_as_written(self):
        uri = "rediss://alex@host:6379/1?ssl_cert_reqs=none"
        with patch.dict(os.environ, {"MSHIP_STATE_STORE": uri, REDIS_PASSWORD_ENV: "pw"}, clear=True):
            assert state_store_env_var() == {"MSHIP_STATE_STORE": uri}

    def test_nothing_to_forward_without_a_uri(self):
        with patch.dict(os.environ, {}, clear=True):
            assert state_store_env_var() == {}

    def test_local_reader_applies_the_password(self):
        env = {"MSHIP_STATE_STORE": "redis://host:6379/0", REDIS_PASSWORD_ENV: "s3cret"}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_state_store_uri() == "redis://:s3cret@host:6379/0"

    @pytest.mark.parametrize("password", ["aB3/xY+9zQ==", "p#w", "p?w", "p@w", "p%41w", "pw:pw"])
    def test_a_reserved_character_password_survives_the_round_trip(self, password):
        env = {"MSHIP_STATE_STORE": "redis://alex@host:6379/0", REDIS_PASSWORD_ENV: password}
        with patch.dict(os.environ, env, clear=True):
            parsed = urlsplit(resolve_state_store_uri())
        assert (parsed.hostname, parsed.port, parsed.path) == ("host", 6379, "/0")
        # redis-py's parse_url unquotes username/password before connecting.
        assert (unquote(parsed.username or ""), unquote(parsed.password or "")) == ("alex", password)

    @pytest.mark.parametrize("uri", ["redis://${REDIS_HOST}:6379/0", "redis://$REDIS_HOST:6379/0"])
    def test_local_reader_expands_an_env_var_in_the_uri(self, uri):
        with patch.dict(os.environ, {"MSHIP_STATE_STORE": uri, "REDIS_HOST": "box"}, clear=True):
            assert resolve_state_store_uri() == "redis://box:6379/0"

    @pytest.mark.parametrize("uri", ["redis://${REDIS_HOST}:6379/0", "redis://$REDIS_HOST:6379/0"])
    def test_unset_var_in_the_uri_fails_naming_it(self, uri):
        with (
            patch.dict(os.environ, {"MSHIP_STATE_STORE": uri}, clear=True),
            pytest.raises(ValueError, match="REDIS_HOST"),
        ):
            resolve_state_store_uri()

    def test_memory_uri_is_untouched(self):
        with patch.dict(os.environ, {REDIS_PASSWORD_ENV: "s3cret"}, clear=True):
            assert resolve_state_store_uri() == "memory://"


class TestCoordinatorCreation:
    def test_the_replica_coordinator_is_created_with_the_store_uri(self):
        with (
            patch.dict(os.environ, {"MSHIP_STATE_STORE": "redis://host:6379/0"}, clear=True),
            patch.object(replica_coordinator.ReplicaCoordinator, "options") as options,
        ):
            replica_coordinator.get_or_create_replica_coordinator()
        assert options.call_args.kwargs["runtime_env"]["env_vars"]["MSHIP_STATE_STORE"] == "redis://host:6379/0"


class TestRejectInlinePassword:
    def test_inline_password_is_rejected(self):
        with pytest.raises(ValueError, match=REDIS_PASSWORD_ENV):
            reject_inline_password("redis://:s3cret@host:6379/0")

    @pytest.mark.parametrize(
        "uri", ["redis://:${MSHIP_REDIS_PASSWORD}@host:6379/0", "redis://host:6379/0?password=s3cret"]
    )
    def test_other_ways_of_writing_one_are_rejected_too(self, uri):
        with pytest.raises(ValueError, match=REDIS_PASSWORD_ENV):
            reject_inline_password(uri)

    @pytest.mark.parametrize("value", [":s3cret@host:6379/0", "host:6379/0?password=s3cret"])
    def test_a_password_an_env_var_expands_to_is_rejected(self, value):
        with (
            patch.dict(os.environ, {"REDIS_URL": value}, clear=True),
            pytest.raises(ValueError, match=REDIS_PASSWORD_ENV),
        ):
            reject_inline_password("redis://${REDIS_URL}")

    @pytest.mark.parametrize(
        "uri",
        [
            "",
            "memory://",
            "redis://host:6379/0",
            "redis://alex@host:6379/0",
            "rediss://host:6379/0?ssl_cert_reqs=none",
        ],
    )
    def test_accepted(self, uri):
        reject_inline_password(uri)

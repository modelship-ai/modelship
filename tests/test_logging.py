"""Tests for the modelship.logging module."""

import json
import logging
import os
import socket
from importlib.metadata import entry_points
from logging.handlers import SysLogHandler
from unittest.mock import patch

import pytest

from modelship.logging import (
    _LIB_ENV_VARS,
    _LIB_LOGGERS,
    _LOWERCASE_LEVEL_LIBS,
    VLLM_CHILD_ENV,
    ModelshipJsonFormatter,
    ModelshipTextFormatter,
    RequestContextFilter,
    _parse_syslog_target,
    _setup_otel,
    configure_logging,
    configure_vllm_child_logging,
    get_logger,
    identity_tier_var,
    identity_var,
    propagate_lib_log_env,
    request_id_var,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset the modelship logger and _configured flag between tests."""
    import modelship.logging as yl

    yl._configured = False
    root = logging.getLogger("modelship")
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    root.propagate = True
    saved_lib_levels = {name: logging.getLogger(name).level for name in _LIB_LOGGERS}
    saved_lib_handlers = {name: list(logging.getLogger(name).handlers) for name in _LIB_LOGGERS}
    saved_lib_propagate = {name: logging.getLogger(name).propagate for name in _LIB_LOGGERS}
    saved_env = {k: os.environ.get(k) for k in _LIB_ENV_VARS}
    # Clear so each test gets a clean setdefault path; importing mship_deploy.py in
    # another test runs propagate_lib_log_env() and leaves these vars set.
    for k in _LIB_ENV_VARS:
        os.environ.pop(k, None)
    token = request_id_var.set(None)
    identity_token = identity_var.set(None)
    identity_tier_token = identity_tier_var.set(None)
    yield
    request_id_var.reset(token)
    identity_var.reset(identity_token)
    identity_tier_var.reset(identity_tier_token)
    yl._configured = False
    root.handlers.clear()
    for name, lvl in saved_lib_levels.items():
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(lvl)
        lib_logger.handlers = saved_lib_handlers[name]
        lib_logger.propagate = saved_lib_propagate[name]
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestGetLogger:
    def test_returns_modelship_prefixed_logger(self):
        log = get_logger("api")
        assert log.name == "modelship.api"

    def test_nested_name(self):
        log = get_logger("infer.vllm")
        assert log.name == "modelship.infer.vllm"


class TestConfigureLogging:
    def test_sets_up_handler(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert root.level == logging.INFO
        assert root.propagate is False

    def test_idempotent(self):
        configure_logging()
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1

    @patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "DEBUG"})
    def test_respects_log_level_env(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert root.level == logging.DEBUG

    def test_lib_loggers_silent_by_default(self):
        from modelship.logging import _LIB_SILENT_LEVEL

        configure_logging()
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == _LIB_SILENT_LEVEL
        for env_var, lib_name in _LIB_ENV_VARS.items():
            expected = "critical" if lib_name in _LOWERCASE_LEVEL_LIBS else "CRITICAL"
            assert os.environ.get(env_var) == expected

    @patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "DEBUG"})
    def test_lib_loggers_mirror_debug(self):
        configure_logging()
        assert logging.getLogger("modelship").level == logging.DEBUG
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG

    def test_lib_loggers_get_modelship_handler(self):
        # Also covers the double-print risk: a pre-existing handler (e.g. Ray's
        # own, attached at import time) must be cleared, not just appended to.
        stray = logging.StreamHandler()
        logging.getLogger("ray").addHandler(stray)

        configure_logging()
        root_formatter = logging.getLogger("modelship").handlers[0].formatter
        for name in _LIB_LOGGERS:
            lib_logger = logging.getLogger(name)
            assert len(lib_logger.handlers) == 1
            # Formatter is shared, but the handler instance must NOT be — ray.init(...)
            # mutates whatever handler it finds on "ray" in place.
            assert lib_logger.handlers[0] is not logging.getLogger("modelship").handlers[0]
            assert lib_logger.handlers[0].formatter is root_formatter
            assert lib_logger.propagate is False

    @patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "TRACE"})
    def test_trace_mode_sets_trace_app_debug_libs(self):
        from modelship.logging import TRACE

        configure_logging()
        assert logging.getLogger("modelship").level == TRACE
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG
        for env_var, lib_name in _LIB_ENV_VARS.items():
            expected = "debug" if lib_name in _LOWERCASE_LEVEL_LIBS else "DEBUG"
            assert os.environ.get(env_var) == expected

    @patch.dict(os.environ, {"MSHIP_LOG_FORMAT": "json"})
    def test_json_format(self):
        configure_logging()
        root = logging.getLogger("modelship")
        handler = root.handlers[0]
        assert isinstance(handler.formatter, ModelshipJsonFormatter)

    def test_text_format_default(self):
        configure_logging()
        root = logging.getLogger("modelship")
        handler = root.handlers[0]
        assert isinstance(handler.formatter, ModelshipTextFormatter)


class TestVllmConfigureLoggingOptOut:
    """vLLM's import-time dictConfig() call closes Ray Serve's MemoryHandler,
    crashing replica health-checks on head-restart recovery. propagate_lib_log_env
    sets VLLM_CONFIGURE_LOGGING=0 before any vLLM import to avoid this."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_CONFIGURE_LOGGING", None)
            propagate_lib_log_env("INFO")
            assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"

    def test_explicit_user_value_wins(self):
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "1"}):
            propagate_lib_log_env("INFO")
            assert os.environ["VLLM_CONFIGURE_LOGGING"] == "1"


class TestVllmChildLogging:
    def test_registered_as_a_vllm_plugin(self):
        [plugin] = [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "modelship_logging"]
        assert plugin.load() is configure_vllm_child_logging

    def test_configures_only_a_marked_process(self):
        with patch.dict(os.environ):
            os.environ.pop(VLLM_CHILD_ENV, None)
            configure_vllm_child_logging()
            assert not logging.getLogger("modelship").handlers

            os.environ[VLLM_CHILD_ENV] = "1"
            configure_vllm_child_logging()
            assert logging.getLogger("vllm").handlers


class TestRequestContextFilter:
    def test_injects_request_id(self):
        filt = RequestContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        request_id_var.set("abc-123")
        filt.filter(record)
        assert record.request_id == "abc-123"

    def test_none_when_not_set(self):
        filt = RequestContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        filt.filter(record)
        assert record.request_id is None
        assert record.identity is None
        assert record.identity_tier is None

    def test_injects_identity_and_tier(self):
        filt = RequestContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        identity_var.set("customer-42")
        identity_tier_var.set("header")
        filt.filter(record)
        assert record.identity == "customer-42"
        assert record.identity_tier == "header"


class TestModelshipJsonFormatter:
    def test_produces_valid_json(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello %s", ("world",), None)
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "modelship.api"
        assert parsed["message"] == "hello world"
        assert "pid" in parsed

    def test_includes_request_id(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "test", (), None)
        record.request_id = "req-456"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "req-456"

    def test_excludes_request_id_when_none(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "test", (), None)
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "request_id" not in parsed

    def test_includes_exception(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord("modelship.api", logging.ERROR, "", 0, "error", (), sys.exc_info())
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError: boom" in parsed["exception"]

    def test_includes_identity_and_tier(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "test", (), None)
        record.identity = "customer-42"
        record.identity_tier = "header"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["identity"] == "customer-42"
        assert parsed["identity_tier"] == "header"

    def test_excludes_identity_when_none(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "test", (), None)
        record.identity = None
        record.identity_tier = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "identity" not in parsed
        assert "identity_tier" not in parsed


class TestModelshipTextFormatter:
    def test_basic_format(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = None
        output = formatter.format(record)
        assert "INFO" in output
        assert "modelship.api" in output
        assert "hello" in output

    def test_includes_request_id(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = "req-789"
        output = formatter.format(record)
        assert "[req-789]" in output

    def test_excludes_request_id_when_none(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = None
        output = formatter.format(record)
        assert "None" not in output

    def test_includes_identity(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.identity = "customer-42"
        output = formatter.format(record)
        assert "id=customer-42" in output

    def test_excludes_identity_when_none(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.identity = None
        output = formatter.format(record)
        assert "id=" not in output


class TestEndToEnd:
    def test_log_output_with_request_id(self, capsys):
        configure_logging()
        log = get_logger("test")
        request_id_var.set("e2e-test-id")
        log.info("end to end")
        captured = capsys.readouterr()
        assert "e2e-test-id" in captured.err
        assert "end to end" in captured.err

    @patch.dict(os.environ, {"MSHIP_LOG_FORMAT": "json"})
    def test_json_log_output(self, capsys):
        configure_logging()
        log = get_logger("test")
        request_id_var.set("json-test-id")
        log.info("json test")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert parsed["request_id"] == "json-test-id"
        assert parsed["message"] == "json test"

    @patch.dict(os.environ, {"MSHIP_LOG_FORMAT": "json"})
    def test_json_log_output_includes_identity(self, capsys):
        configure_logging()
        log = get_logger("test")
        identity_var.set("customer-42")
        identity_tier_var.set("header")
        log.info("identity test")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert parsed["identity"] == "customer-42"
        assert parsed["identity_tier"] == "header"


class TestParseSyslogTarget:
    def test_udp_default(self):
        handler = _parse_syslog_target("syslog://192.168.1.50:514")
        assert handler.address == ("192.168.1.50", 514)
        assert handler.socktype == socket.SOCK_DGRAM

    @patch("modelship.logging.SysLogHandler.createSocket")
    def test_tcp(self, _mock_create):
        handler = _parse_syslog_target("syslog+tcp://127.0.0.1:1514")
        assert handler.address == ("127.0.0.1", 1514)
        assert handler.socktype == socket.SOCK_STREAM

    def test_default_port(self):
        handler = _parse_syslog_target("syslog://127.0.0.1")
        assert handler.address == ("127.0.0.1", 514)
        assert handler.socktype == socket.SOCK_DGRAM

    def test_default_host(self):
        handler = _parse_syslog_target("syslog://")
        assert handler.address == ("localhost", 514)


class TestSyslogLogTarget:
    @patch.dict(os.environ, {"MSHIP_LOG_TARGET": "syslog://127.0.0.1:5140"})
    def test_configure_creates_syslog_handler(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], SysLogHandler)

    def test_console_default(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert not isinstance(root.handlers[0], SysLogHandler)


class TestOtelSetup:
    def test_warns_when_packages_missing(self, capsys):
        root = logging.getLogger("modelship")
        root.handlers.clear()
        sh = logging.StreamHandler()
        root.addHandler(sh)
        root.setLevel(logging.DEBUG)

        _setup_otel(root, "http://localhost:4317", logging.INFO)

        captured = capsys.readouterr()
        assert "opentelemetry packages are not installed" in captured.err
        assert len(root.handlers) == 1

    @patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"})
    @patch("modelship.logging._setup_otel")
    def test_configure_calls_setup_otel(self, mock_setup):
        configure_logging()
        mock_setup.assert_called_once_with(
            logging.getLogger("modelship"),
            "http://collector:4317",
            logging.INFO,
        )

    @patch.dict(os.environ, {}, clear=False)
    def test_configure_skips_otel_when_not_set(self):
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

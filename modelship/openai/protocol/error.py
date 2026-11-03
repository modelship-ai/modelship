"""OpenAI-compatible error models and the error-response factory."""

import json
from http import HTTPStatus
from typing import Any

from pydantic import PrivateAttr, ValidationError

from modelship.logging import get_logger, request_id_var
from modelship.openai.protocol.base import OpenAIBaseModel

logger = get_logger("openai.protocol.error")


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    # OpenAI error code identifier (e.g. "context_length_exceeded"). None when we
    # haven't mapped the underlying failure to a specific OpenAI-known code.
    code: str | None = None


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo

    # HTTP status code carried for the gateway to set on the JSONResponse. Lives
    # outside `error` because OpenAI's spec has no HTTP status field on the
    # error object — the spec exposes status purely via the HTTP layer.
    _http_status: int = PrivateAttr(default=500)


def create_error_response(
    message: str | Exception,
    err_type: str | None = None,
    status_code: HTTPStatus | int = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
    code: str | None = None,
) -> ErrorResponse:
    if isinstance(message, Exception):
        exc = message
        if isinstance(exc, ValueError):
            err_type = "invalid_request_error"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, NotImplementedError):
            err_type = "api_error"
            status_code = HTTPStatus.NOT_IMPLEMENTED
            param = None
        else:
            err_type = "api_error"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None
        message = str(exc)

    resp = ErrorResponse(
        error=ErrorInfo(
            message=message,
            type=err_type or ("api_error" if status_code >= 500 else "invalid_request_error"),
            code=code,
            param=param,
        )
    )
    resp._http_status = int(status_code)
    return resp


def validation_error_from_cause(cause: BaseException) -> ErrorResponse:
    # A pydantic ValidationError's .args is only () locally; once it crosses the
    # Ray pickle boundary it becomes (model_title, [line_errors], ...) — args[0]
    # would be a bare model name like "TranscriptionRequest", not a useful
    # message, so always render those through str() instead.
    if isinstance(cause, ValidationError):
        base = str(cause)
    elif cause.args:
        base = cause.args[0]
    else:
        base = str(cause)
    return create_error_response(
        message=base,
        err_type="invalid_request_error",
        status_code=HTTPStatus.BAD_REQUEST,
        param=getattr(cause, "parameter", None),
    )


ERROR_SSE_PREFIX = 'data: {"error"'


def encode_error_sse(error: ErrorResponse) -> str:
    """Encode a mid-stream `ErrorResponse` as an SSE `data:` line.

    Shared by every loader's streaming chat/responses paths — a pre-generation
    failure is returned as a plain `ErrorResponse` instead (see `BaseInfer.create_chat_completion`);
    this is only for a failure after the stream has already started.
    """
    return f"data: {json.dumps(error.model_dump(mode='json'))}\n\n"


def server_error(status_code: HTTPStatus | int = HTTPStatus.INTERNAL_SERVER_ERROR) -> ErrorResponse:
    return create_error_response(
        f"The server had an error processing your request (request id: {request_id_var.get()}).",
        status_code=status_code,
    )


def client_safe(item: Any) -> Any:
    if isinstance(item, ErrorResponse):
        if item._http_status < 500:
            return item
        logger.warning("Server error replaced for the client: %s", item.error.message)
        return server_error(item._http_status)
    if isinstance(item, str) and item.startswith(ERROR_SSE_PREFIX):
        error = json.loads(item.removeprefix("data: ")).get("error") or {}
        if error.get("type") == "invalid_request_error":
            return item
        logger.warning("Server error replaced for the client: %s", error.get("message"))
        return encode_error_sse(server_error())
    if isinstance(item, dict) and item.get("type") == "response.failed":
        response = item.get("response") or {}
        error = response.get("error") or {}
        if error.get("code") == "invalid_prompt":
            return item
        logger.warning("Server error replaced for the client: %s", error.get("message"))
        safe = {"code": "server_error", "message": server_error().error.message}
        return {**item, "response": {**response, "error": safe}}
    return item


__all__ = [
    "ERROR_SSE_PREFIX",
    "ErrorInfo",
    "ErrorResponse",
    "client_safe",
    "create_error_response",
    "encode_error_sse",
    "server_error",
    "validation_error_from_cause",
]

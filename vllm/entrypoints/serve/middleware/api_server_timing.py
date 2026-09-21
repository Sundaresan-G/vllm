# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from datetime import datetime

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)

GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")


class APIServerRequestTimingMiddleware:
    """Log HTTP request timing from ASGI arrival through the final body."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        root_path = scope.get("root_path", "")
        path = scope["path"].removeprefix(root_path)
        if not path.startswith(GUARDED_PREFIX):
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        request_headers = Headers(scope=scope)
        client_request_id = request_headers.get("X-Request-Id", "-")
        start_time = time.time()
        start = time.perf_counter()
        response_start: float | None = None
        first_body: float | None = None
        response_end: float | None = None
        status_code: int | str = "error"

        logger.info(
            "API server request timing received method=%s path=%s "
            "request_id=%s start=%s",
            method,
            path,
            client_request_id,
            f"{datetime.fromtimestamp(start_time):%m-%d %H:%M:%S.%f}",
        )

        async def send_with_timing(message: Message) -> None:
            nonlocal response_start, first_body, response_end, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
            now = time.perf_counter()
            if message["type"] == "http.response.start":
                response_start = now
            elif message["type"] == "http.response.body":
                if first_body is None:
                    first_body = now
                if not message.get("more_body", False):
                    response_end = now

        try:
            await self.app(scope, receive, send_with_timing)
        finally:
            end = response_end or time.perf_counter()
            end_time = time.time()
            headers_at = response_start or end
            body_at = first_body or end
            state = scope.get("state", {})
            request_metadata = state.get("request_metadata")
            request_id = getattr(request_metadata, "request_id", client_request_id)
            endpoint_start = state.get("api_server_endpoint_start", headers_at)
            endpoint_done = state.get("api_server_endpoint_done", endpoint_start)
            phase_timings_ms = state.get("api_server_phase_timings_ms", {})
            prompt_render_ms = phase_timings_ms.get("prompt_render_ms", 0.0)
            tokenize_ms = phase_timings_ms.get("tokenize_ms", 0.0)
            input_processing_ms = phase_timings_ms.get("input_processing_ms", 0.0)
            endpoint_handler_ms = (endpoint_done - endpoint_start) * 1000.0
            serving_setup_ms = max(
                0.0,
                endpoint_handler_ms
                - prompt_render_ms
                - tokenize_ms
                - input_processing_ms,
            )
            logger.info(
                "API server request timing details method=%s path=%s "
                "request_id=%s status=%s start=%s end=%s: "
                "routing_validation=%.3f prompt_render=%.3f tokenize=%.3f "
                "input_processing=%.3f serving_setup=%.3f "
                "response_setup=%.3f headers_to_first_body=%.3f "
                "body_stream=%.3f total=%.3f",
                method,
                path,
                request_id,
                status_code,
                f"{datetime.fromtimestamp(start_time):%m-%d %H:%M:%S.%f}",
                f"{datetime.fromtimestamp(end_time):%m-%d %H:%M:%S.%f}",
                (endpoint_start - start) * 1000.0,
                prompt_render_ms,
                tokenize_ms,
                input_processing_ms,
                serving_setup_ms,
                (headers_at - endpoint_done) * 1000.0,
                (body_at - headers_at) * 1000.0,
                (end - body_at) * 1000.0,
                (end - start) * 1000.0,
            )

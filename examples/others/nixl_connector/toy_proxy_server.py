# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import time

from vllm.logger import init_logger

logger = init_logger("vllm.shm_proxy_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(timeout=None, base_url=prefiller_base_url),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(timeout=None, base_url=decoder_base_url),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Initialized %d prefill clients and %d decode clients.",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")

def measure_time(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info("%s took %.4f seconds", func.__name__, end_time - start_time)
        return result
    return wrapper

@measure_time
async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_filename": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    # Ask the prefiller to return the tokenized prompt and the sampled token
    # ids so the decoder can be fed pre-tokenized input and skip re-tokenizing
    # the (potentially very large) prompt on its critical path.
    req_data["return_token_ids"] = True
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    return response

def measure_generator_time(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the time before the item is generated
        async for item in func(*args, **kwargs):
            end_time = time.time()  # Record the time after the item is generated
            logger.info("%s generated an item in %.4f seconds", func.__name__, end_time - start_time)
            yield item
            start_time = time.time()  # Record the time before the item is generated again
    return wrapper

@measure_generator_time
async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_single_prompt_completions(
    api: str,
    request: Request,
    req_data: dict,
    request_id: str,
) -> tuple[dict, dict]:
    """
    Prefill + decode for a single-prompt request.
    Returns (prefill_response_json, decode_req_data) so the caller can
    stitch pieces together for multi-prompt batches.
    """
    prefill_client_info = get_next_client(request.app, "prefill")

    response = await send_request_to_service(
        prefill_client_info, api, req_data, request_id
    )
    response_json = response.json()

    # Build the decode request inheriting all fields from the prefill request
    decode_req_data = req_data.copy()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        decode_req_data["kv_transfer_params"] = kv_transfer_params

    choice = response_json["choices"][0]
    prompt_token_ids = choice.get("prompt_token_ids")
    output_token_ids = choice.get("token_ids")

    if prompt_token_ids is not None and output_token_ids is not None:
        # Fast path: feed the decoder pre-tokenized ids (full prompt plus the
        # one token already sampled by the prefiller). This avoids re-tokenizing
        # the whole prompt on the decoder, which otherwise dominates TTFT for
        # long prompts.
        decode_req_data["prompt"] = list(prompt_token_ids) + list(output_token_ids)
    else:
        # Fallback: append the one prefilled token as text so the decoder
        # continues from there (requires re-tokenization on the decoder).
        decode_req_data["prompt"] = decode_req_data["prompt"] + choice["text"]

    # The decoder does not need to echo token ids back.
    decode_req_data.pop("return_token_ids", None)

    # Prefill generated one token already; decrement remaining budget
    if "max_tokens" in decode_req_data:
        decode_req_data["max_tokens"] -= 1

    # Avoid forwarding the large token-id arrays back to the client; restore the
    # original (null) shape of the prefill response chunk.
    choice["prompt_token_ids"] = None
    choice["token_ids"] = None

    return response_json, decode_req_data


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()

        prompts = req_data.get("prompt")

        if isinstance(prompts, list):
            # Split into individual single-prompt requests so that each gets its
            # own kv_transfer_params from the prefiller.  A shared multi-prompt
            # request would give all sub-requests the same remote_block_ids,
            # causing the second sub-request to fail the "block marked busy"
            # assert in start_load_kv after the first sub-request clears the flag.
            import asyncio

            single_reqs = []
            for prompt in prompts:
                single_req = req_data.copy()
                single_req["prompt"] = prompt
                single_reqs.append(single_req)

            # Run all prefills concurrently then all decodes concurrently
            request_ids = [str(uuid.uuid4()) for _ in single_reqs]

            prefill_results = await asyncio.gather(
                *[
                    _handle_single_prompt_completions(api, request, r, rid)
                    for r, rid in zip(single_reqs, request_ids)
                ]
            )

            # prefill_results is a list of (prefill_response_json, decode_req_data)
            decode_client_info = get_next_client(request.app, "decode")

            # Stream all decode responses and merge into a single SSE stream
            async def generate_stream_multi():
                for (prefill_response_json, decode_req_data), rid in zip(prefill_results, request_ids):
                    # Emit the one-token prefill output first
                    import json as _json
                    prefill_output = b"data: " + _json.dumps(prefill_response_json).encode()
                    yield prefill_output
                    logger.info("[%s] Forwarding prefill result to client (%d bytes)", rid, len(prefill_output))

                async def _decode_one(dreq, rid):
                    chunks = []
                    async for chunk in stream_service_response(
                        decode_client_info, api, dreq, request_id=rid
                    ):
                        chunks.append(chunk)
                    return chunks

                decode_chunk_lists = await asyncio.gather(
                    *[
                        _decode_one(dreq, rid)
                        for (_, dreq), rid in zip(prefill_results, request_ids)
                    ]
                )
                for chunks, (prefill_response_json, _), rid in zip(decode_chunk_lists, prefill_results, request_ids):
                    for chunk in chunks:
                        logger.info("[%s] Forwarding decode chunk to client (%d bytes): %s", rid, len(chunk), chunk)
                        yield chunk

            return StreamingResponse(generate_stream_multi(), media_type="application/json")

        else:
            # Single-prompt fast path (original behaviour)
            request_id = str(uuid.uuid4())
            single_req = req_data.copy()

            prefill_response_json, decode_req_data = (
                await _handle_single_prompt_completions(
                    api, request, single_req, request_id
                )
            )

            decode_client_info = get_next_client(request.app, "decode")
            logger.debug("Using decode client %s", decode_client_info)

            async def generate_stream():
                import json as _json
                prefill_output = b"data: " + _json.dumps(prefill_response_json).encode()
                yield prefill_output
                logger.info("[%s] Forwarding prefill result to client (%d bytes)", request_id, len(prefill_output))
                try:
                    async for chunk in stream_service_response(
                        decode_client_info, api, decode_req_data, request_id=request_id
                    ):
                        logger.info("[%s] Forwarding decode chunk to client (%d bytes): %s", request_id, len(chunk), chunk)
                        yield chunk
                except Exception as exc:
                    logger.error("[%s] Error while streaming decode response: %s", request_id, exc, exc_info=True)

            return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        logger.error("Error occurred in disagg prefill proxy server - %s endpoint", api)
        logger.error("%s", e)
        logger.error("%s", "".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)

async def profile(api: str, request: Request):
    req_data = await request.json()
    request_id = str(uuid.uuid4())
    # Get the next prefill client in round-robin fashion
    prefill_client_info = get_next_client(request.app, "prefill")
    # Send request to prefill service
    await send_request_to_service(
        prefill_client_info, api, req_data, request_id
    )
    # Get the next decode client in round-robin fashion
    decode_client_info = get_next_client(request.app, "decode")
    await send_request_to_service(
        decode_client_info, api, req_data, request_id
    )
    return {"status": "ok"}

@app.post("/start_profile")
async def handle_start_profile(request: Request):
    return await profile("/start_profile", request)

@app.post("/stop_profile")
async def handle_stop_profile(request: Request):
    return await profile("/stop_profile", request)


@app.get("/health")
@app.get("/healthcheck")
async def healthcheck():
    """Check proxy is running and all backends are reachable."""
    unhealthy = []

    for client_info in app.state.prefill_clients:
        label = f"prefill:{client_info['host']}:{client_info['port']}"
        try:
            resp = await client_info["client"].get("/health", timeout=5.0)
            if resp.status_code != 200:
                unhealthy.append(f"{label} status={resp.status_code}")
        except Exception as e:
            unhealthy.append(f"{label} error={e}")

    for client_info in app.state.decode_clients:
        label = f"decode:{client_info['host']}:{client_info['port']}"
        try:
            resp = await client_info["client"].get("/health", timeout=5.0)
            if resp.status_code != 200:
                unhealthy.append(f"{label} status={resp.status_code}")
        except Exception as e:
            unhealthy.append(f"{label} error={e}")

    if unhealthy:
        logger.warning("Health check failed: %s", unhealthy)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "unhealthy_backends": unhealthy,
                "prefill_instances": len(app.state.prefill_clients),
                "decode_instances": len(app.state.decode_clients),
            },
        )

    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)

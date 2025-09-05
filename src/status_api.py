from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import os
import socket
import requests
import asyncio
import json

router = APIRouter(prefix="/api")


def check_tcp(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def check_http(url: str, timeout: float = 1.5) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return 200 <= r.status_code < 400
    except Exception:
        return False


def parse_host_port(uri: str):
    # naive parsing for host:port
    if not uri:
        return None, None
    if '://' in uri:
        uri = uri.split('://', 1)[1]
    if '/' in uri:
        uri = uri.split('/', 1)[0]
    if ':' in uri:
        host, port = uri.split(':', 1)
        try:
            return host, int(port)
        except Exception:
            return host, None
    return uri, None


async def _probe_services_async():
    """Async wrapper that runs blocking checks in threadpool and returns a dict."""
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    qdrant_url = os.environ.get('QDRANT_URL', 'http://localhost:6333')
    llm_url = os.environ.get('LLM_URL', 'http://localhost:11434')

    async def _neo_check():
        if neo4j_uri.startswith('bolt') or neo4j_uri.startswith('neo4j'):
            host, port = parse_host_port(neo4j_uri)
            if port is None:
                port = 7687
            return await asyncio.to_thread(check_tcp, host, port, 1.0)
        return await asyncio.to_thread(check_http, neo4j_uri, 1.5)

    async def _chroma_check():
        # Check Qdrant HTTP health endpoint(s)
        try_urls = [qdrant_url.rstrip('/'), qdrant_url.rstrip('/') + '/health']
        for u in try_urls:
            ok = await asyncio.to_thread(check_http, u, 1.5)
            if ok:
                return True
        return False

    async def _llm_check():
        try_urls = [llm_url.rstrip('/'), llm_url.rstrip('/') + '/v1/models', llm_url.rstrip('/') + '/health']
        for u in try_urls:
            ok = await asyncio.to_thread(check_http, u, 1.5)
            if ok:
                return True
        return False

    neo_ok, qdrant_ok, llm_ok = await asyncio.gather(_neo_check(), _chroma_check(), _llm_check())
    return {"neo4j": bool(neo_ok), "qdrant": bool(qdrant_ok), "llm": bool(llm_ok)}


@router.get('/status')
async def api_status():
    """Return current health status for services."""
    state = await _probe_services_async()
    return JSONResponse(state)


async def stream_status_generator(poll_interval: int = 5):
    """Async generator for SSE that yields JSON when service status changes."""
    last = None
    while True:
        current = await _probe_services_async()
        if last is None or current != last:
            data = json.dumps(current)
            yield f"data: {data}\n\n"
            last = current
        await asyncio.sleep(max(1, int(os.environ.get('STATUS_POLL', poll_interval))))


@router.get('/status/stream')
async def status_stream():
    return StreamingResponse(stream_status_generator(), media_type='text/event-stream')

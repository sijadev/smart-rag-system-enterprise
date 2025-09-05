import asyncio
import concurrent.futures
import json
import os
import socket
import threading
import time
from urllib.parse import urlparse

from fastapi import Body, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.web.app import app

# zentralen Config-Helper nutzen (mehrere Fallback-Strategien)
get_config = None
try:
    # 1) Top-level module (wenn working dir auf src gesetzt ist)
    import central_config as _cc
    get_config = _cc.get_config
except Exception:
    try:
        # 2) Paketpfad (when launched as package)
        from src.central_config import get_config as _gc
        get_config = _gc
    except Exception:
        try:
            # 3) relative import (when imported as module in a package)
            from .central_config import get_config as _gc2
            get_config = _gc2
        except Exception:
            # 4) Fallback: lade Datei mittels importlib
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("central_config", os.path.join(os.path.dirname(__file__), 'central_config.py'))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                get_config = module.get_config
            except Exception:
                raise

cfg = get_config()

# Base directory for the application
base_dir = os.path.dirname(os.path.abspath(__file__))

# Folders for static files and templates
static_folder = os.path.join(base_dir, 'web', 'static')
template_folder = os.path.join(base_dir, 'web', 'templates')

# Mount static files and templates
if os.path.isdir(static_folder):
    app.mount('/static', StaticFiles(directory=static_folder), name='static')
templates = Jinja2Templates(directory=template_folder) if os.path.isdir(template_folder) else None

# Simple route to serve the chat UI (renders chat.html if present)


@app.get('/')
async def index(request: Request):
    if templates:
        return templates.TemplateResponse('chat.html', {'request': request})
    return JSONResponse({'detail': 'UI not available, missing templates folder'}, status_code=200)

# Pfad zur Datei mit persistenten Overrides (konfigurierbar via ENV)
# Default: nutze data_path aus CentralConfig
overrides_file = os.environ.get('SIMULATOR_OVERRIDES_FILE', os.path.join(cfg.system.data_path, 'sim_overrides.json'))
# Optionaler Bearer-Token zum Schutz der Simulator-Endpoints
_simulator_token = os.environ.get('SIMULATOR_TOKEN')

# Optionaler Redis-Backend (konfigurierbar)
_simulator_redis_url = os.environ.get('SIMULATOR_REDIS_URL')
_simulator_use_redis = os.environ.get('SIMULATOR_USE_REDIS', '') .lower() in ('1', 'true', 'yes')
_simulator_redis_key = os.environ.get('SIMULATOR_OVERRIDES_KEY', 'sim_overrides')
_redis_client = None

# Thread-sichere In-Memory-Store (wird von _load_overrides initialisiert)
_simulated_overrides = {}
_overrides_lock = threading.Lock()

# Versuche Redis-Client zu initialisieren, falls konfiguriert
if _simulator_redis_url or _simulator_use_redis:
    try:
        import redis as _redis
        try:
            # prefer explicit URL env, otherwise assume default localhost
            url = _simulator_redis_url or os.environ.get('REDIS_URL') or 'redis://127.0.0.1:6379/0'
            _redis_client = _redis.from_url(url, decode_responses=True)
            # einfacher Ping-Test
            _redis_client.ping()
        except Exception:
            _redis_client = None
    except Exception:
        _redis_client = None


def _load_overrides():
    """Load overrides from Redis if available, otherwise from file. Populate _simulated_overrides."""
    global _simulated_overrides
    try:
        if _redis_client:
            raw = _redis_client.get(_simulator_redis_key)
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    with _overrides_lock:
                        _simulated_overrides = {k: (None if v is None else bool(v)) for k, v in data.items()}
                        return
        # Fallback auf Datei
        if os.path.isfile(overrides_file):
            with open(overrides_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    with _overrides_lock:
                        _simulated_overrides = {k: (None if v is None else bool(v)) for k, v in data.items()}
                        return
    except Exception:
        with _overrides_lock:
            _simulated_overrides = {}


def _save_overrides():
    """Persist overrides to Redis if available, otherwise to file (atomic write).
    Non-fatal on failure.
    """
    try:
        # snapshot under lock
        with _overrides_lock:
            snapshot = dict(_simulated_overrides)
        if _redis_client:
            try:
                _redis_client.set(_simulator_redis_key, json.dumps(snapshot))
                return
            except Exception:
                pass
        # Fallback: atomic file write
        tmp = overrides_file + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f)
            os.replace(tmp, overrides_file)
        except Exception:
            # ignore file-write errors
            pass
    except Exception:
        # swallow any unexpected errors
        pass


# Lade Overrides beim Start
_load_overrides()


def require_simulator_token(authorization: str | None = Header(None)):
    """Prüft den Authorization Header gegen SIMULATOR_TOKEN, falls gesetzt.
    Wenn kein SIMULATOR_TOKEN gesetzt ist, erlaubt die Funktion den Zugriff (developer mode).
    """
    if not _simulator_token:
        return True
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing Authorization')
    if not authorization.startswith('Bearer '):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid Authorization header')
    token = authorization.split(' ', 1)[1].strip()
    if token != _simulator_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Forbidden')
    return True

# Globaler Simulator-Override (in-memory) -- bereits oben initialisiert

# Basic status check function (platzhalter, kann durch echte Checks ersetzt werden)


def get_status():
    """Echte, schnelle Health-Checks:
    - Neo4j: wenn neo4j-Driver installiert, führe kurzen Cypher (RETURN 1) mit Timeout aus.
      Falls Treiber nicht verfügbar, Fall back auf TCP-Connect zum Bolt-Port.
    - Chroma / LLM: versuche HTTP GET (timeout kurz). Falls URL ohne Path, versuche /health und dann root.

    Liefert ein Dict mit bools: {"neo4j": bool, "chroma": bool, "llm": bool}.
    """
    # Bevorzugt Werte aus CentralConfig, erlauben aber ENV-Overrides
    neo4j_uri = os.environ.get('NEO4J_URI', cfg.database.neo4j_uri)
    neo4j_user = os.environ.get('NEO4J_USER', cfg.database.neo4j_user)
    neo4j_password = os.environ.get('NEO4J_PASSWORD', cfg.database.neo4j_password)
    chroma_url = os.environ.get('CHROMA_URL')
    llm_url = os.environ.get('LLM_URL')

    def check_tcp(uri: str, default_port: int = 7687) -> bool:
        try:
            p = urlparse(uri)
            host = p.hostname
            port = p.port or default_port
            if not host:
                return False
            with socket.create_connection((host, port), timeout=1):
                return True
        except Exception:
            return False

    def check_http_fallback(url: str) -> bool:
        try:
            import requests

            # versuche zuerst die gegebene URL
            try:
                r = requests.get(url, timeout=1)
                if 200 <= r.status_code < 500:
                    return True
            except Exception:
                pass
            # falls URL ohne Path, versuche /health
            p = urlparse(url)
            base = f"{p.scheme}://{p.netloc}"
            for path in ('/health', '/'):
                try:
                    r = requests.get(base + path, timeout=1)
                    if 200 <= r.status_code < 500:
                        return True
                except Exception:
                    pass
            return False
        except Exception:
            return False

    def check_redis(timeout: float = 0.5) -> bool:
        """Prüfe Redis-Client kurz mit ping und kleinem set/get, ausgeführt mit Timeout."""
        if not _redis_client:
            return False

        def _ping_and_rw():
            try:
                # Ping
                _redis_client.ping()
                # Set a temporary key and delete it
                key = f"healthcheck:{int(time.time() * 1000)}"
                _redis_client.set(key, '1', ex=2)
                val = _redis_client.get(key)
                # best-effort delete
                try:
                    _redis_client.delete(key)
                except Exception:
                    pass
                return val is not None
            except Exception:
                return False

        try:
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                fut = ex.submit(_ping_and_rw)
                return bool(fut.result(timeout=timeout))
            finally:
                try:
                    ex.shutdown(wait=False)
                except Exception:
                    pass
        except Exception:
            return False

    # Defaults: wenn keine URIs/Redis konfiguriert sind -> verhalten konfigurierbar
    # Um zu vermeiden, dass die UI 'verbunden' anzeigt, obwohl z.B. der Neo4j-Container gestoppt ist,
    # prüft diese Einstellung standardmäßig nicht mehr automatisch 'up'.
    assume_up = os.environ.get('ASSUME_SERVICES_UP_IF_NO_CONFIG', 'false').lower() in ('1', 'true', 'yes')
    if not (neo4j_uri or chroma_url or llm_url or _redis_client):
        if assume_up:
            return {"neo4j": True, "chroma": True, "llm": True, "redis": bool(_redis_client)}
        # konservativer Standard: keine Services als erreichbar melden, wenn nichts konfiguriert ist
        return {"neo4j": False, "chroma": False, "llm": False, "redis": bool(_redis_client)}

    # Neo4j: versuche neo4j-Driver wenn vorhanden
    neo4j_ok = False
    if neo4j_uri:
        neo4j_timeout = float(os.environ.get('NEO4J_TIMEOUT', '1'))
        try:
            # lokal importieren, optional dependency
            from neo4j import GraphDatabase

            def _check_neo4j_driver(uri, user, password):
                driver = None
                try:
                    auth = (user, password) if user and password else None
                    driver = GraphDatabase.driver(uri, auth=auth, max_connection_lifetime=30)
                    with driver.session() as session:
                        # einfacher Test-Query
                        result = session.run('RETURN 1')
                        _ = result.single()
                    return True
                finally:
                    try:
                        if driver is not None:
                            driver.close()
                    except Exception:
                        pass

            # Führe den Driver-Check in einem separaten Thread mit Timeout aus
            try:
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    fut = ex.submit(_check_neo4j_driver, neo4j_uri, neo4j_user, neo4j_password)
                    neo4j_ok = bool(fut.result(timeout=neo4j_timeout))
                finally:
                    # don't wait on worker thread if it is still running -> avoid blocking the request
                    try:
                        ex.shutdown(wait=False)
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                # zeitüberschreitung -> nicht erreichbar
                neo4j_ok = False
            except Exception:
                neo4j_ok = False
        except Exception:
            # neo4j-Driver nicht verfügbar oder Fehler -> TCP-Fallback
            neo4j_ok = check_tcp(neo4j_uri, 7687)

    # Chroma / LLM HTTP-Checks
    chroma_ok = True if not chroma_url else check_http_fallback(chroma_url)
    llm_ok = True if not llm_url else check_http_fallback(llm_url)
    # Redis-Check (falls konfiguriert)
    redis_ok = True if not _redis_client else check_redis(float(os.environ.get('REDIS_HEALTH_TIMEOUT', '0.5')))

    # Wende Simulator-Overrides an (falls gesetzt)
    base = {"neo4j": neo4j_ok, "chroma": chroma_ok, "llm": llm_ok, "redis": redis_ok}
    for k, v in _simulated_overrides.items():
        if k in base and v is not None:
            base[k] = bool(v)

    return base


@app.get('/api/status')
async def status_endpoint():
    """Return aggregated service status."""
    return JSONResponse(get_status())

# SSE stream endpoint


@app.get('/api/status/stream')
async def status_stream():
    async def event_generator(poll_interval: float = float(os.environ.get('STATUS_POLL', '5'))):
        last = None
        # send initial event immediately
        while True:
            cur = get_status()
            if last is None or cur != last:
                payload = json.dumps(cur, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                last = cur
            await asyncio.sleep(poll_interval)
    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/api/status/simulate')
async def get_simulator(authorized: bool = Depends(require_simulator_token)):
    """Gebe aktuelle Simulator-Overrides zurück."""
    with _overrides_lock:
        return JSONResponse({"overrides": _simulated_overrides})


@app.post('/api/status/simulate')
async def set_simulator(payload: dict = Body(...), authorized: bool = Depends(require_simulator_token)):
    """Setze Simulator-Overrides per JSON Payload, z.B. {"neo4j": false}
    Setze Wert auf null um Override zu entfernen.
    """
    allowed = {"neo4j", "chroma", "llm", "redis"}
    with _overrides_lock:
        for k, v in payload.items():
            if k in allowed:
                _simulated_overrides[k] = None if v is None else bool(v)
        _save_overrides()
        return JSONResponse({"overrides": _simulated_overrides})


@app.post('/api/status/simulate/toggle')
async def toggle_simulator(service: str, authorized: bool = Depends(require_simulator_token)):
    """Toggle den Override für einen Service (neo4j|chroma|llm). Wenn kein Override vorhanden, invertiert der Toggle den aktuellen realen Status."""
    svc = service.lower()
    if svc not in {"neo4j", "chroma", "llm", "redis"}:
        return JSONResponse({"error": "unknown service"}, status_code=400)
    with _overrides_lock:
        current_real = get_status().get(svc, True)
        current_override = _simulated_overrides.get(svc)
        if current_override is None:
            # Kein Override -> setze auf inverse des realen Werts
            _simulated_overrides[svc] = not bool(current_real)
        else:
            # Override existiert -> toggele es
            _simulated_overrides[svc] = not bool(current_override)
        _save_overrides()
        return JSONResponse({"overrides": _simulated_overrides})


@app.post('/api/status/simulate/reset')
async def reset_simulator(authorized: bool = Depends(require_simulator_token)):
    """Entferne alle Overrides."""
    with _overrides_lock:
        _simulated_overrides.clear()
        _save_overrides()
        return JSONResponse({"overrides": _simulated_overrides})

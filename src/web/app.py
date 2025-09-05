from src.adapters.ollama_adapter import OllamaAdapter
from src.adapters.qdrant_adapter import QdrantAdapter
from src.adapters.neo4j_adapter import Neo4jAdapter
from typing import List, Optional
import json
import time
import io
import contextlib
import logging
import asyncio
import socket
import requests
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from src.bootstrap import register_all_defaults
from src.di_container import resolve
from src.interfaces import IVectorStore, IGraphStore, ILLMService
from fast_import_pipeline_qdrant import FastImportPipeline

# Pipeline-Instanz als Singleton
pipeline = FastImportPipeline()

# Logging-Konfiguration
logging.basicConfig(
    filename='import.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


def log_import(status, filename=None, message=None):
    info = f"Status: {status}"
    if filename:
        info += f", Datei: {filename}"
    if message:
        info += f", Schritt: {message}"
    logging.info(info)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: bootstrap + start single background poller, teardown stops it."""
    try:
        try:
            register_all_defaults()
        except Exception:
            # best-effort bootstrap darf Startup nicht verhindern
            pass

        # --- Background status poller (single shared task for SSE) ---
        # globale Variablen initialisieren
        global _STATUS_POLLER_TASK, _STATUS_SUBSCRIBERS, _STATUS_LAST
        try:
            _STATUS_SUBSCRIBERS
        except NameError:
            _STATUS_SUBSCRIBERS = set()
        try:
            _STATUS_LAST
        except NameError:
            _STATUS_LAST = None
        try:
            _STATUS_POLLER_TASK
        except NameError:
            _STATUS_POLLER_TASK = None

        # Starte den Poller (falls noch nicht gestartet)
        if _STATUS_POLLER_TASK is None:
            interval = max(1, int(os.environ.get('STATUS_POLL', 5)))
            _STATUS_POLLER_TASK = asyncio.create_task(_background_status_poller(interval))

        # Log registered routes once the application has wired routes
        try:
            routes = []
            for r in app.routes:
                methods = ','.join(sorted(getattr(r, 'methods', []) or []))
                routes.append(f"{getattr(r, 'path', '?')} [{methods}]")
            logging.info('Registered routes: ' + '; '.join(routes))
            print('Registered routes:', routes)
        except Exception:
            logging.exception('Failed to list registered routes')

        yield
    finally:
        # Stoppe background poller sauber beim Herunterfahren
        poller_task = globals().get('_STATUS_POLLER_TASK', None)
        if poller_task is not None:
            poller_task.cancel()
            try:
                await poller_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Smart RAG - Ollama UI", lifespan=lifespan)

# CORS für die Entwicklung: erlaubt Zugriffe von anderen Hosts/Ports (z. B. Frontend dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="src/web/templates")


# Fortschritts-Status für Import (einfach, global) — sicherstellen, dass diese Variable vorhanden ist
IMPORT_PROGRESS = {
    "status": "idle",
    "steps": [],
    "stats": {},
    "error": None
}


@app.post("/import", tags=["Import"], include_in_schema=True)
async def import_document(file: UploadFile = File(...)):
    start_time = time.time()
    IMPORT_PROGRESS["status"] = "running"
    IMPORT_PROGRESS["steps"] = ["Import gestartet..."]
    IMPORT_PROGRESS["stats"] = {}
    IMPORT_PROGRESS["error"] = None
    log_import("gestartet", getattr(file, 'filename', None), "Import gestartet...")
    try:
        if not file:
            IMPORT_PROGRESS["status"] = "error"
            IMPORT_PROGRESS["error"] = "No file uploaded."
            log_import("error", None, "No file uploaded.")
            return JSONResponse({"ok": False, "error": "No file uploaded."}, status_code=400)
        IMPORT_PROGRESS["steps"].append(f"Datei empfangen: {getattr(file, 'filename', None)}")
        log_import("empfangen", getattr(file, 'filename', None), "Datei empfangen")
        content = await file.read()
        if not content:
            IMPORT_PROGRESS["status"] = "error"
            IMPORT_PROGRESS["error"] = "File is empty."
            log_import("error", getattr(file, 'filename', None), "File is empty.")
            return JSONResponse({"ok": False, "error": "File is empty."}, status_code=400)
        content_type = getattr(file, "content_type", "") or ""
        text = None
        filename_lower = (file.filename or '').lower()
        if "pdf" in content_type.lower() or filename_lower.endswith('.pdf'):
            IMPORT_PROGRESS["steps"].append("PDF wird gelesen...")
            log_import("fortschritt", file.filename, "PDF wird gelesen...")
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content))
                pages = []
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                text = "\n".join(pages)
                IMPORT_PROGRESS["steps"].append(f"PDF Seiten extrahiert: {len(pages)}")
                log_import("fortschritt", file.filename, f"PDF Seiten extrahiert: {len(pages)}")
            except Exception as pdf_err:
                IMPORT_PROGRESS["steps"].append(f"PDF extraction failed: {pdf_err}")
                log_import("error", file.filename, f"PDF extraction failed: {pdf_err}")
                try:
                    text = content.decode("utf-8")
                except Exception:
                    text = content.decode("utf-8", errors="replace")
        elif filename_lower.endswith('.docx') or 'word' in content_type.lower():
            IMPORT_PROGRESS["steps"].append("DOCX wird gelesen...")
            log_import("fortschritt", file.filename, "DOCX wird gelesen...")
            try:
                from docx import Document
                import io as _io
                doc = Document(_io.BytesIO(content))
                paragraphs = [p.text for p in doc.paragraphs if p.text]
                text = "\n".join(paragraphs)
                IMPORT_PROGRESS["steps"].append(f"DOCX Absätze extrahiert: {len(paragraphs)}")
                log_import("fortschritt", file.filename, f"DOCX Absätze extrahiert: {len(paragraphs)}")
            except Exception as docx_err:
                IMPORT_PROGRESS["steps"].append(f"DOCX extraction failed: {docx_err}")
                log_import("error", file.filename, f"DOCX extraction failed: {docx_err}")
                try:
                    text = content.decode("utf-8")
                except Exception:
                    try:
                        text = content.decode("latin-1")
                    except Exception:
                        text = content.decode("utf-8", errors="replace")
        else:
            IMPORT_PROGRESS["steps"].append("Textdatei wird gelesen...")
            log_import("fortschritt", file.filename, "Textdatei wird gelesen...")
            try:
                text = content.decode("utf-8")
            except Exception:
                try:
                    text = content.decode("latin-1")
                except Exception:
                    text = content.decode("utf-8", errors="replace")
        IMPORT_PROGRESS["steps"].append("Text extrahiert, Vektor-Import startet...")
        log_import("fortschritt", file.filename, "Text extrahiert, Vektor-Import startet...")
        docs = [text]
        metas = [{"source": file.filename}]
        vector_store = get_vector_store()
        graph_store = get_graph_store()
        warnings = None
        try:
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    IMPORT_PROGRESS["steps"].append("Vektoren werden erstellt...")
                    log_import("fortschritt", file.filename, "Vektoren werden erstellt...")
                    # Safety checks: ensure we have a vector store and it's the intended adapter
                    if vector_store is None:
                        raise RuntimeError("No vector store configured (expected Qdrant or other vector adapter)")
                    try:
                        adapter_name = type(vector_store).__name__
                    except Exception:
                        adapter_name = str(vector_store)
                    logging.info("Using vector store adapter: %s", adapter_name)
                    # If adapter does not provide add_documents with (documents, metadatas, embeddings), this will raise
                    await vector_store.add_documents(docs, metas)
                    IMPORT_PROGRESS["steps"].append("Vektoren erfolgreich gespeichert.")
                    log_import("fortschritt", file.filename, "Vektoren erfolgreich gespeichert.")
            finally:
                captured = (buf_out.getvalue() or "") + (buf_err.getvalue() or "")
                if captured.strip():
                    lines = [line.strip() for line in captured.splitlines() if line.strip()]
                    filtered = []
                    for line in lines:
                        if line.startswith('Failed to send telemetry event'):
                            filtered.append('Telemetry sending failed (see server logs)')
                            continue
                        if 'Traceback (most recent call last)' in line or line.startswith('  File'):
                            continue
                        filtered.append(line)
                    if filtered:
                        seen = []
                        for f in filtered:
                            if f not in seen:
                                seen.append(f)
                        summary = '\n'.join(seen[:10])
                        if len(seen) > 10:
                            summary += '\n... (more)'
                        warnings = summary
        except Exception as e:
            IMPORT_PROGRESS["steps"].append(f"Vektor-Import Fehler: {e}")
            IMPORT_PROGRESS["status"] = "error"
            IMPORT_PROGRESS["error"] = f"Vector store error: {e}"
            log_import("error", file.filename, f"Vektor-Import Fehler: {e}")
            return JSONResponse({"ok": False, "error": f"Vector store error: {e}"}, status_code=500)
        try:
            IMPORT_PROGRESS["steps"].append("Graph-Import startet...")
            log_import("fortschritt", file.filename, "Graph-Import startet...")
            entity = [{"name": file.filename, "type": "DOCUMENT", "content": text, "metadata": metas[0]}]
            await graph_store.add_entities(entity)
            IMPORT_PROGRESS["steps"].append("Graph-Import abgeschlossen.")
            log_import("fortschritt", file.filename, "Graph-Import abgeschlossen.")
        except Exception as e:
            IMPORT_PROGRESS["steps"].append(f"Graph-Import Fehler: {e}")
            log_import("error", file.filename, f"Graph-Import Fehler: {e}")
        # Statistiken aus beiden Adaptern abfragen
        try:
            vstats = await vector_store.monitoring_info()
        except Exception as e:
            vstats = {"error": str(e)}
        try:
            gstats = await graph_store.monitoring_info()
        except Exception as e:
            gstats = {"error": str(e)}
        total_time = time.time() - start_time
        IMPORT_PROGRESS["status"] = "done"
        IMPORT_PROGRESS["stats"] = {
            "filename": file.filename,
            "processing_time": round(total_time, 3),
            "warnings": warnings,
            "steps": IMPORT_PROGRESS["steps"],
            "vector_stats": vstats,
            "graph_stats": gstats
        }
        IMPORT_PROGRESS["steps"].append(f"Import abgeschlossen in {round(total_time, 3)}s.")
        log_import("abgeschlossen", file.filename, f"Import abgeschlossen in {round(total_time, 3)}s.")
        resp = {"ok": True, "message": "Imported", "filename": file.filename, "processing_time": round(total_time, 3), "vector_stats": vstats, "graph_stats": gstats}
        if warnings:
            print(f"[IMPORT] Warnings: {warnings}")
        return JSONResponse(resp)
    except Exception as e:
        IMPORT_PROGRESS["status"] = "error"
        IMPORT_PROGRESS["error"] = str(e)
        log_import("error", getattr(file, 'filename', None), f"File read error: {e}")
        return JSONResponse({"ok": False, "error": f"File read error: {e}"}, status_code=400)


@app.post("/api/import", tags=["Import"], include_in_schema=True)
async def import_document_api(file: UploadFile = File(...)):
    """Kompatibilitäts-Wrapper: leitet an /import weiter"""
    return await import_document(file)


@app.post("/legacy/import", tags=["Import"], include_in_schema=True)
async def import_document_legacy(file: UploadFile = File(...)):
    """Kompatibilitäts-Wrapper für ältere Pfade"""
    return await import_document(file)


@app.get("/import/status")
async def import_status():
    """Gibt den aktuellen Fortschritt des Imports zurück."""
    return JSONResponse(IMPORT_PROGRESS)


@app.post("/import/pdf")
async def import_pdf(file: UploadFile = File(...)):
    """Importiert eine PDF-Datei und verarbeitet sie mit der Pipeline"""
    import tempfile
    import os
    if not file:
        return JSONResponse({"ok": False, "error": "No file uploaded."}, status_code=400)
    content = await file.read()
    if not content:
        return JSONResponse({"ok": False, "error": "File is empty."}, status_code=400)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    results = pipeline.import_pdf(tmp_path)
    os.remove(tmp_path)
    return JSONResponse(results)


@app.get("/status")
async def get_status():
    """Gibt den Status der Pipeline zurück"""
    status = {
        "embedding_model_initialized": pipeline.embedding_model is not None,
        "vector_db_initialized": pipeline.vector_db is not None,
        "total_chunks": len(pipeline.chunks),
        "total_connections": pipeline.connection_count,
    }
    return JSONResponse(status)


# Globale Hybrid-Konfiguration für DB und LLM
ACTIVE_CONFIG = {
    "databases": ["neo4j", "qdrant"],
    "llms": ["ollama"]
}

# Hilfsfunktionen für Hybrid-Adapter


def get_vector_stores():
    # Return only vector-oriented stores (Qdrant, Faiss, mock vector stores, ...)
    stores = []
    for db in ACTIVE_CONFIG.get("databases", []):
        if db == "qdrant":
            stores.append(("qdrant", QdrantAdapter()))
        # add other vector store adapters here if present, e.g. 'faiss'
    return stores


def get_graph_stores():
    # Return only graph-oriented stores (Neo4j, mock graph stores, ...)
    stores = []
    for db in ACTIVE_CONFIG.get("databases", []):
        if db == "neo4j":
            stores.append(("neo4j", Neo4jAdapter()))
        # add other graph store adapters here if present
    return stores


def get_llms():
    llms = []
    for llm in ACTIVE_CONFIG.get("llms", []):
        if llm == "ollama":
            llms.append(("ollama", OllamaAdapter()))
    return llms


def get_vector_store():
    # Liefert den ersten VectorStore aus get_vector_stores()
    stores = get_vector_stores()
    return stores[0][1] if stores else None


def get_graph_store():
    # Liefert den ersten GraphStore aus get_graph_stores()
    stores = get_graph_stores()
    return stores[0][1] if stores else None


def get_llm():
    # Liefert den ersten LLM aus get_llms()
    llms = get_llms()
    return llms[0][1] if llms else None


@app.post("/status/db")
async def status_db(request: Request):
    results = []
    for name, store in get_vector_stores():
        try:
            if hasattr(store, "health_check"):
                status = await store.health_check()
                results.append({"db": name, **status})
            else:
                results.append({"db": name, "ok": False, "error": "Kein health_check vorhanden"})
        except Exception as e:
            results.append({"db": name, "ok": False, "error": str(e)})
    for name, store in get_graph_stores():
        try:
            if hasattr(store, "health_check"):
                status = await store.health_check()
                results.append({"db": name, **status})
            else:
                results.append({"db": name, "ok": False, "error": "Kein health_check vorhanden"})
        except Exception as e:
            results.append({"db": name, "ok": False, "error": str(e)})
    return JSONResponse({"databases": results})


@app.post("/status/llm")
async def status_llm_post():
    results = []
    for name, llm in get_llms():
        try:
            status = await llm.health_check()
            results.append({"llm": name, **status})
        except Exception as e:
            results.append({"llm": name, "ok": False, "error": str(e)})
    return JSONResponse({"llms": results})


@app.get("/status/llm")
async def status_llm_get():
    results = []
    for name, llm in get_llms():
        try:
            status = await llm.health_check()
            results.append({"llm": name, **status})
        except Exception as e:
            results.append({"llm": name, "ok": False, "error": str(e)})
    return JSONResponse({"llms": results})


@app.post("/query")
async def query(request: Request, question: Optional[str] = Form(None)):
    req_start = time.time()

    # Accept JSON or form input. JSON body should include {"question": "..."}.
    body_json = None
    try:
        if request.headers.get('content-type', '').lower().startswith('application/json'):
            body_json = await request.json()
    except Exception:
        body_json = None

    if body_json and isinstance(body_json, dict) and 'question' in body_json:
        q_text = str(body_json.get('question') or '').strip()
    else:
        q_text = (question or '').strip()

    if not q_text:
        return JSONResponse({'error': 'question is required'}, status_code=400)

    vector_store = get_vector_store()
    graph_store = get_graph_store()
    llm = get_llm()

    # simple redaction helpers to avoid leaking PII or entire DB records
    import re

    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    DIGIT_RE = re.compile(r"\b\d{4,}\b")

    def redact_and_trim(text: str, max_len: int = 200) -> str:
        if not isinstance(text, str):
            text = str(text)
        t = EMAIL_RE.sub('[REDACTED_EMAIL]', text)
        t = DIGIT_RE.sub('[REDACTED_DIGITS]', t)
        t = ' '.join(t.split())
        return t[:max_len]

    contexts: List[str] = []
    vs_time = None
    try:
        vs_start = time.time()
        vs_results = await vector_store.search_similar(q_text, k=5)
        vs_time = time.time() - vs_start

        def _sanitize(r):
            if isinstance(r, dict):
                meta = r.get('metadata') or r.get('metadatas') or {}
                for key in ('source', 'title', 'name'):
                    if isinstance(meta, dict) and key in meta and isinstance(meta[key], str) and meta[key].strip():
                        return redact_and_trim(meta[key])

                content = r.get('content') or r.get('text') or ''
                if isinstance(content, str) and content.strip():
                    return redact_and_trim(content)

                safe = {k: v for k, v in r.items() if k in ('id', 'score')}
                try:
                    return redact_and_trim(json.dumps(safe))
                except Exception:
                    return redact_and_trim(str(safe))
            return redact_and_trim(str(r))

        contexts = [c for c in (_sanitize(r) for r in vs_results) if c]
    except Exception:
        contexts = []

    graph_summary: List[str] = []
    graph_time = None
    try:
        g_start = time.time()
        graph_results = await graph_store.query_graph("GRAPH_SEARCH", {"query": q_text})
        graph_time = time.time() - g_start
        if isinstance(graph_results, list):
            for node in graph_results[:5]:
                if isinstance(node, dict):
                    for k in ('name', 'title', 'label'):
                        if k in node and isinstance(node[k], str):
                            graph_summary.append(redact_and_trim(node[k]))
                            break
                else:
                    graph_summary.append(redact_and_trim(str(node)))
    except Exception:
        graph_summary = []

    prompt = """
    Use the following contexts to answer the question.

    Question: {question}

    Contexts:
    {contexts}

    GraphResults:
    {graph}
    """.format(question=q_text, contexts="\n---\n".join(contexts), graph=json.dumps(graph_summary))

    llm_time = None
    try:
        llm_start = time.time()
        answer = await llm.generate(prompt)
        llm_time = time.time() - llm_start
    except Exception as e:
        answer = f"LLM generate error: {e}"

    total = time.time() - req_start

    return JSONResponse({
        "question": q_text,
        "answer": answer,
        "contexts": contexts,
        "contexts_count": len(contexts),
        "graph_summary": graph_summary,
        "graph_count": len(graph_summary),
        "processing_time": round(total, 3),
        "timings": {
            "vector_search": round(vs_time, 3) if vs_time is not None else None,
            "graph_search": round(graph_time, 3) if graph_time is not None else None,
            "llm": round(llm_time, 3) if llm_time is not None else None,
        },
    })


@app.get("/status/neo4j")
async def status_neo4j():
    try:
        neo4j_adapter = resolve(IGraphStore)
        status = await neo4j_adapter.health_check()
        return JSONResponse(status)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/status/qdrant")
async def status_qdrant():
    try:
        qdrant_adapter = resolve(IVectorStore)
        status = await qdrant_adapter.health_check()
        return JSONResponse(status)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/monitoring/pipeline")
async def monitoring_pipeline():
    try:
        if hasattr(pipeline, "monitoring_info"):
            info = await pipeline.monitoring_info()
        elif hasattr(pipeline, "status"):
            info = pipeline.status()
        else:
            info = {"ok": True, "message": "Pipeline läuft."}
        return JSONResponse(info)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/monitoring/neo4j")
async def monitoring_neo4j():
    try:
        neo4j_adapter = resolve(IGraphStore)
        info = await neo4j_adapter.monitoring_info()
        return JSONResponse(info)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/monitoring/llm")
async def monitoring_llm():
    try:
        llm_adapter = resolve(ILLMService)
        info = await llm_adapter.monitoring_info()
        return JSONResponse(info)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# --- Service health helpers (used by /api/status and /api/status/stream) ---
def _check_tcp(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _check_http(url: str, timeout: float = 1.5) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return 200 <= r.status_code < 400
    except Exception:
        return False


def _parse_host_port(uri: str):
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


async def _probe_services():
    """Run all checks in threadpool and return a dict with booleans."""
    # read configuration from existing ACTIVE_CONFIG or env fallbacks
    neo4j_uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
    qdrant_url = os.environ.get('QDRANT_URL') or 'http://localhost:6333'
    llm_url = os.environ.get('LLM_URL') or 'http://localhost:11434'

    async def _neo_check():
        if neo4j_uri.startswith('bolt') or neo4j_uri.startswith('neo4j'):
            host, port = _parse_host_port(neo4j_uri)
            if port is None:
                port = 7687
            return await asyncio.to_thread(_check_tcp, host, port, 1.0)
        return await asyncio.to_thread(_check_http, neo4j_uri, 1.5)

    async def _qdrant_check():
        try_urls = [qdrant_url.rstrip('/'), qdrant_url.rstrip('/') + '/api/health', qdrant_url.rstrip('/') + '/collections']
        for u in try_urls:
            ok = await asyncio.to_thread(_check_http, u, 1.5)
            if ok:
                return True
        return False

    async def _llm_check():
        try_urls = [llm_url.rstrip('/'), llm_url.rstrip('/') + '/v1/models', llm_url.rstrip('/') + '/health']
        for u in try_urls:
            ok = await asyncio.to_thread(_check_http, u, 1.5)
            if ok:
                return True
        return False

    neo_ok, qdrant_ok, llm_ok = await asyncio.gather(_neo_check(), _qdrant_check(), _llm_check())
    return {"neo4j": bool(neo_ok), "qdrant": bool(qdrant_ok), "llm": bool(llm_ok)}


# --- Neues: zentraler Background-Poller und Subscriber-Mechanik ---
async def _background_status_poller(interval: int = 5):
    """Periodisch _probe_services ausführen und Unterschiede an Subscriber broadcasten."""
    global _STATUS_LAST, _STATUS_SUBSCRIBERS
    while True:
        try:
            state = await _probe_services()
            if _STATUS_LAST is None or state != _STATUS_LAST:
                _STATUS_LAST = state
                payload = json.dumps(state)
                # Broadcast an alle Subscriber-Queues
                for queue in list(_STATUS_SUBSCRIBERS):
                    try:
                        queue.put_nowait(f"data: {payload}\n\n")
                    except Exception:
                        pass
        except Exception:
            # Fehler ignorieren, Poller weitermachen
            pass
        await asyncio.sleep(max(1, int(os.environ.get('STATUS_POLL', interval))))


# Server-Sent Events stream: nutzt zentrale Subscriber-Queues
async def _status_stream_generator(poll_interval: int = 5):
    q: "asyncio.Queue[str]" = asyncio.Queue()
    _STATUS_SUBSCRIBERS.add(q)
    try:
        if '_STATUS_LAST' in globals() and _STATUS_LAST is not None:
            await q.put(f"data: {json.dumps(_STATUS_LAST)}\n\n")
        while True:
            msg = await q.get()
            yield msg
    finally:
        try:
            _STATUS_SUBSCRIBERS.discard(q)
        except Exception:
            pass


# Simple helper to run with `python -m uvicorn src.web.app:app --reload`
@app.get('/api/status/stream')
async def api_status_stream():
    return StreamingResponse(_status_stream_generator(), media_type='text/event-stream')


@app.get("/status/qdrant/points")
async def status_qdrant_points(limit: int = 20):
    """Debug endpoint: list count and up to `limit` points (id + payload keys + content) from Qdrant collection.
    Safe: imports qdrant_client lazily and returns errors in JSON instead of raising.
    """
    col = os.environ.get('QDRANT_COLLECTION') or os.environ.get('CHROMA_COLLECTION') or 'rag_documents'
    host = os.environ.get('QDRANT_HOST', 'http://localhost:6333')
    port = int(os.environ.get('QDRANT_PORT', 6333))
    try:
        from qdrant_client import QdrantClient
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"qdrant-client not installed: {e}"}, status_code=500)

    try:
        client = QdrantClient(url=host) if str(host).startswith('http') else QdrantClient(host=host, port=port)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to create Qdrant client: {e}"}, status_code=500)

    try:
        cnt = client.count(collection_name=col)
        total = getattr(cnt, 'count', None)
    except Exception as e:
        total = None
        count_error = str(e)
    else:
        count_error = None

    samples = []
    try:
        pts = client.scroll(collection_name=col, limit=limit)
        # pts can be dict or iterable depending on client version
        if isinstance(pts, dict) and 'points' in pts:
            pts_list = pts['points']
        else:
            pts_list = list(pts)
        for p in pts_list:
            # Build a defensive extraction since qdrant-client may return different types/structures
            pid = None
            payload = {}
            raw_repr = None
            try:
                raw_repr = repr(p)
            except Exception:
                raw_repr = str(p)

            # dict-like responses
            if isinstance(p, dict):
                pid = p.get('id') or p.get('point', {}).get('id') if isinstance(p.get('point'), dict) else p.get('id')
                payload = p.get('payload') or (p.get('point', {}) or {}).get('payload') or {}
            else:
                # object-like responses (qdrant-client models)
                # common attributes: id, payload, point
                pid = getattr(p, 'id', None)
                if pid is None:
                    # sometimes nested under .point.id
                    point_obj = getattr(p, 'point', None)
                    if point_obj is not None:
                        pid = getattr(point_obj, 'id', None)
                        payload = getattr(point_obj, 'payload', None) or {}
                else:
                    payload = getattr(p, 'payload', None) or {}
            # ensure payload is a dict
            if not isinstance(payload, dict):
                try:
                    # try to coerce dataclass/obj to dict
                    payload = dict(payload)
                except Exception:
                    payload = {}

            samples.append({
                'id': None if pid is None else str(pid),
                'payload_keys': list(payload.keys()) if isinstance(payload, dict) else None,
                'content': payload.get('content') if isinstance(payload, dict) else None,
                'raw': raw_repr,
            })
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to scroll points: {e}"}, status_code=500)

    return JSONResponse({"ok": True, "collection": col, "total": total, "count_error": count_error, "samples": samples})

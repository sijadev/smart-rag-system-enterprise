# Smart RAG System

Kurzanleitung

- Integrationstests mit Neo4j: siehe docs/INTEGRATION_RUNBOOK.md
- UML-Diagramme: docs/uml/

Repository-Setup

1. Setze das Secret `NEO4J_PASSWORD` im GitHub-Repo (Settings -> Secrets).
2. Lokale Integration: docker compose -f docker-compose.neo4j.yml up -d
3. Tests: pytest -q

Weitere Details: docs/INTEGRATION_RUNBOOK.md


## Web UI (FastAPI) — Schnellstart

Dieses Repository enthält eine kleine Web‑UI (FastAPI) zur Demonstration und schnellen Interaktion
mit dem Smart RAG System (Importieren von Dokumenten, einfache Frage‑Antworten über das LLM).

Voraussetzungen:
- Ollama lokal (optional) oder anderer LLM, der über die Factories registriert ist.
- Python‑Abhängigkeiten aus `requirements.txt` installiert (fastapi, uvicorn, httpx, ...).

Beispiel‑Installation:

    python -m pip install -r requirements.txt

Starten der Dev‑UI:

    # Startet FastAPI auf http://localhost:8000
    uvicorn src.web.app:app --reload --host 0.0.0.0 --port 8000

Die UI ist erreichbar unter: http://localhost:8000 — erlaubt Dokument‑Upload und einfache Abfragen.


## Ollama Integration

Der Adapter `src/adapters/ollama_adapter.py` unterstützt zwei Modi:
- HTTP API (vorzugsweise): konfigurierbar mit `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL`.
- CLI Fallback: ruft lokal `ollama generate` via subprocess auf, falls HTTP nicht erreichbar.

Konfiguration (Beispiel):

    export OLLAMA_HOST=http://localhost
    export OLLAMA_PORT=11434
    export OLLAMA_MODEL=your_model_name

Hinweis: Die embed‑Methode im Adapter ist ein Best‑Effort‑Fallback. Für produktive semantische Suche
sollten echte Embeddings (z. B. externe Embedding‑Service oder ein Modell mit Embedding‑API) verwendet werden.


## Bootstrapping & Dependency Injection

- Default‑Registrierungen (Adapter, Strategien) werden zentral in `src/bootstrap.py` bereitgestellt.
- Bootstrapping wird idempotent ausgeführt und in zwei Kontexten genutzt:
  - ASGI Startup (FastAPI): `src/web/app.py` ruft `register_all_defaults()` im `startup`‑Event auf — empfohlen für Web‑Apps.
  - Beim programmgesteuerten Start über `SmartRAGSystem.initialize()` (siehe `src/rag_system.py`) — empfohlen, wenn du die App programmgesteuert initialisierst.

Um Bootstrapping beim Start zu überspringen (z. B. in Tests), setze die Umgebungsvariable:

    export SKIP_BOOTSTRAP=1


## Entwicklung & Tests

- Linting: flake8 (siehe .flake8 / pyproject.toml)
- Pre‑commit: Ein Hook (scripts/pre-commit.sh) formatiert staged .py Dateien mit autopep8 und prüft mit flake8.
- CI: GitHub Actions Workflow `.github/workflows/ci.yml` führt flake8 + pytest (3.10/3.11) aus.


## Produktionsempfehlungen (Kurz)

- Sicherer Betrieb von Ollama/LLM (Firewall, Auth) falls aus dem Netz erreichbar.
- Replace pseudo‑embeddings durch echte Embedding‑Pipeline (Faiss/Chroma) vor Produktiv‑Nutzung.
- Containerize App + Ollama (Docker Compose) und benutze Reverse Proxy + TLS.

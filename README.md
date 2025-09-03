Smart RAG System

Kurzanleitung

- Integrationstests mit Neo4j: siehe docs/INTEGRATION_RUNBOOK.md
- UML-Diagramme: docs/uml/

Repository-Setup

1. Setze das Secret `NEO4J_PASSWORD` im GitHub-Repo (Settings -> Secrets).
2. Lokale Integration: docker compose -f docker-compose.neo4j.yml up -d
3. Tests: pytest -q

Weitere Details: docs/INTEGRATION_RUNBOOK.md


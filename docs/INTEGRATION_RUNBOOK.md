# Integrationstest Runbook (Neo4j)

Kurz: Dieses Dokument beschreibt, wie du die Neo4j-Integration lokal und in CI ausführst.

Voraussetzungen
- Docker ist installiert (für lokalen Neo4j-Container) oder ein erreichbarer Neo4j-Server.
- Python 3.11, pip
- GitHub Repository Secret `NEO4J_PASSWORD` ist im Repo angelegt (wurde von dir erstellt).

Lokale Ausführung (schnell)
1. Neo4j lokal starten (Docker-Compose ist im Repo):
   docker compose -f docker-compose.neo4j.yml up -d

2. Setze Umgebungsvariablen für diesen Terminal-Run:
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=neo4j123
   (Passe PASSWORD an dein Secret an)

3. Installiere Abhängigkeiten falls nötig:
   python3 -m pip install -r requirements.txt
   python3 -m pip install pytest neo4j

4. Integrationstest ausführen:
   pytest -q tests/test_adapters_integration.py::test_smartrag_system_bootstrap_and_query

CI / GitHub Actions
- Workflow: .github/workflows/integration-neo4j.yml
- Benötigtes Secret im Repo: `NEO4J_PASSWORD` (setze dort das Passwort, z. B. `neo4j123`)
- Der Workflow startet temporär einen Neo4j-Container und führt die Integrationstests aus.

Tipps
- Verwende lokal die gleichen ENV-Variablen wie in CI, damit Verhalten übereinstimmt.
- Wenn du Debug-Ausgaben brauchst, führe pytest mit -vv -s.
- Nach Tests: docker compose -f docker-compose.neo4j.yml down

Wenn du möchtest, kann ich jetzt: A) die Änderungen committen und pushen (ich mache das gleich), oder B) zusätzlich die README anpassen. Du wolltest zuerst B, dann A — ich füge die Datei hinzu und committe/pushe anschließend.


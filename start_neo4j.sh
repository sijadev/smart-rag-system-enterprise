#!/bin/bash

# Smart RAG System - Docker Setup Script
# Startet Neo4j + Qdrant via docker-compose und prüft Health

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "🔁 Loading .env if present..."
if [ -f .env ]; then
  # export all variables from .env (simple loader)
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "✅ .env loaded"
else
  echo "⚠️ .env not found — rely on environment variables"
fi

# choose docker compose command
if command -v docker-compose >/dev/null 2>&1; then
  DC_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  DC_CMD="docker compose"
else
  echo "❌ docker-compose not found. Install Docker Compose or use Docker Engine v2+ (docker compose)."
  exit 1
fi

# --- Determine Qdrant host port (allow overriding via QDRANT_HOST_PORT env) ---
DEFAULT_QPORT="${QDRANT_HOST_PORT:-6333}"
echo "🔎 Determining Qdrant host port (requested: ${DEFAULT_QPORT})..."
if command -v python3 >/dev/null 2>&1; then
  FREE_PORT=$(python3 - <<PY
import socket
req = int("$DEFAULT_QPORT")
# Versuche, den gewünschten Port zu binden; wenn das fehlschlägt, wähle einen freien Ephemeral-Port
s = socket.socket()
try:
    s.bind(('', req))
    s.close()
    print(req)
except Exception:
    s = socket.socket()
    s.bind(('', 0))
    print(s.getsockname()[1])
    s.close()
PY
)
else
  echo "⚠️ python3 not available to detect free port — using requested port ${DEFAULT_QPORT}"
  FREE_PORT="$DEFAULT_QPORT"
fi

export QDRANT_HOST_PORT="$FREE_PORT"
echo "✅ QDRANT_HOST_PORT=$QDRANT_HOST_PORT"

# If user provided a QDRANT_URL in .env, we will override it later for healthcheck to the actual port

echo "📦 Stopping and removing existing containers (if any)..."
$DC_CMD down -v || true

# remove specifically named containers if present
docker rm -f smart_rag_neo4j smart_rag_qdrant 2>/dev/null || true

echo "⬆️  Starting services via compose..."
$DC_CMD up -d --remove-orphans

# wait a bit for services to spin up
echo "⏳ Waiting for services to initialize (sleep 20s)..."
sleep 20

# --- Neo4j health & APOC check ---
NEO_USER="${NEO4J_USER:-neo4j}"
NEO_PWD="${NEO4J_PASSWORD:-}"

if [ -z "$NEO_PWD" ]; then
  echo "⚠️ NEO4J_PASSWORD not set in environment/.env — cannot perform authenticated checks"
else
  echo "🔎 Testing Neo4j connectivity and APOC..."
  if docker exec smart_rag_neo4j cypher-shell -u "$NEO_USER" -p "$NEO_PWD" "RETURN 1" >/dev/null 2>&1; then
    echo "✅ Neo4j basic connectivity OK"
  else
    echo "❌ Neo4j connectivity test FAILED"
  fi

  # Test APOC procedures availability
  if docker exec smart_rag_neo4j cypher-shell -u "$NEO_USER" -p "$NEO_PWD" "CALL apoc.help('apoc') YIELD name RETURN count(name) as apoc_procedures" >/dev/null 2>&1; then
    echo "✅ APOC appears available"
  else
    echo "⚠️ APOC test failed — check neo4j logs for plugin errors"
    docker logs smart_rag_neo4j --tail 40 || true
  fi
fi

# --- Qdrant health check ---
QDRANT_URL="http://localhost:${QDRANT_HOST_PORT}"
export QDRANT_URL
echo "🔎 Testing Qdrant at $QDRANT_URL (host port: ${QDRANT_HOST_PORT})..."

if command -v curl >/dev/null 2>&1; then
  # Try multiple endpoints because Qdrant image variants expose different health paths
  endpoints=("/health" "/api/v1/health" "/collections" "/api/collections")
  max_wait=60
  interval=2
  waited=0
  ok=0
  while [ $waited -lt $max_wait ]; do
    for ep in "${endpoints[@]}"; do
      status_code=$(curl -s -o /dev/null -w "%{http_code}" "$QDRANT_URL${ep}" || true)
      if [ "$status_code" = "200" ]; then
        echo "✅ Qdrant health OK (endpoint: ${ep})"
        ok=1
        break 2
      fi
    done
    sleep $interval
    waited=$((waited + interval))
  done

  if [ $ok -ne 1 ]; then
    echo "❌ Qdrant health check did not return HTTP 200 within ${max_wait}s — showing recent logs"
    docker logs smart_rag_qdrant --tail 80 || true
  fi
else
  echo "⚠️ curl not available — cannot perform HTTP health check for Qdrant"
fi

# Summary and access info
echo "\n📋 Summary"
$DC_CMD ps

echo "\n🌐 Services started. Useful endpoints:"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - Neo4j Bolt: ${NEO4J_URI:-bolt://localhost:7687} (user: ${NEO4J_USER:-neo4j})"
echo "  - Qdrant API: $QDRANT_URL"

echo "\n🔐 Credentials: Neo4j user=${NEO4J_USER:-neo4j} (password from .env/ENV)"

echo "🎯 Next: run your import pipeline, e.g. python src/pipelines/import_orchestrator.py or ./fast_import_pipeline_neo4j.py"

exit 0

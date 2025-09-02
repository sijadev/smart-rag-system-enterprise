#!/bin/bash

# Smart RAG System - Neo4j Docker Setup Script
# Behebt APOC Plugin Probleme automatisch (ohne separate Config-Datei)

echo "🚀 Starting Smart RAG Neo4j Container with APOC Support..."

# Stop und entferne existierende Container
echo "🛑 Stopping existing containers..."
docker-compose down -v

# Entferne alte Container falls vorhanden
docker rm -f smart_rag_neo4j 2>/dev/null || true

# Baue und starte Neo4j Container (nur mit ENV-Variablen)
echo "🔧 Building Neo4j container with APOC configuration via ENV variables..."
docker-compose up -d

# Warte auf Container Start
echo "⏳ Waiting for Neo4j to start (45 seconds)..."
sleep 45

# Teste Container Status
echo "📋 Container Status:"
docker-compose ps

# Teste APOC Installation
echo "🔍 Testing APOC plugin availability..."
if docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "CALL apoc.help('apoc') YIELD name RETURN count(name) as apoc_procedures" 2>/dev/null; then
    echo "✅ APOC plugin successfully installed and configured!"
    echo "✅ Neo4j is ready for Smart RAG Pipeline"
    echo ""
    echo "🌐 Access Neo4j Browser: http://localhost:7474"
    echo "🔌 Bolt Connection: bolt://localhost:7687"
    echo "🔐 Credentials: neo4j / neo4j123"
else
    echo "⚠️ APOC plugin test failed, but container is running"
    echo "📋 Container logs:"
    docker logs smart_rag_neo4j --tail 20
    echo ""
    echo "💡 Try connecting manually and run: CALL apoc.help('apoc')"
fi

echo ""
echo "🎯 Ready to run: python fast_import_pipeline.py"

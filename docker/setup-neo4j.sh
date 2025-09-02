#!/bin/bash
"""
Docker Setup Script für Neo4j mit Graph Data Science Plugin
Behebt das Problem: "gds" is not a known Neo4j plugin
"""

set -e

echo "🐳 Neo4j Docker Setup mit Graph Data Science Plugin"
echo "=" * 60

# Stoppe bestehende Container
echo "🛑 Stopping existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Entferne alte Volumes (optional - nur wenn gewünscht)
read -p "🗑️  Remove existing Neo4j data volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker volume rm $(docker volume ls -q | grep neo4j) 2>/dev/null || true
    echo "✅ Old volumes removed"
fi

# Starte Neo4j mit GDS Plugin
echo "🚀 Starting Neo4j with Graph Data Science plugin..."
docker-compose up -d neo4j

# Warte auf Neo4j startup
echo "⏳ Waiting for Neo4j to start..."
sleep 10

# Prüfe ob Neo4j läuft
echo "🔍 Checking Neo4j status..."
for i in {1..30}; do
    if docker-compose exec -T neo4j cypher-shell -u neo4j -p password123 "RETURN 1" &>/dev/null; then
        echo "✅ Neo4j is running!"
        break
    fi
    echo "   Attempt $i/30: Neo4j not ready yet..."
    sleep 5
done

# Teste GDS Plugin
echo "🧪 Testing Graph Data Science plugin..."
docker-compose exec -T neo4j cypher-shell -u neo4j -p password123 "CALL gds.version()" || {
    echo "⚠️  GDS plugin test failed. Checking available plugins..."
    docker-compose exec -T neo4j cypher-shell -u neo4j -p password123 "CALL dbms.procedures() YIELD name WHERE name STARTS WITH 'gds' RETURN name LIMIT 5"
}

# Zeige verfügbare Plugins
echo "📋 Available plugins:"
docker-compose exec -T neo4j cypher-shell -u neo4j -p password123 "CALL dbms.procedures() YIELD name WHERE name CONTAINS '.' RETURN DISTINCT split(name, '.')[0] AS plugin ORDER BY plugin"

echo "🎉 Setup completed!"
echo "📊 Neo4j Browser: http://localhost:7474"
echo "🔌 Bolt connection: bolt://localhost:7687"
echo "🔑 Credentials: neo4j/password123"

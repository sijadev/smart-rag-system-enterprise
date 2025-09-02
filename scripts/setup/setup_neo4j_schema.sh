#!/bin/bash
# Neo4j RAG Schema Setup Script

echo "🚀 Erstelle RAG Schema für Smart RAG System..."

# Erstelle Constraints für eindeutige Identifikatoren
docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
"

# Erstelle Indizes für bessere Performance
docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
CREATE INDEX document_source IF NOT EXISTS FOR (d:Document) ON (d.source);
"

# Erstelle Beispiel-Entitäten für AI/ML Bereich
docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MERGE (ai:Entity {name: 'Artificial Intelligence', type: 'Concept'})
SET ai.content = 'AI is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans',
    ai.created = datetime(),
    ai.updated = datetime();
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MERGE (ml:Entity {name: 'Machine Learning', type: 'Technique'})
SET ml.content = 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed',
    ml.created = datetime(),
    ml.updated = datetime();
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MERGE (nn:Entity {name: 'Neural Networks', type: 'Technology'})
SET nn.content = 'Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information',
    nn.created = datetime(),
    nn.updated = datetime();
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MERGE (dl:Entity {name: 'Deep Learning', type: 'Technique'})
SET dl.content = 'Deep learning uses multiple layers of neural networks to model and understand complex patterns in data',
    dl.created = datetime(),
    dl.updated = datetime();
"

# Erstelle Beziehungen zwischen Entitäten
docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MATCH (ai:Entity {name: 'Artificial Intelligence'})
MATCH (ml:Entity {name: 'Machine Learning'})
MERGE (ai)-[r:CONTAINS]->(ml)
SET r.strength = 0.9, r.type = 'subset_of';
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MATCH (ml:Entity {name: 'Machine Learning'})
MATCH (nn:Entity {name: 'Neural Networks'})
MERGE (ml)-[r:USES]->(nn)
SET r.strength = 0.8, r.type = 'employs';
"

docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MATCH (nn:Entity {name: 'Neural Networks'})
MATCH (dl:Entity {name: 'Deep Learning'})
MERGE (dl)-[r:BASED_ON]->(nn)
SET r.strength = 0.9, r.type = 'utilizes';
"

echo "✅ RAG Schema erfolgreich erstellt!"
echo "📊 Teste Schema mit Beispiel-Query..."

# Teste das Schema
docker exec smart_rag_neo4j cypher-shell -u neo4j -p neo4j123 "
MATCH (e:Entity)
RETURN e.name as name, e.type as type, e.content as content
ORDER BY e.name;
"

echo "🎉 Neo4j ist bereit für das Smart RAG System!"
echo ""
echo "📝 Verbindungsdetails:"
echo "   URL: bolt://localhost:7687"
echo "   Username: neo4j"
echo "   Password: neo4j123"
echo "   Browser: http://localhost:7474"

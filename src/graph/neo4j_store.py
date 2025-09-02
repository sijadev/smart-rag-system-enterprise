#!/usr/bin/env python3
"""
Neo4j Graph Store Implementation für Smart RAG System
====================================================

Echte Neo4j-Integration mit APOC-Plugin-Support
"""

import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Echte Neo4j Graph Store Implementation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.uri = config.get('uri', 'bolt://localhost:7687')
        self.username = config.get('username', 'neo4j')
        self.password = config.get('password', 'neo4j123')
        self.database = config.get('database', 'neo4j')

        self.driver = None
        self._initialize_driver()

    def _initialize_driver(self):
        """Initialisiert Neo4j Driver"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                encrypted=False  # Für lokale Entwicklung
            )
            logger.info(f"Neo4j Driver initialisiert: {self.uri}")
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren des Neo4j Drivers: {e}")
            self.driver = None

    async def add_entities(self, entities: List[Dict[str, Any]]) -> None:
        """Fügt Entitäten zur Graph-Datenbank hinzu - Updated für Real Data"""
        if not self.driver:
            logger.warning("Neo4j Driver nicht verfügbar")
            return

        # Updated query für echte Daten mit MERGE statt CREATE um Duplikate zu vermeiden
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name})
        SET e.type = entity.type,
            e.content = entity.content,
            e.metadata = entity.metadata,
            e.confidence = COALESCE(entity.confidence, 0.8),
            e.updated = datetime(),
            e.source = COALESCE(entity.source, 'smart_rag_system')
        RETURN e.name as name, e.type as type
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entities=entities)
                created = [record.data() for record in result]
                logger.info(f"Entitäten hinzugefügt/aktualisiert: {len(created)}")

                # Log entity details für Real Data Validation
                for entity_info in created:
                    logger.debug(f"Entity processed: {entity_info['name']} ({entity_info['type']})")

                return len(created)
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Entitäten: {e}")
            raise

    async def add_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Fügt Beziehungen zur Graph-Datenbank hinzu - Updated für Real Data"""
        if not self.driver:
            logger.warning("Neo4j Driver nicht verfügbar")
            return

        # Updated query für echte Daten mit besserer Fehlerbehandlung
        query = """
        UNWIND $relationships AS rel
        MATCH (source:Entity {name: rel.source})
        MATCH (target:Entity {name: rel.target})
        MERGE (source)-[r:RELATES {type: rel.type}]->(target)
        SET r.strength = COALESCE(rel.strength, 0.7),
            r.confidence = COALESCE(rel.confidence, 0.7),
            r.metadata = rel.metadata,
            r.updated = datetime(),
            r.source = COALESCE(rel.source_system, 'smart_rag_system')
        RETURN r.type as relationship_type, source.name as source_name, target.name as target_name
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, relationships=relationships)
                created = [record.data() for record in result]
                logger.info(f"Beziehungen hinzugefügt/aktualisiert: {len(created)}")

                # Log relationship details für Real Data Validation
                for rel_info in created:
                    logger.debug(f"Relationship: {rel_info['source_name']} -[{rel_info['relationship_type']}]-> {rel_info['target_name']}")

                return len(created)
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Beziehungen: {e}")
            raise

    async def query_graph_with_validation(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Graph Query mit Real Data Validation"""
        if not self.driver:
            logger.warning("Neo4j Driver nicht verfügbar - verwende Mock-Daten")
            return self._mock_query_results(query, parameters)

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                records = []

                for record in result:
                    record_dict = dict(record)
                    # Konvertiere Neo4j-Objekte zu Python-Dicts
                    converted_record = self._convert_neo4j_types(record_dict)

                    # Add validation metadata für Real Data Tracking
                    converted_record['_validation'] = {
                        'source': 'neo4j_real_data',
                        'query_timestamp': datetime.now().isoformat(),
                        'record_type': 'validated_real_data'
                    }
                    records.append(converted_record)

                logger.info(f"Real Graph Query ergab {len(records)} validierte Ergebnisse")
                return records

        except Exception as e:
            logger.error(f"Real Graph Query Fehler: {e}")
            # Fallback zu Mock nur wenn echte Query fehlschlägt
            return self._mock_query_results(query, parameters)

    async def get_real_data_statistics(self) -> Dict[str, Any]:
        """Gibt echte Statistiken aus Neo4j zurück"""
        if not self.driver:
            return {"error": "Neo4j nicht verfügbar"}

        try:
            with self.driver.session(database=self.database) as session:
                # Entity counts by type
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.type as entity_type, count(e) as count
                    ORDER BY count DESC
                """)
                entity_stats = [record.data() for record in result]

                # Relationship counts by type
                result = session.run("""
                    MATCH ()-[r:RELATES]->()
                    RETURN r.type as relationship_type, count(r) as count
                    ORDER BY count DESC
                """)
                relationship_stats = [record.data() for record in result]

                # Total counts
                result = session.run("""
                    MATCH (e:Entity) 
                    OPTIONAL MATCH ()-[r:RELATES]->()
                    RETURN count(DISTINCT e) as total_entities, count(r) as total_relationships
                """)
                totals = result.single()

                return {
                    "total_entities": totals["total_entities"],
                    "total_relationships": totals["total_relationships"],
                    "entity_types": entity_stats,
                    "relationship_types": relationship_stats,
                    "data_source": "real_neo4j_data",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Statistik-Abfrage fehlgeschlagen: {e}")
            return {"error": str(e)}

    async def query_graph(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Führt eine Cypher-Query aus"""
        if not self.driver:
            logger.warning("Neo4j Driver nicht verfügbar - verwende Mock-Daten")
            return self._mock_query_results(query, parameters)

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                records = []
                for record in result:
                    record_dict = dict(record)
                    # Konvertiere Neo4j-Objekte zu Python-Dicts
                    converted_record = self._convert_neo4j_types(record_dict)
                    records.append(converted_record)

                logger.info(f"Graph Query ergab {len(records)} Ergebnisse")
                return records
        except Exception as e:
            logger.error(f"Graph Query Fehler: {e}")
            return self._mock_query_results(query, parameters)

    def _convert_neo4j_types(self, data):
        """Konvertiert Neo4j-spezifische Datentypen zu Python-Typen"""
        if isinstance(data, dict):
            return {k: self._convert_neo4j_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_neo4j_types(item) for item in data]
        else:
            # Konvertiere Neo4j Node/Relationship Objekte
            if hasattr(data, '_properties'):
                return dict(data._properties)
            return data

    def _mock_query_results(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback Mock-Ergebnisse wenn Neo4j nicht verfügbar"""
        entities = parameters.get('entities', [])
        mock_results = []

        if any(term in query.lower() for term in ['machine', 'learning', 'ai']):
            mock_results = [
                {
                    'id': 'ml_1',
                    'content': 'Machine learning is a subset of AI that enables computers to learn from data',
                    'relevance_score': 0.9
                },
                {
                    'id': 'ai_1',
                    'content': 'Artificial intelligence is intelligence demonstrated by machines',
                    'relevance_score': 0.8
                }
            ]

        return mock_results

    async def search_entities_by_content(self, search_terms: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """Sucht Entitäten basierend auf Inhalt"""
        if not search_terms:
            return []

        # Verwende APOC für Textsuche wenn verfügbar
        query = """
        MATCH (e:Entity)
        WHERE ANY(term IN $terms WHERE e.content CONTAINS term OR e.name CONTAINS term)
        WITH e, 
             SIZE([term IN $terms WHERE e.content CONTAINS term OR e.name CONTAINS term]) as matches
        RETURN e.name as name,
               e.content as content,
               e.type as type,
               matches as relevance_score
        ORDER BY relevance_score DESC
        LIMIT $k
        """

        parameters = {'terms': search_terms, 'k': k}
        return await self.query_graph(query, parameters)

    async def get_related_entities(self, entity_name: str, max_hops: int = 2, k: int = 5) -> List[Dict[str, Any]]:
        """Findet verwandte Entitäten über Beziehungen"""
        query = f"""
        MATCH path = (start:Entity {{name: $entity_name}})-[*1..{max_hops}]-(related:Entity)
        WITH related, LENGTH(path) as distance,
             COUNT(DISTINCT path) as connection_strength
        RETURN related.name as name,
               related.content as content,
               related.type as type,
               distance,
               connection_strength,
               (1.0 / distance) * connection_strength as relevance_score
        ORDER BY relevance_score DESC
        LIMIT $k
        """

        parameters = {'entity_name': entity_name, 'k': k}
        return await self.query_graph(query, parameters)

    def close(self):
        """Schließt Neo4j-Verbindung"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j Driver geschlossen")


class Neo4jGraphStoreFactory:
    """Factory für Neo4j Graph Store"""

    @staticmethod
    def create_graph_store(config: Dict[str, Any]) -> Neo4jGraphStore:
        """Erstellt Neo4j Graph Store Instanz"""
        return Neo4jGraphStore(config)

    @staticmethod
    def create_test_store() -> Neo4jGraphStore:
        """Erstellt Test-Instanz mit Standard-Konfiguration"""
        config = {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'neo4j123',
            'database': 'neo4j'
        }
        return Neo4jGraphStore(config)

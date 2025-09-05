#!/usr/bin/env python3
"""
Neo4j Schema Setup für RAG System
================================

Erstellt die erwartete Datenbankstruktur für optimale Performance
"""

import asyncio
from typing import Any, Dict, List

from neo4j import GraphDatabase


class Neo4jSchemaSetup:
    """Setup für Neo4j RAG Schema"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    async def setup_complete_schema(self) -> Dict[str, bool]:
        """Erstellt komplette Schema-Struktur für RAG System"""

        results = {}

        # 1. Constraints erstellen
        constraints = [
            "CREATE CONSTRAINT document_name_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            "CREATE CONSTRAINT keyword_term_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.term IS UNIQUE",
        ]

        results["constraints"] = await self._execute_queries(constraints, "Constraints")

        # 2. Indizes erstellen
        indexes = [
            "CREATE INDEX document_created_idx IF NOT EXISTS FOR (d:Document) ON (d.created)",
            "CREATE INDEX concept_domain_idx IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX chunk_embedding_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.embedding_model)",
            "CREATE TEXT INDEX document_content_text_idx IF NOT EXISTS FOR (d:Document) ON (d.content)",
            "CREATE TEXT INDEX chunk_content_text_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.content)",
        ]

        results["indexes"] = await self._execute_queries(indexes, "Indexes")

        # 3. Beispiel-Daten erstellen für Tests
        sample_data = [
            """
            CREATE (doc:Document {
                name: 'sample_document.txt',
                content: 'This is a sample document about machine learning and neural networks.',
                created: datetime(),
                file_path: '/data/documents/sample_document.txt'
            })
            """,
            """
            CREATE (chunk:Chunk {
                content: 'Machine learning is a subset of artificial intelligence.',
                chunk_id: 'chunk_001',
                embedding_model: 'nomic-embed-text',
                created: datetime()
            })
            """,
            """
            CREATE (concept_ml:Concept {
                name: 'Machine Learning',
                description: 'A method of data analysis that automates analytical model building.',
                domain: 'technology',
                created: datetime()
            })
            """,
            """
            CREATE (concept_ai:Concept {
                name: 'Artificial Intelligence',
                description: 'Intelligence demonstrated by machines.',
                domain: 'technology',
                created: datetime()
            })
            """,
            """
            CREATE (keyword_ml:Keyword {
                term: 'machine learning',
                frequency: 5,
                importance: 0.8
            })
            """,
            """
            CREATE (entity:Entity {
                name: 'Neural Networks',
                type: 'TECHNOLOGY',
                description: 'Computing systems inspired by biological neural networks'
            })
            """,
        ]

        results["sample_data"] = await self._execute_queries(sample_data, "Sample Data")

        # 4. Beziehungen erstellen
        relationships = [
            """
            MATCH (chunk:Chunk), (keyword:Keyword {term: 'machine learning'})
            CREATE (chunk)-[:CONTAINS_KEYWORD]->(keyword)
            """,
            """
            MATCH (concept1:Concept {name: 'Machine Learning'}), (concept2:Concept {name: 'Artificial Intelligence'})
            CREATE (concept1)-[:RELATED_TO {strength: 0.9, reason: 'subset_relationship'}]->(concept2)
            """,
            """
            MATCH (doc:Document), (chunk:Chunk)
            CREATE (doc)-[:CONTAINS_CHUNK]->(chunk)
            """,
            """
            MATCH (chunk:Chunk), (entity:Entity {name: 'Neural Networks'})
            CREATE (chunk)-[:MENTIONS_ENTITY]->(entity)
            """,
        ]

        results["relationships"] = await self._execute_queries(
            relationships, "Relationships"
        )

        return results

    async def _execute_queries(self, queries: List[str], category: str) -> bool:
        """Führt eine Liste von Queries aus"""

        print(f"📊 Creating {category}...")
        success_count = 0

        try:
            with self.driver.session() as session:
                for query in queries:
                    try:
                        session.run(query)
                        success_count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to execute query: {str(e)[:100]}...")

            print(f"✅ {category}: {success_count}/{len(queries)} successful")
            return success_count == len(queries)

        except Exception as e:
            print(f"❌ {category} setup failed: {e}")
            return False

    async def verify_schema(self) -> Dict[str, Any]:
        """Überprüft die erstellte Schema-Struktur"""

        verification = {
            "labels": [],
            "relationships": [],
            "constraints": [],
            "indexes": [],
            "sample_counts": {},
        }

        try:
            with self.driver.session() as session:
                # Labels
                labels_result = session.run("CALL db.labels() YIELD label RETURN label")
                verification["labels"] = [record["label"] for record in labels_result]

                # Relationship types
                rel_result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                verification["relationships"] = [
                    record["relationshipType"] for record in rel_result
                ]

                # Constraints
                constraints_result = session.run("SHOW CONSTRAINTS")
                verification["constraints"] = [
                    dict(record) for record in constraints_result
                ]

                # Indexes
                indexes_result = session.run("SHOW INDEXES")
                verification["indexes"] = [dict(record) for record in indexes_result]

                # Sample counts
                for label in verification["labels"]:
                    count_result = session.run(
                        f"MATCH (n:{label}) RETURN count(n) as count"
                    )
                    verification["sample_counts"][label] = count_result.single()[
                        "count"
                    ]

        except Exception as e:
            verification["error"] = str(e)

        return verification

    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.driver:
            self.driver.close()


async def main():
    """Hauptfunktion zum Schema Setup"""

    print("🚀 Setting up Neo4j Schema for RAG System...")

    setup = Neo4jSchemaSetup()

    try:
        # Schema erstellen
        results = await setup.setup_complete_schema()

        # Erfolg anzeigen
        total_success = sum(results.values())
        print("\n📊 Schema Setup Results:")
        print(f"✅ Successful categories: {total_success}/{len(results)}")

        for category, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {category.title()}")

        # Verification
        print("\n🔍 Verifying schema...")
        verification = await setup.verify_schema()

        print(f"📋 Created Labels: {', '.join(verification['labels'])}")
        print(f"🔗 Created Relationships: {', '.join(verification['relationships'])}")
        print(f"📊 Sample Data Counts: {verification['sample_counts']}")

    except Exception as e:
        print(f"❌ Schema setup failed: {e}")

    finally:
        setup.close()


if __name__ == "__main__":
    asyncio.run(main())

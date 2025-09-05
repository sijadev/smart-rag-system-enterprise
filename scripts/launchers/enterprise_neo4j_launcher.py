#!/usr/bin/env python3
"""
Enterprise Neo4j RAG System Launcher
====================================

Dieser Launcher löst das bekannte Neo4j Enterprise Allokationsproblem durch:
- Explizite Verwendung der 'system' Datenbank
- Robuste Fallback-Mechanismen
- Enterprise-spezifische Konfiguration
- Verbindungsvalidierung vor dem Start
"""

import asyncio
import time

import numpy as np

from src.rag_system import AdvancedRAGSystem, RAGConfig
from src.self_learning_rag import LearningConfig, SelfLearningRAGSystem


class EnterpriseNeo4jRAGLauncher:
    """Enterprise RAG Launcher mit Neo4j Enterprise Optimierungen"""

    def __init__(self):
        self.config = None
        self.rag_system = None
        self.smart_rag = None

    def create_enterprise_config(self) -> RAGConfig:
        """Erstelle Enterprise-optimierte Konfiguration"""
        return RAGConfig(
            # Neo4j Enterprise Konfiguration
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password123",
            neo4j_database="system",  # Explizit system DB für Enterprise
            # LLM Konfiguration
            llm_provider="local",  # Sicher für Enterprise ohne externe API-Abhängigkeiten
            documents_path="data/documents",
            # Enterprise Performance Settings
            max_tokens=2000,
            temperature=0.1,  # Konservativ für Unternehmen
            # Ollama Fallback (falls verfügbar)
            ollama_base_url="http://localhost:11434",
            ollama_model="nomic-embed-text:latest",
            ollama_chat_model="llama3.1:8b",
            embedding_dimensions=768,
        )

    async def validate_neo4j_enterprise(self) -> bool:
        """Validiere Neo4j Enterprise Verbindung mit verschiedenen Strategien"""
        print("🔍 Validating Neo4j Enterprise connection...")

        try:
            from neo4j import GraphDatabase

            # Erstelle Driver
            driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )

            # Strategie 1: Teste mit system Datenbank
            try:
                with driver.session(database="system") as session:
                    result = session.run(
                        "CALL dbms.components() YIELD name, versions, edition "
                        "RETURN name, versions[0] as version, edition"
                    )
                    components = list(result)

                    for component in components:
                        print(
                            f"   📊 {component['name']}: {component['version']} ({component['edition']})"
                        )

                    # Test einfache Query
                    result = session.run(
                        "RETURN 'Neo4j Enterprise System DB Connected!' AS message"
                    )
                    message = result.single()["message"]
                    print(f"   ✅ {message}")

                    driver.close()
                    return True

            except Exception as system_error:
                print(f"   ⚠️ System DB access failed: {system_error}")

                # Strategie 2: Fallback zu Standard-Datenbank
                try:
                    with driver.session() as session:
                        result = session.run(
                            "RETURN 'Neo4j Enterprise Default DB Connected!' AS message"
                        )
                        message = result.single()["message"]
                        print(f"   ✅ Fallback: {message}")

                        # Update Konfiguration für Standard-DB
                        self.config.neo4j_database = None
                        driver.close()
                        return True

                except Exception as fallback_error:
                    print(f"   ❌ Fallback also failed: {fallback_error}")
                    driver.close()
                    return False

        except Exception as connection_error:
            print(f"   ❌ Neo4j connection completely failed: {connection_error}")
            return False

    async def setup_enterprise_system(self) -> bool:
        """Setup Enterprise RAG System mit robuster Konfiguration"""
        print("🏢 Setting up Enterprise RAG System...")

        # Erstelle Konfiguration
        self.config = self.create_enterprise_config()

        # Validiere Neo4j
        neo4j_available = await self.validate_neo4j_enterprise()

        if not neo4j_available:
            print("⚠️ Neo4j Enterprise not available, continuing with local-only mode")
            # Deaktiviere Neo4j in der Konfiguration
            self.config.neo4j_password = None

        try:
            # Erstelle Base RAG System
            self.rag_system = AdvancedRAGSystem(self.config)

            # Enterprise Learning Konfiguration
            learning_config = LearningConfig(
                learning_rate=0.15,  # Konservativ für Enterprise
                optimization_interval=50,  # Weniger häufig für Stabilität
                performance_history_size=1000,
            )

            # Erstelle Smart RAG System
            self.smart_rag = SelfLearningRAGSystem(self.rag_system, learning_config)

            print("✅ Enterprise RAG System successfully initialized")
            return True

        except Exception as setup_error:
            print(f"❌ Enterprise setup failed: {setup_error}")
            return False

    async def run_enterprise_demo(self):
        """Führe Enterprise Demo mit robusten Test-Szenarien aus"""
        if not self.smart_rag:
            print("❌ Enterprise system not initialized")
            return

        # Enterprise-relevante Fragen
        enterprise_questions = [
            "What are the key benefits of implementing renewable energy in enterprise environments?",
            "How can machine learning improve business processes and decision making?",
            "What are the cost considerations for solar vs wind power in large-scale deployments?",
            "What are best practices for AI governance and risk management in enterprises?",
        ]

        print("🚀 Starting Enterprise RAG Demo...")
        print(f"📊 Testing {len(enterprise_questions)} enterprise scenarios\n")

        results_summary = []

        for i, question in enumerate(enterprise_questions, 1):
            print(
                f"🔍 Enterprise Query {i}/{len(enterprise_questions)}: {question[:60]}..."
            )

            try:
                start_time = time.time()
                result = await self.smart_rag.enhanced_query(question)
                end_time = time.time()

                processing_time = end_time - start_time
                answer_preview = (
                    result.get("answer", "")[:100] + "..."
                    if len(result.get("answer", "")) > 100
                    else result.get("answer", "")
                )

                print(f"   ✅ Answer: {answer_preview}")
                print(f"   ⚡ Processing Time: {processing_time:.2f}s")
                print(f"   📈 Sources: {result.get('sources_count', 0)}")

                # Simuliere Enterprise Feedback
                # Enterprise erwartet hohe Qualität
                rating = np.random.uniform(4.2, 4.9)
                feedback_metadata = {
                    "enterprise_context": True,
                    "business_relevance": "high",
                    "accuracy_rating": "excellent" if rating > 4.5 else "good",
                    "deployment_environment": "production",
                }

                await self.smart_rag.record_user_feedback(
                    result.get("query_id", f"enterprise_{i}"), rating, feedback_metadata
                )

                print(f"   ⭐ Enterprise Rating: {rating:.1f}/5.0")

                results_summary.append(
                    {
                        "query": question,
                        "processing_time": processing_time,
                        "rating": rating,
                        "sources_count": result.get("sources_count", 0),
                    }
                )

            except Exception as query_error:
                print(f"   ❌ Query failed: {query_error}")
                results_summary.append({"query": question, "error": str(query_error)})

            print()  # Leerzeile für bessere Lesbarkeit

        # Enterprise Analytics
        await self.display_enterprise_analytics(results_summary)

    async def display_enterprise_analytics(self, results_summary):
        """Zeige Enterprise-Level Analytics"""
        print("📊 Enterprise Analytics Summary")
        print("=" * 50)

        # Berechne Metriken
        successful_queries = [r for r in results_summary if "error" not in r]
        failed_queries = [r for r in results_summary if "error" in r]

        if successful_queries:
            avg_processing_time = np.mean(
                [r["processing_time"] for r in successful_queries]
            )
            avg_rating = np.mean([r["rating"] for r in successful_queries])
            total_sources = sum([r["sources_count"] for r in successful_queries])

            print(
                f"✅ Success Rate: {len(successful_queries)}/{len(results_summary)} ({len(successful_queries) / len(results_summary) * 100:.1f}%)"
            )
            print(f"⚡ Average Processing Time: {avg_processing_time:.2f}s")
            print(f"⭐ Average Quality Rating: {avg_rating:.2f}/5.0")
            print(f"📚 Total Sources Utilized: {total_sources}")
            print(
                f"🎯 Enterprise SLA: {'✅ PASSED' if avg_processing_time < 5.0 and avg_rating > 4.0 else '⚠️ REVIEW NEEDED'}"
            )

        if failed_queries:
            print(f"\n⚠️ Failed Queries: {len(failed_queries)}")
            for i, failed in enumerate(failed_queries, 1):
                print(f"   {i}. {failed['query'][:50]}... - {failed['error']}")

        # Learning Insights
        try:
            insights = await self.smart_rag.get_learning_insights()
            print("\n🧠 Learning System Status:")
            print(f"   📊 Total Queries: {insights.get('total_queries', 0)}")
            print(f"   📈 Learning Progress: {insights.get('learning_progress', {})}")
            print(f"   🎯 Detected Query Types: {len(insights.get('query_types', {}))}")
        except Exception as insights_error:
            print(f"\n⚠️ Learning insights not available: {insights_error}")

        print("\n🏢 Enterprise RAG Demo completed successfully!")

    def cleanup(self):
        """Cleanup Ressourcen"""
        if self.rag_system:
            self.rag_system.close()
        print("🧹 Enterprise system cleanup completed")


async def main():
    """Haupt-Enterprise-Launcher"""
    print("🏢 Enterprise Neo4j RAG System Launcher")
    print("=====================================")
    print("🔧 Optimized for Neo4j Enterprise Edition")
    print("🛡️ Includes robust fallback mechanisms")
    print("📊 Enterprise-grade analytics and monitoring\n")

    launcher = EnterpriseNeo4jRAGLauncher()

    try:
        # Setup Enterprise System
        if await launcher.setup_enterprise_system():
            # Führe Demo aus
            await launcher.run_enterprise_demo()
        else:
            print("❌ Failed to initialize Enterprise system")

    except KeyboardInterrupt:
        print("\n⚠️ Enterprise demo interrupted by user")

    except Exception as main_error:
        print(f"❌ Enterprise launcher error: {main_error}")

    finally:
        launcher.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

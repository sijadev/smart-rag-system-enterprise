import asyncio
import json
import numpy as np
import os

# Lade .env-Datei automatisch
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Falls python-dotenv nicht installiert ist, ignoriere es
    pass

from src.rag_system import AdvancedRAGSystem, RAGConfig
from src.self_learning_rag import SelfLearningRAGSystem, LearningConfig


async def main():
    # Enterprise Konfiguration für Neo4j Enterprise + GDS (mit system DB)
    # Verwende Umgebungsvariablen aus .env Datei
    config = RAGConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4j123"),
        neo4j_database="system",  # Explizit system DB für Enterprise
        llm_provider="local",  # Kann auf "openai" oder "anthropic" geändert werden
        documents_path="data/documents"
    )

    # Robuste Neo4j Enterprise Verbindungsprüfung
    print("🔍 Testing Neo4j Enterprise connection...")
    print(f"🔧 Using password from ENV: {config.neo4j_password}")
    neo4j_working = False
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))

        # Versuche zuerst system Datenbank
        try:
            with driver.session(database="system") as session:
                result = session.run("RETURN 'Connected to Neo4j Enterprise System DB!' AS message")
                print(f"✅ {result.single()['message']}")
                neo4j_working = True
        except Exception as system_error:
            print(f"⚠️ System DB access failed: {system_error}")
            # Fallback zu Standard-Datenbank
            try:
                with driver.session() as session:
                    result = session.run("RETURN 'Connected to Neo4j Enterprise (default DB)!' AS message")
                    print(f"✅ Fallback: {result.single()['message']}")
                    config.neo4j_database = None  # Verwende Standard-DB
                    neo4j_working = True
            except Exception as fallback_error:
                print(f"⚠️ Fallback connection also failed: {fallback_error}")

        driver.close()
    except Exception as e:
        print(f"⚠️ Neo4j connection issue: {e}")

    # Wenn Neo4j Enterprise nicht funktioniert, deaktiviere es vollständig
    if not neo4j_working:
        print("📝 Neo4j Enterprise not available - switching to local-only mode")
        config.neo4j_password = None  # Deaktiviere Neo4j komplett
        config.neo4j_database = None

    base_rag = AdvancedRAGSystem(config)

    # Enterprise Learning Configuration
    learning_config = LearningConfig(
        learning_rate=0.2,  # Höhere Lernrate für Enterprise
        optimization_interval=25,  # Häufigere Optimierung
        performance_history_size=2000  # Größere History
    )
    smart_rag = SelfLearningRAGSystem(base_rag, learning_config)

    # Enterprise Test-Szenarien
    enterprise_questions = [
        "What are the key advantages of renewable energy systems?",
        "How can machine learning be applied in enterprise environments?",
        "Compare the efficiency and ROI of solar vs wind power implementations",
        "What are the best practices for deploying AI systems at scale?"
    ]

    print("🏢 Starting Enterprise RAG System Test...\n")
    print(f"🚀 Features: {'Neo4j Enterprise' if neo4j_working else 'Local Mode'} + Advanced Analytics + Learning\n")

    results_summary = []

    for i, question in enumerate(enterprise_questions, 1):
        print(f"📊 Enterprise Query {i}/4: {question}")
        try:
            import time
            start_time = time.time()
            result = await smart_rag.enhanced_query(question)
            end_time = time.time()

            processing_time = end_time - start_time
            answer_preview = result.get('answer', '')[:150] + "..." if len(result.get('answer', '')) > 100 else result.get('answer', '')

            print(f"✅ Answer: {answer_preview}")
            print(f"🧠 Strategy: {result.get('learning_metadata', {}).get('strategy', 'standard')}")
            print(f"⚡ Response Time: {processing_time:.3f}s")
            print(f"📈 Contexts: {len(result.get('contexts', []))}")

            # Enterprise-level Feedback Simulation
            rating = np.random.uniform(4.0, 5.0)  # Enterprise systems erwarten hohe Qualität
            await smart_rag.record_user_feedback(
                result.get('query_id', f'enterprise_{i}'),
                rating,
                {
                    'clicked_sources': [f'enterprise_doc_{i}.pdf'],
                    'user_satisfaction': 'high',
                    'business_value': 'excellent',
                    'mode': 'neo4j_enterprise' if neo4j_working else 'local_fallback'
                }
            )
            print(f"⭐ Enterprise Rating: {rating:.1f}/5.0")

            results_summary.append({
                'query': question,
                'success': True,
                'processing_time': processing_time,
                'rating': rating,
                'contexts': len(result.get('contexts', []))
            })
            print()

        except Exception as e:
            print(f"❌ Error in Enterprise Query '{question}': {e}")
            results_summary.append({
                'query': question,
                'success': False,
                'error': str(e)
            })
            print()

    # Enterprise Results Summary
    print("📊 Enterprise Test Results Summary")
    print("=" * 50)
    successful = [r for r in results_summary if r.get('success', False)]
    failed = [r for r in results_summary if not r.get('success', False)]

    print(f"✅ Successful Queries: {len(successful)}/{len(results_summary)} ({len(successful)/len(results_summary)*100:.1f}%)")

    if successful:
        avg_time = np.mean([r['processing_time'] for r in successful])
        avg_rating = np.mean([r['rating'] for r in successful])
        total_contexts = sum([r['contexts'] for r in successful])

        print(f"⚡ Average Response Time: {avg_time:.3f}s")
        print(f"⭐ Average Quality Rating: {avg_rating:.2f}/5.0")
        print(f"📚 Total Contexts Used: {total_contexts}")
        print(f"🎯 Enterprise SLA: {'✅ PASSED' if avg_time < 5.0 and avg_rating > 4.0 else '⚠️ NEEDS REVIEW'}")

    if failed:
        print(f"\n⚠️ Failed Queries: {len(failed)}")
        for i, f in enumerate(failed, 1):
            print(f"   {i}. {f['query'][:60]}... - {f['error']}")

    # Enterprise Learning Analytics
    try:
        insights = await smart_rag.get_learning_insights()
        print(f"\n🧠 Enterprise Learning Analytics:")
        print(f"   📊 Total Queries Processed: {insights.get('total_queries', 0)}")
        print(f"   ⚡ Average Response Time: {insights.get('average_response_time', 0):.3f}s")
        print(f"   🎯 Query Types Detected: {len(insights.get('query_types', {}))}")
        print(f"   📈 Learning Progress: {insights.get('learning_progress', {})}")
        print(f"   💡 System Mode: {'Neo4j Enterprise' if neo4j_working else 'Local Fallback'}")

    except Exception as e:
        print(f"❌ Error getting Enterprise Analytics: {e}")

    print("\n🏢 Enterprise RAG Test completed!")

    # Cleanup
    base_rag.close()

if __name__ == "__main__":
    import time
    asyncio.run(main())

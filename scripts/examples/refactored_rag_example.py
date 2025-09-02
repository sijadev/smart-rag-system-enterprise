#!/usr/bin/env python3
"""
Example: Verwendung des refactored Smart RAG Systems
==================================================

Demonstriert die Verwendung aller implementierten Design Patterns
"""

import asyncio
import logging
from pathlib import Path

# Refactored System Imports
from src.rag_system import SmartRAGSystem, create_development_system, create_production_system
from src.config.builders import RAGSystemBuilder, create_development_config
from src.interfaces import LLMProvider, RetrievalStrategy

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basis-Verwendung des refactored Systems"""
    print("🚀 Example 1: Basic Usage with Builder Pattern")

    # Verwende Builder Pattern für Konfiguration
    config = (RAGSystemBuilder()
              .with_name("Example-RAG")
              .with_ollama("llama2")
              .with_hybrid_retrieval()
              .with_neo4j(password="password123")
              .with_monitoring(enabled=True)
              .with_learning(enabled=True)
              .build())

    # Erstelle System mit Dependency Injection
    rag_system = SmartRAGSystem(config)

    try:
        # System initialisieren (Factory Pattern intern)
        await rag_system.initialize()

        # Dokumente hinzufügen
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neural networks.",
            "Deep learning uses multiple layers of neural networks."
        ]

        result = await rag_system.add_documents(documents)
        print(f"✅ Added {result['added_documents']} documents")

        # Query mit Strategy Pattern
        response = await rag_system.query(
            "What is machine learning?",
            context={'session_id': 'demo_session'},
            user_id='demo_user'
        )

        print(f"🤖 Answer: {response.answer}")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"⚙️ Strategy: {response.metadata.get('strategy', 'unknown')}")

    finally:
        await rag_system.shutdown()


async def example_advanced_configuration():
    """Erweiterte Konfiguration mit verschiedenen Patterns"""
    print("\n🔧 Example 2: Advanced Configuration")

    # Verwende verschiedene vordefinierte Konfigurationen
    dev_config = create_development_config()

    # Modifiziere Konfiguration für spezielle Anforderungen
    config = (RAGSystemBuilder()
              .with_name("Advanced-RAG")
              .with_anthropic("claude-3", api_key="your-api-key")  # Würde echten API Key benötigen
              .with_semantic_search_strategy()  # Semantic Search Strategy
              .with_retrieval_k(7)
              .with_temperature(0.3)  # Niedrigere Temperature für konsistentere Antworten
              .with_vector_store("chroma", "./data/advanced_vectors")
              .with_neo4j("bolt://localhost:7687", "neo4j", "password123")
              .with_monitoring(enabled=True, retention_days=90)
              .with_learning(enabled=True, optimization_interval=50)
              .with_security(enabled=True, rate_limit=1000)
              .with_caching(enabled=True, ttl=3600)
              .build())

    system = SmartRAGSystem(config)

    try:
        await system.initialize()

        # Test verschiedene Retrieval-Strategien zur Laufzeit
        strategies_to_test = [
            RetrievalStrategy.VECTOR_ONLY,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.SEMANTIC_SEARCH
        ]

        question = "How do neural networks work?"

        for strategy in strategies_to_test:
            response = await system.query(
                question,
                strategy=strategy,
                user_id="advanced_user"
            )
            print(f"Strategy {strategy.value}: Confidence {response.confidence:.2f}")

        # System-Metriken abrufen (Observer Pattern)
        metrics = await system.get_system_metrics()
        print(f"📈 Total Queries: {metrics['system_info']['query_count']}")

    finally:
        await system.shutdown()


async def example_monitoring_and_feedback():
    """Monitoring und Feedback-System demonstrieren"""
    print("\n📊 Example 3: Monitoring and Learning")

    system = create_development_system()

    try:
        await system.initialize()

        # Mehrere Queries für Monitoring
        queries = [
            "What is artificial intelligence?",
            "Explain deep learning concepts.",
            "How do transformers work?",
            "What are the benefits of renewable energy?"
        ]

        query_responses = []

        for i, query in enumerate(queries):
            response = await system.query(
                query,
                user_id=f"user_{i}",
                context={'session_id': f'session_{i}'}
            )
            query_responses.append((response, query))
            print(f"Query {i+1}: {response.confidence:.2f} confidence")

        # Simuliere User-Feedback (Observer Pattern triggert Learning)
        for i, (response, original_query) in enumerate(query_responses):
            feedback = {
                'rating': 4.5 if i % 2 == 0 else 3.5,  # Simuliere verschiedene Ratings
                'helpful': True,
                'comment': f"Response was {'very' if i % 2 == 0 else 'somewhat'} helpful"
            }

            # Feedback wird über Observer Pattern verarbeitet
            await system.record_feedback(
                response.metadata.get('query_id', f'query_{i}'),
                feedback
            )

        # System-Optimierung triggern
        optimization_result = await system.optimize_system()
        print(f"🎯 Optimization triggered: {optimization_result}")

        # Finale Metriken
        final_metrics = await system.get_system_metrics()
        print(f"📊 Final metrics: {final_metrics['system_info']}")

        # Performance-Metriken wenn verfügbar
        if 'performance_metrics' in final_metrics:
            perf = final_metrics['performance_metrics']
            print(f"⚡ Success Rate: {perf.get('success_rate', 0):.2f}")
            print(f"⏱️ Avg Response Time: {perf.get('average_response_time', 0):.3f}s")

    finally:
        await system.shutdown()


async def example_production_setup():
    """Production-Setup demonstrieren"""
    print("\n🏭 Example 4: Production Setup")

    # Production-Konfiguration mit Enterprise-Features
    config = (RAGSystemBuilder()
              .with_name("SmartRAG-Production")
              .with_version("2.0.0")
              .with_openai("gpt-4")  # Würde echten API Key benötigen
              .with_temperature(0.7)
              .with_max_tokens(4096)
              .with_hybrid_retrieval()
              .with_retrieval_k(10)
              .with_context_length(8000)
              .with_vector_store("pinecone")  # Skalierbare Cloud Vector DB
              .with_neo4j_enterprise()  # Enterprise Neo4j Features
              .with_monitoring(enabled=True, retention_days=365)
              .with_learning(enabled=True, optimization_interval=1000)
              .with_security(enabled=True, rate_limit=10000, window=3600)
              .with_caching(enabled=True, ttl=7200)
              .with_parallel_processing(enabled=True, max_workers=16)
              .add_validation_rule(lambda config: config.max_tokens >= 2048)
              .build())

    system = SmartRAGSystem(config)

    print(f"✅ Production system configured:")
    print(f"   - LLM Provider: {config.llm_provider.value}")
    print(f"   - Retrieval Strategy: {config.default_retrieval_strategy.value}")
    print(f"   - Monitoring: {config.enable_monitoring}")
    print(f"   - Security: {config.enable_security}")
    print(f"   - Parallel Processing: {config.parallel_processing}")

    # In Production würde hier die Initialisierung und der Betrieb folgen
    print("🏭 Production system ready (initialization skipped in demo)")


async def example_custom_extensions():
    """Custom Extensions und Erweiterungen"""
    print("\n🔧 Example 5: Custom Extensions")

    # Erweiterte Konfiguration mit Custom Extensions
    config = (RAGSystemBuilder()
              .with_name("Extended-RAG")
              .with_ollama("llama2")
              .with_hybrid_retrieval()
              .add_custom_processor("DomainSpecificProcessor")
              .add_custom_strategy("MultiModalStrategy")
              .with_extension_config("custom_feature_1", {"enabled": True, "threshold": 0.8})
              .with_extension_config("domain_adaptation", {"domain": "medical", "terminology": "strict"})
              .build())

    print(f"🔧 Custom Extensions:")
    print(f"   - Custom Processors: {config.custom_processors}")
    print(f"   - Custom Strategies: {config.custom_strategies}")
    print(f"   - Extension Config: {config.extension_config}")

    # System würde Custom Extensions automatisch laden und registrieren
    print("✅ Custom extensions would be loaded automatically")


async def main():
    """Hauptfunktion für alle Examples"""
    print("🎯 Smart RAG System - Refactored with Design Patterns")
    print("=" * 60)

    try:
        await example_basic_usage()
        await example_advanced_configuration()
        await example_monitoring_and_feedback()
        await example_production_setup()
        await example_custom_extensions()

        print("\n🎉 All examples completed successfully!")
        print("\n📚 Design Patterns implemented:")
        print("   ✅ Strategy Pattern - Retrieval strategies")
        print("   ✅ Factory Pattern - Service creation")
        print("   ✅ Observer Pattern - Monitoring & events")
        print("   ✅ Chain of Responsibility - Query processing")
        print("   ✅ Builder Pattern - Configuration")
        print("   ✅ Dependency Injection - Service management")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Setze Event-Loop-Policy für Windows-Kompatibilität
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())

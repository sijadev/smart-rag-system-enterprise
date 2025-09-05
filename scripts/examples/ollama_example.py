#!/usr/bin/env python3
"""
Beispiel für die Verwendung der Ollama RAG Integration
Demonstriert die Nutzung des nomic-embed-text-v1.5 Modells
"""

import asyncio

from src.rag_system import OllamaRAGSystem, RAGConfig, setup_ollama_models


async def main():
    print("🦙 Ollama RAG System Demo")
    print("=" * 50)

    # 1. Setup benötigte Modelle
    print("🔄 Setting up required Ollama models...")
    success = await setup_ollama_models()
    if not success:
        print("❌ Failed to setup Ollama models. Please install Ollama first:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        return

    # 2. Konfiguration erstellen
    config = RAGConfig(
        llm_provider="ollama",
        ollama_model="nomic-embed-text-v1.5",  # Embedding-Modell
        ollama_chat_model="llama3.1:8b",  # Chat-Modell
        embedding_dimensions=768,
        temperature=0.1,
        max_tokens=1000,
    )

    # 3. RAG System initialisieren
    print("\n🚀 Initializing Ollama RAG System...")
    rag_system = OllamaRAGSystem(config)

    # 4. Gesundheitscheck
    print("\n🔍 Performing health check...")
    health = await rag_system.health_check()
    for key, value in health.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}: {value}")

    # 5. Modell-Informationen anzeigen
    print("\n📊 Model Information:")
    model_info = await rag_system.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")

    # 6. Warte auf Indexierung (asynchron gestartet)
    print("\n⏳ Waiting for document indexing to complete...")
    await asyncio.sleep(5)  # Gib dem System Zeit zu indexieren

    # 7. Beispiel-Dokumente hinzufügen
    print("\n📄 Adding example documents...")
    example_docs = [
        """
        Python ist eine interpretierte, objektorientierte, höhere Programmiersprache mit dynamischer Semantik.
        Sie wurde 1991 von Guido van Rossum entwickelt und zeichnet sich durch ihre einfache, gut lesbare Syntax aus.
        Python unterstützt verschiedene Programmierparadigmen wie objektorientierte, imperative und funktionale Programmierung.
        """,
        """
        Machine Learning (ML) ist ein Teilbereich der Künstlichen Intelligenz (KI), der es Computersystemen ermöglicht,
        automatisch zu lernen und sich zu verbessern, ohne explizit programmiert zu werden.
        ML-Algorithmen erstellen mathematische Modelle basierend auf Trainingsdaten, um Vorhersagen oder
        Entscheidungen zu treffen, ohne für spezifische Aufgaben programmiert zu werden.
        """,
        """
        Retrieval-Augmented Generation (RAG) ist eine Technik in der natürlichen Sprachverarbeitung,
        die die Vorteile von vortrainierten Sprachmodellen mit externen Wissensdatenbanken kombiniert.
        RAG-Systeme rufen relevante Informationen aus einer Datenbank ab und verwenden diese als Kontext
        für die Generierung präziser und faktisch korrekter Antworten.
        """,
    ]

    for i, doc in enumerate(example_docs):
        success = await rag_system.add_document(doc, {"source": f"example_{i + 1}"})
        if success:
            print(f"   ✅ Added document {i + 1}")

    # 8. Beispiel-Queries ausführen
    print("\n❓ Example Queries:")
    print("-" * 30)

    queries = [
        "Was ist Python?",
        "Erkläre Machine Learning",
        "Wie funktioniert RAG?",
        "What are the main features of Python?",
        "Welche Programmierparadigmen unterstützt Python?",
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = await rag_system.query(query, k=3)
            print(f"📝 Answer: {result['answer']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"🔗 Sources: {result['sources_count']}")
            print(f"🤖 Model: {result['model']}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 50)

    # 9. Performance-Test
    print("\n⚡ Performance Test:")
    import time

    start_time = time.time()
    batch_queries = ["Was ist Python?"] * 5

    for i, query in enumerate(batch_queries):
        result = await rag_system.query(query, k=2)
        print(f"   Query {i + 1}: {result['processing_time']:.2f}s")

    total_time = time.time() - start_time
    avg_time = total_time / len(batch_queries)
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per query: {avg_time:.2f}s")

    # 10. Cleanup
    print("\n🧹 Cleaning up...")
    rag_system.close()
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

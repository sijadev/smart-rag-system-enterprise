# Implementierungsstrategie: Interfaces nutzen

Ziel: Klarer, schrittweiser Plan zur vollständigen Nutzung der bestehenden Interfaces (src/interfaces.py) in der Codebasis. Ergebnis: ausführliche Dokumentation + PlantUML-Diagramm (docs/uml/implementation_strategy.puml).

1) Überblick
- Nutze die definierten Interfaces (ILLMService, IVectorStore, IGraphStore, IRetrievalStrategy, IQueryProcessor, IObserver, IMetricsCollector, ILearningSystem) als Contracts.
- Implementierungen (Produktiv + Mocks) müssen diese Contracts erfüllen.
- DIContainer, Factories und Builder orchestrieren Erzeugung und Laufzeit-Wiring.

2) Strategie (Aufgaben & Priorisierung)

A — Kern-Implementierungen (High Priority)
- LLM-Adapter: implementiere konkrete Klassen für OLLAMA, OpenAI, Anthropic, die ILLMService erfüllen.
  - Methoden: async generate(prompt, context), async embed(text), get_provider_info().
  - Fehlerbehandlung, Timeouts, Retry-Policy.

- Vector-Store-Adapter: implementiere/integriere Chromadb/FAISS Adapter, die IVectorStore erfüllen.
  - async add_documents, async search_similar, async delete_documents.

- Graph-Store-Adapter: Neo4j-Adapter, der IGraphStore erfüllt.
  - async add_entities, async add_relationships, async query_graph.

B — Retrieval-Strategien & Tests (High Priority)
- Implementiere/finalisiere VectorOnlyStrategy, GraphOnlyStrategy, HybridStrategy, SemanticSearchStrategy als IRetrievalStrategy-Implementierungen.
- Schreibe Unit-Tests für jede Strategie (Mocks für Stores/LLM).
- Stelle sicher: RetrievalResult.metadata konsistent (keys: strategy, results, raw_scores).

C — Processing-Pipeline (Medium Priority)
- QueryProcessorChain und spezialisierte Prozessoren müssen IQueryProcessor implementieren.
- Implementiere can_handle() für Routing in der Chain.
- Unit- und Integrationstests mit QueryContext-Szenarien.

D — Monitoring & Observers (Medium Priority)
- EventManager implementiert Observer-Registrierung und async notify.
- PerformanceMonitor, MetricsCollector implementieren IObserver/IMetricsCollector.
- Metrics Storage (in-memory + optional persistent backend).

E — DI & Factories (High Priority)
- DIContainer: unterstütze register_singleton/register_transient/register_instance/resolve/create_scope.
- Factories (LLMServiceFactory, DatabaseFactory, RetrievalStrategyFactory) registrieren Implementierungen und liefern Instances via DI.
- Automatisches Wiring in SmartRAGSystem.initialize() anhand von RAGSystemConfig.

F — Learning System (Optional / Low Priority)
- ILearningSystem-Implementierung, die Feedback verarbeitet und optimize_system bereitstellt.

3) Technische Details & Patterns
- Async überall bei I/O (LLM/DB/Graph). Verwende asyncio.
- Use adapters to wrap external SDKs and map to defined interfaces.
- Factories registrieren Implementierungen via a simple registry dict.
- DIContainer nutzt simple constructor-injection (type hints) + factories for external resources.
- Builder (RAGSystemBuilder) produziert RAGSystemConfig, die SmartRAGSystem.initialize() verwendet.

4) Tests & CI
- Unit tests: mocks für LLM/Stores/Graph; tests für Builder, Factories, Strategies, DI.
- Integration tests: minimal config mit Mock-Implementierungen (tests already present).
- CI: pytest in GitHub Actions; Job matrix für Python 3.10/3.11.

5) Taktische Implementierung (3-phasen Plan)
Phase 1 (1-2 Tage)
- Implementiere/prüfe Mocks, DIContainer, Builder, Factories.
- Schreibe Unit-Tests für Builder und DI.
Phase 2 (2-4 Tage)
- Adapter für Vector/Graph/LLM + Strategy-Implementierungen.
- Tests für Retrieval-Strategien.
Phase 3 (1-2 Tage)
- Monitoring, Observers, Metrics Storage, Learning hook.
- Dokumentation + PlantUML erzeugen (PNG/SVG).

6) Dateien (Output)
- docs/uml/implementation_strategy.puml (PlantUML Diagramm: Interfaces -> Implementierungen -> DI/Wiring)
- docs/IMPLEMENTATION_STRATEGY.md (diese Datei)

7) Nächste Schritte (Handlungsoptionen)
- Diagramme rendern (PlantUML -> PNG/SVG). Soll ich rendern und Bilder ablegen?
- Änderungen in Code implementieren (Adapters, Factories, DI improvements). Soll ich mit Phase 1 beginnen und die DI/Factory-Implementierung erstellen und Tests ergänzen?


Kontakt
- Falls du ein konkretes Ziel (z. B. Neo4j-Adapter oder OpenAI-Adapter) priorisieren willst, nenne es kurz.


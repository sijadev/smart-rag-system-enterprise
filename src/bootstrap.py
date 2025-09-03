from .factories import LLMServiceFactory, DatabaseFactory, RetrievalStrategyFactory
from .interfaces import LLMProvider, RetrievalStrategy

# Import adapters lazily to avoid heavy external deps at module import time
def register_default_llm_providers():
    try:
        from .adapters.ollama_adapter import OllamaAdapter
        LLMServiceFactory.register_provider(LLMProvider.OLLAMA, OllamaAdapter)
    except Exception:
        pass

    try:
        from .adapters.openai_adapter import OpenAIAdapter
        LLMServiceFactory.register_provider(LLMProvider.OPENAI, OpenAIAdapter)
    except Exception:
        pass

def register_default_stores_and_strategies():
    import importlib

    # Helper to import adapter modules using package path
    def _register_vector(name: str, module_path: str, class_name: str):
        try:
            mod = importlib.import_module(module_path)
            impl = getattr(mod, class_name)
            DatabaseFactory.register_vector_store(name, impl)
        except Exception:
            pass

    def _register_graph(name: str, module_path: str, class_name: str):
        try:
            mod = importlib.import_module(module_path)
            impl = getattr(mod, class_name)
            DatabaseFactory.register_graph_store(name, impl)
        except Exception:
            pass

    # Mock stores (ensure available for tests)
    _register_vector('mock', 'src.adapters.mock_vector_store', 'MockVectorStore')
    _register_graph('mock', 'src.adapters.mock_graph_store', 'MockGraphStore')

    # Production-capable adapters (best-effort)
    _register_vector('chroma', 'src.adapters.chroma_adapter', 'ChromaAdapter')
    _register_vector('faiss', 'src.adapters.faiss_adapter', 'FaissAdapter')

    # Ensure neo4j adapter is registered if available
    try:
        mod = importlib.import_module('src.adapters.neo4j_adapter')
        if hasattr(mod, 'Neo4jAdapter'):
            DatabaseFactory.register_graph_store('neo4j', getattr(mod, 'Neo4jAdapter'))
    except Exception:
        # best-effort, ignore failures
        pass

    # Ensure mock graph store is registered as fallback
    try:
        mock_mod = importlib.import_module('src.adapters.mock_graph_store')
        if hasattr(mock_mod, 'MockGraphStore'):
            DatabaseFactory.register_graph_store('mock', getattr(mock_mod, 'MockGraphStore'))
    except Exception:
        pass

    # Register strategies
    try:
        strat_mod = importlib.import_module('src.strategies.retrieval_strategies')
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.VECTOR_ONLY, getattr(strat_mod, 'VectorOnlyStrategy'))
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.GRAPH_ONLY, getattr(strat_mod, 'GraphOnlyStrategy'))
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.HYBRID, getattr(strat_mod, 'HybridStrategy'))
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.SEMANTIC_SEARCH, getattr(strat_mod, 'SemanticSearchStrategy'))
    except Exception:
        pass

def register_all_defaults():
    """Register a sensible set of defaults used for bootstrapping and tests."""
    register_default_llm_providers()
    register_default_stores_and_strategies()


if __name__ == "__main__":
    register_all_defaults()
    print("Default implementations registered")

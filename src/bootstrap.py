#!/usr/bin/env python3
"""Bootstrap helper to register default adapters and implementations in factories.

This file is non-invasive: it only registers implementations with the existing
factory registries. It is safe to import and call from initialization code or tests.
"""
from .factories import LLMServiceFactory, DatabaseFactory, RetrievalStrategyFactory
from .interfaces import LLMProvider, RetrievalStrategy

# Import adapters lazily to avoid heavy external deps at module import time
def register_default_llm_providers():
    try:
        from .adapters.ollama_adapter import OllamaAdapter
        LLMServiceFactory.register_provider(LLMProvider.OLLAMA, OllamaAdapter)
    except Exception:
        # best-effort registration; don't crash on import
        pass

    try:
        from .adapters.openai_adapter import OpenAIAdapter
        LLMServiceFactory.register_provider(LLMProvider.OPENAI, OpenAIAdapter)
    except Exception:
        pass

# Neue Registrierungen: Vector/Graph Stores und Retrieval-Strategien

def register_default_stores_and_strategies():
    try:
        from .adapters.mock_vector_store import MockVectorStore
        DatabaseFactory.register_vector_store('mock', MockVectorStore)
    except Exception:
        pass

    try:
        from .adapters.mock_graph_store import MockGraphStore
        DatabaseFactory.register_graph_store('mock', MockGraphStore)
    except Exception:
        pass

    try:
        # Strategien
        from .strategies.retrieval_strategies import (
            VectorOnlyStrategy, GraphOnlyStrategy, HybridStrategy, SemanticSearchStrategy
        )
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.VECTOR_ONLY, VectorOnlyStrategy)
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.GRAPH_ONLY, GraphOnlyStrategy)
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.HYBRID, HybridStrategy)
        RetrievalStrategyFactory.register_strategy(RetrievalStrategy.SEMANTIC_SEARCH, SemanticSearchStrategy)
    except Exception:
        pass


def register_all_defaults():
    """Register a sensible set of defaults used for bootstrapping and tests."""
    register_default_llm_providers()
    register_default_stores_and_strategies()


if __name__ == "__main__":
    register_all_defaults()
    print("Default implementations registered")

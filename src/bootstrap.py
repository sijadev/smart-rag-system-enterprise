import asyncio
import logging
import os

from src.di_container import get_container, register_singleton
from src.interfaces import IGraphStore, ILLMService, IVectorStore

from .factories import (DatabaseFactory, LLMServiceFactory,
                        RetrievalStrategyFactory)
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
    _register_vector("mock", "src.adapters.mock_vector_store", "MockVectorStore")
    _register_graph("mock", "src.adapters.mock_graph_store", "MockGraphStore")

    # Production-capable adapters (best-effort)
    _register_vector("chroma", "src.adapters.chroma_adapter", "ChromaAdapter")
    _register_vector("faiss", "src.adapters.faiss_adapter", "FaissAdapter")

    # Ensure neo4j adapter is registered if available
    try:
        mod = importlib.import_module("src.adapters.neo4j_adapter")
        if hasattr(mod, "Neo4jAdapter"):
            DatabaseFactory.register_graph_store("neo4j", getattr(mod, "Neo4jAdapter"))
    except Exception:
        # best-effort, ignore failures
        pass

    # Ensure mock graph store is registered as fallback
    try:
        mock_mod = importlib.import_module("src.adapters.mock_graph_store")
        if hasattr(mock_mod, "MockGraphStore"):
            DatabaseFactory.register_graph_store(
                "mock", getattr(mock_mod, "MockGraphStore")
            )
    except Exception:
        pass

    # Register strategies
    try:
        strat_mod = importlib.import_module("src.strategies.retrieval_strategies")
        RetrievalStrategyFactory.register_strategy(
            RetrievalStrategy.VECTOR_ONLY, getattr(strat_mod, "VectorOnlyStrategy")
        )
        RetrievalStrategyFactory.register_strategy(
            RetrievalStrategy.GRAPH_ONLY, getattr(strat_mod, "GraphOnlyStrategy")
        )
        RetrievalStrategyFactory.register_strategy(
            RetrievalStrategy.HYBRID, getattr(strat_mod, "HybridStrategy")
        )
        RetrievalStrategyFactory.register_strategy(
            RetrievalStrategy.SEMANTIC_SEARCH,
            getattr(strat_mod, "SemanticSearchStrategy"),
        )
    except Exception:
        pass


def register_all_defaults():
    """Register a sensible set of defaults used for bootstrapping and tests."""
    register_default_llm_providers()
    register_default_stores_and_strategies()
    # Also register runtime service factories into the global DI container.
    # These are lazy factories that create concrete adapter instances when
    # resolved via the DI container. We also register simple teardowns to
    # close resources if adapters expose a close()/aclose() method.
    try:
        # Prepare optional local embedder if configured
        use_local_emb = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
        embedding_dim = int(os.getenv("EMBEDDING_DIMENSIONS", os.getenv("EMBEDDING_DIMENSIONS", "384")))
        embedder_callable = None
        if use_local_emb:
            try:
                from src.embeddings.sentence_transformer import \
                    SentenceTransformerEmbedder

                st = SentenceTransformerEmbedder()
                # embed_texts returns List[List[float]] asynchronously; provide a sync wrapper

                async def _embed_texts_async(texts, batch_size=32):
                    return await st.embed_texts(texts, batch_size=batch_size)

                # We'll pass the async function as the embedder; adapters know to await if coroutine
                embedder_callable = _embed_texts_async
            except Exception:
                embedder_callable = None

        def _make_vector_store():
            # choose best available implementation from DatabaseFactory
            from .factories import DatabaseFactory

            config = {}
            if embedder_callable is not None:
                config["embedder"] = embedder_callable
                config["dimension"] = embedding_dim

            # Prioritize qdrant as configured default; include common adapters and mock as last resort
            for name in ("qdrant", "chroma", "faiss", "mock"):
                try:
                    vs = DatabaseFactory.create_vector_store(name, config)
                    # register teardown to close if available
                    container = get_container()

                    def _teardown():
                        try:
                            if hasattr(vs, "close"):
                                vs.close()
                            elif hasattr(vs, "aclose"):
                                # schedule async close
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        loop.create_task(vs.aclose())
                                    else:
                                        loop.run_until_complete(vs.aclose())
                                except RuntimeError:
                                    pass
                        except Exception:
                            pass

                    try:
                        container.register_teardown(_teardown)
                    except Exception:
                        pass
                    return vs
                except Exception:
                    continue
            raise RuntimeError("No vector store available")

        register_singleton(IVectorStore, _make_vector_store)
    except Exception:
        pass

    try:
        def _make_graph_store():
            from .factories import DatabaseFactory

            for name in ("neo4j", "mock"):
                try:
                    gs = DatabaseFactory.create_graph_store(name, {})
                    container = get_container()

                    def _teardown():
                        try:
                            if hasattr(gs, "close"):
                                gs.close()
                            elif hasattr(gs, "aclose"):
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        loop.create_task(gs.aclose())
                                    else:
                                        loop.run_until_complete(gs.aclose())
                                except RuntimeError:
                                    pass
                        except Exception:
                            pass
                    try:
                        container.register_teardown(_teardown)
                    except Exception:
                        pass
                    return gs
                except Exception:
                    continue
            raise RuntimeError("No graph store available")

        register_singleton(IGraphStore, _make_graph_store)
    except Exception:
        pass

    try:
        def _make_llm_service():
            # prefer configured provider; fallback to any registered provider
            from .factories import LLMServiceFactory

            try:
                svc = LLMServiceFactory.create(LLMProvider.OLLAMA, {})
                try:
                    logging.getLogger(__name__).info("Selected LLM provider: %s", svc.__class__)
                except Exception:
                    pass
                return svc
            except Exception:
                # try other providers
                for p in LLMProvider:
                    try:
                        svc = LLMServiceFactory.create(p, {})
                        try:
                            logging.getLogger(__name__).info("Selected LLM provider: %s", svc.__class__)
                        except Exception:
                            pass
                        return svc
                    except Exception:
                        continue
            raise RuntimeError("No LLM provider available")

        register_singleton(ILLMService, _make_llm_service)
    except Exception:
        pass


if __name__ == "__main__":
    register_all_defaults()
    print("Default implementations registered")

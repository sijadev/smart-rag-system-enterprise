#!/usr/bin/env python3
"""Simple factories for LLM services, databases and retrieval strategies.

Non-invasive: provides registries used by tests and DI wiring.
"""
from typing import Any, Dict, Optional, Type
from .interfaces import LLMProvider, RetrievalStrategy


class LLMServiceFactory:
    _registry: Dict[LLMProvider, Type] = {}

    @classmethod
    def register_provider(cls, provider: LLMProvider, impl: Type) -> None:
        cls._registry[provider] = impl

    @classmethod
    def create(cls, provider: LLMProvider, config: Optional[Dict[str, Any]] = None):
        impl = cls._registry.get(provider)
        if impl is None:
            raise KeyError(f"No LLM provider registered for {provider}")
        return impl(config or {})


class DatabaseFactory:
    _vector_registry: Dict[str, Type] = {}
    _graph_registry: Dict[str, Type] = {}

    @classmethod
    def register_vector_store(cls, name: str, impl: Type) -> None:
        cls._vector_registry[name] = impl

    @classmethod
    def register_graph_store(cls, name: str, impl: Type) -> None:
        cls._graph_registry[name] = impl

    @classmethod
    def create_vector_store(cls, name: str, config: Optional[Dict[str, Any]] = None):
        impl = cls._vector_registry.get(name)
        if impl is None:
            # Best-effort: try to import adapter dynamically from src.adapters
            try:
                module_name = f"src.adapters.{name}_adapter"
                mod = __import__(module_name, fromlist=['*'])
                impl = None
                for attr in dir(mod):
                    if attr.lower().endswith('adapter'):
                        impl = getattr(mod, attr)
                        break
                if impl is not None:
                    cls.register_vector_store(name, impl)
                    return impl(config or {})
            except Exception:
                pass

            # fallback: try to import mock vector store directly and register under requested name
            try:
                from .adapters.mock_vector_store import MockVectorStore
                cls.register_vector_store('mock', MockVectorStore)
                cls.register_vector_store(name, MockVectorStore)
                return MockVectorStore(config or {})
            except Exception:
                raise KeyError(f"No vector store registered for {name}")

        return impl(config or {})

    @classmethod
    def create_graph_store(cls, name: str, config: Optional[Dict[str, Any]] = None):
        impl = cls._graph_registry.get(name)
        if impl is None:
            # Special-case: try Neo4j adapter first for 'neo4j' requests
            if name == 'neo4j':
                try:
                    from .adapters.neo4j_adapter import Neo4jAdapter
                    # Try to instantiate with provided config (best-effort)
                    try:
                        instance = Neo4jAdapter(config or {})
                        # register the class for future creations and return an instance
                        cls.register_graph_store('neo4j', Neo4jAdapter)
                        return instance
                    except Exception:
                        # instantiation may fail if driver not available or config incomplete
                        pass
                except Exception:
                    # import failed - continue to other fallbacks
                    pass

            # Best-effort: try to import adapter dynamically from src.adapters
            try:
                module_name = f"src.adapters.{name}_adapter"
                mod = __import__(module_name, fromlist=['*'])
                # choose first class ending with 'Adapter'
                impl = None
                for attr in dir(mod):
                    if attr.lower().endswith('adapter'):
                        impl = getattr(mod, attr)
                        break
                if impl is not None:
                    # register for future calls and instantiate
                    cls.register_graph_store(name, impl)
                    return impl(config or {})
            except Exception:
                pass

            # fallback: try to import mock graph store directly and register it under requested name
            try:
                from .adapters.mock_graph_store import MockGraphStore
                # register mock under both 'mock' and the requested name to avoid future lookups failing
                cls.register_graph_store('mock', MockGraphStore)
                cls.register_graph_store(name, MockGraphStore)
                return MockGraphStore(config or {})
            except Exception:
                raise KeyError(f"No graph store registered for {name}")

        return impl(config or {})


class RetrievalStrategyFactory:
    _registry: Dict[RetrievalStrategy, Type] = {}

    @classmethod
    def register_strategy(cls, strategy: RetrievalStrategy, impl: Type) -> None:
        cls._registry[strategy] = impl

    @classmethod
    def create(cls, strategy: RetrievalStrategy, *args, **kwargs):
        impl = cls._registry.get(strategy)
        if impl is None:
            raise KeyError(f"No retrieval strategy registered for {strategy}")
        return impl(*args, **kwargs)


# Best-effort default registrations for mocks so tests/bootstrap can rely on them
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

# Best-effort product adapter registrations (non-fatal)
try:
    from .adapters.neo4j_adapter import Neo4jAdapter
    DatabaseFactory.register_graph_store('neo4j', Neo4jAdapter)
except Exception:
    # Try dynamic import as last resort
    try:
        mod = __import__('src.adapters.neo4j_adapter', fromlist=['Neo4jAdapter'])
        if hasattr(mod, 'Neo4jAdapter'):
            DatabaseFactory.register_graph_store('neo4j', getattr(mod, 'Neo4jAdapter'))
    except Exception:
        pass

try:
    from .adapters.chroma_adapter import ChromaAdapter
    DatabaseFactory.register_vector_store('chroma', ChromaAdapter)
except Exception:
    try:
        mod = __import__('src.adapters.chroma_adapter', fromlist=['ChromaAdapter'])
        if hasattr(mod, 'ChromaAdapter'):
            DatabaseFactory.register_vector_store('chroma', getattr(mod, 'ChromaAdapter'))
    except Exception:
        pass

try:
    from .adapters.faiss_adapter import FaissAdapter
    DatabaseFactory.register_vector_store('faiss', FaissAdapter)
except Exception:
    try:
        mod = __import__('src.adapters.faiss_adapter', fromlist=['FaissAdapter'])
        if hasattr(mod, 'FaissAdapter'):
            DatabaseFactory.register_vector_store('faiss', getattr(mod, 'FaissAdapter'))
    except Exception:
        pass

# Ensure at least a mock graph store is registered under 'neo4j' as a last resort
try:
    if 'neo4j' not in DatabaseFactory._graph_registry:
        from .adapters.mock_graph_store import MockGraphStore
        DatabaseFactory.register_graph_store('neo4j', MockGraphStore)
except Exception:
    # ignore if even mock not available
    pass

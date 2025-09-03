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
            raise KeyError(f"No vector store registered for {name}")
        return impl(config or {})

    @classmethod
    def create_graph_store(cls, name: str, config: Optional[Dict[str, Any]] = None):
        impl = cls._graph_registry.get(name)
        if impl is None:
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

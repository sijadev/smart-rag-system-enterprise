#!/usr/bin/env python3
"""
Factory Pattern Implementation für Smart RAG System
==================================================

Erstellt verschiedene Services basierend auf Konfiguration
"""

from typing import Dict, Any, Optional, Type
from .interfaces import (
    ILLMService, IVectorStore, IGraphStore, IRetrievalStrategy,
    LLMProvider, RetrievalStrategy, RAGConfiguration
)
from .llm_services import OllamaLLMService, OllamaConfig, OllamaServiceFactory
import logging

logger = logging.getLogger(__name__)


class LLMServiceFactory:
    """Factory für LLM Services"""

    _providers: Dict[LLMProvider, Type[ILLMService]] = {}

    @classmethod
    def register_provider(cls, provider: LLMProvider, service_class: Type[ILLMService]) -> None:
        """Registriert einen neuen LLM Provider"""
        cls._providers[provider] = service_class
        logger.info(f"Registered LLM provider: {provider.value}")

    @classmethod
    def create_service(cls, provider: LLMProvider, config: Dict[str, Any]) -> ILLMService:
        """Erstellt LLM Service basierend auf Provider"""
        if provider == LLMProvider.OLLAMA:
            # Spezielle Behandlung für Ollama
            ollama_config = OllamaConfig(
                base_url=config.get("base_url", "http://localhost:11434"),
                model=config.get("model", "llama2"),
                embedding_model=config.get("embedding_model", "llama2"),
                timeout=config.get("timeout", 300),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 0.9),
                max_tokens=config.get("max_tokens", 2000)
            )
            return OllamaLLMService(ollama_config)

        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider.value}")

        service_class = cls._providers[provider]
        return service_class(config)

    @classmethod
    def create_ollama_service(cls, **kwargs) -> OllamaLLMService:
        """Erstellt Ollama Service mit spezifischer Konfiguration"""
        return OllamaServiceFactory.create_with_config(**kwargs)

    @classmethod
    def create_ollama_from_env(cls) -> OllamaLLMService:
        """Erstellt Ollama Service aus Umgebungsvariablen"""
        return OllamaServiceFactory.create_from_env()

    @classmethod
    def get_available_providers(cls) -> list[LLMProvider]:
        """Gibt verfügbare Provider zurück"""
        available = list(cls._providers.keys())
        # Ollama ist immer verfügbar, da es direkt implementiert ist
        if LLMProvider.OLLAMA not in available:
            available.append(LLMProvider.OLLAMA)
        return available


class RetrievalStrategyFactory:
    """Factory für Retrieval-Strategien"""

    _strategies: Dict[RetrievalStrategy, Type[IRetrievalStrategy]] = {}

    @classmethod
    def register_strategy(cls, strategy: RetrievalStrategy, strategy_class: Type[IRetrievalStrategy]) -> None:
        """Registriert eine neue Retrieval-Strategie"""
        cls._strategies[strategy] = strategy_class
        logger.info(f"Registered retrieval strategy: {strategy.value}")

    @classmethod
    def create_strategy(cls, strategy: RetrievalStrategy,
                        vector_store: IVectorStore,
                        graph_store: Optional[IGraphStore] = None,
                        llm_service: Optional[ILLMService] = None) -> IRetrievalStrategy:
        """Erstellt Retrieval-Strategie"""
        if strategy not in cls._strategies:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy.value}")

        strategy_class = cls._strategies[strategy]
        return strategy_class(vector_store, graph_store, llm_service)

    @classmethod
    def get_available_strategies(cls) -> list[RetrievalStrategy]:
        """Gibt verfügbare Strategien zurück"""
        return list(cls._strategies.keys())


class DatabaseFactory:
    """Factory für Database Services"""

    @staticmethod
    def create_vector_store(config: Dict[str, Any]) -> IVectorStore:
        """Erstellt Vector Store basierend auf Konfiguration"""
        store_type = config.get('type', 'chroma')

        if store_type == 'chroma':
            from .stores.chroma_store import ChromaVectorStore
            return ChromaVectorStore(config)
        elif store_type == 'faiss':
            from .stores.faiss_store import FAISSVectorStore
            return FAISSVectorStore(config)
        elif store_type == 'pinecone':
            from .stores.pinecone_store import PineconeVectorStore
            return PineconeVectorStore(config)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")

    @staticmethod
    def create_graph_store(config: Dict[str, Any]) -> Optional[IGraphStore]:
        """Erstellt Graph Store basierend auf Konfiguration"""
        if not config.get('enabled', False):
            return None

        store_type = config.get('type', 'neo4j')

        if store_type == 'neo4j':
            from .stores.neo4j_store import Neo4jGraphStore
            return Neo4jGraphStore(config)
        elif store_type == 'networkx':
            from .stores.networkx_store import NetworkXGraphStore
            return NetworkXGraphStore(config)
        else:
            raise ValueError(f"Unsupported graph store type: {store_type}")


class ServiceFactory:
    """Hauptfactory für alle Services"""

    def __init__(self, config: RAGConfiguration):
        self.config = config
        self._llm_service: Optional[ILLMService] = None
        self._vector_store: Optional[IVectorStore] = None
        self._graph_store: Optional[IGraphStore] = None

    def get_llm_service(self) -> ILLMService:
        """Lazy-Loading für LLM Service"""
        if self._llm_service is None:
            self._llm_service = LLMServiceFactory.create_service(
                self.config.llm_provider,
                self._get_llm_config()
            )
        return self._llm_service

    def get_vector_store(self) -> IVectorStore:
        """Lazy-Loading für Vector Store"""
        if self._vector_store is None:
            self._vector_store = DatabaseFactory.create_vector_store(
                self._get_vector_config()
            )
        return self._vector_store

    def get_graph_store(self) -> Optional[IGraphStore]:
        """Lazy-Loading für Graph Store"""
        if self._graph_store is None and hasattr(self.config, 'enable_graph_store'):
            if self.config.enable_graph_store:
                self._graph_store = DatabaseFactory.create_graph_store(
                    self._get_graph_config()
                )
        return self._graph_store

    def create_retrieval_strategy(self, strategy: RetrievalStrategy) -> IRetrievalStrategy:
        """Erstellt Retrieval-Strategie mit verfügbaren Services"""
        return RetrievalStrategyFactory.create_strategy(
            strategy,
            self.get_vector_store(),
            self.get_graph_store(),
            self.get_llm_service()
        )

    def _get_llm_config(self) -> Dict[str, Any]:
        """Extrahiert LLM-Konfiguration"""
        return {
            'model_name': getattr(self.config, 'model_name', 'llama2'),
            'temperature': getattr(self.config, 'temperature', 0.7),
            'max_tokens': getattr(self.config, 'max_tokens', 2048),
            'api_key': getattr(self.config, 'api_key', None),
            'base_url': getattr(self.config, 'base_url', None)
        }

    def _get_vector_config(self) -> Dict[str, Any]:
        """Extrahiert Vector Store-Konfiguration"""
        return {
            'type': getattr(self.config, 'vector_store_type', 'chroma'),
            'persist_directory': getattr(self.config, 'vector_store_path', './data/vectors'),
            'collection_name': getattr(self.config, 'collection_name', 'rag_documents')
        }

    def _get_graph_config(self) -> Dict[str, Any]:
        """Extrahiert Graph Store-Konfiguration"""
        return {
            'enabled': getattr(self.config, 'enable_graph_store', False),
            'type': getattr(self.config, 'graph_store_type', 'neo4j'),
            'uri': getattr(self.config, 'neo4j_uri', 'bolt://localhost:7687'),
            'user': getattr(self.config, 'neo4j_user', 'neo4j'),
            'password': getattr(self.config, 'neo4j_password', 'password123')
        }

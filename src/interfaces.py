#!/usr/bin/env python3
"""
Core Interfaces für Smart RAG System
====================================

Definiert abstrakte Interfaces für alle Hauptkomponenten
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


@dataclass
class QueryContext:
    """Kontext für Query-Verarbeitung"""
    query_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    previous_queries: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class RetrievalResult:
    """Ergebnis einer Retrieval-Operation"""
    contexts: List[str]
    sources: List[Dict[str, Any]]
    confidence_scores: List[float]
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Vollständige RAG-Antwort"""
    answer: str
    contexts: List[str]
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class LLMProvider(Enum):
    """Verfügbare LLM Provider"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class RetrievalStrategy(Enum):
    """Retrieval-Strategien"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    SEMANTIC_SEARCH = "semantic_search"
    CONTEXTUAL = "contextual"


# Abstract Interfaces

class ILLMService(ABC):
    """Interface für LLM Services"""

    @abstractmethod
    async def generate(self, prompt: str, context: Optional[QueryContext] = None) -> str:
        """Generiert Antwort basierend auf Prompt"""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Erstellt Text-Embeddings"""
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Gibt Provider-Informationen zurück"""
        pass


class IVectorStore(ABC):
    """Interface für Vector Stores"""

    @abstractmethod
    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Fügt Dokumente hinzu"""
        pass

    @abstractmethod
    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Sucht ähnliche Dokumente"""
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Löscht Dokumente"""
        pass


class IGraphStore(ABC):
    """Interface für Graph Stores"""

    @abstractmethod
    async def add_entities(self, entities: List[Dict[str, Any]]) -> None:
        """Fügt Entitäten hinzu"""
        pass

    @abstractmethod
    async def add_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Fügt Beziehungen hinzu"""
        pass

    @abstractmethod
    async def query_graph(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Führt Graph-Query aus"""
        pass


class IRetrievalStrategy(ABC):
    """Interface für Retrieval-Strategien"""

    @abstractmethod
    async def retrieve(self, query: str, context: QueryContext, k: int = 5) -> RetrievalResult:
        """Führt Retrieval durch"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Gibt Strategie-Namen zurück"""
        pass


class IQueryProcessor(ABC):
    """Interface für Query-Prozessoren"""

    @abstractmethod
    async def process(self, query: str, context: QueryContext) -> RAGResponse:
        """Verarbeitet Query und gibt Antwort zurück"""
        pass

    @abstractmethod
    def can_handle(self, query: str, context: QueryContext) -> bool:
        """Prüft ob Query verarbeitet werden kann"""
        pass


class IObserver(ABC):
    """Observer Interface für Monitoring"""

    @abstractmethod
    async def notify(self, event: str, data: Dict[str, Any]) -> None:
        """Benachrichtigung über Events"""
        pass


class IMetricsCollector(ABC):
    """Interface für Metriken-Sammlung"""

    @abstractmethod
    async def record_query(self, query: str, response: RAGResponse, context: QueryContext) -> None:
        """Zeichnet Query-Metriken auf"""
        pass

    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Gibt gesammelte Metriken zurück"""
        pass


class ILearningSystem(ABC):
    """Interface für Learning-Systeme"""

    @abstractmethod
    async def learn_from_feedback(self, query_id: str, feedback: Dict[str, Any]) -> None:
        """Lernt aus User-Feedback"""
        pass

    @abstractmethod
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimiert System basierend auf gelernten Patterns"""
        pass


# Configuration Protocols

class RAGConfiguration(Protocol):
    """Protocol für RAG-Konfiguration"""

    llm_provider: LLMProvider
    default_strategy: RetrievalStrategy
    enable_learning: bool
    enable_monitoring: bool
    max_context_length: int
    temperature: float


class DatabaseConfiguration(Protocol):
    """Protocol für Database-Konfiguration"""

    neo4j_uri: Optional[str]
    neo4j_user: Optional[str]
    neo4j_password: Optional[str]
    vector_store_path: Optional[str]
    enable_graph_store: bool

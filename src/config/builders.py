#!/usr/bin/env python3
"""
Builder Pattern für RAG System Konfiguration
===========================================

Implementiert flexible Builder für komplexe Systemkonfigurationen
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from ..interfaces import LLMProvider, RetrievalStrategy


@dataclass
class RAGSystemConfig:
    """Vollständige RAG System Konfiguration"""

    # Core Settings
    system_name: str = "SmartRAG"
    version: str = "2.0.0"

    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    model_name: str = "llama3.2:latest"  # Aktualisiert von llama2
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Retrieval Configuration
    default_retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    max_context_length: int = 4000
    retrieval_k: int = 5

    # Database Configuration
    enable_graph_store: bool = True
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vectors"
    collection_name: str = "rag_documents"
    neo4j_uri: Optional[str] = "bolt://localhost:7687"
    neo4j_user: Optional[str] = "neo4j"
    neo4j_password: Optional[str] = None

    # Monitoring & Learning
    enable_monitoring: bool = True
    enable_learning: bool = True
    metrics_retention_days: int = 30
    optimization_interval: int = 100

    # Security
    enable_security: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # seconds

    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600
    parallel_processing: bool = True
    max_workers: int = 4

    # Custom Extensions
    custom_processors: List[str] = field(default_factory=list)
    custom_strategies: List[str] = field(default_factory=list)
    extension_config: Dict[str, Any] = field(default_factory=dict)


class RAGSystemBuilder:
    """Builder für RAG System Konfiguration"""

    def __init__(self):
        self.config = RAGSystemConfig()
        self._validation_rules: List[Callable[[RAGSystemConfig], bool]] = []

    # Core Configuration
    def with_name(self, name: str) -> 'RAGSystemBuilder':
        """Setzt System-Namen"""
        self.config.system_name = name
        return self

    def with_version(self, version: str) -> 'RAGSystemBuilder':
        """Setzt Version"""
        self.config.version = version
        return self

    # LLM Configuration
    def with_llm_provider(self, provider: LLMProvider, model_name: str = None,
                          api_key: str = None, base_url: str = None) -> 'RAGSystemBuilder':
        """Konfiguriert LLM Provider"""
        self.config.llm_provider = provider
        if model_name:
            self.config.model_name = model_name
        if api_key:
            self.config.api_key = api_key
        if base_url:
            self.config.base_url = base_url
        return self

    def with_ollama(self, model_name: str = "llama3.2:latest", base_url: str = "http://localhost:11434") -> 'RAGSystemBuilder':
        """Konfiguriert Ollama LLM"""
        return self.with_llm_provider(LLMProvider.OLLAMA, model_name, base_url=base_url)

    def with_openai(self, model_name: str = "gpt-4", api_key: str = None) -> 'RAGSystemBuilder':
        """Konfiguriert OpenAI LLM"""
        return self.with_llm_provider(LLMProvider.OPENAI, model_name, api_key=api_key)

    def with_anthropic(self, model_name: str = "claude-3-sonnet-20240229", api_key: str = None) -> 'RAGSystemBuilder':
        """Konfiguriert Anthropic LLM"""
        return self.with_llm_provider(LLMProvider.ANTHROPIC, model_name, api_key=api_key)

    # Retrieval Configuration
    def with_retrieval_strategy(self, strategy: RetrievalStrategy) -> 'RAGSystemBuilder':
        """Setzt Standard-Retrieval-Strategie"""
        self.config.default_retrieval_strategy = strategy
        return self

    def with_hybrid_retrieval(self) -> 'RAGSystemBuilder':
        """Konfiguriert Hybrid-Retrieval (Vector + Graph)"""
        return self.with_retrieval_strategy(RetrievalStrategy.HYBRID)

    def with_vector_only_retrieval(self) -> 'RAGSystemBuilder':
        """Konfiguriert Vector-Only Retrieval"""
        return self.with_retrieval_strategy(RetrievalStrategy.VECTOR_ONLY)

    def with_graph_only_retrieval(self) -> 'RAGSystemBuilder':
        """Konfiguriert Graph-Only Retrieval"""
        return self.with_retrieval_strategy(RetrievalStrategy.GRAPH_ONLY)

    # Database Configuration
    def with_vector_store(self, store_type: str = "chroma", path: str = "./data/vectors") -> 'RAGSystemBuilder':
        """Konfiguriert Vector Store"""
        self.config.vector_store_type = store_type
        self.config.vector_store_path = path
        return self

    def with_neo4j(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = None) -> 'RAGSystemBuilder':
        """Konfiguriert Neo4j Graph Store"""
        self.config.enable_graph_store = True
        self.config.neo4j_uri = uri
        self.config.neo4j_user = user
        self.config.neo4j_password = password
        return self

    def with_neo4j_enterprise(self, uri: str, user: str, password: str) -> 'RAGSystemBuilder':
        """Konfiguriert Neo4j Enterprise"""
        return self.with_neo4j(uri, user, password)

    def without_graph_store(self) -> 'RAGSystemBuilder':
        """Deaktiviert Graph Store"""
        self.config.enable_graph_store = False
        return self

    # Monitoring & Learning
    def with_monitoring(self, enabled: bool = True) -> 'RAGSystemBuilder':
        """Aktiviert/Deaktiviert Monitoring"""
        self.config.enable_monitoring = enabled
        return self

    def with_learning(self, enabled: bool = True) -> 'RAGSystemBuilder':
        """Aktiviert/Deaktiviert Learning"""
        self.config.enable_learning = enabled
        return self

    # Security Configuration
    def with_security(self, enabled: bool = True, rate_limit: int = 100) -> 'RAGSystemBuilder':
        """Konfiguriert Security Settings"""
        self.config.enable_security = enabled
        self.config.rate_limit_requests = rate_limit
        return self

    def without_security(self) -> 'RAGSystemBuilder':
        """Deaktiviert Security Features"""
        self.config.enable_security = False
        return self

    # Performance Configuration
    def with_caching(self, enabled: bool = True, ttl: int = 3600) -> 'RAGSystemBuilder':
        """Konfiguriert Caching"""
        self.config.enable_caching = enabled
        self.config.cache_ttl = ttl
        return self

    def with_parallel_processing(self, enabled: bool = True, max_workers: int = 4) -> 'RAGSystemBuilder':
        """Konfiguriert Parallel Processing"""
        self.config.parallel_processing = enabled
        self.config.max_workers = max_workers
        return self

    # Validation
    def add_validation_rule(self, rule: Callable[[RAGSystemConfig], bool]) -> 'RAGSystemBuilder':
        """Fügt Validierungsregel hinzu"""
        self._validation_rules.append(rule)
        return self

    def build(self) -> RAGSystemConfig:
        """Erstellt finale Konfiguration mit Validierung"""
        # Validate configuration
        for rule in self._validation_rules:
            if not rule(self.config):
                raise ValueError("Configuration validation failed")

        # Set defaults based on configuration
        if self.config.llm_provider == LLMProvider.OLLAMA and not self.config.base_url:
            self.config.base_url = "http://localhost:11434"

        return self.config


# Convenience Factory Functions

def create_development_config() -> RAGSystemConfig:
    """Erstellt Development-Konfiguration"""
    return (RAGSystemBuilder()
            .with_name("SmartRAG-Dev")
            .with_ollama()
            .with_hybrid_retrieval()
            .with_vector_store()
            .with_neo4j()
            .with_monitoring(True)
            .with_learning(True)
            .without_security()
            .with_caching(True)
            .build())


def create_production_config() -> RAGSystemConfig:
    """Erstellt Production-Konfiguration"""
    return (RAGSystemBuilder()
            .with_name("SmartRAG-Prod")
            .with_ollama()
            .with_hybrid_retrieval()
            .with_vector_store("chroma", "./prod/vectors")
            .with_neo4j("bolt://localhost:7687", "neo4j")
            .with_monitoring(True)
            .with_learning(True)
            .with_security(True, 1000)
            .with_caching(True, 7200)
            .with_parallel_processing(True, 8)
            .build())


def create_minimal_config() -> RAGSystemConfig:
    """Erstellt minimale Konfiguration für Tests"""
    return (RAGSystemBuilder()
            .with_name("SmartRAG-Minimal")
            .with_ollama()
            .with_vector_only_retrieval()
            .with_vector_store()
            .without_graph_store()
            .with_monitoring(False)
            .with_learning(False)
            .without_security()
            .build())


def create_enterprise_config() -> RAGSystemConfig:
    """Erstellt Enterprise-Konfiguration"""
    return (RAGSystemBuilder()
            .with_name("SmartRAG-Enterprise")
            .with_ollama()
            .with_hybrid_retrieval()
            .with_vector_store("chroma", "./enterprise/vectors")
            .with_neo4j_enterprise("neo4j://enterprise:7687", "neo4j", "enterprise_password")
            .with_monitoring(True)
            .with_learning(True)
            .with_security(True, 10000)
            .with_caching(True, 14400)
            .with_parallel_processing(True, 16)
            .build())

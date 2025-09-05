#!/usr/bin/env python3
"""
Zentrale Konfiguration mit Dependency Injection
===============================================

Einheitliche Konfigurationsverwaltung für das gesamte Smart RAG System
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

# Lade .env automatisch
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class OllamaConfig:
    """Zentrale Ollama-Konfiguration"""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    embedding_model: str = "nomic-embed-text:latest"
    chat_model: str = "llama3.1:8b"
    timeout: int = 300
    temperature: float = 0.7
    max_tokens: int = 2000
    max_retries: int = 3
    auto_pull_models: bool = True


@dataclass
class DatabaseConfig:
    """Zentrale Datenbank-Konfiguration"""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = None

    # Vector Store
    vector_store_type: str = "chroma"
    # Persist directory used by Chroma; set to ./chroma_db to match local sqlite
    vector_store_path: str = "./chroma_db"
    # Chroma backend implementation to use (sqlite or duckdb+parquet)
    chroma_db_impl: str = "sqlite"
    collection_name: str = "rag_documents"

    # Graph Store
    enable_graph_store: bool = True


@dataclass
class SystemConfig:
    """Zentrale System-Konfiguration"""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    data_path: str = "./data"

    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_workers: int = 4

    # Monitoring
    enable_monitoring: bool = True
    enable_analytics: bool = True

    # Learning
    learning_rate: float = 0.1
    optimization_interval: int = 100
    min_feedback_samples: int = 10


@dataclass
class CentralConfig:
    """Zentrale Hauptkonfiguration - Single Source of Truth"""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Automatische Konfiguration nach Initialisierung"""
        self._load_from_env()
        self._apply_environment_defaults()

    def _load_from_env(self):
        """Lädt alle Werte aus Umgebungsvariablen"""
        # Debug: Umgebungsvariablen prüfen
        print("NEO4J_URI aus env:", os.getenv("NEO4J_URI"))
        print("NEO4J_USER aus env:", os.getenv("NEO4J_USER"))
        print("NEO4J_PASSWORD aus env:", os.getenv("NEO4J_PASSWORD"))

        # System Configuration
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            self.system.environment = Environment(env_name)
        except ValueError:
            self.system.environment = Environment.DEVELOPMENT

        self.system.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.system.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.system.data_path = os.getenv("DATA_PATH", "./data")

        # Ollama Configuration
        self.ollama.base_url = os.getenv("OLLAMA_BASE_URL", self.ollama.base_url)
        self.ollama.model = os.getenv(
            "LLM_MODEL", os.getenv("OLLAMA_MODEL", self.ollama.model)
        )
        self.ollama.embedding_model = os.getenv(
            "EMBED_MODEL",
            os.getenv("OLLAMA_EMBEDDING_MODEL", self.ollama.embedding_model),
        )
        self.ollama.chat_model = os.getenv(
            "ANALYZER_MODEL", os.getenv("OLLAMA_CHAT_MODEL", self.ollama.chat_model)
        )

        # Parse numeric values safely
        try:
            self.ollama.timeout = int(
                os.getenv("OLLAMA_TIMEOUT", str(self.ollama.timeout))
            )
        except ValueError:
            pass

        try:
            self.ollama.temperature = float(
                os.getenv("TEMPERATURE", str(self.ollama.temperature))
            )
        except ValueError:
            pass

        try:
            self.ollama.max_tokens = int(
                os.getenv("MAX_TOKENS", str(self.ollama.max_tokens))
            )
        except ValueError:
            pass

        # Database Configuration
        self.database.neo4j_uri = os.getenv("NEO4J_URI", self.database.neo4j_uri)
        self.database.neo4j_user = os.getenv("NEO4J_USER", self.database.neo4j_user)
        self.database.neo4j_password = os.getenv(
            "NEO4J_PASSWORD", self.database.neo4j_password
        )
        # Allow overriding the vector store type via environment variable
        self.database.vector_store_type = os.getenv("VECTOR_STORE_TYPE", os.getenv("CHROMA_VECTOR_STORE_TYPE", self.database.vector_store_type))
        # Optional overrides for vector store
        self.database.vector_store_path = os.getenv("CHROMA_PERSIST_DIR", self.database.vector_store_path)
        self.database.collection_name = os.getenv("CHROMA_COLLECTION", self.database.collection_name)
        self.database.chroma_db_impl = os.getenv("CHROMA_DB_IMPL", self.database.chroma_db_impl)

        # Learning Configuration
        try:
            self.system.learning_rate = float(
                os.getenv("LEARNING_RATE", str(self.system.learning_rate))
            )
        except ValueError:
            pass

        try:
            self.system.optimization_interval = int(
                os.getenv(
                    "OPTIMIZATION_INTERVAL", str(self.system.optimization_interval)
                )
            )
        except ValueError:
            pass

    def _apply_environment_defaults(self):
        """Wendet umgebungsspezifische Standardwerte an"""
        if self.system.environment == Environment.PRODUCTION:
            self.system.debug = False
            self.system.log_level = "WARNING"
            self.ollama.auto_pull_models = False
            self.ollama.timeout = 30
            self.system.enable_caching = True

        elif self.system.environment == Environment.TESTING:
            self.system.debug = True
            self.system.log_level = "DEBUG"
            self.ollama.auto_pull_models = False
            self.system.enable_monitoring = False
            self.database.enable_graph_store = False

        elif self.system.environment == Environment.DEVELOPMENT:
            self.system.debug = True
            self.ollama.auto_pull_models = True
            self.system.enable_monitoring = True

    def validate(self) -> list[str]:
        """Validiert die gesamte Konfiguration"""
        errors = []

        # Ollama Validation
        if not self.ollama.base_url:
            errors.append("Ollama base_url ist erforderlich")
        if not self.ollama.model:
            errors.append("Ollama model ist erforderlich")
        if not self.ollama.embedding_model:
            errors.append("Ollama embedding_model ist erforderlich")
        if self.ollama.temperature < 0 or self.ollama.temperature > 2:
            errors.append("Ollama temperature muss zwischen 0 und 2 liegen")
        if self.ollama.max_tokens < 1:
            errors.append("Ollama max_tokens muss >= 1 sein")

        # Database Validation
        if self.database.enable_graph_store and not self.database.neo4j_password:
            errors.append(
                "Neo4j password ist erforderlich wenn Graph Store aktiviert ist"
            )
        if not self.database.vector_store_path:
            errors.append("Vector store path ist erforderlich")

        # System Validation
        if self.system.learning_rate <= 0 or self.system.learning_rate > 1:
            errors.append("Learning rate muss zwischen 0 und 1 liegen")
        if self.system.optimization_interval < 1:
            errors.append("Optimization interval muss >= 1 sein")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Serialisierung"""
        return {
            "ollama": {
                "base_url": self.ollama.base_url,
                "model": self.ollama.model,
                "embedding_model": self.ollama.embedding_model,
                "chat_model": self.ollama.chat_model,
                "timeout": self.ollama.timeout,
                "temperature": self.ollama.temperature,
                "max_tokens": self.ollama.max_tokens,
                "max_retries": self.ollama.max_retries,
                "auto_pull_models": self.ollama.auto_pull_models,
            },
            "database": {
                "neo4j_uri": self.database.neo4j_uri,
                "neo4j_user": self.database.neo4j_user,
                "neo4j_password": "***" if self.database.neo4j_password else None,
                "vector_store_type": self.database.vector_store_type,
                "vector_store_path": self.database.vector_store_path,
                "collection_name": self.database.collection_name,
                "chroma_db_impl": self.database.chroma_db_impl,
                "enable_graph_store": self.database.enable_graph_store,
            },
            "system": {
                "environment": self.system.environment.value,
                "debug": self.system.debug,
                "log_level": self.system.log_level,
                "data_path": self.system.data_path,
                "enable_caching": self.system.enable_caching,
                "enable_monitoring": self.system.enable_monitoring,
                "learning_rate": self.system.learning_rate,
                "optimization_interval": self.system.optimization_interval,
            },
        }


# Singleton Pattern für globale Konfiguration
_global_config: Optional[CentralConfig] = None


def get_config() -> CentralConfig:
    """Gibt die globale Konfiguration zurück (Singleton)"""
    global _global_config
    if _global_config is None:
        _global_config = CentralConfig()

        # Validierung
        errors = _global_config.validate()
        if errors:
            logger.warning(f"Konfigurationsfehler gefunden: {', '.join(errors)}")

        logger.info(
            f"Zentrale Konfiguration geladen: {_global_config.system.environment.value}"
        )

    return _global_config


def reset_config():
    """Setzt die globale Konfiguration zurück (für Tests)"""
    global _global_config
    _global_config = None


# Dependency Injection Container
class DIContainer:
    """Einfacher Dependency Injection Container"""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, callable] = {}
        self._config = get_config()

        # Registriere Basis-Services
        self.register_instance(CentralConfig, self._config)
        self.register_instance(OllamaConfig, self._config.ollama)
        self.register_instance(DatabaseConfig, self._config.database)
        self.register_instance(SystemConfig, self._config.system)

    def register_instance(self, interface: Type[T], instance: T) -> "DIContainer":
        """Registriert eine Instanz"""
        self._services[interface] = instance
        return self

    def register_factory(self, interface: Type[T], factory: callable) -> "DIContainer":
        """Registriert eine Factory-Funktion"""
        self._factories[interface] = factory
        return self

    def register_singleton(
        self, interface: Type[T], implementation: Type[T]
    ) -> "DIContainer":
        """Registriert als Singleton"""
        if interface not in self._services:
            instance = self._create_instance(implementation)
            self._services[interface] = instance
        return self

    def resolve(self, interface: Type[T]) -> T:
        """Löst eine Abhängigkeit auf"""
        if interface in self._services:
            return self._services[interface]

        if interface in self._factories:
            instance = self._factories[interface](self)
            self._services[interface] = instance
            return instance

        raise ValueError(f"Service {interface.__name__} nicht registriert")

    def _create_instance(self, implementation: Type[T]) -> T:
        """Erstellt eine Instanz mit automatischer Injection"""
        import inspect

        signature = inspect.signature(implementation.__init__)
        kwargs = {}

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[name] = self.resolve(param.annotation)
                except ValueError:
                    if param.default != inspect.Parameter.empty:
                        kwargs[name] = param.default
                    else:
                        raise ValueError(
                            f"Cannot resolve dependency {param.annotation} for {implementation}"
                        )
            elif param.default != inspect.Parameter.empty:
                kwargs[name] = param.default

        return implementation(**kwargs)


# Globaler Container (Singleton)
_global_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Gibt den globalen DI Container zurück"""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container


def reset_container():
    """Setzt den globalen Container zurück (für Tests)"""
    global _global_container
    _global_container = None


# Decorator für automatische Injection
def inject(func):
    """Decorator für automatische Dependency Injection"""
    import inspect
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        container = get_container()
        signature = inspect.signature(func)

        for name, param in signature.parameters.items():
            if name not in kwargs and param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[name] = container.resolve(param.annotation)
                except ValueError:
                    if param.default == inspect.Parameter.empty:
                        raise ValueError(
                            f"Cannot inject {param.annotation} into {func.__name__}"
                        )

        return func(*args, **kwargs)

    return wrapper


# Configuration Utilities
def configure_logging(config: SystemConfig = None):
    """Konfiguriert Logging basierend auf System Config"""
    if config is None:
        config = get_config().system

    level = getattr(logging, config.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=(
            [
                logging.StreamHandler(),
                logging.FileHandler(f"{config.data_path}/system.log"),
            ]
            if config.debug
            else [logging.StreamHandler()]
        ),
    )

    logger.info(
        f"Logging konfiguriert: Level={config.log_level}, Environment={config.environment.value}"
    )


def print_config_summary():
    """Zeigt eine Zusammenfassung der aktuellen Konfiguration"""
    config = get_config()

    print("🔧 Zentrale Konfiguration - Smart RAG System")
    print("=" * 60)

    print(f"🌍 Environment: {config.system.environment.value}")
    print(f"📊 Debug Mode: {config.system.debug}")
    print(f"📝 Log Level: {config.system.log_level}")

    print("\n🤖 Ollama Configuration:")
    print(f"   Base URL: {config.ollama.base_url}")
    print(f"   Model: {config.ollama.model}")
    print(f"   Embedding Model: {config.ollama.embedding_model}")
    print(f"   Chat Model: {config.ollama.chat_model}")
    print(f"   Temperature: {config.ollama.temperature}")
    print(f"   Auto Pull: {config.ollama.auto_pull_models}")

    print("\n🗄️ Database Configuration:")
    print(f"   Neo4j URI: {config.database.neo4j_uri}")
    print(
        f"   Graph Store: {'Enabled' if config.database.enable_graph_store else 'Disabled'}"
    )
    print(f"   Vector Store: {config.database.vector_store_type}")
    print(f"   Vector Path: {config.database.vector_store_path}")

    print("\n⚙️ System Configuration:")
    print(f"   Caching: {'Enabled' if config.system.enable_caching else 'Disabled'}")
    print(
        f"   Monitoring: {'Enabled' if config.system.enable_monitoring else 'Disabled'}"
    )
    print(f"   Learning Rate: {config.system.learning_rate}")
    print(f"   Data Path: {config.system.data_path}")

    # Validierung
    errors = config.validate()
    if errors:
        print("\n⚠️ Konfigurationsfehler:")
        for error in errors:
            print(f"   • {error}")
    else:
        print("\n✅ Konfiguration ist valid")


if __name__ == "__main__":
    # Test der zentralen Konfiguration
    print_config_summary()

    # Test der Dependency Injection
    container = get_container()

    ollama_config = container.resolve(OllamaConfig)
    print(f"\n🧪 DI Test - Ollama Model: {ollama_config.model}")

    system_config = container.resolve(SystemConfig)
    print(f"🧪 DI Test - Environment: {system_config.environment.value}")

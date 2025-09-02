#!/usr/bin/env python3
"""
Dependency Injection Container für Smart RAG System
=================================================

Implementiert IoC Container für Dependency Management
"""

from typing import Dict, Any, TypeVar, Type, Optional, Callable, Union
import inspect
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ServiceDescriptor:
    """Beschreibt einen registrierten Service"""
    service_type: Type
    implementation: Union[Type, Callable, Any]
    lifetime: str  # 'singleton', 'transient', 'scoped'
    factory: Optional[Callable] = None


class ServiceLifetime:
    """Service Lifetime Konstanten"""
    SINGLETON = 'singleton'
    TRANSIENT = 'transient'
    SCOPED = 'scoped'


class DIContainer:
    """Dependency Injection Container"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_services: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None

    def register_singleton(self, interface: Type[T], implementation: Union[Type[T], Callable[[], T]]) -> 'DIContainer':
        """Registriert Service als Singleton"""
        self._services[interface] = ServiceDescriptor(
            service_type=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON
        )
        logger.debug(f"Registered singleton: {interface.__name__}")
        return self

    def register_transient(self, interface: Type[T], implementation: Union[Type[T], Callable[[], T]]) -> 'DIContainer':
        """Registriert Service als Transient (neue Instanz bei jeder Auflösung)"""
        self._services[interface] = ServiceDescriptor(
            service_type=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT
        )
        logger.debug(f"Registered transient: {interface.__name__}")
        return self

    def register_scoped(self, interface: Type[T], implementation: Union[Type[T], Callable[[], T]]) -> 'DIContainer':
        """Registriert Service als Scoped (eine Instanz pro Scope)"""
        self._services[interface] = ServiceDescriptor(
            service_type=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED
        )
        logger.debug(f"Registered scoped: {interface.__name__}")
        return self

    def register_factory(self, interface: Type[T], factory: Callable[['DIContainer'], T]) -> 'DIContainer':
        """Registriert Service mit Factory-Funktion"""
        self._services[interface] = ServiceDescriptor(
            service_type=interface,
            implementation=None,
            lifetime=ServiceLifetime.SINGLETON,
            factory=factory
        )
        logger.debug(f"Registered factory: {interface.__name__}")
        return self

    def register_instance(self, interface: Type[T], instance: T) -> 'DIContainer':
        """Registriert bereits existierende Instanz"""
        self._services[interface] = ServiceDescriptor(
            service_type=interface,
            implementation=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._singletons[interface] = instance
        logger.debug(f"Registered instance: {interface.__name__}")
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """Löst Service auf"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} is not registered")

        descriptor = self._services[service_type]

        # Singleton
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]

            instance = self._create_instance(descriptor)
            self._singletons[service_type] = instance
            return instance

        # Scoped
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if self._current_scope is None:
                raise ValueError("No active scope for scoped service")

            scope_services = self._scoped_services.get(self._current_scope, {})

            if service_type in scope_services:
                return scope_services[service_type]

            instance = self._create_instance(descriptor)
            scope_services[service_type] = instance
            self._scoped_services[self._current_scope] = scope_services
            return instance

        # Transient
        else:
            return self._create_instance(descriptor)

    def _create_instance(self, descriptor: ServiceDescriptor):
        """Erstellt Service-Instanz"""
        # Factory-Funktion
        if descriptor.factory:
            return descriptor.factory(self)

        # Bereits existierende Instanz
        if not inspect.isclass(descriptor.implementation):
            return descriptor.implementation

        # Für Transient Services immer neue Instanz erstellen
        if descriptor.lifetime == ServiceLifetime.TRANSIENT:
            # Konstruktor-Injection für neue Instanz
            return self._create_with_injection(descriptor.implementation)

        # Konstruktor-Injection
        return self._create_with_injection(descriptor.implementation)

    def _create_with_injection(self, implementation_class: Type):
        """Erstellt Instanz mit automatischer Dependency Injection"""
        try:
            # Analysiere Konstruktor-Parameter
            signature = inspect.signature(implementation_class.__init__)
            parameters = {}

            for name, param in signature.parameters.items():
                if name == 'self':
                    continue

                # Versuche Parameter aufzulösen
                if param.annotation != inspect.Parameter.empty:
                    try:
                        parameters[name] = self.resolve(param.annotation)
                    except ValueError:
                        # Fallback auf Default-Wert wenn vorhanden
                        if param.default != inspect.Parameter.empty:
                            parameters[name] = param.default
                        else:
                            raise ValueError(f"Cannot resolve parameter '{name}' of type {param.annotation}")
                elif param.default != inspect.Parameter.empty:
                    parameters[name] = param.default

            return implementation_class(**parameters)

        except Exception as e:
            logger.error(f"Failed to create instance of {implementation_class.__name__}: {e}")
            raise

    def create_scope(self, scope_id: str = None) -> 'ServiceScope':
        """Erstellt neuen Service-Scope"""
        if scope_id is None:
            import uuid
            scope_id = str(uuid.uuid4())

        return ServiceScope(self, scope_id)

    def _begin_scope(self, scope_id: str) -> None:
        """Beginnt neuen Scope"""
        self._current_scope = scope_id
        self._scoped_services[scope_id] = {}

    def _end_scope(self, scope_id: str) -> None:
        """Beendet Scope"""
        if scope_id in self._scoped_services:
            del self._scoped_services[scope_id]

        if self._current_scope == scope_id:
            self._current_scope = None

    def is_registered(self, service_type: Type) -> bool:
        """Prüft ob Service registriert ist"""
        return service_type in self._services

    def get_services(self) -> Dict[Type, ServiceDescriptor]:
        """Gibt alle registrierten Services zurück"""
        return self._services.copy()

    def register_ollama_service(
        self,
        config_from_env: bool = True,
        base_url: str = None,
        model: str = None,
        **kwargs
    ) -> 'DIContainer':
        """Registriert Ollama LLM Service"""
        from src.llm_services import OllamaLLMService, OllamaServiceFactory
        from src.ollama_config import OllamaConfigurationManager

        def ollama_factory(container: 'DIContainer') -> OllamaLLMService:
            if config_from_env:
                return OllamaServiceFactory.create_from_env()
            else:
                config_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                if base_url:
                    config_kwargs['base_url'] = base_url
                if model:
                    config_kwargs['model'] = model
                return OllamaServiceFactory.create_with_config(**config_kwargs)

        # Registriere als Factory für ILLMService
        from src.interfaces import ILLMService
        self.register_factory(ILLMService, ollama_factory)
        logger.info("Ollama LLM Service als Factory registriert")
        return self

    def register_rag_services(self, use_ollama: bool = True) -> 'DIContainer':
        """Registriert alle RAG-Services mit Ollama-Integration"""
        if use_ollama:
            self.register_ollama_service(config_from_env=True)

        # Hier können weitere RAG-Services registriert werden
        # Beispiel für weitere Services:
        # self.register_singleton(IVectorStore, ChromaVectorStore)
        # self.register_singleton(IGraphStore, Neo4jGraphStore)

        logger.info(f"RAG Services registriert (Ollama: {use_ollama})")
        return self


class ServiceScope:
    """Service Scope Context Manager"""

    def __init__(self, container: DIContainer, scope_id: str):
        self.container = container
        self.scope_id = scope_id

    def __enter__(self):
        self.container._begin_scope(self.scope_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container._end_scope(self.scope_id)

    def resolve(self, service_type: Type[T]) -> T:
        """Löst Service im aktuellen Scope auf"""
        return self.container.resolve(service_type)


class ServiceLocator:
    """Service Locator Pattern (Alternative zu DI)"""

    _instance: Optional['ServiceLocator'] = None

    def __init__(self):
        self._services: Dict[Type, Any] = {}

    @classmethod
    def get_instance(cls) -> 'ServiceLocator':
        """Singleton Instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_service(self, service_type: Type[T], service: T) -> None:
        """Registriert Service"""
        self._services[service_type] = service

    def get_service(self, service_type: Type[T]) -> T:
        """Gibt Service zurück"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not found")
        return self._services[service_type]

    def has_service(self, service_type: Type) -> bool:
        """Prüft ob Service verfügbar ist"""
        return service_type in self._services


# Decorator für automatische Service-Registrierung

def injectable(lifetime: str = ServiceLifetime.TRANSIENT):
    """Decorator für automatische Service-Registrierung"""
    def decorator(cls):
        cls._injectable_lifetime = lifetime
        return cls
    return decorator


def service_provider(container: DIContainer):
    """Decorator für Service Provider Klassen"""
    def decorator(cls):
        # Automatische Registrierung aller injectables
        provider_instance = cls()
        if hasattr(provider_instance, 'configure_services'):
            provider_instance.configure_services(container)
        return cls
    return decorator


# Configuration functions

def configure_rag_services(container: DIContainer, config) -> None:
    """Konfiguriert alle RAG-Services im Container"""
    try:
        # Use absolute imports instead of relative
        from src.interfaces import ILLMService, IVectorStore, IGraphStore, IRetrievalStrategy
        from src.llm_services import OllamaServiceFactory

        logger.info("Configuring RAG services...")

        # LLM Service
        def llm_factory(cont: DIContainer) -> ILLMService:
            return OllamaServiceFactory.create_default()

        container.register_factory(ILLMService, llm_factory)

        # Mock Vector Store für Development
        class MockVectorStore:
            async def add_documents(self, documents, metadata):
                pass
            async def search_similar(self, query, k=5):
                return []

        container.register_instance(IVectorStore, MockVectorStore())

        # Mock Graph Store für Development
        class MockGraphStore:
            async def add_entities(self, entities):
                pass
            async def add_relationships(self, relationships):
                pass
            async def query_graph(self, query, parameters):
                return []

        container.register_instance(IGraphStore, MockGraphStore())

        logger.info("RAG services configured successfully")

    except ImportError as e:
        logger.warning(f"Could not configure full RAG services: {e}")
        # Configure minimal services for demo
        _configure_minimal_services(container)


def _configure_minimal_services(container: DIContainer) -> None:
    """Konfiguriert minimale Services für Demo"""
    # Mock LLM Service
    class MockLLMService:
        async def generate(self, prompt, context=None):
            return "Mock LLM response"
        async def embed(self, text):
            return [0.1] * 384
        def get_provider_info(self):
            return {"provider": "mock"}

    # Register minimal services
    try:
        from src.interfaces import ILLMService, IVectorStore, IGraphStore
        container.register_instance(ILLMService, MockLLMService())
    except ImportError:
        logger.warning("Using completely minimal configuration")

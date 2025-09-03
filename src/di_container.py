#!/usr/bin/env python3
"""Simple Dependency Injection Container

Nicht-invasiv: fügt grundlegende DI-Funktionalität hinzu ohne bestehende Logik zu ändern.
"""
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional


class DIContainer:
    """Ein sehr einfacher DI-Container mit Singleton/Transient/Instance Support.

    API:
      - register_singleton(key, factory)
      - register_transient(key, factory)
      - register_instance(key, instance)
      - resolve(key)
      - create_scope(name)

    Der Container ist bewusst klein gehalten und eignet sich für Tests und Bootstrapping.
    """

    def __init__(self):
        # factories produce instances; lifetime describes behaviour
        self._singletons: Dict[Any, Any] = {}
        self._singleton_factories: Dict[Any, Callable[[], Any]] = {}
        self._transient_factories: Dict[Any, Callable[[], Any]] = {}
        self._instances: Dict[Any, Any] = {}
        # active scope used by register_scoped; set while in create_scope context
        self._active_scope: Optional['DIContainer'] = None

    def register_singleton(self, key: Any, factory: Callable[[], Any]) -> None:
        """Registriert ein Singleton mit zugehöriger Factory (Lazy instantiation)."""
        self._singleton_factories[key] = factory

    def register_transient(self, key: Any, factory: Callable[[], Any]) -> None:
        """Registriert eine Transient-Factory; jedes resolve ruft die Factory neu auf."""
        self._transient_factories[key] = factory

    def register_instance(self, key: Any, instance: Any) -> None:
        """Registriert eine konkrete Instanz."""
        self._instances[key] = instance

    def resolve(self, key: Any) -> Any:
        """Liefert eine Instanz für den gegebenen Key.

        Reihenfolge: instances -> singletons -> singleton_factories -> transients
        """
        # explicit instance
        if key in self._instances:
            return self._instances[key]

        # already created singleton
        if key in self._singletons:
            return self._singletons[key]

        # create singleton if factory vorhanden
        if key in self._singleton_factories:
            inst = self._singleton_factories[key]()
            self._singletons[key] = inst
            return inst

        # transient
        if key in self._transient_factories:
            return self._transient_factories[key]()

        # No fallback instantiation -- require explicit registration
        raise KeyError(f"No registration for key: {key}")

    @contextmanager
    def create_scope(self, name: Optional[str] = None):
        """Kontextmanager für scopes. Bietet temporäre instance-Registries.

        Nutzung:
            with container.create_scope('req') as scope:
                scope.register_instance(...)
        """
        # create a child container view for instances (shallow)
        child = DIContainer()
        # inherit singleton and transient factories
        child._singleton_factories = dict(self._singleton_factories)
        child._transient_factories = dict(self._transient_factories)
        try:
            # set active scope so register_scoped can target this child
            self._active_scope = child
            yield child
        finally:
            # clear active scope and discard child
            self._active_scope = None
            # child will be garbage collected
            pass

    def register_scoped(self, key: Any, factory: Callable[[], Any]) -> None:
        """Registriert eine scoped Factory. Muss innerhalb eines create_scope-Kontexts aufgerufen werden.

        Die Factory wird im aktuellen Scope als Singleton-Factory registriert.
        """
        if self._active_scope is None:
            raise KeyError("No active scope to register scoped service")

        # register as singleton factory inside the active scope
        self._active_scope.register_singleton(key, factory)

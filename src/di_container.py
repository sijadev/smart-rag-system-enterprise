#!/usr/bin/env python3
"""Simple Dependency Injection Container

Nicht-invasiv: fügt grundlegende DI-Funktionalität hinzu ohne bestehende Logik zu ändern.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional
import threading


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
        # parent container reference for scopes; None for root containers
        self._parent: Optional["DIContainer"] = None
        # when acting as a parent container, track the currently active child
        # scope (used by register_scoped called on the parent)
        self._active_child: Optional["DIContainer"] = None
        # simple re-entrant lock to protect registrations and resolves
        # RLock allows nested resolve calls from factories within the same thread
        self._lock = threading.RLock()
        # teardown callbacks called when a scope exits
        self._teardowns: list[Callable[[], Any]] = []
        # async factory registries
        self._async_singleton_factories: Dict[Any, Callable[[], Any]] = {}
        self._async_singletons: Dict[Any, Any] = {}
        self._async_transient_factories: Dict[Any, Callable[[], Any]] = {}

    def register_singleton(self, key: Any, factory: Callable[[], Any]) -> None:
        """Registriert ein Singleton mit zugehöriger Factory (Lazy instantiation)."""
        with self._lock:
            self._singleton_factories[key] = factory

    def register_transient(self, key: Any, factory: Callable[[], Any]) -> None:
        """Registriert eine Transient-Factory; jedes resolve ruft die Factory neu auf."""
        with self._lock:
            self._transient_factories[key] = factory

    def register_instance(self, key: Any, instance: Any) -> None:
        """Registriert eine konkrete Instanz."""
        with self._lock:
            self._instances[key] = instance

    def register_teardown(self, callback: Callable[[], Any]) -> None:
        """Registriert einen Teardown-Callback, der beim Verlassen eines Scopes aufgerufen wird."""
        with self._lock:
            self._teardowns.append(callback)

    def register_async_singleton(self, key: Any, coro_factory: Callable[[], Any]) -> None:
        """Registriert eine asynchrone Singleton-Factory (KOROUTINE-Funktion)."""
        with self._lock:
            self._async_singleton_factories[key] = coro_factory

    def register_async_transient(self, key: Any, coro_factory: Callable[[], Any]) -> None:
        with self._lock:
            self._async_transient_factories[key] = coro_factory

    def resolve(self, key: Any) -> Any:
        """Liefert eine Instanz für den gegebenen Key.

        Reihenfolge: instances -> singletons -> singleton_factories -> transients
        """
        # First check local synchronous registrations under lock
        with self._lock:
            if key in self._instances:
                return self._instances[key]

            if key in self._singletons:
                return self._singletons[key]

            if key in self._singleton_factories:
                inst = self._singleton_factories[key]()
                self._singletons[key] = inst
                return inst

            if key in self._transient_factories:
                return self._transient_factories[key]()

        # Not found locally; if this is a scoped child, delegate to parent
        if self._parent is not None:
            return self._parent.resolve(key)

        # No fallback instantiation -- require explicit registration
        raise KeyError(f"No registration for key: {key}")

    async def resolve_async(self, key: Any) -> Any:
        """Async-aware resolve: supports async factories registered via
        register_async_singleton/register_async_transient as well as delegating
        to parent scopes."""
        # First check local sync registrations under lock
        with self._lock:
            if key in self._instances:
                return self._instances[key]
            if key in self._singletons:
                return self._singletons[key]
            if key in self._singleton_factories:
                inst = self._singleton_factories[key]()
                self._singletons[key] = inst
                return inst
            if key in self._transient_factories:
                return self._transient_factories[key]()

            # async singleton
            if key in self._async_singletons:
                return self._async_singletons[key]
            if key in self._async_singleton_factories:
                coro = self._async_singleton_factories[key]()
                inst = await coro
                with self._lock:
                    self._async_singletons[key] = inst
                return inst
            # async transient
            if key in self._async_transient_factories:
                coro = self._async_transient_factories[key]()
                return await coro

        # delegate to parent if present
        if self._parent is not None:
            return await self._parent.resolve_async(key)

        raise KeyError(f"No registration for key: {key}")

    @contextmanager
    def create_scope(self, name: Optional[str] = None):
        """Kontextmanager für scopes. Bietet temporäre instance-Registries.

        Nutzung:
            with container.create_scope('req') as scope:
                scope.register_instance(...)
        """
        child = DIContainer()
        # parent link and inherit singleton/transient factories (copy under lock)
        child._parent = self
        with self._lock:
            child._singleton_factories = dict(self._singleton_factories)
            child._transient_factories = dict(self._transient_factories)
            # mark this child as active on the parent while context is active
            self._active_child = child

        # set module-level thread-local active container so module wrapper
        # and parent.register_scoped can detect the active child when using
        # container.create_scope directly.
        prev = getattr(_thread_local, "active_container", None)
        _thread_local.active_container = child

        try:
            yield child
        finally:
            # run teardowns in reverse order
            try:
                # copy teardowns under lock
                with child._lock:
                    tds = list(child._teardowns)
                for cb in reversed(tds):
                    try:
                        res = cb()
                        # if coroutine returned, run it
                        import asyncio

                        if hasattr(res, "__await__"):
                            if asyncio.get_event_loop().is_running():
                                # schedule and don't wait
                                asyncio.create_task(res)
                            else:
                                asyncio.run(res)
                    except Exception:
                        pass
            finally:
                # clear parent link and active child to avoid accidental retention
                with self._lock:
                    self._active_child = None
                child._parent = None
                # restore previous thread-local active container
                _thread_local.active_container = prev

    def register_scoped(self, key: Any, factory: Callable[[], Any]) -> None:
        """Registriert eine scoped Factory. Muss innerhalb eines create_scope-Kontexts aufgerufen werden.

        Die Factory wird im aktuellen Scope als Singleton-Factory registriert.
        """
        # If called on a parent container, register into the currently active child.
        if self._parent is None:
            with self._lock:
                child = self._active_child
                if child is None:
                    raise KeyError("No active scope to register scoped service")
                # register into child's singleton factories
                child.register_singleton(key, factory)
            return

        # Called on a scoped child: register locally
        with self._lock:
            self._singleton_factories[key] = factory


# --- Runtime global container & helper API ---------------------------------
# Provide a default container instance that can be used throughout the
# application for runtime registration and resolution of services.
# Use `from src.di_container import get_container, resolve, register_singleton` etc.

_default_container: DIContainer = DIContainer()
_module_lock = threading.RLock()
_thread_local = threading.local()


def _get_effective_container() -> DIContainer:
    """Return the container that should be used for the current thread/context.

    Preference order: thread-local active container (scope) -> global default container.
    """
    active = getattr(_thread_local, "active_container", None)
    if active is not None:
        return active
    with _module_lock:
        return _default_container


def get_container() -> DIContainer:
    """Returns the global DI container used at runtime.

    The function acquires a module-level lock briefly to ensure a consistent
    container instance is observed when called concurrently with use_container().
    """
    with _module_lock:
        return _default_container


def use_container(container: DIContainer) -> None:
    """Replace the global container (useful for testing or advanced bootstrapping).

    This operation is guarded by a module-level lock to avoid races with
    concurrent wrapper calls that fetch the global container.
    """
    global _default_container
    with _module_lock:
        _default_container = container


def register_singleton(key: Any, factory: Callable[[], Any]) -> None:
    c = _get_effective_container()
    c.register_singleton(key, factory)


def register_transient(key: Any, factory: Callable[[], Any]) -> None:
    c = _get_effective_container()
    c.register_transient(key, factory)


def register_instance(key: Any, instance: Any) -> None:
    c = _get_effective_container()
    c.register_instance(key, instance)


def resolve(key: Any) -> Any:
    c = _get_effective_container()
    return c.resolve(key)


async def resolve_async(key: Any) -> Any:
    c = _get_effective_container()
    return await c.resolve_async(key)


@contextmanager
def create_scope(name: Optional[str] = None):
    """Convenience wrapper exposing scope creation on the global container.

    Usage:
        with create_scope('request') as scope:
            scope.register_instance(...)
    """
    # Create a child scope from the current effective container and bind it
    # to the current thread while the context is active so module-level
    # resolve/register functions operate on the scope.
    parent = _get_effective_container()
    with parent.create_scope(name) as scope:
        prev = getattr(_thread_local, "active_container", None)
        _thread_local.active_container = scope
        try:
            yield scope
        finally:
            _thread_local.active_container = prev


def register_scoped(key: Any, factory: Callable[[], Any]) -> None:
    """Register a scoped service into the currently active scope of the global container."""
    c = _get_effective_container()
    c.register_scoped(key, factory)


__all__ = [
    "DIContainer",
    "get_container",
    "use_container",
    "register_singleton",
    "register_transient",
    "register_instance",
    "register_scoped",
    "resolve",
    "create_scope",
]

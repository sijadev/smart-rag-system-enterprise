#!/usr/bin/env python3
"""Neo4j adapter implementing IGraphStore

Lightweight adapter: uses neo4j-driver when available, otherwise provides an in-memory
fallback implementation compatible with IGraphStore for testing and bootstrapping.
"""

import os
import json
from typing import Any, Dict, List, Optional

from ..interfaces import IGraphStore


class Neo4jAdapter(IGraphStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Wenn keine Konfiguration übergeben wird, nutze zentrale Konfiguration
        if config is None:
            try:
                from src.central_config import get_config

                central_db = get_config().database
                config = {
                    "uri": getattr(central_db, "neo4j_uri", None),
                    "neo4j_uri": getattr(central_db, "neo4j_uri", None),
                    "user": getattr(central_db, "neo4j_user", None),
                    "neo4j_user": getattr(central_db, "neo4j_user", None),
                    "password": getattr(central_db, "neo4j_password", None),
                    "neo4j_password": getattr(central_db, "neo4j_password", None),
                    "database": getattr(central_db, "neo4j_database", None),
                }
            except Exception as e:
                config = {}
        self.config = config or {}
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Neo4j-Konfiguration: uri={self.config.get('uri') or self.config.get('neo4j_uri')}, user={self.config.get('user') or self.config.get('neo4j_user')}"
        )
        self._mode = self.config.get("mode", "production")  # NEU: Modus setzen
        self._use_driver = False
        self._driver = None
        # Connection / retry configuration
        self._connect_retries = int(self.config.get("connect_retries", 3))
        self._connect_backoff = float(self.config.get("connect_backoff", 2.0))
        self._connect_timeout = float(self.config.get("connect_timeout", 10.0))
        try:
            # Prefer the async driver when available (neo4j v4+/v5+)
            from neo4j import AsyncGraphDatabase  # type: ignore

            uri = (
                self.config.get("uri")
                or self.config.get("neo4j_uri")
                or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            )
            user = (
                self.config.get("user")
                or self.config.get("neo4j_user")
                or os.environ.get("NEO4J_USER", "neo4j")
            )
            password = (
                self.config.get("password")
                or self.config.get("neo4j_password")
                or os.environ.get("NEO4J_PASSWORD")
            )
            # don't create driver synchronously here; defer to ensure_driver
            # for retries
            self._async_driver_factory = (AsyncGraphDatabase, uri, (user, password))
            # initial state stays False until ensure_driver is called
            self._use_driver = False
        except Exception:
            # Treiber ist verpflichtend; kein In-Memory-Fallback mehr
            raise RuntimeError("Neo4j-Driver nicht installiert oder importierbar. Bitte installiere das 'neo4j' Paket.")

    async def ensure_driver(self) -> bool:
        """Ensure async driver is available and can connect. Returns True if ready."""
        if self._use_driver and self._driver is not None:
            return True

        try:
            from neo4j import AsyncGraphDatabase  # type: ignore
        except Exception:
            raise RuntimeError("Neo4j-Driver nicht installiert oder importierbar.")

        uri = (
            self.config.get("uri")
            or self.config.get("neo4j_uri")
            or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        )
        user = (
            self.config.get("user")
            or self.config.get("neo4j_user")
            or os.environ.get("NEO4J_USER", "neo4j")
        )
        password = (
            self.config.get("password")
            or self.config.get("neo4j_password")
            or os.environ.get("NEO4J_PASSWORD")
        )

        if not password:
            raise RuntimeError("Neo4j-Passwort ist nicht gesetzt. Setze NEO4J_PASSWORD oder konfiguriere das Passwort in der Config.")

        attempt = 0
        while attempt < self._connect_retries:
            try:
                self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
                # quick connectivity probe
                async with self._driver.session() as session:
                    result = await session.run("RETURN 1 AS v")
                    record = await result.single()
                    if record is not None:
                        self._use_driver = True
                        return True
            except Exception:
                attempt += 1
                await __import__("asyncio").sleep(self._connect_backoff**attempt)

        # failed to connect
        self._use_driver = False
        return False

    async def add_entities(self, entities: List[Dict[str, Any]]) -> None:
        if not await self.ensure_driver():
            raise RuntimeError("Neo4j-Driver nicht verfügbar!")
        async with self._driver.session() as session:

            async def _tx(tx):
                for e in entities:
                    props = self._sanitize_props(e if isinstance(e, dict) else {"value": e})
                    await tx.run("CREATE (n:Entity $props)", props=props)

            await session.execute_write(_tx)

    async def add_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        if not await self.ensure_driver():
            raise RuntimeError("Neo4j-Driver nicht verfügbar!")
        async with self._driver.session() as session:

            async def _tx(tx):
                for r in relationships:
                    # Stelle sicher, dass a/b primitive Typen sind
                    a = r.get("a")
                    b = r.get("b")
                    if not isinstance(a, (str, int, float, bool)):
                        a = str(a)
                    if not isinstance(b, (str, int, float, bool)):
                        b = str(b)
                    await tx.run(
                        "MERGE (a {id: $a}) MERGE (b {id: $b}) MERGE (a)-[:REL]->(b)",
                        a=a,
                        b=b,
                    )

            await session.execute_write(_tx)

    async def query_graph(
        self, query: str, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not await self.ensure_driver():
            raise RuntimeError("Neo4j-Driver nicht verfügbar!")
        async with self._driver.session() as session:

            # einfache Heuristik: wenn das Query Schreib-Keywords enthält, führe es in einem Schreib-Transaction aus
            q_upper = (query or "").upper()
            write_keywords = ("CREATE", "MERGE", "DELETE", "SET", "DETACH", "REMOVE", "LOAD CSV", "UNWIND", "CALL")
            is_write = any(k in q_upper for k in write_keywords)

            params = self._sanitize_props(parameters or {})

            async def _run_read(tx):
                result = await tx.run(query, **params)
                records = []
                async for rec in result:
                    try:
                        records.append(rec.data())
                    except Exception:
                        records.append(dict(rec))
                return records

            async def _run_write(tx):
                result = await tx.run(query, **params)
                # manchmal liefert ein Schreib-Query keine Records; konsistent zurückgeben
                records = []
                async for rec in result:
                    try:
                        records.append(rec.data())
                    except Exception:
                        records.append(dict(rec))
                return records

            if is_write:
                rows = await session.execute_write(_run_write)
            else:
                rows = await session.execute_read(_run_read)

            return rows

    # Hilfsmethode: stellt sicher, dass alle Properties primitive Typen oder Arrays davon sind
    def _sanitize_props(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Konvertiert verschachtelte Maps/Objekte in JSON-Strings und entfernt None-Werte.

        Neo4j-Properties dürfen nur Primitive (str, int, float, bool) oder Arrays dieser Typen
        sein. Diese Methode macht eine defensive Umwandlung und wird vor allen db-Schreib-/Leseaufrufen angewendet.
        """
        sanitized: Dict[str, Any] = {}
        for k, v in (props or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, list):
                if all(isinstance(x, (str, int, float, bool)) for x in v):
                    sanitized[k] = v
                else:
                    try:
                        sanitized[k] = json.dumps(v, ensure_ascii=False, default=str)
                    except Exception:
                        sanitized[k] = str(v)
            elif isinstance(v, dict):
                try:
                    sanitized[k] = json.dumps(v, ensure_ascii=False, default=str)
                except Exception:
                    sanitized[k] = str(v)
            else:
                try:
                    sanitized[k] = str(v)
                except Exception:
                    sanitized[k] = json.dumps(v, ensure_ascii=False, default=str)
        return sanitized

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Erstellt Document-Knoten aus der Dokumentliste und wendet Sanitizing an."""
        if not await self.ensure_driver():
            raise RuntimeError("Neo4j-Driver nicht verfügbar!")
        async with self._driver.session() as session:

            async def _tx(tx):
                for d in documents:
                    props = dict(d) if isinstance(d, dict) else {"content": d}
                    props = self._sanitize_props(props)
                    await tx.run("CREATE (n:Document $props)", props=props)

            await session.execute_write(_tx)

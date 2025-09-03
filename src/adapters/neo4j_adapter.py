#!/usr/bin/env python3
"""Neo4j adapter implementing IGraphStore

Lightweight adapter: uses neo4j-driver when available, otherwise provides an in-memory
fallback implementation compatible with IGraphStore for testing and bootstrapping.
"""
from typing import Any, Dict, List, Optional
import os
from ..interfaces import IGraphStore


class Neo4jAdapter(IGraphStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._use_driver = False
        self._nodes: List[Dict[str, Any]] = []
        self._relationships: List[Dict[str, Any]] = []
        try:
            from neo4j import GraphDatabase  # type: ignore
            self._use_driver = True
            uri = self.config.get('uri') or self.config.get('neo4j_uri') or 'bolt://localhost:7687'
            uri = self.config.get('uri') or self.config.get('neo4j_uri') or os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
            user = self.config.get('user') or self.config.get('neo4j_user') or os.environ.get('NEO4J_USER', 'neo4j')
            password = self.config.get('password') or self.config.get('neo4j_password') or os.environ.get('NEO4J_PASSWORD')
                # don't attempt real connection without password in bootstrap
                self._use_driver = False
            else:
                self._driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception:
            # driver not available or connection info missing - fallback to in-memory
            self._use_driver = False

    async def add_entities(self, entities: List[Dict[str, Any]]) -> None:
        if self._use_driver:
            # Real implementation would run a write transaction
            def _tx(tx):
                for e in entities:
                    tx.run("CREATE (n:Entity $props)", props=e)
            with self._driver.session() as session:
                session.write_transaction(_tx)
        else:
            for e in entities:
                nid = f'node_{len(self._nodes)}'
                node = {'id': nid, **e}
                self._nodes.append(node)

    async def add_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        if self._use_driver:
            def _tx(tx):
                for r in relationships:
                    tx.run("MERGE (a {id: $a}) MERGE (b {id: $b}) MERGE (a)-[:REL]->(b)", a=r.get('a'), b=r.get('b'))
            with self._driver.session() as session:
                session.write_transaction(_tx)
        else:
            for r in relationships:
                self._relationships.append(r)

    async def query_graph(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._use_driver:
            with self._driver.session() as session:
                res = session.run(query)
                return [dict(record) for record in res]
        else:
            # naive search over in-memory nodes
            terms = []
            if isinstance(parameters, dict):
                for v in parameters.values():
                    if isinstance(v, (list, tuple)):
                        terms.extend([str(x).lower() for x in v])
                    else:
                        terms.append(str(v).lower())
            results = []
            for n in self._nodes:
                content = ' '.join([str(n.get(k, '')).lower() for k in n.keys()])
                score = 1.0 if any(t in content for t in terms) else 0.5
                results.append({'id': n.get('id'), 'content': n.get('content', ''), 'relevance_score': score})
            results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
            return results[: parameters.get('limit', 5) if isinstance(parameters, dict) else 5]

    async def get_real_data_statistics(self) -> Dict[str, int]:
        if self._use_driver:
            # Placeholder: real implementation would query counts
            return {'total_entities': 0, 'total_relationships': 0}
        else:
            return {'total_entities': len(self._nodes), 'total_relationships': len(self._relationships)}



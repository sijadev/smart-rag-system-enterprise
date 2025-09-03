#!/usr/bin/env python3
"""Mock Graph Store adapter implementing IGraphStore

Used for tests and as a safe default registration.
"""
from typing import Any, Dict, List, Optional
from ..interfaces import IGraphStore


class MockGraphStore(IGraphStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._nodes: List[Dict[str, Any]] = []
        self._relationships: List[Dict[str, Any]] = []

    async def add_entities(self, entities: List[Dict[str, Any]]) -> None:
        for e in entities:
            nid = f'node_{len(self._nodes)}'
            node = {'id': nid, **e}
            self._nodes.append(node)

    async def add_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        for r in relationships:
            self._relationships.append(r)

    async def query_graph(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        # naive implementation: return nodes containing any parameter tokens
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
        limit = parameters.get('limit', 5) if isinstance(parameters, dict) else 5
        return results[:limit]

    async def get_real_data_statistics(self) -> Dict[str, int]:
        return {
            'total_entities': len(self._nodes),
            'total_relationships': len(self._relationships)
        }


# For compatibility with older imports
MockGraphStore = MockGraphStore


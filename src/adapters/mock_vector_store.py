#!/usr/bin/env python3
"""Mock Vector Store adapter implementing IVectorStore

Used for tests and as a safe default registration.
"""
from typing import Any, Dict, List, Optional
from ..interfaces import IVectorStore


class MockVectorStore(IVectorStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._docs: List[Dict[str, Any]] = []

    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        for i, doc in enumerate(documents):
            self._docs.append({'id': f'doc_{len(self._docs)}', 'content': doc, 'metadata': metadata[i] if i < len(metadata) else {}})

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # naive substring match for deterministic behavior
        results = []
        for doc in self._docs:
            score = 1.0 if query.lower() in doc['content'].lower() else 0.5
            results.append({'id': doc['id'], 'content': doc['content'], 'score': score})
        # sort by score desc
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:k]

    async def delete_documents(self, doc_ids: List[str]) -> None:
        self._docs = [d for d in self._docs if d['id'] not in doc_ids]


# For compatibility with older tests expecting class MockVectorStore at top-level
MockVectorStore = MockVectorStore


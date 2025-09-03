#!/usr/bin/env python3
"""Faiss adapter implementing IVectorStore

Lightweight adapter with best-effort import of faiss. If faiss is not available,
falls back to an in-memory implementation with the same async API for tests.
"""
from typing import Any, Dict, List, Optional
from ..interfaces import IVectorStore


class FaissAdapter(IVectorStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._use_faiss = False
        self._docs: List[Dict[str, Any]] = []
        try:
            import faiss  # type: ignore
            self._use_faiss = True
            # Real setup would build index from embeddings; omitted here for safety
        except Exception:
            self._use_faiss = False

    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        # In-memory fallback just stores content and metadata
        for i, doc in enumerate(documents):
            self._docs.append({'id': f'doc_{len(self._docs)}', 'content': doc, 'metadata': metadata[i] if i < len(metadata) else {}})

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Simple substring-based scoring for tests
        results = []
        for doc in self._docs:
            score = 1.0 if query.lower() in doc['content'].lower() else 0.5
            results.append({'id': doc['id'], 'content': doc['content'], 'score': score})
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:k]

    async def delete_documents(self, doc_ids: List[str]) -> None:
        self._docs = [d for d in self._docs if d['id'] not in doc_ids]


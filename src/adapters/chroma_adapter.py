#!/usr/bin/env python3
"""Chroma adapter implementing IVectorStore with best-effort import.

If chromadb is available the adapter will wrap it; otherwise falls back to an
in-memory store with compatible async methods for tests and bootstrapping.
"""
from typing import Any, Dict, List, Optional
from ..interfaces import IVectorStore


class ChromaAdapter(IVectorStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._use_chromadb = False
        self._docs: List[Dict[str, Any]] = []
        try:
            import chromadb  # type: ignore
            # If available, set up a minimal client (best-effort)
            self._use_chromadb = True
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(self.config.get('collection_name', 'rag_documents'))
        except Exception:
            self._use_chromadb = False

    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        if self._use_chromadb:
            # best-effort: insert documents into chroma collection
            ids = [f'doc_{i}' for i in range(len(documents))]
            self._collection.add(documents=documents, metadatas=metadata, ids=ids)
        else:
            for i, doc in enumerate(documents):
                self._docs.append({'id': f'doc_{len(self._docs)}', 'content': doc, 'metadata': metadata[i] if i < len(metadata) else {}})

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._use_chromadb:
            # best-effort: use simple query
            results = self._collection.query(query_texts=[query], n_results=k)
            # normalize into expected dicts
            out = []
            for i, docs in enumerate(results.get('documents', [[]])):
                for j, d in enumerate(docs):
                    out.append({'id': results.get('ids', [[]])[i][j], 'content': d, 'score': 1.0})
            return out[:k]
        else:
            results = []
            for doc in self._docs:
                score = 1.0 if query.lower() in doc['content'].lower() else 0.5
                results.append({'id': doc['id'], 'content': doc['content'], 'score': score})
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            return results[:k]

    async def delete_documents(self, doc_ids: List[str]) -> None:
        if self._use_chromadb:
            try:
                self._collection.delete(ids=doc_ids)
            except Exception:
                pass
        else:
            self._docs = [d for d in self._docs if d['id'] not in doc_ids]


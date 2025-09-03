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
        self._collection = None
        try:
            import chromadb  # type: ignore
            # If available, set up a minimal client (best-effort)
            self._use_chromadb = True
            self._client = chromadb.Client()
            collection_name = self.config.get('collection_name', 'rag_documents')
            # use get_or_create_collection if available
            try:
                self._collection = self._client.get_or_create_collection(collection_name)
            except Exception:
                # fallback to direct collection retrieval/creation
                try:
                    self._collection = self._client.create_collection(name=collection_name)
                except Exception:
                    self._collection = None
        except Exception:
            self._use_chromadb = False

    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        # wrapper that delegates to batch_add_documents
        await self.batch_add_documents(documents, metadata)

    async def batch_add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Fügt Dokumente als Batch hinzu. Einheitliche API für Chroma / in-memory fallback."""
        meta = metadata or [{} for _ in documents]
        if self._use_chromadb and self._collection is not None:
            # best-effort: provide ids and metadatas
            ids = [f"doc_{i}" for i in range(len(documents))]
            try:
                # chroma's API may vary; use add if present
                if hasattr(self._collection, 'add'):
                    self._collection.add(documents=documents, metadatas=meta, ids=ids)
                else:
                    # fallback behavior: try collection.upsert
                    if hasattr(self._collection, 'upsert'):
                        self._collection.upsert(ids=ids, documents=documents, metadatas=meta)
            except Exception:
                # degrade to in-memory
                for i, doc in enumerate(documents):
                    self._docs.append({'id': f'doc_{len(self._docs)}', 'content': doc, 'metadata': meta[i]})
        else:
            for i, doc in enumerate(documents):
                self._docs.append({'id': f'doc_{len(self._docs)}', 'content': doc, 'metadata': meta[i] if i < len(meta) else {}})

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._use_chromadb and self._collection is not None:
            try:
                # Use query API if available
                res = None
                if hasattr(self._collection, 'query'):
                    res = self._collection.query(query_texts=[query], n_results=k)
                    # normalize
                    docs = []
                    documents = res.get('documents', [[]])[0] if isinstance(res, dict) else []
                    ids = res.get('ids', [[]])[0] if isinstance(res, dict) else []
                    scores = res.get('distances', [[]])[0] if isinstance(res, dict) else []
                    for i, d in enumerate(documents):
                        docs.append({'id': ids[i] if i < len(ids) else f'doc_{i}', 'content': d, 'score': float(scores[i]) if i < len(scores) else 1.0})
                    return docs[:k]
                # fallback: empty
                return []
            except Exception:
                # degrade to in-memory search
                pass
        # in-memory fallback
        results = []
        q = query.lower()
        for doc in self._docs:
            content = doc.get('content', '')
            score = 1.0 if q in content.lower() else 0.5
            results.append({'id': doc['id'], 'content': content, 'score': score})
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:k]

    async def delete_documents(self, doc_ids: List[str]) -> None:
        if self._use_chromadb and self._collection is not None:
            try:
                if hasattr(self._collection, 'delete'):
                    self._collection.delete(ids=doc_ids)
                    return
            except Exception:
                pass
        # fallback
        self._docs = [d for d in self._docs if d['id'] not in doc_ids]

    # Persistence helpers (non-breaking, best-effort)
    def persist_collection(self, path: str) -> None:
        """Best-effort persist collection metadata/state for in-memory fallback."""
        try:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._docs, f)
        except Exception:
            pass

    def load_collection(self, path: str) -> None:
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                self._docs = json.load(f)
        except Exception:
            pass


# For compatibility with older tests expecting class name at top-level
ChromaAdapter = ChromaAdapter

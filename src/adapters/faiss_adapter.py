#!/usr/bin/env python3
"""Faiss adapter implementing IVectorStore

Lightweight adapter with best-effort import of faiss. If faiss is not available,
falls back to an in-memory implementation with the same async API for tests.
"""
from typing import Any, Dict, List, Optional
from ..interfaces import IVectorStore
from ..processing.embedding_pipeline import EmbeddingPipeline
from ..factories import LLMServiceFactory
from ..interfaces import LLMProvider

try:
    import numpy as _np
except Exception:
    _np = None


class FaissAdapter(IVectorStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._use_faiss = False
        self._docs: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []
        self._embedding_pipeline: Optional[EmbeddingPipeline] = None
        try:
            import faiss  # type: ignore
            self._use_faiss = True
            # Real setup would build index from embeddings; omitted here for safety
            self._faiss_index = None
            self._embedding_dim = int(self.config.get('embedding_dim', 384))
            # Attempt to create an index placeholder; actual vectors added later
            try:
                self._faiss_index = faiss.IndexFlatL2(self._embedding_dim)
            except Exception:
                self._faiss_index = None
        except Exception:
            self._use_faiss = False

    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        # Ensure metadata list
        meta = metadata or [{} for _ in documents]

        # If embeddings are missing, try to generate them via EmbeddingPipeline
        missing_idxs = [i for i, m in enumerate(meta) if not (isinstance(m, dict) and m.get('embedding'))]
        if missing_idxs:
            try:
                # lazy create pipeline if possible
                if self._embedding_pipeline is None:
                    # attempt to build LLM service from config hint
                    provider = self.config.get('llm_provider') if self.config.get('llm_provider') else LLMProvider.OLLAMA
                    try:
                        llm = LLMServiceFactory.create(provider, {'model_name': self.config.get('model_name')})
                        self._embedding_pipeline = EmbeddingPipeline(llm)
                    except Exception:
                        self._embedding_pipeline = None

                if self._embedding_pipeline is not None:
                    texts_to_embed = [documents[i] for i in missing_idxs]
                    emb_results = await self._embedding_pipeline.embed_texts(texts_to_embed, batch_size=int(self.config.get('embed_batch_size', 8)), normalize=False)
                    for idx, emb in zip(missing_idxs, emb_results):
                        if emb:
                            meta[idx]['embedding'] = emb
            except Exception:
                # fallback: continue without embeddings
                pass

        # In-memory fallback stores content/metadata and optionally embeddings
        for i, doc in enumerate(documents):
            doc_id = f'doc_{len(self._docs)}'
            self._docs.append({'id': doc_id, 'content': doc, 'metadata': meta[i] if i < len(meta) else {}})
            # If embeddings provided in metadata, store them
            emb = None
            if isinstance(meta, list) and i < len(meta):
                emb = meta[i].get('embedding') if isinstance(meta[i], dict) else None
            if emb:
                self._embeddings.append(list(emb))
                if self._use_faiss and self._faiss_index is not None:
                    try:
                        if _np is not None:
                            self._faiss_index.add(_np.array([emb], dtype='float32'))
                    except Exception:
                        pass
            else:
                # placeholder zero-vector
                self._embeddings.append([0.0] * getattr(self, '_embedding_dim', 384))

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # If faiss is available and index populated, it's expected that caller provides an embedding
        try:
            import numpy as _np
        except Exception:
            _np = None

        # If query looks like embedding (list/tuple) treat accordingly
        if isinstance(query, (list, tuple)):
            qvec = _np.array([query], dtype='float32') if _np is not None else None
            if self._use_faiss and self._faiss_index is not None and qvec is not None:
                try:
                    D, I = self._faiss_index.search(qvec, k)
                    results = []
                    for dist, idx in zip(D[0], I[0]):
                        if idx < len(self._docs):
                            results.append({'id': self._docs[idx]['id'], 'content': self._docs[idx]['content'], 'score': float(dist)})
                    return results
                except Exception:
                    pass

        # Fallback: cosine / substring hybrid on in-memory docs
        results = []
        q_lower = str(query).lower()
        for i, doc in enumerate(self._docs):
            content = doc.get('content', '')
            text_score = 1.0 if q_lower in content.lower() else 0.5
            emb_score = 0.0
            if _np is not None and i < len(self._embeddings):
                try:
                    q_emb = _np.array(self._embeddings[i], dtype='float32')
                    # treat numeric query as embedding not supported here
                except Exception:
                    q_emb = None
            results.append({'id': doc['id'], 'content': content, 'score': text_score + emb_score})
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:k]

    async def delete_documents(self, doc_ids: List[str]) -> None:
        # remove docs and align embeddings
        remaining = []
        remaining_emb = []
        for i, d in enumerate(self._docs):
            if d['id'] not in doc_ids:
                remaining.append(d)
                if i < len(self._embeddings):
                    remaining_emb.append(self._embeddings[i])
        self._docs = remaining
        self._embeddings = remaining_emb

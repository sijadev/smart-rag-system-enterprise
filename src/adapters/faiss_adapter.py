#!/usr/bin/env python3
"""Faiss-backed Vector Store adapter implementing IVectorStore.

Best-effort: uses faiss (if installed) and numpy; otherwise falls back to
an in-memory Python implementation with linear search. Embeddings are
obtained by calling an embedder: either a callable provided in config as
"embedder" or by resolving ILLMService from DI and calling embed().
"""
import asyncio
import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

from src.di_container import resolve
from src.interfaces import ILLMService, IVectorStore

logger = logging.getLogger(__name__)


class FaissAdapter(IVectorStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._embedder = self.config.get("embedder")  # optional callable
        # support both 'embedding_dim' and legacy 'dimension' keys
        self._dimension = int(self.config.get("embedding_dim", self.config.get("dimension", 384)))
        self._ids: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._vectors: List[List[float]] = []
        # provide alias expected by tests
        self._embeddings = self._vectors
        # store original docs/records for persistence/roundtrip tests
        self._docs: List[Dict[str, Any]] = []

        # Try to import faiss & numpy
        try:
            import faiss  # type: ignore
            import numpy as _np  # type: ignore

            self._np = _np
            self._faiss = faiss
            self._use_faiss = True
            # create an index placeholder
            self._index = None
        except Exception:
            self._use_faiss = False
            self._np = None
            self._faiss = None
            self._index = None
            logger.info("Faiss not available; using pure-python fallback store")

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if callable(self._embedder):
            # embedder may be sync or async
            res = self._embedder(texts)
            if asyncio.iscoroutine(res):
                return await res
            return res

        # fallback: try to resolve an ILLMService and use .embed / embed_texts
        try:
            llm: ILLMService = resolve(ILLMService)
            # prefer embed_texts if available
            if hasattr(llm, "embed_texts"):
                res = await llm.embed_texts(texts)
            else:
                # call embed per text
                res = []
                for t in texts:
                    emb = await llm.embed(t)
                    res.append(emb)
            # normalize to list-of-lists
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                return res
            if isinstance(res, list):
                # single embedding returned
                return [res]
        except Exception:
            pass

        # final fallback: deterministic pseudo-embeddings
        vecs = []
        for text in texts:
            vec = [float((ord(c) % 100) / 100.0) for c in (text[: self._dimension] or "")]
            if len(vec) < self._dimension:
                vec.extend([0.0] * (self._dimension - len(vec)))
            vecs.append(vec[: self._dimension])
        return vecs

    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        # compute embeddings for documents, but prefer any embedding provided
        # in the metadata under the key 'embedding'. This allows tests to
        # supply deterministic embeddings without contacting an embedder.
        emb = await self._get_embeddings(documents)
        # store ids
        for i, doc in enumerate(documents):
            new_id = f"doc_{len(self._ids)}"
            self._ids.append(new_id)
            meta = metadata[i] if i < len(metadata) and isinstance(metadata[i], dict) and len(metadata[i]) > 0 else {"content": doc}
            self._metadatas.append(meta)
            # use embedded vector from metadata if present
            vec = None
            if isinstance(meta, dict) and "embedding" in meta and isinstance(meta["embedding"], list):
                vec = [float(x) for x in meta["embedding"]]
            else:
                vec = emb[i]
            # ensure vector has correct dimensionality
            if len(vec) < self._dimension:
                vec = list(vec) + [0.0] * (self._dimension - len(vec))
            self._vectors.append(vec)
            # maintain docs list for persistence/roundtrip tests
            try:
                self._docs.append({"id": new_id, "content": doc, "metadata": meta})
            except Exception:
                # defensive: ensure _docs remains a list
                pass

        # ensure alias remains pointing to the same list object
        self._embeddings = self._vectors

        # build or rebuild faiss index
        if self._use_faiss:
            self._rebuild_index()

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = await self._get_embeddings([query])
        q = q_emb[0]
        if self._use_faiss and self._index is not None and len(self._vectors) > 0:
            import numpy as np

            v = np.array([q], dtype=np.float32)
            # normalize for cosine
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            v = v / norms
            D, I = self._index.search(v, min(k, len(self._vectors)))
            results = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0 or idx >= len(self._ids):
                    continue
                meta = self._metadatas[idx]
                results.append({"id": self._ids[idx], "content": meta.get("content", ""), "score": float(score), "metadata": meta})
            return results

        # fallback linear search
        best = []
        for idx, vec in enumerate(self._vectors):
            # cosine similarity
            dot = sum(a * b for a, b in zip(q, vec))
            norm_q = sum(a * a for a in q) ** 0.5
            norm_v = sum(a * a for a in vec) ** 0.5
            score = dot / (norm_q * norm_v) if norm_q and norm_v else 0.0
            best.append((score, idx))
        best.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, idx in best[:k]:
            meta = self._metadatas[idx]
            results.append({"id": self._ids[idx], "content": meta.get("content", ""), "score": float(score), "metadata": meta})
        return results

    async def delete_documents(self, doc_ids: List[str]):
        # remove by id and rebuild index
        id_set = set(doc_ids)
        keep_ids = []
        keep_meta = []
        keep_vecs = []
        keep_docs = []
        for i, did in enumerate(self._ids):
            if did in id_set:
                continue
            keep_ids.append(did)
            keep_meta.append(self._metadatas[i])
            keep_vecs.append(self._vectors[i])
            # keep docs in same order
            if i < len(self._docs):
                keep_docs.append(self._docs[i])
        self._ids = keep_ids
        self._metadatas = keep_meta
        self._vectors = keep_vecs
        self._docs = keep_docs
        # keep alias in sync
        self._embeddings = self._vectors
        if self._use_faiss:
            self._rebuild_index()

    def _rebuild_index(self):
        try:
            import faiss
            import numpy as np

            if len(self._vectors) == 0:
                # empty index
                self._index = faiss.IndexFlatIP(self._dimension)
                return

            arr = np.array(self._vectors, dtype=np.float32)
            # normalize rows
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(arr)
        except Exception as e:
            logger.warning(f"Faiss index rebuild failed: {e}")
            self._use_faiss = False
            self._index = None

    def update_index_safe(self) -> bool:
        """Attempt to rebuild the underlying index. Returns True when the
        adapter successfully has a usable faiss index (or False otherwise).
        This is used by tests to trigger a safe index rebuild after loading
        persisted vectors.
        """
        try:
            # If faiss is available, try to rebuild; otherwise return False
            if not self._use_faiss:
                # try to import faiss now
                try:
                    import faiss  # type: ignore
                    self._faiss = faiss
                    self._use_faiss = True
                except Exception:
                    return False

            # call rebuild and check index
            self._rebuild_index()
            return self._index is not None
        except Exception:
            logger.exception("update_index_safe failed")
            return False

    # Persistence and index management helpers
    def save_index(self, path: str) -> bool:
        """Persist faiss index to disk (best-effort). Returns True if saved."""
        try:
            logger.debug("save_index: attempt saving index to %s", path)
            if not self._use_faiss or self._index is None:
                logger.info("save_index: faiss not available or index is empty")
                return False
            import faiss  # type: ignore

            faiss.write_index(self._index, path)
            logger.info("save_index: wrote faiss index to %s", path)
            return True
        except Exception as e:
            logger.error("save_index failed: %s\n%s", e, traceback.format_exc())
            return False

    def load_index(self, path: str) -> bool:
        """Load faiss index from disk (best-effort). Returns True if loaded."""
        try:
            logger.debug("load_index: loading index from %s", path)
            import faiss  # type: ignore

            idx = faiss.read_index(path)
            self._index = idx
            self._use_faiss = True
            logger.info("load_index: loaded faiss index from %s", path)
            return True
        except Exception as e:
            logger.error("load_index failed: %s\n%s", e, traceback.format_exc())
            return False

    def persist_state(self, path_prefix: str) -> bool:
        """Persist docs+embeddings as JSON and optionally the faiss index.
        path_prefix is used to write path_prefix+'.meta.json' and path_prefix+'.index'
        """
        try:
            logger.debug(
                "persist_state: persisting metadata to %s.meta.json", path_prefix
            )
            meta_path = f"{path_prefix}.meta.json"
            # write both keys for compatibility
            payload = {"ids": self._ids, "metadatas": self._metadatas, "vectors": self._vectors, "embeddings": self._embeddings, "docs": self._docs}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            logger.info("persist_state: wrote meta to %s", meta_path)
            if self._use_faiss and self._index is not None:
                idx_path = f"{path_prefix}.index"
                ok = self.save_index(idx_path)
                logger.info("persist_state: save_index returned %s", ok)
                return ok
            return True
        except Exception as e:
            logger.error("persist_state failed: %s\n%s", e, traceback.format_exc())
            return False

    def load_state(self, path_prefix: str) -> bool:
        """Load meta JSON and faiss index if present."""
        try:
            meta_path = f"{path_prefix}.meta.json"
            logger.debug("load_state: loading meta from %s", meta_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._ids = meta.get("ids", [])
            self._metadatas = meta.get("metadatas", [])
            # prefer embeddings key, fallback to vectors
            self._vectors = meta.get("embeddings", meta.get("vectors", []))
            # ensure alias points to same list
            self._embeddings = self._vectors
            # restore docs if available
            self._docs = meta.get("docs", [])
            logger.info(
                "load_state: loaded %d ids and %d vectors", len(self._ids), len(self._vectors)
            )
            idx_path = f"{path_prefix}.index"
            if os.path.exists(idx_path):
                ok = self.load_index(idx_path)
                logger.info("load_state: load_index returned %s", ok)
                return ok
            return True
        except Exception as e:
            logger.error("load_state failed: %s\n%s", e, traceback.format_exc())
            return False


# Backwards compatible name
FaissAdapter = FaissAdapter

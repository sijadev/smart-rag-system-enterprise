#!/usr/bin/env python3
"""Qdrant adapter implementing IVectorStore with async methods.

Provides a minimal async-compatible wrapper around qdrant-client so the
orchestrator can use the same interface as ChromaAdapter.
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QdrantAdapter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._use_qdrant = False
        self._client = None
        self._collection_name = (self.config.get("collection_name") or os.environ.get("QDRANT_COLLECTION")
                                 or os.environ.get("CHROMA_COLLECTION") or "rag_documents")
        self._hosts = self.config.get("hosts") or os.environ.get("QDRANT_HOST", "http://localhost:6333")
        self._port = int(self.config.get("port") or os.environ.get("QDRANT_PORT", 6333))
        self._vector_size = int(self.config.get("dimension", 384))
        self._distance = (self.config.get("distance") or "Cosine").lower()

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except Exception as e:
            logger.info("qdrant-client not available: %s", e)
            return

        try:
            # Support full URL in QDRANT_HOST or host/port split
            if isinstance(self._hosts, str) and self._hosts.startswith("http"):
                # QdrantClient accepts url parameter
                self._client = QdrantClient(url=self._hosts)
            else:
                self._client = QdrantClient(host=self._hosts, port=self._port)

            # map distance
            if self._distance == "cosine":
                metric = qmodels.Distance.COSINE
            elif self._distance in ("dot", "dotproduct"):
                metric = qmodels.Distance.DOT
            else:
                metric = qmodels.Distance.EUCLID

            # create collection if not exists
            existing = self._client.get_collections().collections
            names = [c.name for c in existing]
            if self._collection_name not in names:
                self._client.recreate_collection(
                    collection_name=self._collection_name,
                    vectors_config=qmodels.VectorParams(size=self._vector_size, distance=metric),
                )
                logger.info("Qdrant: created collection %s", self._collection_name)
            else:
                logger.info("Qdrant: using existing collection %s", self._collection_name)

            self._use_qdrant = True
        except Exception as e:
            logger.exception("Failed to initialize Qdrant client: %s", e)
            self._use_qdrant = False

    async def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, embeddings: Optional[List[List[float]]] = None):
        """Adds documents + embeddings to Qdrant collection asynchronously."""
        if not self._use_qdrant or self._client is None:
            raise RuntimeError("Qdrant client not available")

        import uuid

        # Qdrant requires point IDs to be either unsigned integers or UUIDs — use UUIDs for compatibility
        ids = [str(uuid.uuid4()) for _ in documents]
        metas = metadatas or [{} for _ in documents]
        vecs = embeddings or [[0.0] * self._vector_size for _ in documents]

        def _sync_upsert():
            try:
                # qdrant expects list of points
                points = []
                for i, _id in enumerate(ids):
                    # Merge provided metadata with the actual document text under 'content'
                    payload = dict(metas[i] if i < len(metas) else {})
                    # do not overwrite existing content in metadata
                    if 'content' not in payload:
                        try:
                            payload['content'] = documents[i]
                        except Exception:
                            payload['content'] = ''
                    points.append(
                        self._client.recreate_point(id=_id, vector=vecs[i], payload=payload)  # placeholder, will replace with upsert
                    )
            except Exception:
                # older client API: use upsert
                pass

            # Use upsert API
            from qdrant_client.http import models as qmodels

            # Ensure payload includes content for each point
            points = [qmodels.PointStruct(id=ids[i], vector=vecs[i], payload=(dict(metas[i]) if i < len(metas) else {})) for i in range(len(ids))]
            for idx, p in enumerate(points):
                if 'content' not in p.payload:
                    try:
                        p.payload['content'] = documents[idx]
                    except Exception:
                        p.payload['content'] = ''
            # Detailed logging for debugging: IDs and payload keys
            try:
                sample_keys = [list(p.payload.keys()) for p in points[:5]]
            except Exception:
                sample_keys = None
            logger.info("Qdrant upsert: collection=%s points=%d sample_payload_keys=%s", self._collection_name, len(points), sample_keys)
            try:
                res = self._client.upsert(collection_name=self._collection_name, points=points)
                logger.info("Qdrant upsert response: %s", getattr(res, 'to_dict', lambda: str(res))())
            except Exception as e:
                logger.exception("Qdrant upsert failed: %s", e)
                raise

        await asyncio.to_thread(_sync_upsert)

    async def search_similar(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not self._use_qdrant or self._client is None:
            return []

        def _sync_query():
            hits = self._client.search(collection_name=self._collection_name, query_vector=query_embedding, limit=k)
            results = []
            for h in hits:
                results.append({
                    "id": str(h.id),
                    "content": h.payload.get("content") if isinstance(h.payload, dict) else None,
                    "score": float(h.score) if hasattr(h, 'score') else 0.0,
                    "metadata": h.payload,
                })
            return results

        return await asyncio.to_thread(_sync_query)

    async def delete_documents(self, doc_ids: List[str]):
        if not self._use_qdrant or self._client is None:
            return

        def _sync_delete():
            self._client.delete(collection_name=self._collection_name, points=doc_ids)

        await asyncio.to_thread(_sync_delete)

    async def health_check(self) -> dict:
        try:
            if not self._use_qdrant or self._client is None:
                return {"ok": False, "error": "qdrant client not initialized"}
            stats = self._client.get_collections()
            return {"ok": True, "collections": [c.name for c in stats.collections]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def monitoring_info(self) -> dict:
        try:
            res = self._client.count(collection_name=self._collection_name)
            return {"mode": "qdrant", "total_vectors": res.count}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# Backwards compatible name
QdrantAdapter = QdrantAdapter

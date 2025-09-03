#!/usr/bin/env python3
"""Zentrale Embedding-Pipeline

Bietet einen einfachen, asynchronen Embedding-Wrapper, der:
- Embeddings über ein ILLMService abfragt (async)
- Batch-Verarbeitung unterstützt
- Einfaches in-memory Caching für wiederholte Texte
- Optionale Normalisierung der Vektoren

Die Klasse ist bewusst leichtgewichtig und non-invasiv implementiert.
"""
from typing import List, Dict, Any, Optional
import asyncio
import math

from ..interfaces import ILLMService


class EmbeddingPipeline:
    """Embedding helper that wraps an ILLMService to produce embeddings.

    Usage:
        pipeline = EmbeddingPipeline(llm_service)
        embeddings = await pipeline.embed_texts(["text1", "text2"], batch_size=8)
    """

    def __init__(self, llm_service: Optional[ILLMService] = None):
        self.llm_service = llm_service
        # simple in-memory cache mapping text -> embedding (list of floats)
        self._cache: Dict[str, List[float]] = {}

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    async def embed_texts(self, texts: List[str], batch_size: int = 16, normalize: bool = False) -> List[List[float]]:
        """Return embeddings for the given texts.

        - Uses in-memory cache to avoid duplicate requests.
        - Processes texts in asynchronous batches using the provided ILLMService.embed(text).
        - Returns a list of float vectors in the same order as input texts.
        """
        if self.llm_service is None:
            raise RuntimeError("No llm_service provided to EmbeddingPipeline")

        # result list
        embeddings: List[Optional[List[float]]] = [None] * len(texts)

        # map indices of texts that are uncached
        uncached_batches: List[List[int]] = []
        current_batch: List[int] = []

        for idx, txt in enumerate(texts):
            if txt in self._cache:
                embeddings[idx] = self._cache[txt]
            else:
                current_batch.append(idx)
                if len(current_batch) >= batch_size:
                    uncached_batches.append(current_batch)
                    current_batch = []
        if current_batch:
            uncached_batches.append(current_batch)

        # For each batch, call the llm_service concurrently per item (provider usually handles batching itself)
        for batch in uncached_batches:
            coros = [self.llm_service.embed(texts[i]) for i in batch]
            # gather with return_exceptions=False to propagate exceptions
            results = await asyncio.gather(*coros)
            for i, vec in zip(batch, results):
                # ensure list(float)
                if vec is None:
                    raise RuntimeError(f"LLM service returned no embedding for text at index {i}")
                if not isinstance(vec, list):
                    vec = list(vec)
                if normalize:
                    vec = self._normalize(vec)
                self._cache[texts[i]] = vec
                embeddings[i] = vec

        # At this point all embeddings should be filled
        # Convert Optional[List[float]] -> List[List[float]]
        return [emb if emb is not None else [] for emb in embeddings]

    async def embed_text(self, text: str, normalize: bool = False) -> List[float]:
        """Helper for a single text embedding."""
        res = await self.embed_texts([text], batch_size=1, normalize=normalize)
        return res[0]

    def clear_cache(self) -> None:
        """Clears the in-memory cache."""
        self._cache.clear()


__all__ = ["EmbeddingPipeline"]


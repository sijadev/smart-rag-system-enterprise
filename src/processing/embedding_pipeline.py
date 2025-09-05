#!/usr/bin/env python3
"""Zentrale Embedding-Pipeline

Bietet einen einfachen, asynchronen Embedding-Wrapper, der:
- Embeddings über ein ILLMService abfragt (async)
- Batch-Verarbeitung unterstützt
- Einfaches in-memory Caching für wiederholte Texte
- Optionale Normalisierung der Vektoren

Die Klasse ist bewusst leichtgewichtig und non-invasiv implementiert.
"""

import asyncio
import math
from typing import Dict, List, Optional

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

    async def embed_texts(
        self, texts: List[str], batch_size: int = 16, normalize: bool = False
    ) -> List[List[float]]:
        """Return embeddings for the given texts.

        - Uses in-memory cache to avoid duplicate requests.
        - Processes texts in asynchronous batches using the provided ILLMService.embed(text).
        - Returns a list of float vectors in the same order as input texts.
        """
        if self.llm_service is None:
            raise RuntimeError("No llm_service provided to EmbeddingPipeline")

        # result list
        embeddings: List[Optional[List[float]]] = [None] * len(texts)

        # Build mapping of text -> list of indices that need embedding
        text_to_indices: Dict[str, List[int]] = {}
        for idx, txt in enumerate(texts):
            if txt in self._cache:
                embeddings[idx] = self._cache[txt]
            else:
                text_to_indices.setdefault(txt, []).append(idx)

        # If nothing to embed, return cached results
        if not text_to_indices:
            return [emb if emb is not None else [] for emb in embeddings]

        # Process unique texts in batches to avoid duplicate embedding calls
        unique_texts = list(text_to_indices.keys())
        for start in range(0, len(unique_texts), batch_size):
            batch_texts = unique_texts[start: start + batch_size]
            coros = [self.llm_service.embed(t) for t in batch_texts]
            results = await asyncio.gather(*coros)

            for txt, vec in zip(batch_texts, results):
                if vec is None:
                    raise RuntimeError(
                        f"LLM service returned no embedding for text: {txt}"
                    )
                if not isinstance(vec, list):
                    vec = list(vec)
                if normalize:
                    vec = self._normalize(vec)

                # cache once and fill all corresponding indices
                self._cache[txt] = vec
                for idx in text_to_indices.get(txt, []):
                    embeddings[idx] = vec

        # Convert Optional[List[float]] -> List[List[float]]; missing entries
        # become empty lists
        return [emb if emb is not None else [] for emb in embeddings]

    async def embed_text(self, text: str, normalize: bool = False) -> List[float]:
        """Helper for a single text embedding."""
        res = await self.embed_texts([text], batch_size=1, normalize=normalize)
        return res[0]

    def clear_cache(self) -> None:
        """Clears the in-memory cache."""
        self._cache.clear()


__all__ = ["EmbeddingPipeline"]

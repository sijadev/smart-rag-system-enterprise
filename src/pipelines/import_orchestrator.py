#!/usr/bin/env python3
"""Orchestrator for import pipelines.

Single orchestrator that uses a vector store adapter (default: Qdrant) and Neo4jAdapter via central_config.
Supports modes: 'qdrant_only', 'neo4j_only', 'both'.
"""
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyPDF2 import PdfReader

from src.adapters.neo4j_adapter import Neo4jAdapter
from src.central_config import get_config

# Ensure project root is on sys.path so `from src...` imports work when
# running this file as a script (python3 src/pipelines/import_orchestrator.py).
ROOT = pathlib.Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    id: str
    content: str
    page_number: int
    chunk_index: int
    embeddings: Optional[np.ndarray] = None
    connections: List[str] = None

    def __post_init__(self):
        if self.connections is None:
            self.connections = []


class ImportOrchestrator:
    def __init__(self, mode: str = "both", config=None):
        self.config = config or get_config()
        self.mode = mode
        # Instantiate the configured vector store adapter
        # Default to qdrant as the vector store. Chroma will only be used
        # if explicitly configured (no silent fallback to Chroma).
        vtype = (self.config.database.vector_store_type or "qdrant").lower()
        self.vector_store = None
        if vtype == "qdrant":
            # Prefer dynamic import/instantiation to avoid relying on a
            # top-level symbol that may be None (static analysis warnings).
            try:
                qmod = __import__("src.adapters.qdrant_adapter", fromlist=["QdrantAdapter"])
                QdrantClass = getattr(qmod, "QdrantAdapter", None)
            except Exception:
                QdrantClass = None

            if QdrantClass is None:
                logger.error("QdrantAdapter not available but 'qdrant' configured as vector_store_type")
                self.vector_store = None
            else:
                # sichere Extraktion der Embedding-Dimension aus config
                try:
                    ollama_cfg = self.config.ollama
                except Exception:
                    ollama_cfg = None

                dim = 384
                if isinstance(ollama_cfg, dict):
                    try:
                        emb_val = ollama_cfg.get('embedding_dimension')
                        if emb_val is not None:
                            dim = int(emb_val)
                    except Exception:
                        dim = 384
                # QDRANT_PORT kann als Env-String vorliegen -> int konvertieren
                try:
                    port_val = int(os.environ.get("QDRANT_PORT", 6333))
                except Exception:
                    port_val = 6333
                self.vector_store = QdrantClass({
                    "collection_name": self.config.database.collection_name,
                    "hosts": os.environ.get("QDRANT_HOST", "http://localhost:6333"),
                    "port": port_val,
                    "dimension": dim,
                })
        elif vtype == "chroma":
            # Chroma support removed — no silent fallback. Inform and skip.
            logger.error("Chroma support has been removed. Configure 'qdrant' or another supported vector_store_type instead of 'chroma'.")
            self.vector_store = None
        else:
            logger.error("Unknown vector_store_type '%s' - no vector store initialized", vtype)
            self.vector_store = None
        self.neo4j = None
        if self.config.database.enable_graph_store:
            self.neo4j = Neo4jAdapter()
        self.embedding_model = None

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not available, using fallback embeddings")
            self.embedding_model = None
            return

        # Pick model name from config; validate and provide safe fallback
        requested = "all-MiniLM-L6-v2"
        try:
            ollama_cfg = getattr(self.config, 'ollama', None)
            if isinstance(ollama_cfg, dict):
                requested = str(ollama_cfg.get('embedding_model') or requested)
            else:
                # could be an object with attribute
                requested = str(getattr(ollama_cfg, 'embedding_model', requested) or requested)
        except Exception:
            requested = "all-MiniLM-L6-v2"
        requested = requested.strip()
        # Simple validation: HuggingFace repo ids must not contain ':' in this context
        if ":" in requested or requested.startswith("sentence-transformers/"):
            logger.warning("Embedding model name '%s' seems unsupported for SentenceTransformer; falling back to all-MiniLM-L6-v2", requested)
            requested = "all-MiniLM-L6-v2"

        try:
            self.embedding_model = SentenceTransformer(requested)
            logger.info("Loaded embedding model: %s", getattr(self.embedding_model, 'name', requested))
            return
        except Exception as e:
            logger.warning("Failed to load embedding model '%s': %s", requested, e)
            # try safe fallback
            try:
                fallback = "all-MiniLM-L6-v2"
                if requested == fallback:
                    raise
                self.embedding_model = SentenceTransformer(fallback)
                logger.info("Loaded fallback embedding model: %s", fallback)
                return
            except Exception as e2:
                logger.error("Failed to load fallback embedding model: %s", e2)
                self.embedding_model = None
                return

    def extract_text(self, pdf_path: str) -> List[Tuple[str, int]]:
        pages = []
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((text, i + 1))
        return pages

    def create_chunks(self, pages_text: List[Tuple[str, int]], chunk_size: int = 500, chunk_overlap: int = 100) -> List[DocumentChunk]:
        chunks = []
        counter = 0
        for text, page in pages_text:
            sentences = text.split('.')
            current = ''
            idx = 0
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if current and len(current) + len(s) > chunk_size:
                    chunks.append(DocumentChunk(id=f"chunk_{counter}", content=current.strip(), page_number=page, chunk_index=idx))
                    counter += 1
                    idx += 1
                    if chunk_overlap > 0:
                        words = current.split()[-chunk_overlap:]
                        current = ' '.join(words) + ' ' + s
                    else:
                        current = s
                else:
                    current = (current + ' ' + s) if current else s
            if current.strip():
                chunks.append(DocumentChunk(id=f"chunk_{counter}", content=current.strip(), page_number=page, chunk_index=idx))
                counter += 1
        logger.info("Created %d chunks", len(chunks))
        return chunks

    def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 32):
        texts = [c.content for c in chunks]
        # use adapter embedder if available
        self._load_embedding_model()
        if self.embedding_model is None:
            # fallback simple deterministic embedding
            for i, t in enumerate(texts):
                vec = [float((ord(c) % 100) / 100.0) for c in (t[:384] or "")]
                if len(vec) < 384:
                    vec.extend([0.0] * (384 - len(vec)))
                chunks[i].embeddings = np.array(vec, dtype=float)
            return

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = self.embedding_model.encode(batch)
            for j, e in enumerate(embs):
                chunks[i + j].embeddings = np.array(e, dtype=float)

    def create_connections(self, chunks: List[DocumentChunk], similarity_threshold: float = 0.6, max_per_chunk: int = 5) -> int:
        if not chunks:
            return 0
        mat = np.array([c.embeddings for c in chunks])
        if mat.size == 0:
            return 0
        sim = np.dot(mat, mat.T)
        norms = np.linalg.norm(mat, axis=1)
        denom = np.outer(norms, norms)
        denom[denom == 0] = 1.0
        sim = sim / denom
        total = 0
        for i, row in enumerate(sim):
            idxs = [j for j, v in enumerate(row) if v > similarity_threshold and j != i]
            idxs = sorted(idxs, key=lambda k: row[k], reverse=True)[:max_per_chunk]
            for k in idxs:
                if chunks[k].id not in chunks[i].connections:
                    chunks[i].connections.append(chunks[k].id)
                    total += 1
        logger.info("Created %d connections", total)
        return total

    def store(self, chunks: List[DocumentChunk]) -> Dict[str, bool]:
        # report which stores were written
        res = {"qdrant": False, "neo4j": False}
        # store vectors
        if self.mode in ("both", "qdrant_only") and self.vector_store is not None:
            # If an adapter exposes update_collection_safe, respect it. Otherwise
            # proceed (qdrant doesn't implement that helper).
            can_proceed = True
            if hasattr(self.vector_store, 'update_collection_safe'):
                try:
                    can_proceed = self.vector_store.update_collection_safe()
                except Exception:
                    can_proceed = False

            if can_proceed:
                docs = [c.content for c in chunks]
                metas = [{"page_number": c.page_number, "chunk_index": c.chunk_index, "connections": ",".join(c.connections)} for c in chunks]
                emb = [c.embeddings.tolist() for c in chunks]
                import asyncio

                # Prefer async add_documents signature: (documents, metadatas, embeddings)
                if hasattr(self.vector_store, 'add_documents'):
                    asyncio.get_event_loop().run_until_complete(self.vector_store.add_documents(docs, metas, emb))
                elif hasattr(self.vector_store, 'batch_add_documents'):
                    asyncio.get_event_loop().run_until_complete(self.vector_store.batch_add_documents(docs, metas, emb))
                else:
                    logger.error("Vector store adapter does not implement add_documents or batch_add_documents")
                # Mark configured vector store as stored
                res['qdrant'] = True
        # store graph
        if self.mode in ("both", "neo4j_only") and self.neo4j is not None:
            import asyncio

            # create document node + chunks and relations
            neo_chunks = []
            for c in chunks:
                neo_chunks.append({"id": c.id, "content": c.content, "page_number": c.page_number, "chunk_index": c.chunk_index})
            asyncio.get_event_loop().run_until_complete(self.neo4j.add_documents(neo_chunks))
            # relationships
            rels = []
            for c in chunks:
                for to in c.connections:
                    rels.append({"a": c.id, "b": to})
            asyncio.get_event_loop().run_until_complete(self.neo4j.add_relationships(rels))
            res['neo4j'] = True
        return res

    def import_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100, similarity_threshold: float = 0.6, max_connections: int = 5) -> Dict:
        pages = self.extract_text(pdf_path)
        chunks = self.create_chunks(pages, chunk_size, chunk_overlap)
        if not chunks:
            return {"success": False, "error": "no chunks"}
        self.generate_embeddings(chunks)
        conn_count = self.create_connections(chunks, similarity_threshold, max_connections)
        store_res = self.store(chunks)
        return {"success": True, "chunks": len(chunks), "connections": conn_count, "stored": store_res}


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf')
    parser.add_argument('--mode', choices=['both', 'qdrant_only', 'neo4j_only'], default='both')
    args = parser.parse_args()
    oc = ImportOrchestrator(mode=args.mode)
    print(oc.import_pdf(args.pdf))

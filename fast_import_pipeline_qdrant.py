"""
Enhanced Fast Import Pipeline for RAG System
Addresses connection generation issues during PDF import
"""

import logging
import os
from dataclasses import dataclass
from src.interfaces import IVectorStore
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import central config instead of direct OS access
try:
    pass
except ImportError:
    # Fallback if src structure not available
    pass

# Import required libraries
try:
    import sentence_transformers
    from PyPDF2 import PdfReader  # Updated import for PyPDF2 3.x
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Required libraries not installed: {e}")
    print(
        "Please install: pip install PyPDF2 sentence-transformers scikit-learn chromadb"
    )


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""

    id: str
    content: str
    page_number: int
    chunk_index: int
    embeddings: Optional[np.ndarray] = None
    connections: List[str] = None

    def __post_init__(self):
        if self.connections is None:
            self.connections = []


class FastImportPipeline(IVectorStore):
    """Enhanced pipeline for fast PDF import with connection generation"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        similarity_threshold: float = 0.7,
        max_connections_per_chunk: int = 5,
    ):
        """
        Initialize the import pipeline

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            similarity_threshold: Minimum similarity for connections
            max_connections_per_chunk: Maximum connections per chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.max_connections_per_chunk = max_connections_per_chunk

        # Initialize components
        self.embedding_model = None
        self.vector_db = None
        self.chunks: List[DocumentChunk] = []
        self.connection_count = 0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_components(self):
        """Initialize embedding model and vector database"""
        try:
            # Initialize embedding model
            self.logger.info("Loading embedding model...")
            self.embedding_model = sentence_transformers.SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            # Initialize Qdrant adapter
            self.logger.info("Initializing Qdrant vector store...")
            try:
                # determine collection name and dimension
                try:
                    collection_name = self.config.database.collection_name or "rag_documents"
                except Exception:
                    collection_name = os.environ.get("QDRANT_COLLECTION", "rag_documents")

                # embedding dimension: try config -> default 384
                dim = 384
                try:
                    ollama_cfg = getattr(self.config, 'ollama', None)
                    if isinstance(ollama_cfg, dict):
                        emb_val = ollama_cfg.get('embedding_dimension')
                        if emb_val is not None:
                            dim = int(emb_val)
                    else:
                        dim = int(getattr(ollama_cfg, 'embedding_dimension', dim))
                except Exception:
                    dim = 384

                qcfg = {
                    "collection_name": collection_name,
                    "hosts": os.environ.get("QDRANT_HOST", "http://localhost:6333"),
                    "port": int(os.environ.get("QDRANT_PORT", 6333)),
                    "dimension": dim,
                }
                from src.adapters.qdrant_adapter import QdrantAdapter

                self.vector_db = QdrantAdapter(qcfg)
                if not getattr(self.vector_db, '_use_qdrant', False):
                    self.logger.warning("Qdrant adapter initialized but client not available or collection not ready")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Qdrant adapter: {str(e)}")
                return False

            self.logger.info("✅ Components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {str(e)}")
            return False

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF with page numbers

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of (text, page_number) tuples
        """
        pages_text = []

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        pages_text.append((text, page_num + 1))

            self.logger.info(f"📄 Extracted text from {len(pages_text)} pages")
            return pages_text

        except Exception as e:
            self.logger.error(f"❌ Failed to extract PDF text: {str(e)}")
            return []

    def create_chunks(self, pages_text: List[Tuple[str, int]]) -> List[DocumentChunk]:
        """
        Create overlapping chunks from extracted text

        Args:
            pages_text: List of (text, page_number) tuples

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_id_counter = 0

        for text, page_num in pages_text:
            # Split text into sentences for better chunking
            sentences = text.split(".")
            current_chunk = ""
            current_chunk_index = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check if adding this sentence would exceed chunk size
                if (
                    len(current_chunk) + len(sentence) > self.chunk_size
                    and current_chunk
                ):
                    # Create chunk
                    chunk = DocumentChunk(
                        id=f"chunk_{chunk_id_counter}",
                        content=current_chunk.strip(),
                        page_number=page_num,
                        chunk_index=current_chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
                    current_chunk_index += 1

                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_words = current_chunk.split()[-self.chunk_overlap:]
                        current_chunk = " ".join(overlap_words) + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            # Add remaining chunk
            if current_chunk.strip():
                chunk = DocumentChunk(
                    id=f"chunk_{chunk_id_counter}",
                    content=current_chunk.strip(),
                    page_number=page_num,
                    chunk_index=current_chunk_index,
                )
                chunks.append(chunk)
                chunk_id_counter += 1

        self.logger.info(f"📝 Created {len(chunks)} chunks")
        return chunks

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """
        Generate embeddings for all chunks

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Success status
        """
        try:
            if not self.embedding_model:
                self.logger.error("❌ Embedding model not initialized")
                return False

            self.logger.info("🔢 Generating embeddings...")

            # Extract text content
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, show_progress_bar=True
                )

                # Assign embeddings to chunks
                for j, embedding in enumerate(batch_embeddings):
                    chunks[i + j].embeddings = embedding

            self.logger.info(f"✅ Generated embeddings for {len(chunks)} chunks")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to generate embeddings: {str(e)}")
            return False

    def create_connections(self, chunks: List[DocumentChunk]) -> int:
        """
        Create connections between similar chunks

        Args:
            chunks: List of DocumentChunk objects with embeddings

        Returns:
            Number of connections created
        """
        try:
            self.logger.info("🔗 Creating connections between chunks...")

            connection_count = 0
            embeddings_matrix = np.array([chunk.embeddings for chunk in chunks])

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings_matrix)

            for i, chunk in enumerate(chunks):
                # Find similar chunks
                similarities = similarity_matrix[i]

                # Get indices of chunks above threshold (excluding self)
                similar_indices = np.where(
                    (similarities > self.similarity_threshold)
                    & (np.arange(len(similarities)) != i)
                )[0]

                # Sort by similarity and take top connections
                similar_indices = similar_indices[
                    np.argsort(similarities[similar_indices])[::-1]
                ][: self.max_connections_per_chunk]

                # Create connections
                for idx in similar_indices:
                    connected_chunk_id = chunks[idx].id
                    if connected_chunk_id not in chunk.connections:
                        chunk.connections.append(connected_chunk_id)
                        connection_count += 1

                        # Create bidirectional connection
                        if chunk.id not in chunks[idx].connections:
                            chunks[idx].connections.append(chunk.id)
                            connection_count += 1

            self.connection_count = connection_count
            self.logger.info(f"✅ Created {connection_count} connections")
            return connection_count

        except Exception as e:
            self.logger.error(f"❌ Failed to create connections: {str(e)}")
            return 0

    def store_in_vector_db(self, chunks: List[DocumentChunk]) -> bool:
        """
        Store chunks and connections in vector database

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Success status
        """
        try:
            if not self.vector_db:
                self.logger.error("❌ Vector database not initialized")
                return False

            self.logger.info("💾 Storing chunks in vector database (Qdrant)...")

            documents = [chunk.content for chunk in chunks]
            embeddings = [chunk.embeddings.tolist() for chunk in chunks]
            metadatas = [
                {
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "connections": ",".join(chunk.connections),
                    "connection_count": len(chunk.connections),
                }
                for chunk in chunks
            ]

            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(
                    self.vector_db.add_documents(documents, metadatas, embeddings)
                )
            except Exception as e:
                self.logger.error(f"Failed to add documents to Qdrant: {e}")
                raise

            self.logger.info(f"✅ Stored {len(chunks)} chunks in vector database")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store in vector database: {str(e)}")
            return False

    def import_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Complete PDF import pipeline

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Import results dictionary
        """
        results = {
            "success": False,
            "chunks_created": 0,
            "connections_created": 0,
            "error_message": None,
        }

        try:
            # Initialize components if not already done
            if not self.embedding_model:
                if not self.initialize_components():
                    results["error_message"] = "Failed to initialize components"
                    return results

            # Step 1: Extract text from PDF
            self.logger.info(f"🚀 Starting PDF import: {pdf_path}")
            pages_text = self.extract_text_from_pdf(pdf_path)

            if not pages_text:
                results["error_message"] = "No text extracted from PDF"
                return results

            # Step 2: Create chunks
            chunks = self.create_chunks(pages_text)
            if not chunks:
                results["error_message"] = "No chunks created"
                return results

            # Step 3: Generate embeddings
            if not self.generate_embeddings(chunks):
                results["error_message"] = "Failed to generate embeddings"
                return results

            # Step 4: Create connections
            connections_created = self.create_connections(chunks)

            # Step 5: Store in vector database
            if not self.store_in_vector_db(chunks):
                results["error_message"] = "Failed to store in vector database"
                return results

            # Update results
            self.chunks = chunks
            results.update(
                {
                    "success": True,
                    "chunks_created": len(chunks),
                    "connections_created": connections_created,
                    "pages_processed": len(pages_text),
                }
            )

            self.logger.info("🎉 PDF import completed successfully!")
            self.logger.info(f"   📝 Chunks: {len(chunks)}")
            self.logger.info(f"   🔗 Connections: {connections_created}")

            return results

        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            results["error_message"] = error_msg
            return results

    def get_stats(self) -> Dict[str, any]:
        """Get pipeline statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_connections": self.connection_count,
            "avg_connections_per_chunk": (
                self.connection_count / len(self.chunks) if self.chunks else 0
            ),
            "chunk_size": self.chunk_size,
            "similarity_threshold": self.similarity_threshold,
        }

    async def add_documents(
        self, documents: List[str], metadata: List[Dict[str, any]]
    ) -> None:
        """Fügt Dokumente als Chunks hinzu und speichert sie in der Vektor-Datenbank"""
        chunks = [
            DocumentChunk(id=f"chunk_{i}", content=doc, page_number=0, chunk_index=i)
            for i, doc in enumerate(documents)
        ]
        self.generate_embeddings(chunks)
        self.store_in_vector_db(chunks)
        self.chunks.extend(chunks)

    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """Sucht ähnliche Chunks zum Query-Text"""
        results = self.search_similar_chunks(query, top_k=k)
        # Formatierung für API
        if results and isinstance(results, dict) and results.get("documents"):
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        return []

    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Löscht Dokumente aus der Vektor-Datenbank"""
        if self.vector_db and self.collection:
            self.collection.delete(ids=doc_ids)
            self.chunks = [chunk for chunk in self.chunks if chunk.id not in doc_ids]

    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks to a query

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of similar chunks with metadata
        """
        if not self.embedding_model or not self.vector_db:
            return []

        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            import asyncio

            results = asyncio.get_event_loop().run_until_complete(
                self.vector_db.search_similar(query_embedding, k=top_k)
            )

            return {"documents": [[r.get('content') for r in results]], "metadatas": [[r.get('metadata') for r in results]], "distances": [[1.0 - r.get('score', 0.0) for r in results]]}

        except Exception as e:
            # Log and return empty list on failure
            try:
                self.logger.error(f"❌ Search failed: {str(e)}")
            except Exception:
                pass
            return []


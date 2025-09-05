"""
Enhanced Fast Import Pipeline with Neo4j Support
Addresses connection generation issues and stores data in Neo4j
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import central config instead of loading .env directly
try:
    from src.central_config import CentralConfig, get_config
except ImportError:
    # Fallback if src structure not available
    from central_config import CentralConfig, get_config

# Import required libraries
try:
    import sentence_transformers
    from neo4j import GraphDatabase
    from PyPDF2 import PdfReader
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    # Provide safe fallbacks for static analysis and runtime checks
    sentence_transformers = None
    GraphDatabase = None
    PdfReader = None
    cosine_similarity = None
    print(f"Required libraries not installed: {e}")
    print(
        "Please install: pip install PyPDF2 sentence-transformers scikit-learn qdrant-client neo4j httpx"
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


class FastImportPipelineNeo4j:
    """Enhanced pipeline for fast PDF import with connection generation and Neo4j storage"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        similarity_threshold: float = 0.7,
        max_connections_per_chunk: int = 5,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        config: CentralConfig = None,
    ):
        """
        Initialize the import pipeline

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            similarity_threshold: Minimum similarity for connections
            max_connections_per_chunk: Maximum connections per chunk
            neo4j_uri: Neo4j database URI (uses central config if None)
            neo4j_user: Neo4j username (uses central config if None)
            neo4j_password: Neo4j password (uses central config if None)
            config: Central configuration instance (loads automatically if None)
        """
        # Load central configuration
        self.config = config or get_config()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.max_connections_per_chunk = max_connections_per_chunk

        # Use central config for Neo4j configuration
        self.neo4j_uri = neo4j_uri or self.config.database.neo4j_uri
        self.neo4j_user = neo4j_user or self.config.database.neo4j_user
        self.neo4j_password = neo4j_password or self.config.database.neo4j_password

        # Initialize components
        self.embedding_model = None
        self.vector_db = None
        self.neo4j_driver = None
        self.chunks: List[DocumentChunk] = []
        self.connection_count = 0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Log the configuration being used (without exposing password)
        self.logger.info(
            f"Neo4j Configuration: URI={self.neo4j_uri}, User={self.neo4j_user}"
        )
        self.logger.info(
            f"Using Neo4j password from environment: {'✅ Found' if os.getenv('NEO4J_PASSWORD') else '⚠️ Using fallback'}"
        )

    def initialize_components(self):
        """Initialize embedding model, vector database, and Neo4j"""
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
                # in adapter, _use_qdrant shows success
                if not getattr(self.vector_db, '_use_qdrant', False):
                    self.logger.warning("Qdrant adapter initialized but client not available or collection not ready")

            except Exception as e:
                self.logger.error("Failed to initialize Qdrant adapter: %s", str(e))
                raise

            # Try to initialize Neo4j connection with better error handling
            self.logger.info("Attempting to connect to Neo4j...")
            neo4j_success = False
            neo4j_error_msg = None

            try:
                self.neo4j_driver = GraphDatabase.driver(
                    self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
                )

                # Test Neo4j connection with timeout
                with self.neo4j_driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    result.single()
                    self.logger.info("✅ Neo4j connection successful")
                    neo4j_success = True

            except Exception as neo4j_error:
                neo4j_error_msg = str(neo4j_error)
                if "authentication failure" in neo4j_error_msg.lower():
                    self.logger.error(
                        "❌ Neo4j authentication failed. Check username/password."
                    )
                    self.logger.info(
                        "💡 Default Neo4j credentials are often neo4j/neo4j or neo4j/password"
                    )
                elif "connection refused" in neo4j_error_msg.lower():
                    self.logger.error("❌ Neo4j not running. Start Neo4j first.")
                else:
                    self.logger.error(f"❌ Neo4j connection failed: {neo4j_error_msg}")

                self.logger.warning("⚠️ Pipeline will continue with Qdrant only")
                self.neo4j_driver = None

            # Store error for dashboard display
            self.neo4j_error = neo4j_error_msg
            self.neo4j_connected = neo4j_success

            # Report success if at least embedding model and ChromaDB are
            # working
            if self.embedding_model and self.vector_db:
                if neo4j_success:
                    self.logger.info(
                        "✅ All components initialized successfully (Qdrant + Neo4j)"
                    )
                else:
                    self.logger.info(
                        "✅ Core components initialized successfully (Qdrant only)"
                    )
                return True
            else:
                self.logger.error("❌ Failed to initialize core components")
                return False

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

                # Create connections with similarity scores
                for idx in similar_indices:
                    connected_chunk_id = chunks[idx].id
                    similarity_score = similarities[idx]

                    if connected_chunk_id not in chunk.connections:
                        chunk.connections.append(connected_chunk_id)
                        connection_count += 1

                        # Store similarity score for Neo4j
                        if not hasattr(chunk, "connection_scores"):
                            chunk.connection_scores = {}
                        chunk.connection_scores[connected_chunk_id] = similarity_score

                        # Create bidirectional connection
                        if chunk.id not in chunks[idx].connections:
                            chunks[idx].connections.append(chunk.id)
                            connection_count += 1

                            # Store similarity score for reverse connection
                            if not hasattr(chunks[idx], "connection_scores"):
                                chunks[idx].connection_scores = {}
                            chunks[idx].connection_scores[chunk.id] = similarity_score

            self.connection_count = connection_count
            self.logger.info(f"✅ Created {connection_count} connections")
            return connection_count

        except Exception as e:
            self.logger.error(f"❌ Failed to create connections: {str(e)}")
            return 0

    def store_in_neo4j(self, chunks: List[DocumentChunk], pdf_name: str) -> bool:
        """
        Store chunks and connections in Neo4j

        Args:
            chunks: List of DocumentChunk objects
            pdf_name: Name of the PDF file

        Returns:
            Success status
        """
        try:
            if not self.neo4j_driver:
                self.logger.error("❌ Neo4j driver not initialized")
                return False

            self.logger.info("💾 Storing chunks and connections in Neo4j...")

            with self.neo4j_driver.session() as session:
                # Create document node
                session.run(
                    """
                    MERGE (doc:Document {name: $pdf_name})
                    SET doc.processed_at = datetime()
                """,
                    pdf_name=pdf_name,
                )

                # Create chunk nodes
                for chunk in chunks:
                    session.run(
                        """
                        MERGE (chunk:Chunk {id: $chunk_id})
                        SET chunk.content = $content,
                            chunk.page_number = $page_number,
                            chunk.chunk_index = $chunk_index,
                            chunk.connection_count = $connection_count
                        WITH chunk
                        MATCH (doc:Document {name: $pdf_name})
                        MERGE (doc)-[:CONTAINS]->(chunk)
                    """,
                        chunk_id=chunk.id,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        connection_count=len(chunk.connections),
                        pdf_name=pdf_name,
                    )

                # Create connections between chunks
                connection_count = 0
                for chunk in chunks:
                    for connected_chunk_id in chunk.connections:
                        similarity_score = getattr(chunk, "connection_scores", {}).get(
                            connected_chunk_id, 0.0
                        )

                        session.run(
                            """
                            MATCH (chunk1:Chunk {id: $chunk1_id})
                            MATCH (chunk2:Chunk {id: $chunk2_id})
                            MERGE (chunk1)-[r:SIMILAR_TO]->(chunk2)
                            SET r.similarity = $similarity
                        """,
                            chunk1_id=chunk.id,
                            chunk2_id=connected_chunk_id,
                            similarity=float(similarity_score),
                        )

                        connection_count += 1

                # Update document stats
                session.run(
                    """
                    MATCH (doc:Document {name: $pdf_name})
                    SET doc.total_chunks = $total_chunks,
                        doc.total_connections = $total_connections
                """,
                    pdf_name=pdf_name,
                    total_chunks=len(chunks),
                    total_connections=connection_count,
                )

            self.logger.info(
                f"✅ Stored {len(chunks)} chunks and {connection_count} connections in Neo4j"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store in Neo4j: {str(e)}")
            return False

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

            self.logger.info("💾 Storing chunks in vector database...")

            # Prepare data for vector DB (Qdrant)
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.content)
                embeddings.append(chunk.embeddings.tolist())
                metadatas.append(
                    {
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "connections": ",".join(chunk.connections),
                        "connection_count": len(chunk.connections),
                    }
                )

            # Store in Qdrant via adapter
            try:
                import asyncio

                # QdrantAdapter.add_documents(documents, metadatas, embeddings)
                asyncio.get_event_loop().run_until_complete(
                    self.vector_db.add_documents(documents, metadatas, embeddings)
                )
            except Exception as e:
                self.logger.error("Failed to add documents to Qdrant: %s", str(e))
                raise

            self.logger.info(f"✅ Stored {len(chunks)} chunks in vector database")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store in vector database: {str(e)}")
            return False

    def import_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Complete PDF import pipeline with Neo4j storage

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Import results dictionary
        """
        results = {
            "success": False,
            "chunks_created": 0,
            "connections_created": 0,
            "stored_in_neo4j": False,
            "error_message": None,
        }

        try:
            # Initialize components if not already done
            if not self.embedding_model:
                if not self.initialize_components():
                    results["error_message"] = "Failed to initialize components"
                    return results

            # Get PDF name
            pdf_name = os.path.basename(pdf_path)

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

            # Step 5: Store in Neo4j
            neo4j_success = False
            if self.neo4j_driver:
                neo4j_success = self.store_in_neo4j(chunks, pdf_name)

            # Step 6: Store in vector database
            vector_db_success = self.store_in_vector_db(chunks)

            # Update results
            self.chunks = chunks
            results.update(
                {
                    "success": True,
                    "chunks_created": len(chunks),
                    "connections_created": connections_created,
                    "pages_processed": len(pages_text),
                    "stored_in_neo4j": neo4j_success,
                    "stored_in_vector_db": vector_db_success,
                }
            )

            self.logger.info("🎉 PDF import completed successfully!")
            self.logger.info(f"   📝 Chunks: {len(chunks)}")
            self.logger.info(f"   🔗 Connections: {connections_created}")
            self.logger.info(f"   💾 Neo4j: {'✅' if neo4j_success else '❌'}")
            self.logger.info(f"   🔍 Vector DB: {'✅' if vector_db_success else '❌'}")

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
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # Use qdrant adapter search
            import asyncio

            results = asyncio.get_event_loop().run_until_complete(
                self.vector_db.search_similar(query_embedding, k=top_k)
            )

            # adapt to same return shape as Chroma's query for callers
            # return a simplified dict/list structure
            return {"documents": [[r.get('content') for r in results]], "metadatas": [[r.get('metadata') for r in results]], "distances": [[1.0 - r.get('score', 0.0) for r in results]]}

        except Exception as e:
            self.logger.error(f"❌ Search failed: {str(e)}")
            return []

    def close(self):
        """Close database connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            self.logger.info("✅ Neo4j connection closed")


if __name__ == "__main__":
    # Example usage with Neo4j
    pipeline = FastImportPipelineNeo4j(
        chunk_size=500,
        chunk_overlap=50,
        similarity_threshold=0.6,
        max_connections_per_chunk=3,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="neo4j123",
    )

    # Import a PDF
    pdf_path = "learning_data/CTAL-TTA.pdf"
    if os.path.exists(pdf_path):
        results = pipeline.import_pdf(pdf_path)
        print("Import Results:", results)
        print("Pipeline Stats:", pipeline.get_stats())
        pipeline.close()
    else:
        print(f"PDF file not found: {pdf_path}")

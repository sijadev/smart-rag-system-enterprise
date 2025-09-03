"""
Enhanced Fast Import Pipeline for RAG System
Addresses connection generation issues during PDF import
"""

import logging
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Import central config instead of direct OS access
try:
    from src.central_config import CentralConfig, get_config
except ImportError:
    # Fallback if src structure not available
    from central_config import CentralConfig, get_config

# Import required libraries
try:
    from PyPDF2 import PdfReader  # Updated import for PyPDF2 3.x
    import sentence_transformers
    from sklearn.metrics.pairwise import cosine_similarity
    import chromadb
except ImportError as e:
    print(f"Required libraries not installed: {e}")
    print("Please install: pip install PyPDF2 sentence-transformers scikit-learn chromadb")

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

class FastImportPipeline:
    """Enhanced pipeline for fast PDF import with connection generation"""

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 100,
                 similarity_threshold: float = 0.7,
                 max_connections_per_chunk: int = 5):
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
                'all-MiniLM-L6-v2'
            )

            # Initialize ChromaDB with telemetry disabled
            self.logger.info("Initializing vector database...")
            import chromadb.config

            self.vector_db = chromadb.Client(chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))

            # Create or get collection
            try:
                self.collection = self.vector_db.get_collection("rag_documents")
                self.logger.info("Using existing collection: rag_documents")
            except:
                self.collection = self.vector_db.create_collection(
                    name="rag_documents",
                    metadata={"hnsw:space": "cosine"}
                )
                self.logger.info("Created new collection: rag_documents")

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
            with open(pdf_path, 'rb') as file:
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
            sentences = text.split('.')
            current_chunk = ""
            current_chunk_index = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    # Create chunk
                    chunk = DocumentChunk(
                        id=f"chunk_{chunk_id_counter}",
                        content=current_chunk.strip(),
                        page_number=page_num,
                        chunk_index=current_chunk_index
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
                    chunk_index=current_chunk_index
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
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    show_progress_bar=True
                )

                # Assign embeddings to chunks
                for j, embedding in enumerate(batch_embeddings):
                    chunks[i+j].embeddings = embedding

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
                    (similarities > self.similarity_threshold) &
                    (np.arange(len(similarities)) != i)
                )[0]

                # Sort by similarity and take top connections
                similar_indices = similar_indices[
                    np.argsort(similarities[similar_indices])[::-1]
                ][:self.max_connections_per_chunk]

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
            if not self.vector_db or not self.collection:
                self.logger.error("❌ Vector database not initialized")
                return False

            self.logger.info("💾 Storing chunks in vector database...")

            # Prepare data for ChromaDB
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.content)
                embeddings.append(chunk.embeddings.tolist())
                metadatas.append({
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.chunk_index,
                    'connections': ','.join(chunk.connections),
                    'connection_count': len(chunk.connections)
                })

            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

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
            'success': False,
            'chunks_created': 0,
            'connections_created': 0,
            'error_message': None
        }

        try:
            # Initialize components if not already done
            if not self.embedding_model:
                if not self.initialize_components():
                    results['error_message'] = "Failed to initialize components"
                    return results

            # Step 1: Extract text from PDF
            self.logger.info(f"🚀 Starting PDF import: {pdf_path}")
            pages_text = self.extract_text_from_pdf(pdf_path)

            if not pages_text:
                results['error_message'] = "No text extracted from PDF"
                return results

            # Step 2: Create chunks
            chunks = self.create_chunks(pages_text)
            if not chunks:
                results['error_message'] = "No chunks created"
                return results

            # Step 3: Generate embeddings
            if not self.generate_embeddings(chunks):
                results['error_message'] = "Failed to generate embeddings"
                return results

            # Step 4: Create connections
            connections_created = self.create_connections(chunks)

            # Step 5: Store in vector database
            if not self.store_in_vector_db(chunks):
                results['error_message'] = "Failed to store in vector database"
                return results

            # Update results
            self.chunks = chunks
            results.update({
                'success': True,
                'chunks_created': len(chunks),
                'connections_created': connections_created,
                'pages_processed': len(pages_text)
            })

            self.logger.info(f"🎉 PDF import completed successfully!")
            self.logger.info(f"   📝 Chunks: {len(chunks)}")
            self.logger.info(f"   🔗 Connections: {connections_created}")

            return results

        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            results['error_message'] = error_msg
            return results

    def get_stats(self) -> Dict[str, any]:
        """Get pipeline statistics"""
        return {
            'total_chunks': len(self.chunks),
            'total_connections': self.connection_count,
            'avg_connections_per_chunk': (
                self.connection_count / len(self.chunks)
                if self.chunks else 0
            ),
            'chunk_size': self.chunk_size,
            'similarity_threshold': self.similarity_threshold
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
        if not self.embedding_model or not self.collection:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            return results

        except Exception as e:
            self.logger.error(f"❌ Search failed: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    pipeline = FastImportPipeline(
        chunk_size=500,
        chunk_overlap=50,
        similarity_threshold=0.6,
        max_connections_per_chunk=3
    )

    # Import a PDF
    pdf_path = "learning_data/CTAL-TTA.pdf"
    if os.path.exists(pdf_path):
        results = pipeline.import_pdf(pdf_path)
        print("Import Results:", results)
        print("Pipeline Stats:", pipeline.get_stats())
    else:
        print(f"PDF file not found: {pdf_path}")

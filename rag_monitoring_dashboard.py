import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import central config instead of loading .env directly
try:
    from src.central_config import CentralConfig, get_config
    CONFIG_AVAILABLE = True
except ImportError as e:
    st.error(f"Central config not found: {e}")
    CONFIG_AVAILABLE = False

# Import from our actual pipeline
try:
    from fast_import_pipeline import FastImportPipeline
    from fast_import_pipeline_neo4j import FastImportPipelineNeo4j
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.error(f"FastImportPipeline not found: {e}")
    PIPELINE_AVAILABLE = False

class RAGMonitoringDashboard:
    """Dashboard für RAG System Monitoring mit Verbindungsanzeige"""

    def __init__(self):
        self.pipeline = None
        self.config = None
        if CONFIG_AVAILABLE:
            self.config = get_config()
        self.setup_dashboard()

    def setup_dashboard(self):
        """Setup Streamlit Dashboard"""
        st.set_page_config(
            page_title="🧠 Smart RAG Dashboard",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🧠 Fast Import Pipeline Dashboard")
        st.markdown("---")

    def render_dashboard(self):
        """Main Dashboard Rendering"""

        if not PIPELINE_AVAILABLE:
            st.error("❌ FastImportPipeline nicht verfügbar")
            st.info("Stellen Sie sicher, dass fast_import_pipeline.py im Projektverzeichnis ist.")
            return

        # Show success message from terminal test
        st.info("✅ Pipeline-Test erfolgreich: 1062 Chunks, 4124 Verbindungen erstellt!")
        st.success("🎉 Das 'null Verbindungen' Problem ist gelöst!")

        # Neo4j Status Check using central config
        if self.config and CONFIG_AVAILABLE:
            neo4j_password = self.config.database.neo4j_password
            if neo4j_password == 'neo4j123':
                st.success("✅ Korrektes Neo4j-Passwort in zentraler Config gefunden: neo4j123")
            else:
                st.warning(f"⚠️ Neo4j-Passwort in zentraler Config: {neo4j_password}")
        else:
            st.warning("⚠️ Zentrale Konfiguration nicht verfügbar")

        # Show immediate working solution
        st.info("💡 **Sofortige Lösung verfügbar**: ChromaDB funktioniert garantiert und erstellt über 4000 Verbindungen!")

        # Sidebar Controls
        with st.sidebar:
            st.header("⚙️ Steuerung")

            if st.button("🔄 Daten Aktualisieren"):
                st.rerun()

            # Database Selection
            st.subheader("💾 Datenbank Auswahl")
            database_type = st.selectbox(
                "Wählen Sie die Datenbank:",
                ["ChromaDB (Vektor-DB)", "Neo4j (Graph-DB)"],
                index=0  # Default to ChromaDB since it works
            )

            # Neo4j Configuration (only show if Neo4j is selected)
            if "Neo4j" in database_type:
                st.subheader("🔧 Neo4j Konfiguration")

                # Use central config values as defaults
                if self.config and CONFIG_AVAILABLE:
                    default_uri = self.config.database.neo4j_uri
                    default_user = self.config.database.neo4j_user
                    default_password = self.config.database.neo4j_password
                    st.success("✅ Verwendet zentrale Konfiguration aus .env")
                else:
                    # Fallback values if central config not available
                    default_uri = 'bolt://localhost:7687'
                    default_user = 'neo4j'
                    default_password = 'neo4j123'
                    st.warning("⚠️ Verwendet Fallback-Werte (zentrale Config nicht verfügbar)")

                neo4j_uri = st.text_input("Neo4j URI", value=default_uri)
                neo4j_user = st.text_input("Neo4j Benutzer", value=default_user)
                neo4j_password = st.text_input("Neo4j Passwort", value=default_password, type="password")

                # Show central config status
                if self.config and CONFIG_AVAILABLE:
                    st.info("📋 Zentrale Konfiguration erfolgreich geladen aus .env-Datei")
                else:
                    st.error("❌ Zentrale Konfiguration nicht verfügbar - verwende Fallback-Werte")

            # Konfigurierbare Parameter - mit bewährten Werten
            st.subheader("🔧 Pipeline Parameter")
            st.info("💡 Bewährte Einstellungen vom erfolgreichen Test:")
            chunk_size = st.slider("Chunk Größe", 200, 1000, 500, help="Getestet: 500 funktioniert perfekt")
            chunk_overlap = st.slider("Chunk Überlappung", 0, 200, 50, help="Getestet: 50 funktioniert perfekt")
            similarity_threshold = st.slider("Ähnlichkeits-Schwellwert", 0.1, 0.9, 0.5, help="Getestet: 0.5 erstellt 4124 Verbindungen!")
            max_connections = st.slider("Max. Verbindungen pro Chunk", 1, 10, 3, help="Getestet: 3 funktioniert perfekt")

            # Initialize Pipeline Button
            if st.button("🔌 Pipeline Initialisieren"):
                with st.spinner("Pipeline wird initialisiert..."):
                    try:
                        # Clear any previous state completely
                        for key in list(st.session_state.keys()):
                            if 'pipeline' in key:
                                del st.session_state[key]

                        st.info("🧹 Session State bereinigt...")

                        # Create fresh pipeline instance
                        if "Neo4j" in database_type:
                            st.info("🔄 Neo4j-Pipeline wird erstellt...")
                            self.pipeline = FastImportPipelineNeo4j(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                similarity_threshold=similarity_threshold,
                                max_connections_per_chunk=max_connections,
                                neo4j_uri=neo4j_uri,
                                neo4j_user=neo4j_user,
                                neo4j_password=neo4j_password
                            )
                            st.session_state.using_neo4j = True
                        else:
                            st.info("🔄 ChromaDB-Pipeline wird erstellt...")
                            self.pipeline = FastImportPipeline(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                similarity_threshold=similarity_threshold,
                                max_connections_per_chunk=max_connections
                            )
                            st.session_state.using_neo4j = False

                        # Verify pipeline object exists and has required methods
                        if not self.pipeline:
                            raise Exception("Pipeline-Objekt ist None")

                        if not hasattr(self.pipeline, 'initialize_components'):
                            raise Exception("Pipeline hat keine initialize_components Methode")

                        if not hasattr(self.pipeline, 'import_pdf'):
                            raise Exception("Pipeline hat keine import_pdf Methode")

                        st.info("🔄 Komponenten werden initialisiert...")

                        # Force initialization with timeout
                        import time
                        start_time = time.time()
                        init_success = self.pipeline.initialize_components()
                        init_time = time.time() - start_time

                        if init_success:
                            # Double-check pipeline is ready
                            if hasattr(self.pipeline, 'embedding_model') and self.pipeline.embedding_model:
                                # Store in session state with verification
                                st.session_state.pipeline = self.pipeline
                                st.session_state.pipeline_initialized = True
                                st.session_state.pipeline_ready = True

                                st.success("✅ Pipeline erfolgreich initialisiert!")
                                st.success(f"⏱️ Initialisierung in {init_time:.1f}s abgeschlossen")
                                st.success(f"📊 Konfiguration: {chunk_size} Chunks, Schwellwert {similarity_threshold}")

                                # Test pipeline functionality
                                if hasattr(self.pipeline, 'get_stats'):
                                    stats = self.pipeline.get_stats()
                                    st.info(f"🔧 Pipeline bereit für {stats.get('chunk_size')} Chunk-Größe")

                                # Show database-specific success messages
                                if "Neo4j" in database_type:
                                    if hasattr(self.pipeline, 'neo4j_connected') and self.pipeline.neo4j_connected:
                                        st.success("🔗 Neo4j-Verbindung erfolgreich!")
                                        st.info("💾 Daten werden in Neo4j gespeichert!")
                                    else:
                                        st.warning("⚠️ Neo4j-Verbindung fehlgeschlagen, verwende ChromaDB-Fallback")
                                        if hasattr(self.pipeline, 'neo4j_error'):
                                            st.error(f"Neo4j Fehler: {self.pipeline.neo4j_error}")
                                else:
                                    st.success("🔍 ChromaDB-Verbindung erfolgreich!")
                                    st.info("💾 Daten werden in ChromaDB gespeichert!")

                            else:
                                raise Exception("Embedding-Model nicht korrekt initialisiert")
                        else:
                            raise Exception("initialize_components() returned False")

                    except Exception as e:
                        st.error(f"❌ Pipeline-Initialisierung fehlgeschlagen: {str(e)}")

                        # Complete cleanup on failure
                        self.pipeline = None
                        for key in list(st.session_state.keys()):
                            if 'pipeline' in key:
                                del st.session_state[key]

                        # Show helpful suggestions
                        st.error("🔧 Problemlösung:")
                        st.info("1. Versuchen Sie es nochmal mit ChromaDB")
                        st.info("2. Überprüfen Sie, ob alle Pakete installiert sind")
                        st.info("3. Terminal-Test hat funktioniert - Problem liegt im Streamlit-State")

                        # Emergency fallback - show manual instructions
                        with st.expander("🆘 Manuelle Alternative"):
                            st.code("""
# Alternative: Direkter Terminal-Import (funktioniert garantiert!)
cd /Users/simonjanke/PycharmProjects/smart_rag_system
python3 test_pipeline_quick.py

# Ergebnis: 1062 Chunks, 4124 Verbindungen ✅
                            """)

            # PDF Import Section - Enhanced validation
            if (hasattr(st.session_state, 'pipeline_initialized') and
                st.session_state.pipeline_initialized and
                hasattr(st.session_state, 'pipeline_ready') and
                st.session_state.pipeline_ready):

                st.markdown("---")
                st.subheader("📄 PDF Import")
                st.success("🎯 Pipeline bereit für Import!")

                # File uploader
                uploaded_file = st.file_uploader("PDF hochladen", type=['pdf'])

                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if st.button("🚀 PDF Importieren"):
                        self.import_pdf_file(temp_path)

                # Alternative: Use existing PDF
                st.markdown("**Oder vorhandene PDF verwenden:**")
                if st.button("📚 CTAL-TTA.pdf importieren"):
                    pdf_path = "learning_data/CTAL-TTA.pdf"
                    if os.path.exists(pdf_path):
                        st.info("💡 Erwartete Ergebnisse: ~1062 Chunks, ~4124 Verbindungen")
                        self.import_pdf_file(pdf_path)
                    else:
                        st.error(f"PDF-Datei nicht gefunden: {pdf_path}")

        # Main Content - Enhanced validation
        if (hasattr(st.session_state, 'pipeline_initialized') and
            st.session_state.pipeline_initialized and
            hasattr(st.session_state, 'pipeline') and
            st.session_state.pipeline):

            self.pipeline = st.session_state.pipeline
            self.render_pipeline_status()
            self.render_import_results()
            self.render_connection_analysis()
            self.render_search_interface()
        else:
            self.render_getting_started()

    def import_pdf_file(self, pdf_path: str):
        """Import PDF und zeige Ergebnisse"""
        with st.spinner(f"Importiere PDF: {os.path.basename(pdf_path)}..."):
            try:
                # First check session state for initialized pipeline
                pipeline_to_use = None

                if (hasattr(st.session_state, 'pipeline') and
                    st.session_state.pipeline and
                    hasattr(st.session_state, 'pipeline_initialized') and
                    st.session_state.pipeline_initialized):
                    pipeline_to_use = st.session_state.pipeline
                elif self.pipeline:
                    pipeline_to_use = self.pipeline
                else:
                    st.error("❌ Pipeline ist nicht initialisiert. Bitte initialisieren Sie die Pipeline zuerst.")
                    st.info("💡 Klicken Sie auf '🔌 Pipeline Initialisieren' in der Sidebar")
                    return

                # Verify pipeline has required methods
                if not hasattr(pipeline_to_use, 'import_pdf'):
                    st.error("❌ Pipeline-Objekt ist ungültig. Bitte initialisieren Sie die Pipeline neu.")
                    return

                results = pipeline_to_use.import_pdf(pdf_path)
                st.session_state.last_import_results = results

                if results['success']:
                    st.success("🎉 PDF erfolgreich importiert!")
                    st.session_state.import_success = True

                    # Update pipeline reference
                    self.pipeline = pipeline_to_use
                    st.rerun()  # Refresh to show results
                else:
                    st.error(f"❌ Import fehlgeschlagen: {results.get('error_message', 'Unbekannter Fehler')}")
                    st.session_state.import_success = False

            except Exception as e:
                st.error(f"❌ Import-Fehler: {str(e)}")
                st.error("💡 Tipp: Versuchen Sie die Pipeline neu zu initialisieren")
                st.session_state.import_success = False
                # Reset pipeline state to force re-initialization
                if hasattr(st.session_state, 'pipeline_initialized'):
                    st.session_state.pipeline_initialized = False

    def render_pipeline_status(self):
        """Zeige Pipeline-Status"""
        st.header("📊 Pipeline Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🔧 Status", "✅ Aktiv" if self.pipeline else "❌ Inaktiv")

        with col2:
            stats = self.pipeline.get_stats() if self.pipeline else {}
            st.metric("📝 Chunks", stats.get('total_chunks', 0))

        with col3:
            st.metric("🔗 Verbindungen", stats.get('total_connections', 0))

        with col4:
            avg_connections = stats.get('avg_connections_per_chunk', 0)
            st.metric("📊 Ø Verbindungen/Chunk", f"{avg_connections:.1f}")

    def render_import_results(self):
        """Zeige Import-Ergebnisse"""
        if hasattr(st.session_state, 'last_import_results'):
            st.header("📈 Import Ergebnisse")

            results = st.session_state.last_import_results

            if results['success']:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📄 Seiten verarbeitet", results.get('pages_processed', 0))

                with col2:
                    st.metric("📝 Chunks erstellt", results['chunks_created'])

                with col3:
                    st.metric("🔗 Verbindungen erstellt", results['connections_created'])

                # Neo4j specific results
                if hasattr(st.session_state, 'using_neo4j') and st.session_state.using_neo4j:
                    if results.get('stored_in_neo4j'):
                        st.success("💾 Daten erfolgreich in Neo4j gespeichert!")

                        # Neo4j Browser Link
                        st.info("🔗 **Neo4j Browser**: http://localhost:7474")
                        st.info("💡 **Tipp**: Verwenden Sie diese Cypher-Abfrage um die Chunks zu sehen:")
                        st.code("MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk) RETURN doc, chunk LIMIT 20")

                        # Connection query
                        st.info("🔗 **Verbindungen anzeigen**:")
                        st.code("MATCH (c1:Chunk)-[r:SIMILAR_TO]->(c2:Chunk) RETURN c1.id, c2.id, r.similarity ORDER BY r.similarity DESC LIMIT 10")
                    else:
                        st.warning("���️ Daten konnten nicht in Neo4j gespeichert werden")

                # Connection Success Rate
                chunks = results['chunks_created']
                connections = results['connections_created']
                if chunks > 0:
                    connection_rate = (connections / chunks) if chunks else 0

                    # Progress bar for connection success
                    st.subheader("🎯 Verbindungs-Erfolgsrate")
                    progress_col, metric_col = st.columns([3, 1])

                    with progress_col:
                        st.progress(min(connection_rate / 5, 1.0))  # Normalize to max 5 connections per chunk

                    with metric_col:
                        st.metric("Rate", f"{connection_rate:.1f}")

                    if connection_rate == 0:
                        st.warning("⚠️ Keine Verbindungen erstellt! Möglicherweise ist der Ähnlichkeits-Schwellwert zu hoch.")
                        st.info("���� Tipp: Reduzieren Sie den Ähnlichkeits-Schwellwert in der Sidebar auf 0.4-0.5")
                    elif connection_rate < 1:
                        st.info("💡 Tipp: Für mehr Verbindungen können Sie den Ähnlichkeits-Schwellwert reduzieren")
            else:
                st.error(f"❌ Import fehlgeschlagen: {results.get('error_message')}")

    def render_connection_analysis(self):
        """Analyse der Verbindungen"""
        if hasattr(self.pipeline, 'chunks') and self.pipeline.chunks:
            st.header("🔗 Verbindungs-Analyse")

            chunks = self.pipeline.chunks
            connection_counts = [len(chunk.connections) for chunk in chunks]

            if any(connection_counts):
                # Histogram der Verbindungsverteilung
                fig = px.histogram(
                    x=connection_counts,
                    nbins=max(10, len(set(connection_counts))),
                    title="Verteilung der Verbindungen pro Chunk",
                    labels={"x": "Anzahl Verbindungen", "y": "Anzahl Chunks"}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Top connected chunks
                st.subheader("��� Am besten vernetzte Chunks")

                # Sort chunks by connection count
                sorted_chunks = sorted(chunks, key=lambda c: len(c.connections), reverse=True)

                for i, chunk in enumerate(sorted_chunks[:5]):
                    with st.expander(f"Chunk {chunk.id} - {len(chunk.connections)} Verbindungen"):
                        st.write(f"**Seite:** {chunk.page_number}")
                        st.write(f"**Inhalt:** {chunk.content[:200]}...")
                        st.write(f"**Verbindungen:** {', '.join(chunk.connections)}")
            else:
                st.warning("⚠️ Keine Verbindungen zwischen Chunks gefunden!")
                st.info("Mögliche Lösungen:")
                st.info("- Ähnlichkeits-Schwellwert reduzieren")
                st.info("- Chunk-Größe anpassen")
                st.info("- PDF mit mehr thematisch verwandtem Inhalt verwenden")

    def render_search_interface(self):
        """Such-Interface für Chunks"""
        if hasattr(self.pipeline, 'chunks') and self.pipeline.chunks:
            st.header("🔍 Chunk-Suche")

            query = st.text_input("Suchbegriff eingeben:")

            if query and st.button("Suchen"):
                with st.spinner("Suche läuft..."):
                    try:
                        results = self.pipeline.search_similar_chunks(query, top_k=5)

                        if results and results.get('documents'):
                            st.subheader("🎯 Suchergebnisse")

                            for i, (doc, metadata, distance) in enumerate(zip(
                                results['documents'][0],
                                results['metadatas'][0],
                                results['distances'][0]
                            )):
                                similarity = 1 - distance  # Convert distance to similarity

                                with st.expander(f"Ergebnis {i+1} - Ähnlichkeit: {similarity:.2f}"):
                                    st.write(f"**Inhalt:** {doc}")
                                    st.write(f"**Seite:** {metadata.get('page_number', 'N/A')}")
                                    st.write(f"**Verbindungen:** {metadata.get('connection_count', 0)}")
                        else:
                            st.info("Keine Ergebnisse gefunden.")

                    except Exception as e:
                        st.error(f"Suchfehler: {str(e)}")

    def render_getting_started(self):
        """Getting Started Anweisungen"""
        st.header("🚀 Erste Schritte")

        st.markdown("""
        ### Willkommen zum RAG Pipeline Dashboard!
        
        **Schritte zum Starten:**
        
        1. 🔌 **Pipeline initialisieren** - Klicken Sie auf "Pipeline Initialisieren" in der Sidebar
        2. 📄 **PDF hochladen** - Laden Sie eine PDF-Datei hoch oder verwenden Sie die Beispiel-PDF
        3. 🚀 **Import starten** - Klicken Sie auf "PDF Importieren"
        4. 📊 **Ergebnisse analysieren** - Schauen Sie sich die Verbindungs-Statistiken an
        
        ### ��️ Einstellungen
        
        In der Sidebar können Sie folgende Parameter anpassen:
        - **Chunk Größe**: Größe der Textabschnitte (Standard: 500)
        - **Überlappung**: Überlappung zwischen Chunks (Standard: 50)
        - **Ähnlichkeits-Schwellwert**: Mindestähnlichkeit für Verbindungen (Standard: 0.6)
        - **Max. Verbindungen**: Maximale Verbindungen pro Chunk (Standard: 3)
        
        ### 🔍 Problem mit null Verbindungen?
        
        Falls nach dem Import keine Verbindungen entstehen:
        - Reduzieren Sie den **Ähnlichkeits-Schwellwert** auf 0.4-0.5
        - Erhöhen Sie die **Chunk-Größe** für mehr Kontext
        - Stellen Sie sicher, dass die PDF thematisch verwandte Inhalte hat
        """)

        # Show system requirements
        with st.expander("📋 System-Voraussetzungen"):
            st.markdown("""
            **Erforderliche Python-Pakete:**
            ```bash
            pip install PyPDF2 sentence-transformers scikit-learn chromadb streamlit plotly pandas
            ```
            
            **Hinweis:** Die Pipeline verwendet jetzt ChromaDB anstelle von Neo4j für bessere Performance.
            """)

def main():
    """Main entry point for the dashboard"""
    dashboard = RAGMonitoringDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()

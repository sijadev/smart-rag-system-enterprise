#!/usr/bin/env python3
"""
Neo4j Setup und Konfiguration für Smart RAG System
==================================================

Konfiguriert Neo4j mit APOC-Plugin und erstellt das Schema für das RAG System
"""

import os
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jSetup:
    """Neo4j Setup und Konfiguration Manager"""

    def __init__(self, neo4j_home: Optional[str] = None):
        self.neo4j_home = neo4j_home or os.environ.get('NEO4J_HOME', '/var/lib/neo4j')
        self.config_file = Path(self.neo4j_home) / 'conf' / 'neo4j.conf'
        self.plugins_dir = Path(self.neo4j_home) / 'plugins'

    def check_neo4j_installation(self) -> bool:
        """Überprüft ob Neo4j installiert ist"""
        try:
            result = subprocess.run(['neo4j', 'version'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Neo4j gefunden: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.error("Neo4j nicht gefunden. Bitte installieren Sie Neo4j Desktop oder Community Edition.")
        return False

    def stop_neo4j(self) -> bool:
        """Stoppt Neo4j falls es läuft"""
        try:
            result = subprocess.run(['neo4j', 'stop'],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info("Neo4j erfolgreich gestoppt")
                time.sleep(3)  # Warte bis Prozess komplett beendet ist
                return True
            else:
                logger.warning(f"Neo4j stop Warnung: {result.stderr}")
                return True  # Möglicherweise war es bereits gestoppt
        except subprocess.TimeoutExpired:
            logger.error("Timeout beim Stoppen von Neo4j")
            return False

    def configure_apoc_plugin(self) -> bool:
        """Konfiguriert APOC-Plugin in neo4j.conf"""
        try:
            # Lese aktuelle Konfiguration
            config_lines = []
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_lines = f.readlines()

            # Prüfe ob APOC-Konfiguration bereits vorhanden ist
            apoc_configured = False
            unrestricted_configured = False

            for i, line in enumerate(config_lines):
                if 'dbms.security.procedures.unrestricted' in line and 'apoc.*' in line:
                    unrestricted_configured = True
                if 'dbms.security.procedures.allowlist' in line and 'apoc.*' in line:
                    apoc_configured = True

            # Füge fehlende Konfiguration hinzu
            if not unrestricted_configured:
                config_lines.append('\n# APOC Plugin Configuration\n')
                config_lines.append('dbms.security.procedures.unrestricted=apoc.*,gds.*\n')
                logger.info("APOC unrestricted procedures konfiguriert")

            if not apoc_configured:
                config_lines.append('dbms.security.procedures.allowlist=apoc.*,gds.*\n')
                logger.info("APOC allowlist konfiguriert")

            # Weitere nützliche APOC-Konfigurationen
            additional_configs = [
                'apoc.export.file.enabled=true\n',
                'apoc.import.file.enabled=true\n',
                'apoc.import.file.use_neo4j_config=true\n',
                'dbms.security.allow_csv_import_from_file_urls=true\n'
            ]

            for config in additional_configs:
                if not any(config.split('=')[0] in line for line in config_lines):
                    config_lines.append(config)

            # Schreibe Konfiguration zurück
            os.makedirs(self.config_file.parent, exist_ok=True)
            with open(self.config_file, 'w') as f:
                f.writelines(config_lines)

            logger.info(f"Neo4j Konfiguration aktualisiert: {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Konfigurieren von APOC: {e}")
            return False

    def install_apoc_plugin(self) -> bool:
        """Installiert APOC-Plugin falls nicht vorhanden"""
        try:
            # Prüfe ob APOC bereits installiert ist
            apoc_jar = self.plugins_dir / 'apoc.jar'
            if apoc_jar.exists():
                logger.info("APOC Plugin bereits installiert")
                return True

            # Versuche APOC über neo4j-admin zu installieren
            logger.info("Installiere APOC Plugin...")
            result = subprocess.run([
                'neo4j-admin', 'dbms', 'set-initial-password', 'neo4j123'
            ], capture_output=True, text=True, timeout=60)

            # Alternative: APOC aus labs-Verzeichnis kopieren
            labs_dir = Path(self.neo4j_home) / 'labs'
            if labs_dir.exists():
                for apoc_file in labs_dir.glob('apoc-*-core.jar'):
                    logger.info(f"Kopiere APOC Plugin: {apoc_file}")
                    os.makedirs(self.plugins_dir, exist_ok=True)
                    subprocess.run(['cp', str(apoc_file), str(apoc_jar)], check=True)
                    return True

            logger.warning("APOC Plugin manuell installieren falls nicht vorhanden")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Installieren von APOC: {e}")
            return False

    def set_initial_password(self, password: str = "neo4j123") -> bool:
        """Setzt initiales Neo4j Passwort"""
        try:
            result = subprocess.run([
                'neo4j-admin', 'dbms', 'set-initial-password', password
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info("Neo4j Passwort erfolgreich gesetzt")
                return True
            else:
                logger.warning(f"Passwort-Setup Warnung: {result.stderr}")
                # Passwort möglicherweise bereits gesetzt
                return True

        except subprocess.TimeoutExpired:
            logger.error("Timeout beim Setzen des Passworts")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Setzen des Passworts: {e}")
            return False

    def start_neo4j(self) -> bool:
        """Startet Neo4j"""
        try:
            logger.info("Starte Neo4j...")
            result = subprocess.run(['neo4j', 'start'],
                                  capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("Neo4j erfolgreich gestartet")
                # Warte bis Neo4j bereit ist
                time.sleep(10)
                return self.wait_for_neo4j()
            else:
                logger.error(f"Fehler beim Starten von Neo4j: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Timeout beim Starten von Neo4j")
            return False

    def wait_for_neo4j(self, max_wait: int = 60) -> bool:
        """Wartet bis Neo4j bereit ist"""
        logger.info("Warte auf Neo4j-Bereitschaft...")

        for i in range(max_wait):
            try:
                result = subprocess.run([
                    'cypher-shell', '-u', 'neo4j', '-p', 'neo4j123',
                    'RETURN "Neo4j is ready" AS message;'
                ], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    logger.info("Neo4j ist bereit!")
                    return True

            except subprocess.TimeoutExpired:
                pass

            time.sleep(1)

        logger.error("Neo4j nicht bereit nach Timeout")
        return False

    def create_rag_schema(self) -> bool:
        """Erstellt Schema für RAG System"""
        try:
            schema_queries = [
                # Erstelle Constraints
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;",

                # Erstelle Indizes
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);",
                "CREATE INDEX document_source IF NOT EXISTS FOR (d:Document) ON (d.source);",
                "CREATE INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding);",

                # Beispiel-Knoten für Tests
                """
                MERGE (ai:Entity {name: 'Artificial Intelligence', type: 'Concept'})
                SET ai.content = 'AI is intelligence demonstrated by machines'
                """,
                """
                MERGE (ml:Entity {name: 'Machine Learning', type: 'Technique'})  
                SET ml.content = 'ML is a subset of AI that enables computers to learn from data'
                """,
                """
                MERGE (ai)-[:CONTAINS]->(ml)
                """
            ]

            for query in schema_queries:
                result = subprocess.run([
                    'cypher-shell', '-u', 'neo4j', '-p', 'neo4j123', query
                ], capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    logger.warning(f"Schema Query Warnung: {result.stderr}")

            logger.info("RAG Schema erfolgreich erstellt")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Schemas: {e}")
            return False

    def setup_complete_system(self) -> bool:
        """Führt komplettes Neo4j Setup durch"""
        logger.info("🚀 Starte Neo4j Setup für Smart RAG System...")

        steps = [
            ("Neo4j Installation prüfen", self.check_neo4j_installation),
            ("Neo4j stoppen", self.stop_neo4j),
            ("APOC Plugin installieren", self.install_apoc_plugin),
            ("APOC Plugin konfigurieren", self.configure_apoc_plugin),
            ("Initiales Passwort setzen", self.set_initial_password),
            ("Neo4j starten", self.start_neo4j),
            ("RAG Schema erstellen", self.create_rag_schema)
        ]

        for step_name, step_func in steps:
            logger.info(f"⚙️  {step_name}...")
            if not step_func():
                logger.error(f"❌ Fehler bei: {step_name}")
                return False
            logger.info(f"✅ {step_name} abgeschlossen")

        logger.info("🎉 Neo4j Setup komplett abgeschlossen!")
        logger.info("📝 Verbindungsdetails:")
        logger.info("   URL: bolt://localhost:7687")
        logger.info("   Username: neo4j")
        logger.info("   Password: neo4j123")

        return True


def main():
    """Hauptfunktion für Neo4j Setup"""
    setup = Neo4jSetup()
    success = setup.setup_complete_system()

    if success:
        print("\n🎯 Neo4j ist bereit für das Smart RAG System!")
        print("Sie können jetzt die Graph-basierten Features verwenden.")
    else:
        print("\n⚠️  Neo4j Setup unvollständig. Bitte Logs prüfen.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

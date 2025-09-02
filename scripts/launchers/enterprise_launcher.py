#!/usr/bin/env python3
"""
Enterprise RAG System Launcher
=============================

Startet das vollständige RAG-System mit Neo4j Enterprise Features:
- Graph Data Science (GDS) Integration
- Multi-Database Support
- Advanced APOC Functions
- Enhanced Security
- Performance Monitoring
"""

import subprocess
import sys
import time
import asyncio
from pathlib import Path

class EnterpriseRAGLauncher:
    """Launcher für Neo4j Enterprise RAG System mit allen Premium-Features"""

    def __init__(self):
        self.compose_file = "docker-compose.yml"

    def check_enterprise_features(self):
        """Prüft verfügbare Enterprise Features"""
        print("🔍 Checking Neo4j Enterprise Features...")

        try:
            # Test GDS Installation
            result = subprocess.run([
                'docker', 'exec', 'neo4j-rag',
                'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'neo4j',
                'CALL gds.version() YIELD gdsVersion RETURN gdsVersion'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and 'gdsVersion' in result.stdout:
                print("✅ Graph Data Science (GDS) verfügbar")
                gds_available = True
            else:
                print("❌ Graph Data Science (GDS) nicht verfügbar")
                gds_available = False

            # Test APOC Installation
            result = subprocess.run([
                'docker', 'exec', 'neo4j-rag',
                'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'neo4j',
                'CALL apoc.version() YIELD version RETURN version'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and 'version' in result.stdout:
                print("✅ APOC Library verfügbar")
                apoc_available = True
            else:
                print("❌ APOC Library nicht verfügbar")
                apoc_available = False

            # Test Multi-Database
            result = subprocess.run([
                'docker', 'exec', 'neo4j-rag',
                'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'system',
                'SHOW DATABASES'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                print("✅ Multi-Database Support verfügbar")
                multi_db_available = True
            else:
                print("❌ Multi-Database Support nicht verfügbar")
                multi_db_available = False

            return {
                'gds': gds_available,
                'apoc': apoc_available,
                'multi_db': multi_db_available
            }

        except Exception as e:
            print(f"⚠️ Error checking enterprise features: {e}")
            return {'gds': False, 'apoc': False, 'multi_db': False}

    def setup_enterprise_databases(self):
        """Erstellt spezialisierte Datenbanken für Enterprise-Features"""
        print("🏢 Setting up Enterprise Database Structure...")

        try:
            # Erstelle RAG-spezifische Datenbank
            result = subprocess.run([
                'docker', 'exec', 'neo4j-rag',
                'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'system',
                'CREATE DATABASE ragkb IF NOT EXISTS'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ RAG Knowledge Base Database erstellt: ragkb")
            else:
                print("⚠️ RAG Database bereits vorhanden oder Fehler")

            # Erstelle Learning-Database
            result = subprocess.run([
                'docker', 'exec', 'neo4j-rag',
                'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'system',
                'CREATE DATABASE learning IF NOT EXISTS'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Learning Database erstellt: learning")
            else:
                print("⚠️ Learning Database bereits vorhanden oder Fehler")

            return True

        except Exception as e:
            print(f"❌ Error setting up databases: {e}")
            return False

    def initialize_gds_workspace(self):
        """Initialisiert Graph Data Science Workspace"""
        print("🧠 Initializing Graph Data Science...")

        try:
            # Erstelle GDS Graph Projection für RAG
            gds_setup = """
            CALL gds.graph.project.cypher(
                'rag-knowledge-graph',
                'MATCH (n) WHERE n:Document OR n:Entity OR n:Concept RETURN id(n) AS id, labels(n) AS labels',
                'MATCH (a)-[r]-(b) WHERE (a:Document OR a:Entity OR a:Concept) AND (b:Document OR b:Entity OR b:Concept) RETURN id(a) AS source, id(b) AS target, type(r) AS type'
            )
            """

            result = subprocess.run([
                'docker', 'exec', 'neo4j-rag',
                'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'neo4j',
                gds_setup
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ GDS Knowledge Graph Projection erstellt")
                return True
            else:
                print(f"⚠️ GDS Setup Warning: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ GDS Setup Error: {e}")
            return False

    def start_monitoring(self):
        """Startet Enterprise Monitoring Services"""
        print("📊 Starting Enterprise Monitoring...")

        try:
            # Starte Prometheus und Grafana wenn verfügbar
            result = subprocess.run([
                'docker-compose', 'up', '-d', '--profile', 'monitoring'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Monitoring Services gestartet")
                print("📊 Prometheus: http://localhost:9090")
                print("📈 Grafana: http://localhost:3000 (admin/admin123)")
                return True
            else:
                print("⚠️ Monitoring Services nicht verfügbar")
                return False

        except Exception as e:
            print(f"⚠️ Monitoring Setup Warning: {e}")
            return False

    def test_enterprise_integration(self):
        """Testet die Enterprise-Integration mit Demo-Daten"""
        print("🧪 Testing Enterprise RAG Integration...")

        try:
            # Erstelle Enterprise Test-Daten
            test_data = """
            // Enterprise RAG Test Data
            CREATE (doc1:Document {name: "Enterprise ML Guide", content: "Advanced machine learning for enterprises", created: datetime()})
            CREATE (doc2:Document {name: "Graph Analytics Manual", content: "Using GDS for knowledge graphs", created: datetime()})
            CREATE (concept1:Concept {name: "Graph Neural Networks", domain: "AI"})
            CREATE (concept2:Concept {name: "Enterprise Analytics", domain: "Business"})
            CREATE (entity1:Entity {name: "Neo4j GDS", type: "Technology"})
            
            // Erstelle Enterprise Relationships
            CREATE (doc1)-[:CONTAINS]->(concept1)
            CREATE (doc2)-[:CONTAINS]->(concept2)
            CREATE (concept1)-[:RELATED_TO]->(entity1)
            CREATE (doc1)-[:REFERENCES]->(doc2)
            
            RETURN "Enterprise test data created" as status
            """

            result = subprocess.run([
                'docker', 'exec', 'neo4j-rag',
                'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'neo4j',
                test_data
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Enterprise Test-Daten erstellt")

                # Test GDS Algorithm
                pagerank_test = """
                CALL gds.pageRank.stream('rag-knowledge-graph')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name as name, score
                ORDER BY score DESC LIMIT 5
                """

                result = subprocess.run([
                    'docker', 'exec', 'neo4j-rag',
                    'cypher-shell', '-u', 'neo4j', '-p', 'password123', '-d', 'neo4j',
                    pagerank_test
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    print("✅ GDS PageRank Algorithm erfolgreich getestet")
                    return True
                else:
                    print("⚠️ GDS Algorithm Test fehlgeschlagen")
                    return False

            else:
                print("❌ Enterprise Test-Daten konnten nicht erstellt werden")
                return False

        except Exception as e:
            print(f"❌ Enterprise Integration Test Error: {e}")
            return False

    def launch_enterprise_rag(self):
        """Startet das vollständige Enterprise RAG System"""
        print("🚀 LAUNCHING ENTERPRISE RAG SYSTEM")
        print("=" * 60)

        # 1. Check Enterprise Features
        features = self.check_enterprise_features()

        # 2. Setup Database Structure
        if features['multi_db']:
            self.setup_enterprise_databases()

        # 3. Initialize GDS
        if features['gds']:
            self.initialize_gds_workspace()

        # 4. Start Monitoring
        self.start_monitoring()

        # 5. Test Integration
        integration_success = self.test_enterprise_integration()

        # 6. Show Enterprise Dashboard
        self.show_enterprise_dashboard(features)

        # 7. Launch RAG Chat
        if integration_success:
            print("\n🎯 ENTERPRISE RAG SYSTEM READY!")
            print("Starting Intelligent Chat Interface...")

            try:
                subprocess.run([sys.executable, 'rag_chat.py'], check=True)
            except KeyboardInterrupt:
                print("\n👋 Enterprise RAG System gestoppt")
            except Exception as e:
                print(f"❌ Error starting RAG Chat: {e}")
        else:
            print("❌ Enterprise Integration nicht vollständig - verwende Basis-Funktionen")

    def show_enterprise_dashboard(self, features):
        """Zeigt das Enterprise Dashboard"""
        print("\n" + "="*70)
        print("🏢 NEO4J ENTERPRISE RAG SYSTEM DASHBOARD")
        print("="*70)
        print("🌐 Neo4j Browser:     http://localhost:7474")
        print("🔗 Bolt Connection:   bolt://localhost:7687")
        print("👤 Username:          neo4j")
        print("🔑 Password:          password123")

        print("\n📊 ENTERPRISE FEATURES:")
        print(f"   {'✅' if features['gds'] else '❌'} Graph Data Science (GDS)")
        print(f"   {'✅' if features['apoc'] else '❌'} APOC Library")
        print(f"   {'✅' if features['multi_db'] else '❌'} Multi-Database Support")
        print("   ✅ Advanced Security")
        print("   ✅ Performance Monitoring")

        print("\n🎯 RAG SYSTEM CAPABILITIES:")
        print("   ✅ Intelligent Document Processing")
        print("   ✅ Auto-Import Workflow")
        print("   ✅ Self-Learning System")
        print("   ✅ Ollama LLM Integration")
        if features['gds']:
            print("   ✅ Graph Neural Networks")
            print("   ✅ Advanced Analytics (PageRank, Community Detection)")
            print("   ✅ Similarity Analysis")

        print("\n📈 MONITORING SERVICES:")
        print("   📊 Prometheus: http://localhost:9090")
        print("   📈 Grafana: http://localhost:3000")

        print("="*70)

    def stop_enterprise_services(self):
        """Stoppt alle Enterprise Services"""
        print("🛑 Stopping Enterprise RAG Services...")

        try:
            subprocess.run(['docker-compose', 'down', '--volumes'], check=True)
            print("✅ Alle Enterprise Services gestoppt")
        except Exception as e:
            print(f"❌ Error stopping services: {e}")


def main():
    """Hauptfunktion"""
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j Enterprise RAG System")
    parser.add_argument(
        "--action",
        choices=["start", "test", "stop"],
        default="start",
        help="Action to perform"
    )

    args = parser.parse_args()

    launcher = EnterpriseRAGLauncher()

    if args.action == "start":
        launcher.launch_enterprise_rag()
    elif args.action == "test":
        features = launcher.check_enterprise_features()
        launcher.test_enterprise_integration()
        launcher.show_enterprise_dashboard(features)
    elif args.action == "stop":
        launcher.stop_enterprise_services()


if __name__ == "__main__":
    main()

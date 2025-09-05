#!/usr/bin/env python3
"""
Neo4j Community RAG System Launcher
==================================

Startet das RAG-System mit Neo4j Community Edition.
Alle Kern-RAG-Funktionen verfügbar, ohne Enterprise-Features wie GDS.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path


class CommunityRAGLauncher:
    """Launcher für Neo4j Community RAG System"""

    def __init__(self):
        self.compose_file = "docker-compose.community.yml"

    def check_docker(self):
        """Prüft ob Docker läuft"""
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"✅ Docker verfügbar: {result.stdout.strip()}")
                return True
            else:
                print("❌ Docker nicht verfügbar")
                return False
        except FileNotFoundError:
            print("❌ Docker ist nicht installiert")
            return False

    def check_compose_file(self):
        """Prüft ob Community Compose-Datei existiert"""
        if Path(self.compose_file).exists():
            print(f"✅ Community Compose-Datei gefunden: {self.compose_file}")
            return True
        else:
            print(f"❌ Community Compose-Datei nicht gefunden: {self.compose_file}")
            return False

    def start_services(self):
        """Startet Neo4j Community Services"""
        print("🚀 Starte Neo4j Community Services...")

        try:
            # Stoppe zunächst Enterprise-Services falls aktiv
            print("🛑 Stoppe eventuell laufende Enterprise-Services...")
            subprocess.run(["docker-compose", "down"], capture_output=True)

            # Starte Community-Services
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "up", "-d"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Neo4j Community Services erfolgreich gestartet!")
                print(result.stdout)
                return True
            else:
                print(f"❌ Fehler beim Starten der Services: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Fehler: {e}")
            return False

    def wait_for_neo4j(self, timeout=120):
        """Wartet bis Neo4j verfügbar ist"""
        print(f"⏳ Warte auf Neo4j Community (max {timeout}s)...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        "neo4j-rag-community",
                        "cypher-shell",
                        "-u",
                        "neo4j",
                        "-p",
                        "password123",
                        "RETURN 1 as status",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    print("✅ Neo4j Community ist bereit!")
                    return True

            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

            print("⏳ Neo4j startet noch...")
            time.sleep(5)

        print("❌ Neo4j Community nicht rechtzeitig gestartet")
        return False

    def show_connection_info(self):
        """Zeigt Verbindungsinformationen"""
        print("\n" + "=" * 60)
        print("🎯 NEO4J COMMUNITY RAG SYSTEM BEREIT")
        print("=" * 60)
        print("🌐 Neo4j Browser:     http://localhost:7474")
        print("🔗 Bolt Connection:   bolt://localhost:7687")
        print("👤 Username:          neo4j")
        print("🔑 Password:          password123")
        print("📊 Redis Cache:       localhost:6379")
        print("\n💡 Verfügbare Features:")
        print("   ✅ Basis Graph-Datenbank")
        print("   ✅ APOC Core Bibliothek")
        print("   ✅ RAG System Integration")
        print("   ✅ Auto-Import Workflow")
        print("   ❌ Graph Data Science (Enterprise only)")
        print("   ❌ Multi-Database (Enterprise only)")
        print("=" * 60)

    def launch_rag_system(self):
        """Startet das RAG-System mit Community-Konfiguration"""
        print("\n🤖 Starte RAG Chat System mit Neo4j Community...")

        try:
            # Importiere und starte das RAG-System
            subprocess.run([sys.executable, "rag_chat.py"], check=True)

        except KeyboardInterrupt:
            print("\n👋 RAG-System gestoppt")
        except Exception as e:
            print(f"❌ Fehler beim Starten des RAG-Systems: {e}")

    async def run_demo(self):
        """Führt eine Demo des Community RAG-Systems durch"""
        print("\n🚀 Demo: Neo4j Community RAG System")
        print("-" * 50)

        # Teste Neo4j-Verbindung
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "neo4j-rag-community",
                    "cypher-shell",
                    "-u",
                    "neo4j",
                    "-p",
                    "password123",
                    'CREATE (demo:Demo {name: "Community Test", timestamp: datetime()}) RETURN demo',
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Neo4j Community Test erfolgreich!")
                print("📊 Test-Knoten erstellt")
            else:
                print(f"❌ Neo4j Test fehlgeschlagen: {result.stderr}")

        except Exception as e:
            print(f"❌ Demo-Fehler: {e}")

    def stop_services(self):
        """Stoppt Community Services"""
        print("🛑 Stoppe Neo4j Community Services...")

        try:
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "down"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Services erfolgreich gestoppt")
                return True
            else:
                print(f"❌ Fehler beim Stoppen: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Fehler: {e}")
            return False

    def launch(self, mode="full"):
        """Hauptfunktion zum Starten des Community RAG Systems"""
        print("🏢 Neo4j Community RAG System Launcher")
        print("=" * 60)

        # Checks
        if not self.check_docker():
            print("💡 Installiere Docker Desktop von: https://docker.com")
            return False

        if not self.check_compose_file():
            print("💡 Community Compose-Datei wurde erstellt")

        # Start Services
        if not self.start_services():
            return False

        # Wait for Neo4j
        if not self.wait_for_neo4j():
            print("🔧 Versuche Services zu stoppen...")
            self.stop_services()
            return False

        self.show_connection_info()

        if mode == "demo":
            asyncio.run(self.run_demo())
        elif mode == "chat":
            self.launch_rag_system()
        elif mode == "full":
            asyncio.run(self.run_demo())
            input("\n👀 Drücke Enter um das RAG Chat-System zu starten...")
            self.launch_rag_system()

        return True


def main():
    """Hauptfunktion"""
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j Community RAG System Launcher")
    parser.add_argument(
        "--mode",
        choices=["full", "demo", "chat", "stop"],
        default="full",
        help="Launch mode (default: full)",
    )

    args = parser.parse_args()

    launcher = CommunityRAGLauncher()

    if args.mode == "stop":
        launcher.stop_services()
    else:
        launcher.launch(args.mode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Interactive RAG Chat Interface
=============================

Eine benutzerfreundliche Chat-Oberfläche für das Smart RAG System.
Unterstützt sowohl Terminal- als auch Streamlit-basierte Interaktion.
"""

import asyncio
import time
from datetime import datetime

from intelligent_data_import import create_intelligent_rag_system

from src.rag_system import RAGConfig


class RAGChatInterface:
    """Interaktive Chat-Oberfläche für das RAG-System"""

    def __init__(self):
        self.rag_system = None
        self.smart_rag = None
        self.chat_history = []
        self.session_start = datetime.now()

    async def initialize_system(self):
        """Initialisiere das RAG-System mit robuster Konfiguration"""
        print("🚀 Initialisiere Smart RAG Chat System mit Intelligent Data Import...")

        # Robuste Enterprise-Konfiguration
        config = RAGConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password123",
            neo4j_database="neo4j",  # Verwende die reparierte neo4j Datenbank
            llm_provider="ollama",  # Verwende Ollama für bessere Antworten
            documents_path="data/documents",
        )

        # Teste Neo4j Verbindung
        neo4j_available = await self._test_neo4j_connection(config)
        if not neo4j_available:
            print("📝 Neo4j nicht verfügbar - verwende lokalen Modus")
            config.neo4j_password = None

        # Erstelle intelligentes RAG System mit Auto-Import
        self.smart_rag = create_intelligent_rag_system(config)

        print("✅ Intelligentes RAG Chat System erfolgreich initialisiert!")
        print("💡 Features: Self-Learning + Automatic Knowledge Gap Filling")
        print(f"🗄️ Modus: {'Neo4j Enterprise' if neo4j_available else 'Lokal'}")
        return True

    async def _test_neo4j_connection(self, config):
        """Teste Neo4j Verbindung"""
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
            )

            # Versuche system DB
            try:
                with driver.session(database="system") as session:
                    session.run("RETURN 1")
                driver.close()
                return True
            except BaseException:
                # Fallback zu Standard-DB
                try:
                    with driver.session() as session:
                        session.run("RETURN 1")
                    config.neo4j_database = None
                    driver.close()
                    return True
                except BaseException:
                    driver.close()
                    return False
        except BaseException:
            return False

    async def start_chat(self):
        """Starte den interaktiven Chat"""
        if not await self.initialize_system():
            print("❌ System konnte nicht initialisiert werden")
            return

        print("\n" + "=" * 60)
        print("🤖 Smart RAG Chat Interface")
        print("=" * 60)
        print("💡 Stellen Sie mir Fragen zu Ihren Dokumenten!")
        print("🔄 Das System lernt von jedem Gespräch")
        print("❓ Verfügbare Befehle:")
        print("   /help    - Hilfe anzeigen")
        print("   /stats   - Statistiken anzeigen")
        print("   /history - Chat-Verlauf")
        print("   /clear   - Verlauf löschen")
        print("   /quit    - Chat beenden")
        print("=" * 60)

        while True:
            try:
                # Benutzereingabe
                user_input = input("\n🧑‍💻 Sie: ").strip()

                if not user_input:
                    continue

                # Befehle verarbeiten
                if user_input.startswith("/"):
                    if await self._handle_command(user_input):
                        break
                    continue

                # RAG Query
                await self._process_user_query(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Chat wurde unterbrochen. Auf Wiedersehen!")
                break
            except Exception as e:
                print(f"\n❌ Fehler: {e}")

    async def _handle_command(self, command):
        """Verarbeite Chat-Befehle"""
        cmd = command.lower()

        if cmd == "/quit":
            print("\n👋 Auf Wiedersehen!")
            return True

        elif cmd == "/help":
            print("\n📚 Verfügbare Befehle:")
            print("   /help    - Diese Hilfe")
            print("   /stats   - System-Statistiken")
            print("   /history - Chat-Verlauf anzeigen")
            print("   /clear   - Verlauf löschen")
            print("   /quit    - Chat beenden")
            print("\n💡 Tipp: Stellen Sie Fragen zu:")
            print("   • Erneuerbaren Energien")
            print("   • Machine Learning")
            print("   • Ihren hochgeladenen Dokumenten")

        elif cmd == "/stats":
            await self._show_statistics()

        elif cmd == "/history":
            self._show_chat_history()

        elif cmd == "/clear":
            self.chat_history.clear()
            print("🧹 Chat-Verlauf wurde gelöscht")

        else:
            print(f"❓ Unbekannter Befehl: {command}")
            print("💡 Verwenden Sie /help für eine Liste der verfügbaren Befehle")

        return False

    async def _process_user_query(self, query):
        """Verarbeite Benutzeranfrage mit intelligenter Auto-Import-Funktionalität"""
        print(
            "\n🤖 KI: Einen Moment, ich analysiere Ihre Frage und durchsuche meine Wissensdatenbank..."
        )

        start_time = time.time()

        try:
            # Verwende intelligente Enhanced Query mit Auto-Import
            result = await self.smart_rag.intelligent_enhanced_query(query)

            end_time = time.time()
            processing_time = end_time - start_time

            # Antwort anzeigen
            answer = result.get(
                "answer", "Entschuldigung, ich konnte keine Antwort finden."
            )

            print(f"\n🤖 KI: {answer}")

            # Erweiterte Metadaten anzeigen
            contexts = result.get("contexts", [])
            sources_count = len(contexts)
            workflow_type = result.get("workflow_type", "standard")

            # Basis-Info
            info_parts = [
                f"{processing_time:.2f}s",
                f"{sources_count} Quellen",
                f"Strategie: {result.get('learning_metadata', {}).get('strategy', 'standard')}",
            ]

            # Auto-Import Info falls ausgelöst
            if result.get("auto_import_triggered", False):
                imported_concepts = result.get("imported_concepts", [])
                print(
                    f"\n🔄 WISSEN ERWEITERT: Neue Konzepte hinzugefügt: {', '.join(imported_concepts)}"
                )
                enhancement = result.get("knowledge_enhancement", {})
                print(
                    f"📈 Qualitätsverbesserung: {enhancement.get('original_quality', 0):.2f} → Erweitert"
                )
                info_parts.append("Auto-Import: ✅")
            else:
                info_parts.append("Auto-Import: -")

            print(f"\n📊 Info: {' | '.join(info_parts)}")

            # In Chat-History speichern mit erweiterten Daten
            self.chat_history.append(
                {
                    "timestamp": datetime.now(),
                    "user_query": query,
                    "ai_response": answer,
                    "processing_time": processing_time,
                    "sources_count": sources_count,
                    "strategy": result.get("learning_metadata", {}).get(
                        "strategy", "standard"
                    ),
                    "workflow_type": workflow_type,
                    "auto_import_triggered": result.get("auto_import_triggered", False),
                    "imported_concepts": result.get("imported_concepts", []),
                }
            )

            # Bewertung erfragen (optional)
            await self._ask_for_feedback(result.get("query_id", ""), answer)

        except Exception as e:
            print(
                f"\n❌ KI: Es tut mir leid, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten: {e}"
            )
            import traceback

            traceback.print_exc()  # Für Debugging

    async def _ask_for_feedback(self, query_id, answer):
        """Einfaches Feedback-System"""
        try:
            feedback = (
                input(
                    "\n⭐ War diese Antwort hilfreich? (j/n/Enter zum Überspringen): "
                )
                .strip()
                .lower()
            )

            if feedback in ["j", "ja", "y", "yes"]:
                rating = 5.0
                print("✅ Danke für Ihr positives Feedback!")
            elif feedback in ["n", "nein", "no"]:
                rating = 2.0
                print(
                    "📝 Danke für Ihr Feedback. Ich werde versuchen, beim nächsten Mal besser zu werden!"
                )
            else:
                return  # Kein Feedback

            # Feedback an das Learning System senden
            await self.smart_rag.record_user_feedback(
                query_id, rating, {"manual_feedback": True, "interface": "chat"}
            )

        except BaseException:
            pass  # Feedback ist optional

    async def _show_statistics(self):
        """Zeige System-Statistiken"""
        print("\n📊 Chat-Statistiken:")
        print(f"   🕒 Sitzungsdauer: {datetime.now() - self.session_start}")
        print(f"   💬 Anzahl Fragen: {len(self.chat_history)}")

        if self.chat_history:
            avg_time = sum(h["processing_time"] for h in self.chat_history) / len(
                self.chat_history
            )
            total_sources = sum(h["sources_count"] for h in self.chat_history)

            print(f"   ⚡ Durchschn. Antwortzeit: {avg_time:.2f}s")
            print(f"   📚 Verwendete Quellen: {total_sources}")

        try:
            # Learning System Insights
            insights = await self.smart_rag.get_learning_insights()
            print(f"   🧠 System-Queries total: {insights.get('total_queries', 0)}")
            print(f"   🎯 Erkannte Query-Typen: {len(insights.get('query_types', {}))}")
        except BaseException:
            print("   🧠 Learning-Insights nicht verfügbar")

    def _show_chat_history(self):
        """Zeige Chat-Verlauf"""
        if not self.chat_history:
            print("\n📝 Noch keine Gespräche geführt")
            return

        print(f"\n📜 Chat-Verlauf ({len(self.chat_history)} Einträge):")
        print("-" * 60)

        for i, entry in enumerate(self.chat_history[-10:], 1):  # Letzte 10 Einträge
            timestamp = entry["timestamp"].strftime("%H:%M:%S")
            query_preview = (
                entry["user_query"][:40] + "..."
                if len(entry["user_query"]) > 40
                else entry["user_query"]
            )
            response_preview = (
                entry["ai_response"][:60] + "..."
                if len(entry["ai_response"]) > 60
                else entry["ai_response"]
            )

            print(f"{i:2d}. [{timestamp}] Sie: {query_preview}")
            print(f"    KI: {response_preview}")
            print(
                f"    📊 {entry['processing_time']:.2f}s | {entry['sources_count']} Quellen"
            )
            print()


async def main():
    """Hauptfunktion für Chat Interface"""
    print("🤖 Smart RAG Chat Interface wird gestartet...")

    chat = RAGChatInterface()
    await chat.start_chat()


if __name__ == "__main__":
    asyncio.run(main())

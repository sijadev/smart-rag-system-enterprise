#!/usr/bin/env python3
"""
Modernisierte RAG Chat Anwendung
===============================

Nutzt zentrale Konfiguration mit Dependency Injection
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Lade src zum Python-Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.central_config import (
    get_config, get_container, inject, configure_logging,
    print_config_summary, CentralConfig, OllamaConfig
)
from src.modern_factory import ModernLLMServiceFactory, register_llm_services
from src.interfaces import ILLMService, QueryContext

logger = logging.getLogger(__name__)


class ModernRAGChatSession:
    """Modernisierte Chat-Session mit DI"""

    def __init__(self):
        self.session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history: List[Dict[str, str]] = []
        self.llm_service: Optional[ILLMService] = None
        self.config: Optional[CentralConfig] = None

    async def initialize(self) -> bool:
        """Initialisiert die Chat-Session mit DI"""
        try:
            # Hole zentrale Konfiguration direkt
            container = get_container()
            self.config = container.resolve(CentralConfig)

            print("🚀 Initialisiere Modern RAG Chat System...")
            print(f"🌍 Environment: {self.config.system.environment.value}")

            # Registriere Services
            register_llm_services()

            # Hole LLM Service über DI
            self.llm_service = container.resolve(ILLMService)

            # Initialisiere Service
            await self.llm_service.__aenter__()
            await self.llm_service.initialize()

            provider_info = self.llm_service.get_provider_info()
            print(f"✅ LLM Service bereit:")
            print(f"   🤖 Model: {provider_info['model']}")
            print(f"   🔢 Embedding Model: {provider_info['embedding_model']}")
            print(f"   🌐 Provider: {provider_info['provider']}")

            return True

        except Exception as e:
            logger.error(f"Fehler bei Initialisierung: {e}")
            print(f"❌ Initialisierung fehlgeschlagen: {e}")
            return False

    async def chat(self, user_input: str) -> str:
        """Verarbeitet Chat mit zentraler Konfiguration"""
        if not self.llm_service:
            return "❌ System nicht initialisiert"

        try:
            # Query-Kontext mit zentraler Config erstellen
            context = QueryContext(
                query_id=f"query_{len(self.conversation_history) + 1}",
                session_id=self.session_id,
                previous_queries=[msg["user"] for msg in self.conversation_history[-3:]],
                metadata={
                    "system_message": "Du bist ein hilfsbereiches AI-Assistant. Antworte präzise und freundlich auf Deutsch.",
                    "environment": self.config.system.environment.value,
                    "debug": self.config.system.debug
                }
            )

            # Generiere Antwort
            response = await self.llm_service.generate(user_input, context)

            # Speichere in Verlauf
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })

            return response

        except Exception as e:
            logger.error(f"Fehler bei Chat-Verarbeitung: {e}")
            return f"❌ Fehler bei der Antwort-Generierung: {e}"

    async def cleanup(self):
        """Aufräumen der Session"""
        if self.llm_service:
            await self.llm_service.__aexit__(None, None, None)

    def get_session_info(self) -> Dict[str, Any]:
        """Gibt Session-Informationen zurück"""
        return {
            "session_id": self.session_id,
            "total_messages": len(self.conversation_history),
            "environment": self.config.system.environment.value if self.config else "unknown",
            "model": self.llm_service.get_provider_info()['model'] if self.llm_service else "unknown"
        }


async def main():
    """Modernisierte Hauptfunktion mit zentraler Konfiguration"""

    # Lade und konfiguriere System
    config = get_config()
    configure_logging(config.system)

    print("🔧 Smart RAG System - Zentrale Konfiguration")
    print("=" * 60)

    # Zeige Konfiguration wenn Debug-Modus
    if config.system.debug:
        print_config_summary()
        print()

    # Validiere Konfiguration
    errors = config.validate()
    if errors:
        print("⚠️ Konfigurationsfehler gefunden:")
        for error in errors:
            print(f"   • {error}")
        print()

    # Session erstellen und initialisieren
    session = ModernRAGChatSession()

    if not await session.initialize():  # Entferne den config Parameter
        print("💥 Kann Chat-System nicht starten")
        return 1

    print("🚀 Modern RAG Chat gestartet! Geben Sie Ihre erste Frage ein...")
    print("💡 Verwenden Sie /help für Hilfe oder /quit zum Beenden\n")

    try:
        while True:
            try:
                # Benutzer-Input
                user_input = input("👤 Sie: ").strip()

                if not user_input:
                    continue

                # Befehle verarbeiten
                if user_input.startswith('/'):
                    command = user_input[1:].lower()

                    if command in ['quit', 'q', 'exit']:
                        print("👋 Auf Wiedersehen!")
                        break

                    elif command in ['help', 'h']:
                        print("""
🆘 Modern RAG Chat - Hilfe
=========================

💬 Chat-Befehle:
   /status  - System-Status anzeigen
   /config  - Aktuelle Konfiguration anzeigen  
   /info    - Session-Informationen
   /clear   - Verlauf löschen
   /quit    - Chat beenden

🔧 Features:
   • Zentrale Konfiguration mit .env-Support
   • Dependency Injection für alle Services
   • Automatische Modell-Auswahl basierend auf Environment
   • Kontextualisierte Konversationen
""")
                        continue

                    elif command == 'status':
                        if session.llm_service:
                            provider_info = session.llm_service.get_provider_info()
                            print(f"\n🔧 System-Status:")
                            print(f"   🤖 Model: {provider_info['model']}")
                            print(f"   🔢 Embedding Model: {provider_info['embedding_model']}")
                            print(f"   🌐 Provider: {provider_info['provider']}")
                            print(f"   📊 Verfügbare Modelle: {len(provider_info['available_models'])}")
                            print(f"   🌍 Environment: {config.system.environment.value}")
                            print(f"   📝 Debug Mode: {config.system.debug}")
                        continue

                    elif command == 'config':
                        print_config_summary()
                        continue

                    elif command == 'info':
                        info = session.get_session_info()
                        print(f"\n📊 Session-Informationen:")
                        print(f"   🆔 Session ID: {info['session_id']}")
                        print(f"   💬 Nachrichten: {info['total_messages']}")
                        print(f"   🌍 Environment: {info['environment']}")
                        print(f"   🤖 Model: {info['model']}")
                        continue

                    elif command == 'clear':
                        session.conversation_history.clear()
                        print("🗑️ Konversations-Verlauf gelöscht")
                        continue

                    else:
                        print(f"❌ Unbekannter Befehl: /{command}")
                        print("💡 Verwenden Sie /help für verfügbare Befehle")
                        continue

                # Chat-Verarbeitung
                print("🤖 AI: ", end="", flush=True)
                start_time = datetime.now()

                response = await session.chat(user_input)

                duration = (datetime.now() - start_time).total_seconds()

                print(response)

                # Zeige Performance-Info im Debug-Modus
                if config.system.debug:
                    print(f"⏱️ Antwortzeit: {duration:.2f}s")
                print()

            except KeyboardInterrupt:
                print("\n👋 Chat durch Strg+C beendet")
                break
            except EOFError:
                print("\n👋 Chat beendet")
                break

    finally:
        await session.cleanup()

    return 0


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n👋 Programm beendet")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Unerwarteter Fehler: {e}")
        logger.error(f"Unerwarteter Fehler: {e}")
        sys.exit(1)

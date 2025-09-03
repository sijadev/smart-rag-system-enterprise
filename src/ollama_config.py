#!/usr/bin/env python3
"""
Ollama-spezifische Konfiguration und Utilities
==============================================

Konfiguration und Hilfsfunktionen für Ollama-Integration
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from .llm_services import OllamaLLMService, OllamaConfig
from .interfaces import LLMProvider

# Lade .env-Datei automatisch
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Falls python-dotenv nicht installiert ist, ignoriere es
    pass

logger = logging.getLogger(__name__)


@dataclass
class OllamaSystemConfig:
    """System-weite Ollama-Konfiguration"""
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2:latest"  # Aktualisiert von llama2
    default_embedding_model: str = "nomic-embed-text:latest"  # Aktualisiert von llama2
    chat_model: str = None  # Für Chat-Completion
    timeout: int = 300
    max_retries: int = 3
    auto_pull_models: bool = True
    recommended_models: List[str] = None

    # Enterprise-spezifische Einstellungen
    embedding_dimensions: int = 768
    embedding_batch_size: int = 32
    use_local_embeddings: bool = True

    def __post_init__(self):
        # Lade Werte aus .env-Datei
        self.base_url = os.getenv("OLLAMA_BASE_URL", self.base_url)
        self.default_model = os.getenv("LLM_MODEL", self.default_model)
        self.default_embedding_model = os.getenv("EMBED_MODEL", self.default_embedding_model)
        self.chat_model = os.getenv("ANALYZER_MODEL", self.chat_model)

        # Numerische Werte sicher parsen
        try:
            self.timeout = int(os.getenv("OLLAMA_TIMEOUT", str(self.timeout)))
        except ValueError:
            pass

        try:
            self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", str(self.max_retries)))
        except ValueError:
            pass

        if self.recommended_models is None:
            self.recommended_models = [
                "llama3.2:latest",  # Aktualisiert von llama2
                "llama3.1:8b",
                "llama3.1:latest",
                "mistral:latest",
                "codellama:latest",
                "codellama:7b",
                "nomic-embed-text:latest"
            ]

        # Chat-Model Fallback
        if self.chat_model is None:
            self.chat_model = self.default_model

class OllamaHealthChecker:
    """Ollama Gesundheitscheck und Monitoring"""

    def __init__(self, config: OllamaSystemConfig):
        self.config = config

    async def check_ollama_status(self) -> Dict[str, Any]:
        """Überprüft den Status von Ollama"""
        try:
            # Temporären Service für Status-Check erstellen
            temp_config = OllamaConfig(base_url=self.config.base_url, timeout=30)

            async with OllamaLLMService(temp_config) as service:
                await service.initialize()

                # Informationen sammeln
                provider_info = service.get_provider_info()
                running_models = await service.list_running_models()

                return {
                    "status": "healthy",
                    "base_url": self.config.base_url,
                    "available_models": provider_info.get("available_models", []),
                    "running_models": running_models,
                    "default_model": self.config.default_model,
                    "connection": "successful"
                }

        except Exception as e:
            logger.error(f"Ollama Health Check fehlgeschlagen: {e}")
            return {
                "status": "unhealthy",
                "base_url": self.config.base_url,
                "error": str(e),
                "connection": "failed"
            }

    async def ensure_models_available(self, models: List[str] = None) -> Dict[str, bool]:
        """Stellt sicher, dass erforderliche Modelle verfügbar sind"""
        models = models or [self.config.default_model, self.config.default_embedding_model]
        results = {}

        try:
            temp_config = OllamaConfig(base_url=self.config.base_url)

            async with OllamaLLMService(temp_config) as service:
                await service.initialize()
                available_models = service.get_provider_info().get("available_models", [])

                for model in models:
                    model_base = model.split(":")[0]  # Entferne Tag falls vorhanden
                    if model_base in available_models:
                        results[model] = True
                        logger.info(f"Modell {model} ist verfügbar")
                    elif self.config.auto_pull_models:
                        try:
                            logger.info(f"Lade Modell {model} herunter...")
                            await service._pull_model(model)
                            results[model] = True
                            logger.info(f"Modell {model} erfolgreich geladen")
                        except Exception as e:
                            logger.error(f"Fehler beim Laden von Modell {model}: {e}")
                            results[model] = False
                    else:
                        results[model] = False
                        logger.warning(f"Modell {model} nicht verfügbar und auto_pull deaktiviert")

        except Exception as e:
            logger.error(f"Fehler bei Modell-Check: {e}")
            for model in models:
                results[model] = False

        return results


class OllamaConfigurationManager:
    """Manager für Ollama-Konfigurationen"""

    @staticmethod
    def load_from_env() -> OllamaSystemConfig:
        """Lädt Konfiguration aus Umgebungsvariablen - nutzt bestehende .env-Dateien"""

        # Legacy-Support für bestehende ENV-Variablen
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Primäre Modell-Konfiguration - priorisiere spezifische OLLAMA_ Variablen, dann LLM_MODEL
        default_model = (
            os.getenv("OLLAMA_MODEL") or
            os.getenv("LLM_MODEL") or
            "llama3.2:latest"  # Besserer Fallback statt llama2
        )

        # Embedding-Modell-Konfiguration
        embedding_model = (
            os.getenv("OLLAMA_EMBEDDING_MODEL") or
            os.getenv("EMBED_MODEL") or
            "nomic-embed-text:latest"  # Besserer Fallback
        )

        # Chat-Modell (für Enterprise-Setup)
        chat_model = (
            os.getenv("OLLAMA_CHAT_MODEL") or
            os.getenv("ANALYZER_MODEL") or
            default_model
        )

        # Timeout-Konfiguration
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))

        # Enterprise-Einstellungen mit robustem Parsing
        def parse_int_with_comment(value: str, default: int) -> int:
            """Parst Integer-Werte und ignoriert Kommentare"""
            if not value:
                return default
            try:
                # Entferne Kommentare nach #
                clean_value = value.split('#')[0].strip()
                return int(clean_value) if clean_value else default
            except (ValueError, AttributeError):
                return default

        embedding_dimensions = parse_int_with_comment(os.getenv("EMBEDDING_DIMENSIONS"), 768)
        embedding_batch_size = parse_int_with_comment(os.getenv("EMBEDDING_BATCH_SIZE"), 32)
        use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"

        # Auto-Pull basierend auf Environment
        environment = os.getenv("ENVIRONMENT", "development")
        auto_pull = os.getenv("OLLAMA_AUTO_PULL", "true" if environment == "development" else "false").lower() == "true"

        # Empfohlene Modelle aus ENV
        recommended_models_str = os.getenv("OLLAMA_RECOMMENDED_MODELS", "")
        recommended_models = [m.strip() for m in recommended_models_str.split(",") if m.strip()] if recommended_models_str else None

        logger.info(f"Lade Ollama-Konfiguration aus .env-Dateien:")
        logger.info(f"  Base URL: {base_url}")
        logger.info(f"  Default Model: {default_model}")
        logger.info(f"  Embedding Model: {embedding_model}")
        logger.info(f"  Chat Model: {chat_model}")
        logger.info(f"  Environment: {environment}")

        return OllamaSystemConfig(
            base_url=base_url,
            default_model=default_model,
            default_embedding_model=embedding_model,
            chat_model=chat_model,
            timeout=timeout,
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
            auto_pull_models=auto_pull,
            recommended_models=recommended_models,
            embedding_dimensions=embedding_dimensions,
            embedding_batch_size=embedding_batch_size,
            use_local_embeddings=use_local_embeddings
        )

    @staticmethod
    def create_llm_config(system_config: OllamaSystemConfig, **overrides) -> OllamaConfig:
        """Erstellt LLM-Service-Konfiguration aus System-Konfiguration"""
        base_config = {
            "base_url": system_config.base_url,
            "model": system_config.default_model,
            "embedding_model": system_config.default_embedding_model,
            "timeout": system_config.timeout,
            "max_retries": system_config.max_retries
        }

        # Überschreibungen anwenden
        base_config.update(overrides)

        return OllamaConfig(**base_config)

    @staticmethod
    def create_chat_config(system_config: OllamaSystemConfig, **overrides) -> OllamaConfig:
        """Erstellt Chat-spezifische Konfiguration"""
        chat_config = {
            "base_url": system_config.base_url,
            "model": system_config.chat_model,
            "embedding_model": system_config.default_embedding_model,
            "timeout": system_config.timeout,
            "max_retries": system_config.max_retries,
            "temperature": float(os.getenv("TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "1500"))
        }

        chat_config.update(overrides)
        return OllamaConfig(**chat_config)


class OllamaIntegrationHelper:
    """Hilfsfunktionen für die Ollama-Integration"""

    @staticmethod
    async def setup_ollama_for_rag(
        system_config: Optional[OllamaSystemConfig] = None,
        models_needed: List[str] = None
    ) -> Dict[str, Any]:
        """Komplette Ollama-Setup für RAG System"""

        if system_config is None:
            system_config = OllamaConfigurationManager.load_from_env()

        if models_needed is None:
            models_needed = [
                system_config.default_model,
                system_config.default_embedding_model
            ]

        logger.info("Starte Ollama-Setup für RAG System...")

        # Health Check
        health_checker = OllamaHealthChecker(system_config)
        health_status = await health_checker.check_ollama_status()

        if health_status["status"] != "healthy":
            logger.error(f"Ollama nicht gesund: {health_status}")
            return {
                "success": False,
                "error": "Ollama Health Check fehlgeschlagen",
                "details": health_status
            }

        # Modelle überprüfen/laden
        model_status = await health_checker.ensure_models_available(models_needed)

        missing_models = [model for model, available in model_status.items() if not available]
        if missing_models:
            logger.warning(f"Nicht verfügbare Modelle: {missing_models}")

        # LLM Service erstellen und testen
        try:
            llm_config = OllamaConfigurationManager.create_llm_config(system_config)

            async with OllamaLLMService(llm_config) as service:
                await service.initialize()

                # Test-Generation
                test_response = await service.generate("Hallo, funktioniert die Ollama-Integration?")

                # Test-Embedding
                test_embedding = await service.embed("Test Text für Embeddings")

                logger.info("Ollama-Integration erfolgreich getestet")

                return {
                    "success": True,
                    "health_status": health_status,
                    "model_status": model_status,
                    "test_generation": len(test_response) > 0,
                    "test_embedding": len(test_embedding) > 0,
                    "embedding_dimensions": len(test_embedding),
                    "available_models": health_status.get("available_models", []),
                    "config": asdict(system_config)
                }

        except Exception as e:
            logger.error(f"Ollama-Integration Test fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": f"Service Test fehlgeschlagen: {e}",
                "health_status": health_status,
                "model_status": model_status
            }

    @staticmethod
    def get_startup_recommendations() -> List[str]:
        """Gibt Empfehlungen für den Ollama-Start zurück"""
        return [
            "Stelle sicher, dass Ollama läuft: 'ollama serve'",
            "Empfohlene Modelle für RAG: llama3.2, llama3.1, mistral",  # Aktualisiert von llama2
            "Für Embeddings: nomic-embed-text oder andere spezialisierte Embedding-Modelle",  # Aktualisiert
            "Für Code-Generierung: codellama",
            "Überprüfe verfügbaren Speicher für größere Modelle",
            "Konfiguriere OLLAMA_BASE_URL wenn Ollama nicht lokal läuft"
        ]


async def quick_ollama_test() -> bool:
    """Schneller Test der Ollama-Verbindung"""
    try:
        config = OllamaConfigurationManager.load_from_env()
        health_checker = OllamaHealthChecker(config)

        status = await health_checker.check_ollama_status()
        return status["status"] == "healthy"

    except Exception as e:
        logger.error(f"Schneller Ollama-Test fehlgeschlagen: {e}")
        return False


if __name__ == "__main__":
    # Beispiel-Verwendung
    async def main():
        print("🔍 Ollama Integration Test...")

        result = await OllamaIntegrationHelper.setup_ollama_for_rag()

        if result["success"]:
            print("✅ Ollama-Integration erfolgreich!")
            print(f"📊 Verfügbare Modelle: {result['available_models']}")
            print(f"🔢 Embedding Dimensionen: {result['embedding_dimensions']}")
        else:
            print("❌ Ollama-Integration fehlgeschlagen!")
            print(f"❗ Fehler: {result['error']}")

        print("\n💡 Empfehlungen:")
        for rec in OllamaIntegrationHelper.get_startup_recommendations():
            print(f"   • {rec}")

    asyncio.run(main())

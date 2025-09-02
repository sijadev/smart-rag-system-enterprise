#!/usr/bin/env python3
"""
LLM Services für Smart RAG System
=================================

Implementierungen für verschiedene LLM-Provider inkl. Ollama
Nutzt zentrale Konfiguration mit Dependency Injection
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from dataclasses import dataclass

from src.interfaces import ILLMService, QueryContext
from src.central_config import get_config, inject, OllamaConfig

logger = logging.getLogger(__name__)


@dataclass
class OllamaServiceConfig:
    """Legacy-Wrapper für OllamaConfig"""
    base_url: str
    model: str
    embedding_model: str
    timeout: int = 300
    max_retries: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2000


class OllamaLLMService(ILLMService):
    """Ollama LLM Service Implementation mit zentraler Konfiguration"""

    def __init__(self, config: Optional[OllamaServiceConfig] = None):
        if config is None:
            # Nutze zentrale Konfiguration
            central_config = get_config()
            config = OllamaServiceConfig(
                base_url=central_config.ollama.base_url,
                model=central_config.ollama.model,
                embedding_model=central_config.ollama.embedding_model,
                timeout=central_config.ollama.timeout,
                max_retries=central_config.ollama.max_retries,
                temperature=central_config.ollama.temperature,
                max_tokens=central_config.ollama.max_tokens
            )

        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout)
        )
        self._available_models: Optional[List[str]] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.aclose()

    async def initialize(self):
        """Initialisiert den Ollama Service"""
        try:
            # Teste Verbindung zu Ollama
            await self._check_connection()

            # Lade verfügbare Modelle
            self._available_models = await self._get_available_models()

            # Stelle sicher, dass das gewünschte Modell verfügbar ist
            if self.config.model not in self._available_models:
                logger.warning(f"Model '{self.config.model}' nicht verfügbar. Versuche zu pullen...")
                await self._pull_model(self.config.model)

            if self.config.embedding_model not in self._available_models:
                logger.warning(f"Embedding model '{self.config.embedding_model}' nicht verfügbar. Versuche zu pullen...")
                await self._pull_model(self.config.embedding_model)

            logger.info(f"Ollama Service initialisiert mit Model: {self.config.model}")

        except Exception as e:
            logger.error(f"Fehler bei Ollama-Initialisierung: {e}")
            raise

    async def generate(self, prompt: str, context: Optional[QueryContext] = None) -> str:
        """Generiert Antwort basierend auf Prompt"""
        try:
            # Erstelle Ollama-Request
            request_data = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens
                }
            }

            # Füge Kontext hinzu falls vorhanden
            if context and context.metadata:
                system_message = context.metadata.get("system_message")
                if system_message:
                    request_data["system"] = system_message

            logger.debug(f"Sende Anfrage an Ollama: {self.config.model}")

            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.post(
                        "/api/generate",
                        json=request_data
                    )
                    response.raise_for_status()

                    result = response.json()
                    answer = result.get("response", "").strip()

                    if answer:
                        logger.debug(f"Ollama Antwort erhalten ({len(answer)} Zeichen)")
                        return answer
                    else:
                        raise ValueError("Leere Antwort von Ollama erhalten")

                except httpx.HTTPStatusError as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(f"HTTP Fehler (Versuch {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)

                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(f"Fehler bei Ollama-Anfrage (Versuch {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"Fehler bei Text-Generierung mit Ollama: {e}")
            raise

    async def embed(self, text: str) -> List[float]:
        """Erstellt Text-Embeddings mit Ollama"""
        try:
            request_data = {
                "model": self.config.embedding_model,
                "prompt": text
            }

            logger.debug(f"Erstelle Embeddings mit Ollama: {self.config.embedding_model}")

            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.post(
                        "/api/embeddings",
                        json=request_data
                    )
                    response.raise_for_status()

                    result = response.json()
                    embeddings = result.get("embedding", [])

                    if embeddings and isinstance(embeddings, list):
                        logger.debug(f"Embeddings erstellt: {len(embeddings)} Dimensionen")
                        return embeddings
                    else:
                        raise ValueError("Keine gültigen Embeddings von Ollama erhalten")

                except httpx.HTTPStatusError as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(f"HTTP Fehler beim Embedding (Versuch {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)

                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(f"Fehler beim Embedding (Versuch {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"Fehler bei Embedding-Erstellung mit Ollama: {e}")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """Gibt Ollama Provider-Informationen zurück"""
        return {
            "provider": "ollama",
            "base_url": self.config.base_url,
            "model": self.config.model,
            "embedding_model": self.config.embedding_model,
            "available_models": self._available_models or [],
            "config": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens
            }
        }

    async def _check_connection(self):
        """Prüft Verbindung zu Ollama"""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            logger.info("Verbindung zu Ollama erfolgreich")
        except Exception as e:
            logger.error(f"Keine Verbindung zu Ollama möglich: {e}")
            raise ConnectionError(f"Ollama nicht erreichbar unter {self.config.base_url}: {e}")

    async def _get_available_models(self) -> List[str]:
        """Lädt verfügbare Modelle von Ollama"""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [model.get("name", "").split(":")[0] for model in data.get("models", [])]
            models = list(set(filter(None, models)))  # Entferne Duplikate und leere Namen

            logger.info(f"Verfügbare Ollama Modelle: {models}")
            return models

        except Exception as e:
            logger.warning(f"Konnte verfügbare Modelle nicht laden: {e}")
            return []

    async def _pull_model(self, model_name: str):
        """Lädt ein Modell herunter"""
        try:
            logger.info(f"Lade Modell herunter: {model_name}")

            request_data = {"name": model_name}

            async with self.client.stream(
                "POST",
                "/api/pull",
                json=request_data
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                logger.info(f"Pull Status: {data['status']}")
                        except json.JSONDecodeError:
                            continue

            logger.info(f"Modell erfolgreich geladen: {model_name}")

        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells {model_name}: {e}")
            raise

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Gibt Informationen zu einem spezifischen Modell zurück"""
        model = model_name or self.config.model
        try:
            response = await self.client.post(
                "/api/show",
                json={"name": model}
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Konnte Model-Info für {model} nicht laden: {e}")
            return {}

    async def list_running_models(self) -> List[Dict[str, Any]]:
        """Listet aktuell laufende Modelle auf"""
        try:
            response = await self.client.get("/api/ps")
            response.raise_for_status()

            data = response.json()
            return data.get("models", [])

        except Exception as e:
            logger.error(f"Konnte laufende Modelle nicht abrufen: {e}")
            return []

    @classmethod
    def create_from_central_config(cls, config: OllamaConfig) -> 'OllamaLLMService':
        """Erstellt Service aus zentraler Konfiguration mit DI"""
        service_config = OllamaServiceConfig(
            base_url=config.base_url,
            model=config.model,
            embedding_model=config.embedding_model,
            timeout=config.timeout,
            max_retries=config.max_retries,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        return cls(service_config)


class OllamaServiceFactory:
    """Factory für OllamaLLMService Instanzen"""

    @staticmethod
    def create_service(config: Optional[OllamaServiceConfig] = None) -> OllamaLLMService:
        """Erstellt OllamaLLMService Instanz"""
        return OllamaLLMService(config)

    @staticmethod
    def create_from_config(config: OllamaConfig) -> OllamaLLMService:
        """Erstellt Service aus OllamaConfig"""
        return OllamaLLMService.create_from_central_config(config)

    @staticmethod
    def create_default() -> OllamaLLMService:
        """Erstellt Service mit Standard-Konfiguration"""
        return OllamaLLMService()


# Mock LLM Service für Tests
class MockLLMService(ILLMService):
    """Mock LLM Service für Tests und Development"""

    def __init__(self):
        self.call_count = 0

    async def generate(self, prompt: str, context: Optional[QueryContext] = None) -> str:
        """Mock-Implementierung für generate"""
        self.call_count += 1
        return f"Mock response {self.call_count} for prompt: {prompt[:50]}..."

    async def embed(self, text: str) -> List[float]:
        """Mock-Implementierung für embed"""
        # Erstelle deterministische Mock-Embeddings basierend auf Text
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Konvertiere zu 384-dimensionalem Vektor (Standard für viele Modelle)
        mock_embedding = []
        for i in range(384):
            # Pseudo-zufällige Werte basierend auf Hash
            value = int(text_hash[i % len(text_hash)], 16) / 15.0 - 0.5
            mock_embedding.append(value)
        return mock_embedding

    def get_provider_info(self) -> Dict[str, Any]:
        """Mock Provider Info"""
        return {
            "provider": "mock",
            "model": "mock-model",
            "call_count": self.call_count
        }


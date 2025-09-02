#!/usr/bin/env python3
"""
Modernisierte Factory mit zentraler Konfiguration
================================================

Factory Pattern mit Dependency Injection für das Smart RAG System
"""

from typing import Dict, Any, Optional, Type
from src.interfaces import ILLMService
from src.central_config import get_container, OllamaConfig, CentralConfig
from src.llm_services import OllamaLLMService
import logging

logger = logging.getLogger(__name__)


class ModernLLMServiceFactory:
    """Modernisierte LLM Service Factory mit zentraler Konfiguration"""

    @staticmethod
    def create_ollama_service(ollama_config: OllamaConfig = None) -> OllamaLLMService:
        """Erstellt Ollama Service mit zentraler Konfiguration"""
        if ollama_config is None:
            container = get_container()
            ollama_config = container.resolve(OllamaConfig)

        logger.info(f"Erstelle Ollama Service mit Model: {ollama_config.model}")
        return OllamaLLMService.create_from_central_config(ollama_config)

    @staticmethod
    def create_default_service() -> ILLMService:
        """Erstellt Standard LLM Service aus zentraler Konfiguration"""
        container = get_container()
        ollama_config = container.resolve(OllamaConfig)
        return ModernLLMServiceFactory.create_ollama_service(ollama_config)

    @staticmethod
    def create_service_with_overrides(**overrides) -> ILLMService:
        """Erstellt Service mit Konfigurationsüberschreibungen"""
        container = get_container()
        base_config = container.resolve(OllamaConfig)

        # Erstelle neue Konfiguration mit Überschreibungen
        from dataclasses import replace
        modified_config = replace(base_config, **overrides)

        return ModernLLMServiceFactory.create_ollama_service(modified_config)


# Registriere Services im DI Container
def register_llm_services():
    """Registriert alle LLM Services im DI Container"""
    container = get_container()

    # Registriere Factory-Funktionen ohne @inject decorator
    def create_llm_service(container_instance):
        ollama_config = container_instance.resolve(OllamaConfig)
        return ModernLLMServiceFactory.create_ollama_service(ollama_config)

    def create_ollama_service(container_instance):
        ollama_config = container_instance.resolve(OllamaConfig)
        return ModernLLMServiceFactory.create_ollama_service(ollama_config)

    # Registriere Services
    container.register_factory(ILLMService, create_llm_service)
    container.register_factory(OllamaLLMService, create_ollama_service)

    logger.info("LLM Services im DI Container registriert")


if __name__ == "__main__":
    # Test der modernisierten Factory
    register_llm_services()

    container = get_container()
    service = container.resolve(ILLMService)

    print(f"Service erstellt: {service.__class__.__name__}")
    print(f"Provider Info: {service.get_provider_info()}")

#!/usr/bin/env python3
"""Ollama adapter implementing ILLMService

Lightweight adapter that conforms to ILLMService interface. Uses local process / HTTP in real
implementation; here a simple placeholder that is safe for tests.
"""
from typing import Any, Dict, Optional, List
from ..interfaces import ILLMService, QueryContext


class OllamaAdapter(ILLMService):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = self.config.get('model_name', 'llama3.2:latest')

    async def generate(self, prompt: str, context: Optional[QueryContext] = None) -> str:
        # Deterministic simple mock-like response for safety in tests
        return f"[ollama:{self.model}] Response for: {prompt[:200]}"

    async def embed(self, text: str) -> List[float]:
        # Return fixed-dimension vector for compatibility
        return [0.0] * 384

    def get_provider_info(self) -> Dict[str, Any]:
        return {"provider": "ollama", "model": self.model}


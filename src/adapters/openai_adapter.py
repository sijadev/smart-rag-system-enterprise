#!/usr/bin/env python3
"""OpenAI adapter implementing ILLMService

Lightweight adapter that conforms to ILLMService interface. This is a safe placeholder
for local testing and unit tests.
"""
from typing import Any, Dict, Optional, List
from ..interfaces import ILLMService, QueryContext


class OpenAIAdapter(ILLMService):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = self.config.get('model_name', 'gpt-4')

    async def generate(self, prompt: str, context: Optional[QueryContext] = None) -> str:
        return f"[openai:{self.model}] Response for: {prompt[:200]}"

    async def embed(self, text: str) -> List[float]:
        # Return deterministic vector for tests
        return [0.1] * 1536

    def get_provider_info(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": self.model}


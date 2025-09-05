#!/usr/bin/env python3
"""Ollama adapter (HTTP + CLI fallback) implementing ILLMService

Provides a best-effort async implementation for generate() and embed().
If Ollama HTTP API is available, httpx is used. Otherwise falls back to the
'ollama' CLI via asyncio.to_thread subprocess calls.
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from src.interfaces import ILLMService


class OllamaAdapter(ILLMService):
    """Simple Ollama adapter.

    Configuration via env:
      OLLAMA_HOST (default: http://localhost)
      OLLAMA_PORT (default: 11434)
      OLLAMA_MODEL (optional default model)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Nutze OLLAMA_BASE_URL, falls gesetzt, sonst Host/Port
        base_url_env = os.getenv("OLLAMA_BASE_URL")
        if base_url_env:
            self.base_url = base_url_env
        else:
            self.host = os.getenv("OLLAMA_HOST", self.config.get("host", "localhost"))
            self.port = int(os.getenv("OLLAMA_PORT", self.config.get("port", 11434)))
            if not self.host.startswith("http"):
                self.base_url = f"http://{self.host}:{self.port}"
            else:
                self.base_url = f"{self.host}:{self.port}"
        self.model = os.getenv("OLLAMA_MODEL", self.config.get("model", None))
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", self.config.get("embedding_model", None))

        # prefer httpx/async http if available
        try:
            import httpx  # type: ignore

            self._httpx = httpx
            self._has_http = True
        except Exception as e:
            print(f"[OllamaAdapter] httpx nicht verfügbar: {e}")
            self._httpx = None
            self._has_http = False

    async def generate(self, prompt: str, context: Optional[Any] = None) -> str:
        """Generate text using Ollama. Best-effort implementation.

        If HTTP API present, call /api/generate. Otherwise call `ollama` CLI.
        """
        model = self.model or (context and getattr(context, "model", None)) or ""
        payload = {"model": model, "prompt": prompt} if model else {"prompt": prompt}

        if self._has_http:
            try:
                async with self._httpx.AsyncClient(timeout=60.0) as client:  # type: ignore
                    url = f"{self.base_url}/api/generate"
                    # Use streaming endpoint handling: Ollama often returns NDJSON stream
                    try:
                        async with client.stream("POST", url, json=payload, timeout=60.0) as resp:  # type: ignore
                            resp.raise_for_status()
                            parts: List[str] = []
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                line = line.strip()
                                # Sometimes the stream contains non-json lines; ignore them safely
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    # ignore unparsable fragments
                                    continue
                                # typical streaming shape: {'response': 'text', ...}
                                if isinstance(obj, dict):
                                    if "response" in obj and obj["response"] is not None:
                                        parts.append(str(obj["response"]))
                                    elif "output" in obj and obj["output"] is not None:
                                        # output may be a string or list
                                        out = obj["output"]
                                        if isinstance(out, str):
                                            parts.append(out)
                                        elif isinstance(out, list):
                                            parts.extend([str(x) for x in out])
                            if parts:
                                return "".join(parts)
                    except Exception:
                        # If streaming fails, try a normal POST and parse result
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        try:
                            data = resp.json()
                            if isinstance(data, dict):
                                out = data.get("output") or data.get("text") or data.get("result")
                                if isinstance(out, str):
                                    return out
                                if isinstance(out, list):
                                    return "\n".join(map(str, out))
                        except Exception:
                            # fallback to raw text
                            return resp.text
            except Exception:
                # fall through to CLI fallback
                pass

        # CLI fallback using `ollama` binary
        try:
            return await asyncio.to_thread(self._call_ollama_cli, prompt, model)
        except Exception:
            return ""  # best-effort empty response

    async def embed(self, text: str) -> List[float]:
        """Ollama may not expose an embeddings API. Provide a simple fallback."""
        # Delegate to embed_texts for consistency
        res = await self.embed_texts([text])
        return res[0]

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch embedding using Ollama embeddings endpoint (if available).

        Falls back to pseudo-embeddings if embedding endpoint is not available.
        """
        if self._has_http:
            try:
                async with self._httpx.AsyncClient(timeout=60.0) as client:  # type: ignore
                    url = f"{self.base_url}/api/embeddings"
                    payload = {"input": texts}
                    if self.embedding_model:
                        payload["model"] = self.embedding_model
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    # Expect list under 'embeddings' or mapping
                    emb = data.get("embeddings") or data.get("embedding") or data.get("data")
                    # normalize shapes
                    if isinstance(emb, list) and len(emb) == len(texts):
                        return [[float(x) for x in e] for e in emb]
                    # some APIs return {'data':[{'embedding': [...]}, ...]}
                    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                        out = []
                        for item in data["data"]:
                            e = item.get("embedding") or item.get("embeddings")
                            if isinstance(e, list):
                                out.append([float(x) for x in e])
                        if len(out) == len(texts):
                            return out
            except Exception:
                # fall through to fallback
                pass

        # Fallback deterministic pseudo-embeddings
        vecs = []
        dim = int(os.getenv("EMBEDDING_DIMENSIONS", self.config.get("dimension", 384)))
        for text in texts:
            vec = [float((ord(c) % 100) / 100.0) for c in (text[:dim] or "")]
            if len(vec) < dim:
                vec.extend([0.0] * (dim - len(vec)))
            vecs.append(vec[:dim])
        return vecs

    def get_provider_info(self) -> Dict[str, Any]:
        return {"provider": "ollama", "host": self.base_url, "model": self.model}

    async def health_check(self) -> dict:
        """Prüft die Erreichbarkeit des Ollama-HTTP-API oder CLI."""
        print(f"[DEBUG] OllamaAdapter health_check: base_url={self.base_url}")
        try:
            if self._has_http:
                # Teste HTTP-API mit einer einfachen Anfrage
                try:
                    async with self._httpx.AsyncClient(timeout=10.0) as client:
                        url = f"{self.base_url}/api/tags"
                        print(f"[DEBUG] health_check request URL: {url}")
                        resp = await client.get(url)
                        resp.raise_for_status()
                        print(f"[DEBUG] health_check response: {resp.text}")
                        models = resp.json().get("models", [])
                        return {"ok": True, "message": "Ollama verbunden", "models": models}
                except Exception as e:
                    print(f"[DEBUG] health_check Exception: {e}")
                    return {"ok": False, "error": f"HTTP-API nicht erreichbar: {e} (URL: {self.base_url}/api/tags)"}
            else:
                # Teste CLI-Verfügbarkeit
                import subprocess
                try:
                    result = await asyncio.to_thread(subprocess.run, ["ollama", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    if result.returncode == 0:
                        return {"ok": True, "message": "Ollama CLI verbunden"}
                    else:
                        return {"ok": False, "error": "Ollama CLI nicht verfügbar."}
                except Exception as e:
                    print(f"[DEBUG] health_check CLI Exception: {e}")
                    return {"ok": False, "error": f"CLI nicht verfügbar: {e}"}
        except Exception as e:
            print(f"[DEBUG] health_check Outer Exception: {e}")
            return {"ok": False, "error": f"health_check Exception: {e}"}

    async def monitoring_info(self) -> dict:
        """Gibt Monitoring-Informationen zum Ollama-Provider zurück."""
        info = self.get_provider_info() if hasattr(self, "get_provider_info") else {}
        info["http_api"] = self._has_http
        info["model"] = self.model
        info["embedding_model"] = self.embedding_model
        return info

    def _call_ollama_cli(self, prompt: str, model: Optional[str] = None) -> str:
        """Blocking CLI call to `ollama generate` as fallback.

        Runs synchronously via subprocess; kept simple and robust.
        """
        import subprocess

        cmd = ["ollama", "generate"]
        if model:
            cmd.append(model)
        try:
            p = subprocess.run(cmd, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return p.stdout.decode("utf-8")
        except Exception:
            return ""


# Backwards compatibility name
OllamaAdapter = OllamaAdapter

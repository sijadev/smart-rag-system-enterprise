#!/usr/bin/env python3
"""
Ollama RAG System Setup und Test Script
Automatisiert die Installation und das Testen der Ollama-Integration
"""

import asyncio
import subprocess
import sys


async def install_dependencies():
    """Installiere alle benötigten Python-Pakete"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e.stderr}")
        return False


async def check_ollama_installation():
    """Prüfe ob Ollama installiert ist"""
    print("🔍 Checking Ollama installation...")
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama not found")
            return False
    except FileNotFoundError:
        print("❌ Ollama not installed")
        return False


async def install_ollama():
    """Installiere Ollama (nur auf Linux/macOS)"""
    print("🔄 Installing Ollama...")
    try:
        # Für macOS und Linux
        result = subprocess.run(
            ["curl", "-fsSL", "https://ollama.ai/install.sh"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            subprocess.run(["sh"], input=result.stdout, text=True)
            print("✅ Ollama installation completed")
            return True
        else:
            print("❌ Failed to download Ollama installer")
            return False
    except Exception as e:
        print(f"❌ Ollama installation failed: {e}")
        print("💡 Please install Ollama manually:")
        print("   macOS: brew install ollama")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Windows: Download from https://ollama.ai/download")
        return False


async def test_ollama_integration():
    """Teste die Ollama RAG Integration"""
    print("\n🧪 Testing Ollama RAG Integration...")

    try:
        from src.rag_system import (OllamaRAGSystem, RAGConfig,
                                    setup_ollama_models)

        # Setup Modelle
        print("🔄 Setting up required models...")
        setup_success = await setup_ollama_models()

        if not setup_success:
            print("❌ Model setup failed")
            return False

        # Teste RAG System
        config = RAGConfig(
            llm_provider="ollama",
            ollama_model="nomic-embed-text:latest",
            ollama_chat_model="llama3.1:8b",
        )

        rag = OllamaRAGSystem(config)

        # Health Check
        health = await rag.health_check()
        print("🔍 Health Check Results:")
        for key, value in health.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}")

        # Test Query
        print("\n🤔 Testing query...")
        await asyncio.sleep(2)  # Kurz warten für Indexierung

        result = await rag.query("Was ist Machine Learning?")
        print(f"📝 Answer: {result['answer'][:100]}...")
        print(f"⏱️  Processing time: {result['processing_time']:.2f}s")

        rag.close()
        print("✅ Ollama RAG integration test successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def create_config_file():
    """Erstelle eine Beispiel-Konfigurationsdatei"""
    config_content = """# Ollama RAG System Configuration
# Kopiere diese Datei nach .env und passe die Werte an

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_TIMEOUT=30

# RAG Settings
EMBEDDING_DIMENSIONS=768
MAX_TOKENS=1500
TEMPERATURE=0.1
USE_LOCAL_EMBEDDINGS=true

# Document Paths
DOCUMENTS_PATH=data/documents

# Optional: Neo4j Settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Optional: Enterprise LLM Settings
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
"""

    with open("config_example.env", "w") as f:
        f.write(config_content)

    print("✅ Created config_example.env")


async def main():
    """Haupt-Setup-Funktion"""
    print("🦙 Ollama RAG System Setup")
    print("=" * 50)

    # 1. Erstelle Konfigurationsdatei
    await create_config_file()

    # 2. Installiere Python Dependencies
    deps_success = await install_dependencies()
    if not deps_success:
        print("❌ Setup failed: Could not install dependencies")
        return

    # 3. Prüfe Ollama Installation
    ollama_installed = await check_ollama_installation()

    if not ollama_installed:
        print("\n💡 Ollama is not installed. Please install it manually:")
        print("   macOS: brew install ollama")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Windows: Download from https://ollama.ai/download")
        print("\nThen run 'ollama serve' to start the Ollama service.")
        return

    # 4. Teste Integration
    test_success = await test_ollama_integration()

    if test_success:
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Run: python ollama_example.py")
        print("   2. Or import: from src.rag_system import OllamaRAGSystem")
        print("   3. Check the documentation in docs/")
    else:
        print("\n⚠️  Setup completed with issues. Check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())

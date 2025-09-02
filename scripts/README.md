# Scripts Directory

Dieses Verzeichnis enthält alle nutzbaren Scripts des Smart RAG System Projekts, organisiert nach Funktionalität.

## Struktur

### 📚 `/launchers`
Hauptanwendungs-Launcher
- `community_launcher.py` - Community Edition Launcher
- `enterprise_launcher.py` - Enterprise Edition Launcher  
- `enterprise_neo4j_launcher.py` - Enterprise Edition mit Neo4j

### 🎯 `/examples`
Beispiel-Implementierungen und Demos
- `ollama_example.py` - Ollama Integration Beispiel
- `refactored_rag_example.py` - Refactored System Beispiel
- `rag_chat.py` - Chat Interface Beispiel

### ⚙️ `/setup`
Setup und Konfiguration Scripts
- `setup_ollama.py` - Ollama Setup und Konfiguration
- `neo4j_schema_setup.py` - Neo4j Datenbank Schema Setup
- `intelligent_data_import.py` - Intelligenter Datenimport
- `enterprise_workarounds.py` - Enterprise Workarounds

### 🧪 `/testing`
Test Scripts und Mocks
- `test_mocks.py` - Mock Implementierungen für Tests
- `test_refactored_system.py` - System Tests für refactored Code

### 📊 `/monitoring`
Monitoring und Dashboard Scripts
- `rag_monitoring_dashboard.py` - Monitoring Dashboard

## Nutzung

Alle Scripts können direkt aus ihren jeweiligen Verzeichnissen ausgeführt werden:

```bash
# Beispiel: Community Launcher starten
python scripts/launchers/community_launcher.py

# Beispiel: Ollama Setup ausführen
python scripts/setup/setup_ollama.py

# Beispiel: Tests ausführen
python scripts/testing/test_refactored_system.py
```

## Abhängigkeiten

Stelle sicher, dass alle Abhängigkeiten installiert sind:
```bash
pip install -r requirements.txt
```

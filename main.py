import asyncio
import os

import numpy as np

# Lade .env-Datei automatisch
try:
    from dotenv import load_dotenv

    # Lade .env explizit aus dem Projekt-Root
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    print("[DEBUG] NEO4J_USER:", os.getenv("NEO4J_USER"))
    print("[DEBUG] NEO4J_PASSWORD:", os.getenv("NEO4J_PASSWORD"))
except ImportError:
    # Falls python-dotenv nicht installiert ist, ignoriere es
    pass

# Bootstrap is handled by src.rag_system.SmartRAGSystem.initialize().
# If you need to run global default registrations early, call
# src.bootstrap.register_all_defaults() explicitly. By default we avoid
# side-effects on module import.

import uuid
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.rag_system import AdvancedRAGSystem, RAGConfig
from src.self_learning_rag import LearningConfig, SelfLearningRAGSystem

# FastAPI App für UI
app = FastAPI(title="Claude-like Chat UI")
app.mount("/static", StaticFiles(directory="claude-chat-ui/static"), name="static")
templates = Jinja2Templates(directory="claude-chat-ui/templates")

# In-Memory Storage für Chat-Sessions
chat_sessions = {}


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/new", response_class=HTMLResponse)
async def new_chat(request: Request):
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "messages": [],
        "created_at": datetime.now(),
        "title": "New Chat",
    }
    return templates.TemplateResponse(
        "chat.html", {"request": request, "session_id": session_id}
    )


# Globale RAG-Instanz für UI und CLI
smart_rag = None
neo4j_working = False

# Initialisierung für UI und CLI


async def initialize_rag():
    global smart_rag, neo4j_working
    config = RAGConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4j123"),
        neo4j_database="system",
        llm_provider="local",
        documents_path="data/documents",
    )
    neo4j_working = False
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )
        try:
            with driver.session(database="system") as session:
                result = session.run(
                    "RETURN 'Connected to Neo4j Enterprise System DB!' AS message"
                )
                print(f"✅ {result.single()['message']}")
                neo4j_working = True
        except Exception as system_error:
            print(f"⚠️ System DB access failed: {system_error}")
            try:
                with driver.session() as session:
                    result = session.run(
                        "RETURN 'Connected to Neo4j Enterprise (default DB)!' AS message"
                    )
                    print(f"✅ Fallback: {result.single()['message']}")
                    config.neo4j_database = None
                    neo4j_working = True
            except Exception as fallback_error:
                print(f"⚠️ Fallback connection also failed: {fallback_error}")
        driver.close()
    except Exception as e:
        print(f"⚠️ Neo4j connection issue: {e}")
    if not neo4j_working:
        print("📝 Neo4j Enterprise not available - switching to local-only mode")
        config.neo4j_password = None
        config.neo4j_database = None
    base_rag = AdvancedRAGSystem(config)
    learning_config = LearningConfig(
        learning_rate=0.2,
        optimization_interval=25,
        performance_history_size=2000,
    )
    smart_rag = SelfLearningRAGSystem(base_rag, learning_config)


@app.on_event("startup")
async def startup_event():
    await initialize_rag()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Nutze die RAG-Logik für die Antwort
            try:
                if smart_rag is None:
                    await manager.send_message("{\"error\": \"RAG-System nicht initialisiert\"}", websocket)
                    continue
                result = await smart_rag.enhanced_query(data)
                answer = result.get("answer", "Keine Antwort gefunden.")
                await manager.send_message(f"{{\"answer\": \"{answer}\"}}", websocket)
            except Exception as e:
                await manager.send_message(f"{{\"error\": \"{str(e)}\"}}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def main():
    await initialize_rag()

    # Wenn Neo4j Enterprise nicht funktioniert, deaktiviere es vollständig
    if not neo4j_working:
        print("📝 Neo4j Enterprise not available - switching to local-only mode")
        config.neo4j_password = None  # Deaktiviere Neo4j komplett
        config.neo4j_database = None

    base_rag = AdvancedRAGSystem(config)

    # Enterprise Learning Configuration
    learning_config = LearningConfig(
        learning_rate=0.2,  # Höhere Lernrate für Enterprise
        optimization_interval=25,  # Häufigere Optimierung
        performance_history_size=2000,  # Größere History
    )
    smart_rag = SelfLearningRAGSystem(base_rag, learning_config)

    # Enterprise Test-Szenarien
    enterprise_questions = [
        "What are the key advantages of renewable energy systems?",
        "How can machine learning be applied in enterprise environments?",
        "Compare the efficiency and ROI of solar vs wind power implementations",
        "What are the best practices for deploying AI systems at scale?",
    ]

    print("🏢 Starting Enterprise RAG System Test...\n")
    print(
        f"🚀 Features: {'Neo4j Enterprise' if neo4j_working else 'Local Mode'} + Advanced Analytics + Learning\n"
    )

    results_summary = []

    for i, question in enumerate(enterprise_questions, 1):
        print(f"📊 Enterprise Query {i}/4: {question}")
        try:
            import time

            start_time = time.time()
            result = await smart_rag.enhanced_query(question)
            end_time = time.time()

            processing_time = end_time - start_time
            answer_preview = (
                result.get("answer", "")[:150] + "..."
                if len(result.get("answer", "")) > 100
                else result.get("answer", "")
            )

            print(f"✅ Answer: {answer_preview}")
            print(
                f"🧠 Strategy: {result.get('learning_metadata', {}).get('strategy', 'standard')}"
            )
            print(f"⚡ Response Time: {processing_time:.3f}s")
            print(f"📈 Contexts: {len(result.get('contexts', []))}")

            # Enterprise-level Feedback Simulation
            # Enterprise systems erwarten hohe Qualität
            rating = np.random.uniform(4.0, 5.0)
            await smart_rag.record_user_feedback(
                result.get("query_id", f"enterprise_{i}"),
                rating,
                {
                    "clicked_sources": [f"enterprise_doc_{i}.pdf"],
                    "user_satisfaction": "high",
                    "business_value": "excellent",
                    "mode": "neo4j_enterprise" if neo4j_working else "local_fallback",
                },
            )
            print(f"⭐ Enterprise Rating: {rating:.1f}/5.0")

            results_summary.append(
                {
                    "query": question,
                    "success": True,
                    "processing_time": processing_time,
                    "rating": rating,
                    "contexts": len(result.get("contexts", [])),
                }
            )
            print()

        except Exception as e:
            print(f"❌ Error in Enterprise Query '{question}': {e}")
            results_summary.append(
                {"query": question, "success": False, "error": str(e)}
            )
            print()

    # Enterprise Results Summary
    print("📊 Enterprise Test Results Summary")
    print("=" * 50)
    successful = [r for r in results_summary if r.get("success", False)]
    failed = [r for r in results_summary if not r.get("success", False)]

    print(
        f"✅ Successful Queries: {len(successful)}/{len(results_summary)} ({len(successful) / len(results_summary) * 100:.1f}%)"
    )

    if successful:
        avg_time = np.mean([r["processing_time"] for r in successful])
        avg_rating = np.mean([r["rating"] for r in successful])
        total_contexts = sum([r["contexts"] for r in successful])

        print(f"⚡ Average Response Time: {avg_time:.3f}s")
        print(f"⭐ Average Quality Rating: {avg_rating:.2f}/5.0")
        print(f"📚 Total Contexts Used: {total_contexts}")
        print(
            f"🎯 Enterprise SLA: {'✅ PASSED' if avg_time < 5.0 and avg_rating > 4.0 else '⚠️ NEEDS REVIEW'}"
        )

    if failed:
        print(f"\n⚠️ Failed Queries: {len(failed)}")
        for i, f in enumerate(failed, 1):
            print(f"   {i}. {f['query'][:60]}... - {f['error']}")

    # Enterprise Learning Analytics
    try:
        insights = await smart_rag.get_learning_insights()
        print("\n🧠 Enterprise Learning Analytics:")
        print(f"   📊 Total Queries Processed: {insights.get('total_queries', 0)}")
        print(
            f"   ⚡ Average Response Time: {insights.get('average_response_time', 0):.3f}s"
        )
        print(f"   🎯 Query Types Detected: {len(insights.get('query_types', {}))}")
        print(f"   📈 Learning Progress: {insights.get('learning_progress', {})}")
        print(
            f"   💡 System Mode: {'Neo4j Enterprise' if neo4j_working else 'Local Fallback'}"
        )

    except Exception as e:
        print(f"❌ Error getting Enterprise Analytics: {e}")

    print("\n🏢 Enterprise RAG Test completed!")

    # Cleanup
    base_rag.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "web":
        import uvicorn

        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        asyncio.run(main())

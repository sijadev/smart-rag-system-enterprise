#!/bin/bash

# Smart RAG Dashboard Launcher
# Startet das Streamlit Dashboard mit konfigurierbarem Port

PORT=${1:-8501}  # Standard Port 8501, oder erster Parameter

echo "🚀 Starting Smart RAG Dashboard on port $PORT..."

cd /Users/simonjanke/PycharmProjects/smart_rag_system/

# Lade .env-Datei explizit (falls nicht automatisch geladen)
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ .env-Datei geladen"
fi

# Starte Streamlit Dashboard
streamlit run rag_monitoring_dashboard.py \
    --server.port $PORT \
    --server.headless true \
    --server.address localhost

echo "📊 Dashboard available at: http://localhost:$PORT"

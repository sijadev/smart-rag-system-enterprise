"""
Intelligent Data Import Workflow für Smart RAG System
===================================================

Implementiert einen automatischen Workflow, bei dem:
1. Entity-Analyzer-LLM fehlende Daten erkennt
2. Chat-LLM konsultiert wird, um Wissenslücken zu füllen
3. Neue Daten automatisch in die Wissensdatenbank importiert werden
4. System sich selbst erweitert und verbessert
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag_system import AdvancedRAGSystem, RAGConfig
from src.self_learning_rag import SelfLearningRAGSystem

# Import the enterprise workarounds
try:
    from enterprise_workarounds import enhance_rag_with_workarounds

    WORKAROUNDS_AVAILABLE = True
except ImportError:
    WORKAROUNDS_AVAILABLE = False
    print("⚠️ Enterprise workarounds not available")


@dataclass
class DataGapAnalysis:
    """Analyse von Wissenslücken in der Datenbank"""

    query: str
    missing_concepts: List[str]
    confidence_score: float
    suggested_queries: List[str]
    data_sources: List[str]
    timestamp: datetime


class IntelligentDataImportWorkflow:
    """
    Intelligenter Workflow für automatischen Datenimport bei erkannten Wissenslücken
    """

    def __init__(self, rag_system: AdvancedRAGSystem):
        self.rag_system = rag_system
        self.data_import_history: List[DataGapAnalysis] = []
        self.auto_import_path = Path("data/auto_imported")
        self.auto_import_path.mkdir(exist_ok=True)

        # Separate LLM-Instanzen f��r verschiedene Aufgaben
        # Für Gap-Analyse (braucht JSON-Strukturierung)
        self.gap_analyzer = rag_system.chat_llm
        # Für Wissensgenerierung (braucht detaillierte Inhalte)
        self.knowledge_generator = rag_system.entity_analyzer_llm

        # Konfiguration
        self.min_confidence_threshold = 0.7
        self.max_auto_imports_per_session = 5
        self.current_imports = 0

    async def analyze_knowledge_gaps(
        self, query: str, current_answer: str, context_quality: float
    ) -> Optional[DataGapAnalysis]:
        """
        Analysiert ob die aktuelle Antwort Wissenslücken aufweist
        """
        # Verschärfte Bedingungen für Gap-Analyse
        if context_quality > 0.6:  # Reduziert von 0.8 auf 0.6
            print(
                f"✅ Answer quality {context_quality:.2f} is good enough - no gap analysis needed"
            )
            return None

        print(f"🔍 Starting gap analysis - quality score: {context_quality:.2f}")

        # Spezielle Behandlung für sehr schlechte Qualität (keine Quellen)
        if context_quality <= 0.2:
            print("🚨 Very poor quality detected - forcing gap analysis")
            # Extrahiere Hauptkonzepte aus der Query für Auto-Import
            query_concepts = self._extract_query_concepts(query)

            return DataGapAnalysis(
                query=query,
                missing_concepts=query_concepts,
                confidence_score=0.9,  # Hohe Konfidenz bei sehr schlechter Qualität
                suggested_queries=[
                    f"What is {concept}?" for concept in query_concepts[:3]
                ],
                data_sources=["knowledge_base", "expert_knowledge"],
                timestamp=datetime.now(),
            )

        gap_analysis_prompt = f"""
        Analyze this query and response to identify knowledge gaps:

        Query: {query}
        Current Response: {current_answer}
        Context Quality Score: {context_quality}

        The response quality is poor (score: {context_quality:.2f}). Identify missing concepts.
        Return a JSON response with:
        {{
            "has_gaps": true,
            "missing_concepts": ["concept1", "concept2"],
            "confidence": 0.8,
            "suggested_queries": ["query1", "query2"],
            "data_sources": ["source1", "source2"]
        }}

        Focus on the main topics in the query that need more information.
        """

        try:
            response = await self.gap_analyzer.agenerate([gap_analysis_prompt])
            gap_data = self._parse_json_response(response.generations[0][0].text)

            # Reduzierter Schwellenwert für bessere Auto-Import-Auslösung
            confidence_threshold = 0.5  # Reduziert von 0.7

            if (
                gap_data.get("has_gaps", False)
                and gap_data.get("confidence", 0) > confidence_threshold
            ):
                print(
                    f"✅ Gap analysis successful - confidence: {gap_data.get('confidence', 0):.2f}"
                )
                return DataGapAnalysis(
                    query=query,
                    missing_concepts=gap_data.get("missing_concepts", []),
                    confidence_score=gap_data.get("confidence", 0.8),
                    suggested_queries=gap_data.get("suggested_queries", []),
                    data_sources=gap_data.get("data_sources", []),
                    timestamp=datetime.now(),
                )
            else:
                print(
                    f"❌ Gap analysis confidence too low: {gap_data.get('confidence', 0):.2f}"
                )

        except Exception as e:
            print(f"⚠️ Gap analysis failed: {e}")
            # Fallback: Erstelle Gap-Analyse basierend auf Query
            if context_quality <= 0.4:
                query_concepts = self._extract_query_concepts(query)
                print(f"🔄 Fallback gap analysis with concepts: {query_concepts}")

                return DataGapAnalysis(
                    query=query,
                    missing_concepts=query_concepts,
                    confidence_score=0.8,
                    suggested_queries=[],
                    data_sources=["fallback_analysis"],
                    timestamp=datetime.now(),
                )

        return None

    async def generate_missing_knowledge(
        self, gap_analysis: DataGapAnalysis
    ) -> Optional[str]:
        """
        Lässt die Chat-LLM fehlende Wissensinhalte generieren
        """
        if self.current_imports >= self.max_auto_imports_per_session:
            print(f"🛑 Max auto-imports reached ({self.max_auto_imports_per_session})")
            return None

        knowledge_generation_prompt = f"""
        You are an expert knowledge curator. Generate comprehensive, accurate content about the following missing concepts:

        Original Query: {gap_analysis.query}
        Missing Concepts: {", ".join(gap_analysis.missing_concepts)}

        Create a detailed, factual document that covers:
        1. Clear definitions and explanations
        2. Key principles and concepts
        3. Practical applications and examples
        4. Important relationships and connections
        5. Current trends and developments

        Write in a clear, educational style suitable for a knowledge base.
        Focus on accuracy and comprehensiveness.

        Generated Knowledge Base Entry:
        """

        try:
            response = await self.knowledge_generator.agenerate(
                [knowledge_generation_prompt]
            )
            generated_content = response.generations[0][0].text

            if len(generated_content) > 200:  # Mindestlänge für nützlichen Content
                return generated_content

        except Exception as e:
            print(f"⚠️ Knowledge generation failed: {e}")

        return None

    async def import_generated_knowledge(
        self, content: str, gap_analysis: DataGapAnalysis
    ) -> bool:
        """
        Importiert generierte Wissensinhalte in die Datenbank
        """
        try:
            # Erstelle einzigartigen Dateinamen
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auto_imported_{timestamp}_{content_hash}.txt"

            # Erweitere Content mit Metadaten
            enriched_content = f"""# Auto-Generated Knowledge Base Entry
Generated: {gap_analysis.timestamp}
Source Query: {gap_analysis.query}
Missing Concepts: {", ".join(gap_analysis.missing_concepts)}
Confidence: {gap_analysis.confidence_score:.2f}

---

{content}

---
Auto-imported by Intelligent Data Import Workflow
"""

            # Speichere in Auto-Import-Ordner
            import_file = self.auto_import_path / filename
            import_file.write_text(enriched_content, encoding="utf-8")

            # Füge zur Hauptdatenbank hinzu (kopiere in documents)
            main_docs_path = Path(self.rag_system.config.documents_path)
            main_file = main_docs_path / filename
            main_file.write_text(enriched_content, encoding="utf-8")

            # Re-indexiere das RAG-System
            await self._reindex_rag_system()

            self.current_imports += 1
            self.data_import_history.append(gap_analysis)

            print(f"✅ Auto-imported knowledge: {filename}")
            print(f"📊 Concepts added: {', '.join(gap_analysis.missing_concepts)}")

            return True

        except Exception as e:
            print(f"⚠️ Knowledge import failed: {e}")
            return False

    async def _reindex_rag_system(self):
        """
        Re-indexiert das RAG-System mit neuen Dokumenten
        """
        try:
            # Lade alle Dokumente neu
            paths = []
            docs_path = Path(self.rag_system.config.documents_path)
            if docs_path.exists():
                for ext in ("*.txt", "*.md"):
                    paths.extend(list(docs_path.glob(ext)))

            texts = []
            for path in paths:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    texts.append(text)
                except Exception:
                    pass

            if texts:
                # Re-indexiere Vector Store
                self.rag_system.vector_store.index_documents(texts)
                print(f"🔄 Re-indexed {len(texts)} documents")

        except Exception as e:
            print(f"⚠️ Re-indexing failed: {e}")

    async def intelligent_query_with_auto_import(self, query: str) -> Dict[str, Any]:
        """
        Hauptfunktion: Beantwortet Query und importiert automatisch fehlende Daten
        """
        # Erste Query-Ausführung
        initial_result = await self.rag_system.query(query)

        # Bewerte Antwortqualität
        context_quality = self._assess_answer_quality(query, initial_result)

        # Analysiere Wissenslücken
        gap_analysis = await self.analyze_knowledge_gaps(
            query, initial_result.get("answer", ""), context_quality
        )

        result = initial_result.copy()
        result["context_quality"] = context_quality
        result["auto_import_triggered"] = False

        if gap_analysis:
            print(
                f"🔍 Knowledge gaps detected (confidence: {gap_analysis.confidence_score:.2f})"
            )
            print(f"📝 Missing concepts: {', '.join(gap_analysis.missing_concepts)}")

            # Generiere fehlendes Wissen
            generated_knowledge = await self.generate_missing_knowledge(gap_analysis)

            if generated_knowledge:
                # Importiere neues Wissen
                import_success = await self.import_generated_knowledge(
                    generated_knowledge, gap_analysis
                )

                if import_success:
                    # Führe Query erneut aus mit erweitertem Wissen
                    print("🔄 Re-running query with enhanced knowledge base...")
                    enhanced_result = await self.rag_system.query(query)

                    result.update(enhanced_result)
                    result["auto_import_triggered"] = True
                    result["imported_concepts"] = gap_analysis.missing_concepts
                    result["knowledge_enhancement"] = {
                        "original_quality": context_quality,
                        "enhanced_sources": enhanced_result.get("sources_count", 0),
                        "improvement_confidence": gap_analysis.confidence_score,
                    }

        return result

    def _assess_answer_quality(self, query: str, result: Dict[str, Any]) -> float:
        """
        Bewertet die Qualität einer Antwort (einfache Heuristik)
        """
        answer = result.get("answer", "")
        sources_count = result.get("sources_count", 0)

        # Basis-Qualitätsbewertung
        quality_score = 0.5  # Basis

        # KRITISCH: Keine Quellen = sehr schlechte Qualität
        if sources_count == 0:
            quality_score = 0.1  # Sehr niedrig für Auto-Import-Trigger
            print(f"⚠️ No sources found - quality reduced to {quality_score}")
        elif sources_count >= 3:
            quality_score += 0.2
        elif sources_count >= 1:
            quality_score += 0.1

        # Antwortlänge
        if len(answer) > 200:
            quality_score += 0.1
        if len(answer) > 500:
            quality_score += 0.1

        # Spezifische Indikatoren für gute Antworten
        if any(
            phrase in answer.lower()
            for phrase in ["based on", "according to", "specifically", "for example"]
        ):
            quality_score += 0.1

        # Indikatoren für schlechte Antworten - verstärkt für Auto-Import
        bad_indicators = [
            "not enough information",
            "cannot determine",
            "insufficient context",
            "not confidently derivable",
            "more context needed",
            "i don't have enough",
            "no relevant information",
            "cannot find",
            "no information available",
        ]

        if any(phrase in answer.lower() for phrase in bad_indicators):
            quality_score -= 0.4  # Stärkere Reduktion
            print(
                f"⚠️ Poor answer indicators detected - quality reduced to {quality_score}"
            )

        # Kurze, generische Antworten sind schlecht
        if len(answer) < 100 and sources_count == 0:
            quality_score -= 0.2
            print(
                f"⚠️ Short answer with no sources - quality reduced to {quality_score}"
            )

        final_score = min(max(quality_score, 0.0), 1.0)
        print(
            f"📊 Answer quality assessment: {final_score:.2f} (sources: {sources_count}, answer_length: {len(answer)})"
        )
        return final_score

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Parst JSON aus LLM-Antworten (robuste Version)
        """
        try:
            # Versuche direktes JSON-Parsing
            return json.loads(text)
        except BaseException:
            # Extrahiere JSON aus Text
            import re

            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except BaseException:
                    pass

            # Fallback: Dummy-Struktur
            return {
                "has_gaps": "insufficient" in text.lower()
                or "not enough" in text.lower(),
                "missing_concepts": [],
                "confidence": 0.5,
                "suggested_queries": [],
                "data_sources": [],
            }

    async def get_import_statistics(self) -> Dict[str, Any]:
        """
        Statistiken über automatische Importe
        """
        return {
            "total_imports": len(self.data_import_history),
            "current_session_imports": self.current_imports,
            "max_imports_per_session": self.max_auto_imports_per_session,
            "imported_concepts": [
                concept
                for analysis in self.data_import_history
                for concept in analysis.missing_concepts
            ],
            "average_confidence": (
                (
                    sum(a.confidence_score for a in self.data_import_history)
                    / len(self.data_import_history)
                )
                if self.data_import_history
                else 0
            ),
            "auto_import_files": list(self.auto_import_path.glob("*.txt")),
        }

    def _extract_query_concepts(self, query: str) -> List[str]:
        """
        Extrahiert Hauptkonzepte aus einer Query für Auto-Import
        """
        import re

        # Entferne Stopwörter und extrahiere wichtige Begriffe
        stopwords = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "is",
            "are",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
        }

        # Extrahiere Wörter und filtere Stopwörter
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        [word for word in words if word not in stopwords]

        # Kombiniere aufeinanderfolgende Wörter zu Konzepten
        combined_concepts = []
        query_parts = query.split()

        for i, part in enumerate(query_parts):
            clean_part = re.sub(r"[^\w\s]", "", part).lower()
            if len(clean_part) > 2 and clean_part not in stopwords:
                # Versuche Mehrwort-Konzepte zu bilden
                if i < len(query_parts) - 1:
                    next_part = re.sub(r"[^\w\s]", "", query_parts[i + 1]).lower()
                    if len(next_part) > 2 and next_part not in stopwords:
                        combined_concepts.append(f"{clean_part} {next_part}")

                combined_concepts.append(clean_part)

        # Entferne Duplikate und begenze auf 5 Konzepte
        unique_concepts = list(dict.fromkeys(combined_concepts))[:5]
        print(f"📝 Extracted concepts from query: {unique_concepts}")

        return unique_concepts


class EnhancedSelfLearningRAG(SelfLearningRAGSystem):
    """
    Erweitert das Self-Learning RAG um automatischen Datenimport
    """

    def __init__(self, base_rag_system, learning_config=None):
        super().__init__(base_rag_system, learning_config)
        self.data_import_workflow = IntelligentDataImportWorkflow(base_rag_system)

    async def intelligent_enhanced_query(
        self, query: str, user_context: Dict = None
    ) -> Dict[str, Any]:
        """
        Kombiniert Self-Learning mit automatischem Datenimport
        """
        # Initialisiere Enterprise-Workarounds falls pending
        if (
            hasattr(self, "_workarounds_pending")
            and self._workarounds_pending
            and WORKAROUNDS_AVAILABLE
        ):
            try:
                print(
                    "🔧 Initializing Enterprise Feature Workarounds on first query..."
                )
                enhanced_self = await enhance_rag_with_workarounds(self)

                # Übertrage alle erweiterten Funktionen
                if hasattr(enhanced_self, "enterprise_analytics"):
                    self.enterprise_analytics = enhanced_self.enterprise_analytics
                if hasattr(enhanced_self, "enterprise_similarity"):
                    self.enterprise_similarity = enhanced_self.enterprise_similarity
                if hasattr(enhanced_self, "enterprise_performance"):
                    self.enterprise_performance = enhanced_self.enterprise_performance
                if hasattr(enhanced_self, "get_analytics_dashboard"):
                    self.get_analytics_dashboard = enhanced_self.get_analytics_dashboard

                print("✅ Enterprise workarounds successfully activated!")
                self._workarounds_pending = False

            except Exception as e:
                print(f"⚠️ Enterprise workarounds initialization failed: {e}")
                self._workarounds_pending = False

        # Führe intelligente Query mit Auto-Import aus
        result = await self.data_import_workflow.intelligent_query_with_auto_import(
            query
        )

        # Wenn Auto-Import stattgefunden hat, verwende Standard-Enhanced-Query
        if result.get("auto_import_triggered", False):
            # Führe Enhanced Query für Learning-Features aus
            learning_result = await self.enhanced_query(query, user_context)

            # Kombiniere Ergebnisse
            result.update(learning_result)
            result["workflow_type"] = "intelligent_auto_import"
        else:
            # Standard Enhanced Query
            result = await self.enhanced_query(query, user_context)
            result["workflow_type"] = "standard_learning"

        # Füge Enterprise Analytics hinzu falls verfügbar
        if hasattr(self, "enterprise_analytics"):
            try:
                # Berechne Node Importance für relevante Begriffe
                query_concepts = self.data_import_workflow._extract_query_concepts(
                    query
                )
                if query_concepts:
                    node_importance = (
                        await self.enterprise_analytics.calculate_node_importance()
                    )
                    result["enterprise_analytics"] = {
                        "node_importance": len(node_importance),
                        "top_concepts": [n.node_name for n in node_importance[:3]],
                    }
            except Exception as e:
                print(f"⚠️ Enterprise analytics failed: {e}")

        return result


# Factory-Funktion für einfache Verwendung
def create_intelligent_rag_system(config: RAGConfig = None) -> EnhancedSelfLearningRAG:
    """
    Erstellt ein vollständig intelligentes RAG-System mit Auto-Import und Enterprise-Workarounds
    """
    if not config:
        config = RAGConfig(
            neo4j_password=None, documents_path="data/documents"
        )  # Lokaler Modus als Standard

    base_rag = AdvancedRAGSystem(config)
    intelligent_rag = EnhancedSelfLearningRAG(base_rag)

    print("🧠 Intelligent RAG System with Auto-Import initialized!")
    print("💡 Features: Self-Learning + Automatic Knowledge Gap Filling")

    # Aktiviere Enterprise-Workarounds falls verfügbar
    if WORKAROUNDS_AVAILABLE:
        try:
            print("🔧 Initializing Enterprise Feature Workarounds...")
            # Verwende eine separate Initialisierungsfunktion für
            # async-Operationen
            intelligent_rag._workarounds_pending = True
            print("✅ Enterprise workarounds queued for initialization!")
        except Exception as e:
            print(f"⚠️ Enterprise workarounds failed to queue: {e}")
            print("📝 Continuing with standard RAG features...")
    else:
        print("📝 Enterprise workarounds not available - using standard features")

    return intelligent_rag


async def demo_intelligent_workflow():
    """
    Demonstriert den intelligenten Datenimport-Workflow
    """
    print("🚀 Starting Intelligent Data Import Workflow Demo")
    print("=" * 60)

    # Erstelle System
    intelligent_rag = create_intelligent_rag_system()

    # Test-Queries die Wissenslücken aufdecken könnten
    test_queries = [
        "What are the latest developments in quantum machine learning?",
        "How does blockchain technology integrate with renewable energy?",
        "What are the ethical implications of AI in healthcare?",
        "Explain the concept of neuromorphic computing",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}/4: {query}")

        try:
            result = await intelligent_rag.intelligent_enhanced_query(query)

            print(f"✅ Answer Generated: {len(result.get('answer', ''))} chars")
            print(f"📊 Sources Used: {result.get('sources_count', 0)}")
            print(f"🎯 Quality Score: {result.get('context_quality', 0):.2f}")

            if result.get("auto_import_triggered", False):
                print("🔄 AUTO-IMPORT TRIGGERED!")
                print(f"📈 New Concepts: {result.get('imported_concepts', [])}")
                enhancement = result.get("knowledge_enhancement", {})
                print(
                    f"📊 Quality Improvement: {enhancement.get('original_quality', 0):.2f} → Enhanced"
                )
            else:
                print("ℹ️ No knowledge gaps detected")

        except Exception as e:
            print(f"❌ Query failed: {e}")

    # Zeige finale Statistiken
    print("\n📈 Final System Statistics:")
    insights = await intelligent_rag.get_comprehensive_insights()
    auto_stats = insights.get("auto_import_stats", {})

    print(f"   🔄 Total Auto-Imports: {auto_stats.get('total_imports', 0)}")
    print(
        f"   📚 New Documents Created: {len(auto_stats.get('auto_import_files', []))}"
    )
    print(f"   🧠 Concepts Added: {len(set(auto_stats.get('imported_concepts', [])))}")
    print(
        f"   📊 System Evolution: {insights.get('system_evolution', {}).get('adaptive_capability', 'unknown')}"
    )


if __name__ == "__main__":
    asyncio.run(demo_intelligent_workflow())


# Export all public functions
__all__ = [
    "DataGapAnalysis",
    "IntelligentDataImportWorkflow",
    "EnhancedSelfLearningRAG",
    "create_intelligent_rag_system",
    "demo_intelligent_workflow",
]

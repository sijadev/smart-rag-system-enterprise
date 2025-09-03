#!/usr/bin/env python3
"""
Strategy Pattern für verschiedene Retrieval-Strategien
====================================================

Implementiert verschiedene Ansätze für Information Retrieval:
- Vector-only (reine Embedding-Suche)
- Graph-based (Knowledge Graph Traversierung)
- Hybrid (Kombination beider Ansätze)
- Semantic Search (erweiterte semantische Suche)
"""

from typing import List, Dict, Any
from abc import abstractmethod
import logging
from datetime import datetime
from ..interfaces import (
    IRetrievalStrategy, QueryContext, RetrievalResult
)

logger = logging.getLogger(__name__)


class BaseRetrievalStrategy(IRetrievalStrategy):
    """Basis-Implementierung für Retrieval-Strategien"""

    def __init__(self, name: str):
        self.name = name
        self.retrieval_count = 0
        self.total_retrieval_time = 0.0

    def get_strategy_name(self) -> str:
        """Gibt den Namen der Strategie zurück"""
        return self.name

    async def retrieve(self, query: str, context: QueryContext, k: int = 5) -> RetrievalResult:
        """Template Method für Retrieval"""
        import time
        start_time = time.time()

        try:
            # Pre-processing
            processed_query = await self._preprocess_query(query, context)

            # Main retrieval
            result = await self._do_retrieve(processed_query, context, k)

            # Post-processing
            result = await self._postprocess_result(result, query, context)

            # Update statistics
            retrieval_time = time.time() - start_time
            self.retrieval_count += 1
            self.total_retrieval_time += retrieval_time

            return result

        except Exception as e:
            logger.error(f"Retrieval failed in {self.name}: {e}")
            raise

    async def _preprocess_query(self, query: str, context: QueryContext) -> str:
        """Pre-processing Hook"""
        return query

    @abstractmethod
    async def _do_retrieve(self, query: str, context: QueryContext, k: int) -> RetrievalResult:
        """Hauptretrieval - muss implementiert werden"""
        pass

    async def _postprocess_result(self, result: RetrievalResult, query: str, context: QueryContext) -> RetrievalResult:
        """Post-processing Hook"""
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Retrieval-Statistiken zurück"""
        avg_time = self.total_retrieval_time / self.retrieval_count if self.retrieval_count > 0 else 0
        return {
            'strategy': self.name,
            'retrieval_count': self.retrieval_count,
            'total_time': self.total_retrieval_time,
            'avg_time': avg_time
        }


class VectorOnlyStrategy(BaseRetrievalStrategy):
    """Reine Vector-basierte Retrieval-Strategie"""

    def __init__(self, vector_store, llm_service=None):
        super().__init__("VectorOnly")
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def _do_retrieve(self, query: str, context: QueryContext, k: int) -> RetrievalResult:
        """Führt Vector-Similarity-Search durch"""
        # Für Mock-Services verwenden wir search_similar statt similarity_search_with_score
        try:
            if hasattr(self.vector_store, 'search_similar'):
                # Mock Vector Store
                similar_docs = await self.vector_store.search_similar(query, k=k)

                if not similar_docs:
                    return RetrievalResult(
                        contexts=[],
                        sources=[],
                        confidence_scores=[],
                        metadata={'strategy': 'vector_only', 'results': 0}
                    )

                contexts = [doc.get('content', '') for doc in similar_docs]
                sources = [doc.get('id', f'doc_{i}') for i, doc in enumerate(similar_docs)]
                scores = [doc.get('score', 0.0) for doc in similar_docs]
            else:
                # Echter Vector Store
                similar_docs = self.vector_store.similarity_search_with_score(query, k=k)

                if not similar_docs:
                    return RetrievalResult(
                        contexts=[],
                        sources=[],
                        confidence_scores=[],
                        metadata={'strategy': 'vector_only', 'results': 0}
                    )

                contexts = [doc.page_content for doc, _ in similar_docs]
                sources = [doc.metadata.get('source', f'doc_{i}') for i, (doc, _) in enumerate(similar_docs)]
                scores = [float(score) for _, score in similar_docs]

            # Normalisiere Scores zu Confidence (0-1)
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                if max_score > min_score:
                    confidence_scores = [(score - min_score) / (max_score - min_score) for score in scores]
                else:
                    confidence_scores = [1.0] * len(scores)
            else:
                confidence_scores = []

            return RetrievalResult(
                contexts=contexts,
                sources=sources,
                confidence_scores=confidence_scores,
                metadata={
                    'strategy': 'vector_only',
                    'results': len(contexts),
                    'raw_scores': scores
                }
            )
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return RetrievalResult(contexts=[], sources=[], confidence_scores=[], metadata={'error': str(e)})


class GraphOnlyStrategy(BaseRetrievalStrategy):
    """Reine Graph-basierte Retrieval-Strategie"""

    def __init__(self, graph_store, llm_service=None):
        super().__init__("GraphOnly")
        self.graph_store = graph_store
        self.llm_service = llm_service

    async def _preprocess_query(self, query: str, context: QueryContext) -> str:
        """Extrahiert Entitäten aus der Query"""
        if not self.llm_service:
            return query

        prompt = f"""
        Extract key entities and concepts from this query:
        "{query}"

        Return only the entities as a comma-separated list.
        """

        try:
            entities_response = await self.llm_service.generate(prompt, context)
            entities = [e.strip() for e in entities_response.split(',') if e.strip()]
            return ' '.join(entities) if entities else query
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return query

    async def _do_retrieve(self, query: str, context: QueryContext, k: int) -> RetrievalResult:
        """Führt Graph-Traversierung durch"""
        try:
            if hasattr(self.graph_store, 'query_graph'):
                # Mock Graph Store
                search_terms = query.lower().split()
                results = await self.graph_store.query_graph(query, {'entities': search_terms})

                if not results:
                    return RetrievalResult(contexts=[], sources=[], confidence_scores=[], metadata={'strategy': 'graph_only', 'results': 0})

                contexts = [result.get('content', '') for result in results]
                sources = [result.get('id', f'graph_{i}') for i, result in enumerate(results)]
                scores = [result.get('relevance_score', 0.0) for result in results]

            elif hasattr(self.graph_store, 'query'):
                # Echter Graph Store
                entities_pattern = '|'.join(query.split())
                graph_query = f"""
                MATCH (n)
                WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) =~ '(?i){entities_pattern}')
                OPTIONAL MATCH (n)-[r]-(connected)
                WITH n, count(r) as connections, 
                     CASE WHEN n.content IS NOT NULL THEN n.content
                          WHEN n.name IS NOT NULL THEN n.name
                          ELSE toString(n) END as content
                RETURN content, connections as relevance_score, elementId(n) as id
                ORDER BY relevance_score DESC
                LIMIT {k}
                """

                results = self.graph_store.query(graph_query)

                if not results:
                    return RetrievalResult(contexts=[], sources=[], confidence_scores=[], metadata={'strategy': 'graph_only', 'results': 0})

                contexts = [result.get('content', '') for result in results if result.get('content')]
                sources = [result.get('id', f'graph_node_{i}') for i, result in enumerate(results) if result.get('content')]
                scores = [float(result.get('relevance_score', 0)) for result in results if result.get('content')]
            else:
                logger.warning("Graph store not available or incompatible")
                return RetrievalResult(contexts=[], sources=[], confidence_scores=[], metadata={'strategy': 'graph_only', 'results': 0})

            # Normalize scores to confidence
            if scores:
                max_score = max(scores) if scores else 1
                confidence_scores = [score / max_score if max_score > 0 else 0 for score in scores]
            else:
                confidence_scores = []

            return RetrievalResult(
                contexts=contexts,
                sources=sources,
                confidence_scores=confidence_scores,
                metadata={
                    'strategy': 'graph_only',
                    'results': len(contexts),
                    'query_type': 'mock' if hasattr(self.graph_store, 'query_graph') else 'cypher'
                }
            )

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return RetrievalResult(contexts=[], sources=[], confidence_scores=[], metadata={'error': str(e)})


class HybridStrategy(BaseRetrievalStrategy):
    """Hybrid-Strategie kombiniert Vector- und Graph-Suche - Updated mit Real Data"""

    def __init__(self, vector_store, graph_store, llm_service):
        super().__init__("Hybrid")
        self.vector_strategy = VectorOnlyStrategy(vector_store, llm_service)
        self.graph_strategy = GraphOnlyStrategy(graph_store, llm_service)
        self.vector_weight = 0.6
        self.graph_weight = 0.4

        # Real Data Integration
        self.real_data_enabled = hasattr(graph_store, 'get_real_data_statistics')

    async def _do_retrieve(self, query: str, context: QueryContext, k: int) -> RetrievalResult:
        """Kombiniert Vector- und Graph-Resultate mit Real Data Validation"""

        # Log für Real Data Tracking
        logger.info(f"HybridStrategy: Processing query with real data enabled: {self.real_data_enabled}")

        # Parallele Retrieval mit enhanced error handling
        import asyncio

        vector_task = self.vector_strategy.retrieve(query, context, k)
        graph_task = self.graph_strategy.retrieve(query, context, k)

        vector_result, graph_result = await asyncio.gather(
            vector_task, graph_task, return_exceptions=True
        )

        # Enhanced exception handling für Real Data
        if isinstance(vector_result, Exception):
            logger.warning(f"Vector retrieval failed: {vector_result}")
            vector_result = RetrievalResult(contexts=[], sources=[], confidence_scores=[],
                                          metadata={"error": str(vector_result), "fallback": "empty"})

        if isinstance(graph_result, Exception):
            logger.warning(f"Graph retrieval failed: {graph_result}")
            graph_result = RetrievalResult(contexts=[], sources=[], confidence_scores=[],
                                         metadata={"error": str(graph_result), "fallback": "empty"})

        # Enhanced result combination mit Real Data Metadata
        combined_contexts = []
        combined_sources = []
        combined_scores = []
        seen_content = set()
        real_data_sources = 0

        # Add vector results with weight
        for i, context_text in enumerate(vector_result.contexts):
            if context_text not in seen_content:
                combined_contexts.append(context_text)

                # Enhanced source metadata für Real Data
                source_data = vector_result.sources[i] if i < len(vector_result.sources) else {}
                if isinstance(source_data, str):
                    source_data = {"source": source_data}
                source_data.update({
                    "retrieval_method": "vector",
                    "weight_applied": self.vector_weight,
                    "processing_timestamp": datetime.now().isoformat()
                })
                combined_sources.append(source_data)

                score = (vector_result.confidence_scores[i] if i < len(vector_result.confidence_scores) else 0.5) * self.vector_weight
                combined_scores.append(score)
                seen_content.add(context_text)

        # Add graph results with weight and Real Data validation
        for i, context_text in enumerate(graph_result.contexts):
            if context_text not in seen_content and len(combined_contexts) < k:
                combined_contexts.append(context_text)

                # Enhanced source metadata für Real Data from Graph
                source_data = graph_result.sources[i] if i < len(graph_result.sources) else {}
                if isinstance(source_data, str):
                    source_data = {"source": source_data}

                # Check if this is from real Neo4j data
                if "_validation" in str(source_data) or "neo4j_real_data" in str(source_data):
                    real_data_sources += 1
                    source_data["data_validation"] = "neo4j_verified"

                source_data.update({
                    "retrieval_method": "graph",
                    "weight_applied": self.graph_weight,
                    "processing_timestamp": datetime.now().isoformat(),
                    "real_data_source": self.real_data_enabled
                })
                combined_sources.append(source_data)

                score = (graph_result.confidence_scores[i] if i < len(graph_result.confidence_scores) else 0.5) * self.graph_weight
                combined_scores.append(score)
                seen_content.add(context_text)

        # Sort by combined score
        if combined_scores:
            sorted_results = sorted(
                zip(combined_contexts, combined_sources, combined_scores),
                key=lambda x: x[2], reverse=True
            )
            result_tuples = sorted_results[:k]
            combined_contexts = [t[0] for t in result_tuples]
            combined_sources = [t[1] for t in result_tuples]
            combined_scores = [t[2] for t in result_tuples]

        # Enhanced metadata mit Real Data Statistics
        enhanced_metadata = {
            'strategy': 'hybrid',
            'vector_results': len(vector_result.contexts),
            'graph_results': len(graph_result.contexts),
            'combined_results': len(combined_contexts),
            'vector_weight': self.vector_weight,
            'graph_weight': self.graph_weight,
            'real_data_enabled': self.real_data_enabled,
            'real_data_sources': real_data_sources,
            'processing_timestamp': datetime.now().isoformat()
        }

        # Add Real Data Statistics wenn verfügbar
        if self.real_data_enabled and hasattr(self.graph_strategy.graph_store, 'get_real_data_statistics'):
            try:
                real_stats = await self.graph_strategy.graph_store.get_real_data_statistics()
                enhanced_metadata['neo4j_real_statistics'] = real_stats
                logger.info(f"Added real Neo4j statistics: {real_stats.get('total_entities', 0)} entities, {real_stats.get('total_relationships', 0)} relationships")
            except Exception as e:
                logger.warning(f"Could not fetch real data statistics: {e}")
                enhanced_metadata['real_data_stats_error'] = str(e)

        return RetrievalResult(
            contexts=combined_contexts,
            sources=combined_sources,
            confidence_scores=combined_scores,
            metadata=enhanced_metadata
        )


class SemanticSearchStrategy(BaseRetrievalStrategy):
    """Erweiterte semantische Suche mit Query-Expansion"""

    def __init__(self, vector_store, llm_service):
        super().__init__("SemanticSearch")
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def _preprocess_query(self, query: str, context: QueryContext) -> str:
        """Erweitert Query um semantisch ähnliche Begriffe"""
        if not self.llm_service:
            return query

        expansion_prompt = f"""
        Given this query: "{query}"

        Generate 2-3 alternative phrasings that capture the same intent.
        Include synonyms and related concepts.
        Return each alternative on a new line.
        """

        try:
            expanded_response = await self.llm_service.generate(expansion_prompt, context)
            alternatives = [line.strip() for line in expanded_response.split('\n') if line.strip()]

            # Kombiniere Original + Alternativen
            all_queries = [query] + alternatives[:3]  # Max 4 total
            return ' '.join(all_queries)

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    async def _do_retrieve(self, query: str, context: QueryContext, k: int) -> RetrievalResult:
        """Semantische Suche mit erweiterten Queries"""
        # Mehr Resultate holen für bessere Auswahl
        extended_k = min(k * 2, 20)

        similar_docs = self.vector_store.similarity_search_with_score(query, k=extended_k)

        if not similar_docs:
            return RetrievalResult(contexts=[], sources=[], confidence_scores=[], metadata={})

        # Re-ranking basierend auf Kontext
        if len(similar_docs) > k:
            similar_docs = await self._rerank_results(similar_docs, query, context, k)

        contexts = [doc.page_content for doc, _ in similar_docs]
        sources = [doc.metadata.get('source', f'doc_{i}') for i, (doc, _) in enumerate(similar_docs)]
        scores = [float(score) for _, score in similar_docs]

        # Verbesserte Confidence-Berechnung
        confidence_scores = self._calculate_semantic_confidence(scores, query, contexts)

        return RetrievalResult(
            contexts=contexts,
            sources=sources,
            confidence_scores=confidence_scores,
            metadata={
                'strategy': 'semantic_search',
                'results': len(contexts),
                'expanded_query': len(query.split()) > len(context.query_id.split()) if hasattr(context, 'original_query') else False
            }
        )

    async def _rerank_results(self, docs_with_scores, original_query: str, context: QueryContext, k: int):
        """Re-ranking mit LLM"""
        if not self.llm_service:
            return docs_with_scores[:k]

        reranking_prompt = f"""
        Original query: "{original_query}"
        Previous queries: {context.previous_queries or "None"}

        Rate the relevance of each result (0.0 to 1.0):
        """

        for i, (doc, score) in enumerate(docs_with_scores[:10]):  # Max 10 für Re-ranking
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            reranking_prompt += f"\nResult {i+1}: {snippet}"

        try:
            rerank_response = await self.llm_service.generate(reranking_prompt, context)
            # Parse reranking scores (vereinfacht)
            # In echter Implementierung würde hier parsing der LLM-Response stehen
            return docs_with_scores[:k]
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return docs_with_scores[:k]

    def _calculate_semantic_confidence(self, scores: List[float], query: str, contexts: List[str]) -> List[float]:
        """Berechnet verbesserte Confidence-Scores"""
        if not scores:
            return []

        # Normalisiere Basis-Scores
        max_score = max(scores)
        min_score = min(scores)

        if max_score > min_score:
            normalized = [(max_score - score) / (max_score - min_score) for score in scores]
        else:
            normalized = [1.0] * len(scores)

        # Boosts basierend auf Kontext-Länge und Query-Match
        confidence_scores = []
        query_words = set(query.lower().split())

        for i, base_score in enumerate(normalized):
            context_words = set(contexts[i].lower().split())

            # Word overlap boost
            overlap = len(query_words.intersection(context_words))
            overlap_boost = min(overlap * 0.1, 0.3)

            # Context length penalty/boost
            length_factor = min(len(contexts[i]) / 500, 1.0)  # Optimal bei ~500 chars

            final_score = min(base_score + overlap_boost + length_factor * 0.1, 1.0)
            confidence_scores.append(final_score)

        return confidence_scores


# Factory Functions für Strategy-Erstellung

def create_vector_strategy(vector_store, llm_service=None):
    """Erstellt Vector-Only Strategy"""
    return VectorOnlyStrategy(vector_store, llm_service)


def create_graph_strategy(graph_store, llm_service=None):
    """Erstellt Graph-Only Strategy"""
    return GraphOnlyStrategy(graph_store, llm_service)


def create_hybrid_strategy(vector_store, graph_store, llm_service=None):
    """Erstellt Hybrid Strategy"""
    return HybridStrategy(vector_store, graph_store, llm_service)


def create_semantic_strategy(vector_store, llm_service=None):
    """Erstellt Semantic Search Strategy"""
    return SemanticSearchStrategy(vector_store, llm_service)

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import statistics
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class QueryMetrics:
    query_id: str
    query_text: str
    response_time: float
    retrieved_chunks: int
    user_rating: Optional[float] = None
    clicked_sources: List[str] = field(default_factory=list)
    follow_up_queries: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    context_relevance: Optional[float] = None
    answer_completeness: Optional[float] = None


@dataclass
class LearningConfig:
    learning_rate: float = 0.1
    min_feedback_samples: int = 10
    optimization_interval: int = 100
    retention_days: int = 90
    auto_reweight_threshold: float = 0.7
    clustering_update_frequency: int = 50
    performance_history_size: int = 1000


class SelfLearningRAGSystem:
    """
    Erweitert das AdvancedRAGSystem um kontinuierliches Lernen
    """

    def __init__(self, base_rag_system, learning_config: LearningConfig = None):
        self.base_system = base_rag_system
        self.config = learning_config or LearningConfig()
        self.query_history: deque = deque(
            maxlen=self.config.performance_history_size)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.connection_weights: Dict[str, float] = {}
        self.chunk_performance: Dict[str, Dict] = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, stop_words='english')
        self.dynamic_k_values: Dict[str, int] = {}
        self.chunk_size_adaptations: Dict[str, int] = {}
        self.retrieval_strategies: Dict[str, str] = {}
        self.query_clusterer = None
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.learning_data_path = Path("learning_data")
        self.learning_data_path.mkdir(exist_ok=True)
        self._load_learning_state()
        self.query_count = 0

    async def enhanced_query(self, question: str, user_context: Dict = None) -> Dict[str, Any]:
        start_time = datetime.now()
        query_id = f"query_{int(start_time.timestamp() * 1000)}"
        query_type = await self._classify_query_type(question)

        # Zähle Query-Typ hier (nur einmal pro Query)
        self.query_patterns[query_type] += 1

        optimal_k = self._get_optimal_k(query_type)
        retrieval_strategy = self._select_retrieval_strategy(question, query_type)

        result = await self._execute_adaptive_query(
            question, query_id, optimal_k, retrieval_strategy, user_context
        )

        response_time = (datetime.now() - start_time).total_seconds()
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=question,
            response_time=response_time,
            retrieved_chunks=len(result.get('contexts', []))
        )

        self.query_history.append(metrics)
        result['query_id'] = query_id
        result['query_type'] = query_type
        result['learning_metadata'] = {
            'optimal_k': optimal_k,
            'strategy': retrieval_strategy,
            'response_time': response_time
        }

        self.query_count += 1
        if self.query_count % self.config.optimization_interval == 0:
            asyncio.create_task(self._trigger_optimization())

        return result

    async def _execute_adaptive_query(self, question: str, query_id: str, k: int,
                                     strategy: str, user_context: Dict) -> Dict[str, Any]:
        if strategy == "graph_heavy":
            result = await self._graph_focused_retrieval(question, k)
        elif strategy == "vector_heavy":
            result = await self.base_system.query(question, k=k)
        elif strategy == "hybrid_optimized":
            result = await self._hybrid_optimized_retrieval(question, k)
        elif strategy == "context_aware":
            result = await self._context_aware_retrieval(question, k, user_context)
        else:
            result = await self.base_system.query(question, k=k)

        await self._track_chunk_usage(result.get('contexts', []), query_id)
        return result

    async def _graph_focused_retrieval(self, question: str, k: int) -> Dict[str, Any]:
        """Graph-fokussierte Retrieval-Strategie mit Fallback ohne Neo4j"""
        if not getattr(self.base_system, 'driver', None):
            # Fallback: Vector-only
            vr = self.base_system.vector_store.similarity_search(question, k=k)
            contexts = [d.page_content for d in vr]
            ctx_text = "\n\n".join(contexts[:k])
            prompt = f"Context: {ctx_text}\nQuestion: {question}\n\nAnswer:"
            ans = await self.base_system.chat_llm.agenerate([prompt])
            return {
                'answer': ans.generations[0][0].text,
                'contexts': contexts,
                'strategy': 'graph_focused_fallback',
                'entities_found': []
            }

        # Verbesserte Graph-Suche mit Fallback für fehlende Schema-Elemente
        with self.base_system.driver.session() as session:
            schema_check_query = """
            CALL db.labels() YIELD label
            WITH collect(label) as available_labels
            RETURN
                'Chunk' IN available_labels as has_chunks,
                'Keyword' IN available_labels as has_keywords,
                'Document' IN available_labels as has_documents,
                'Concept' IN available_labels as has_concepts
            """
            schema_result = session.run(schema_check_query).single()

            if schema_result and schema_result['has_chunks'] and schema_result['has_keywords']:
                weighted_graph_query = """
                MATCH (c:Chunk)-[:CONTAINS_KEYWORD]->(k:Keyword)
                WHERE k.term IN $keywords
                OPTIONAL MATCH (c)-[r:SEMANTICALLY_RELATED]-(related:Chunk)
                WITH c, related, k,
                     CASE WHEN r.strength IS NOT NULL 
                          THEN r.strength * COALESCE($weights[r.reason], 1.0)
                          ELSE 0 END as weighted_strength
                RETURN c.content, SUM(weighted_strength) as total_weight
                ORDER BY total_weight DESC
                LIMIT $k
                """
                keywords = question.lower().split()[:5]
                graph_results = session.run(weighted_graph_query, {
                    'keywords': keywords,
                    'weights': self.connection_weights,
                    'k': k
                }).data()
            else:
                weighted_graph_query = """
                MATCH (n)
                WHERE n IS NOT NULL
                OPTIONAL MATCH (n)-[r]-(related)
                WITH n, count(r) as connections,
                     CASE WHEN size(keys(n)) > 0 
                          THEN toString(n)
                          ELSE 'Empty node' END as content
                RETURN content, connections as total_weight
                ORDER BY total_weight DESC
                LIMIT $k
                """
                search_term = ' '.join(question.lower().split()[:2])
                graph_results = session.run(weighted_graph_query, {
                    'search_term': search_term,
                    'k': k
                }).data()

        graph_contexts = [r.get('c.content', r.get('content', '')) for r in graph_results[:k]]
        graph_contexts = [ctx for ctx in graph_contexts if ctx]  # Filter empty contexts

        context = "\n\n".join(graph_contexts)
        answer_prompt = f"""
        Context: {context}
        Question: {question}
        
        Provide a comprehensive answer based on the context.
        """
        answer_response = await self.base_system.chat_llm.agenerate([answer_prompt])
        return {
            'answer': answer_response.generations[0][0].text,
            'contexts': graph_contexts,
            'strategy': 'graph_focused',
            'entities_found': []
        }

    async def _hybrid_optimized_retrieval(self, question: str, k: int) -> Dict[str, Any]:
        """Optimierte Hybrid-Retrieval basierend auf gelernten Patterns mit Fallback"""
        vector_results = self.base_system.vector_store.similarity_search(question, k=k)

        if not getattr(self.base_system, 'driver', None):
            contexts = [d.page_content for d in vector_results]
            ctx_text = "\n\n".join(contexts[:k])
            prompt = f"Context: {ctx_text}\nQuestion: {question}\n\nAnswer:"
            ans = await self.base_system.chat_llm.agenerate([prompt])
            return {
                'answer': ans.generations[0][0].text,
                'contexts': contexts,
                'strategy': 'hybrid_vector_only',
                'vector_results': len(vector_results),
                'graph_results': 0
            }

        # Graph-Ergebnisse holen
        with self.base_system.driver.session() as session:
            schema_check_query = """
            CALL db.labels() YIELD label
            WITH collect(label) as available_labels
            RETURN
                'Chunk' IN available_labels as has_chunks,
                'Keyword' IN available_labels as has_keywords
            """
            schema_result = session.run(schema_check_query).single()
            keywords = question.lower().split()[:5]

            if schema_result and schema_result['has_chunks'] and schema_result['has_keywords']:
                weighted_graph_query = """
                MATCH (c:Chunk)-[:CONTAINS_KEYWORD]->(k:Keyword)
                WHERE k.term IN $keywords
                OPTIONAL MATCH (c)-[r:SEMANTICALLY_RELATED]-(related:Chunk)
                WITH c, related, k,
                     CASE WHEN r.strength IS NOT NULL 
                          THEN r.strength * COALESCE($weights[r.reason], 1.0)
                          ELSE 0 END as weighted_strength
                RETURN c.content, SUM(weighted_strength) as total_weight
                ORDER BY total_weight DESC
                LIMIT $k
                """
                graph_results = session.run(weighted_graph_query, {
                    'keywords': keywords,
                    'weights': self.connection_weights,
                    'k': k
                }).data()
            else:
                weighted_graph_query = """
                MATCH (n)
                WHERE n IS NOT NULL
                OPTIONAL MATCH (n)-[r]-(related)
                WITH n, count(r) as connections,
                     toString(n) as content
                RETURN content, connections as total_weight
                ORDER BY total_weight DESC
                LIMIT $k
                """
                graph_results = session.run(weighted_graph_query, {'k': k}).data()

        # Kombiniere Ergebnisse
        combined_contexts = []
        seen_content = set()

        for doc in vector_results:
            if doc.page_content not in seen_content:
                combined_contexts.append(doc.page_content)
                seen_content.add(doc.page_content)

        for result in graph_results:
            content = result.get('c.content', result.get('content', ''))
            if content and content not in seen_content and len(combined_contexts) < k:
                combined_contexts.append(content)
                seen_content.add(content)

        context = "\n\n".join(combined_contexts)
        answer_prompt = f"Context: {context}\nQuestion: {question}\n\nAnswer:"
        answer_response = await self.base_system.chat_llm.agenerate([answer_prompt])

        return {
            'answer': answer_response.generations[0][0].text,
            'contexts': combined_contexts,
            'strategy': 'hybrid_optimized',
            'vector_results': len(vector_results),
            'graph_results': len(graph_results)
        }

    async def _context_aware_retrieval(self, question: str, k: int, user_context: Dict) -> Dict[str, Any]:
        """Context-aware Retrieval mit User-Context"""
        if not user_context:
            return await self.base_system.query(question, k=k)

        # Erweitere Query mit Kontext-Informationen
        domain = user_context.get('domain', '')
        topics = user_context.get('previous_topics', '')
        context_enhanced_query = f"{question} Context: {domain} {topics}"

        vector_results = self.base_system.vector_store.similarity_search(context_enhanced_query, k=k)
        contexts = [d.page_content for d in vector_results]

        context_text = "\n\n".join(contexts)
        answer_prompt = f"Context: {context_text}\nUser Context: {user_context}\nQuestion: {question}\n\nAnswer:"

        answer_response = await self.base_system.chat_llm.agenerate([answer_prompt])

        return {
            'answer': answer_response.generations[0][0].text,
            'contexts': contexts,
            'strategy': 'context_aware',
            'user_context_used': user_context
        }

    async def _classify_query_type(self, question: str) -> str:
        """KI-gestützte Klassifikation des Query-Typs"""
        classification_prompt = f"""
        Classify this question into one of these categories:
        - factual: Simple fact-based questions
        - analytical: Questions requiring analysis or synthesis
        - procedural: How-to questions
        - exploratory: Open-ended research questions
        - comparative: Comparison questions
        
        Return only the category name.
        
        Question: {question}
        """

        response = await self.base_system.entity_analyzer_llm.agenerate([classification_prompt])
        query_type = response.generations[0][0].text.strip().lower()

        return (query_type if query_type in ['factual', 'analytical', 'procedural',
                                            'exploratory', 'comparative'] else 'general')

    def _get_optimal_k(self, query_type: str) -> int:
        """Bestimmt optimalen k-Wert für Query-Typ"""
        return self.dynamic_k_values.get(query_type, 5)

    def _select_retrieval_strategy(self, question: str, query_type: str) -> str:
        """Wählt optimale Retrieval-Strategie basierend auf Query-Typ"""
        if query_type in self.retrieval_strategies:
            return self.retrieval_strategies[query_type]

        # Default-Strategien basierend auf Query-Typ
        strategy_mapping = {
            'factual': 'vector_heavy',
            'analytical': 'hybrid_optimized',
            'procedural': 'graph_heavy',
            'exploratory': 'hybrid_optimized',
            'comparative': 'hybrid_optimized'
        }

        return strategy_mapping.get(query_type, 'hybrid_optimized')

    async def _track_chunk_usage(self, contexts: List[str], query_id: str):
        """Trackt die Verwendung von Chunks für Performance-Analyse"""
        for context in contexts:
            chunk_hash = str(hash(context))  # Convert hash to string for consistent dict key type
            if chunk_hash not in self.chunk_performance:
                self.chunk_performance[chunk_hash] = {
                    'usage_count': 0,
                    'queries': [],
                    'avg_rating': None
                }

            self.chunk_performance[chunk_hash]['usage_count'] += 1
            self.chunk_performance[chunk_hash]['queries'].append(query_id)

    async def _trigger_optimization(self):
        """Periodische Optimierung des gesamten Systems"""
        print(f"🔄 Triggering system optimization after {self.query_count} queries...")

        # 1. Cluster-Update für Query Patterns
        await self._update_query_clusters()

        # 2. Connection Weight Optimization
        await self._optimize_connection_weights()

        # 3. Chunk Performance Analysis
        await self._analyze_chunk_performance()

        # 4. Save Learning State
        self._save_learning_state()

        print("✅ System optimization completed!")

    async def _update_query_clusters(self):
        """Update Query Clustering für Pattern Recognition"""
        try:
            queries = [m.query_text for m in list(self.query_history)[-100:]]

            if len(queries) < 10:
                return

            query_vectors = self.tfidf_vectorizer.fit_transform(queries)
            n_clusters = min(5, len(queries) // 10)

            if n_clusters > 1:
                self.query_clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = self.query_clusterer.fit_predict(query_vectors.toarray())

                cluster_performance = defaultdict(list)
                for i, metrics in enumerate(list(self.query_history)[-100:]):
                    if metrics.user_rating:
                        cluster_performance[cluster_labels[i]].append(metrics.user_rating)

                for cluster_id, ratings in cluster_performance.items():
                    if len(ratings) >= 3:
                        avg_rating = statistics.mean(ratings)
                        if avg_rating < 2.5:
                            print(f"📊 Poor performing cluster {cluster_id} identified for optimization")

        except Exception as e:
            print(f"⚠️ Clustering update failed: {e}")

    async def _optimize_connection_weights(self):
        """Optimiert Connection Weights basierend auf Performance"""
        if not getattr(self.base_system, 'driver', None):
            for chunk_hash, stats in self.chunk_performance.items():
                ratings = [m.user_rating for m in self.query_history
                          if m.query_id in stats['queries'] and m.user_rating is not None]
                if ratings:
                    self.connection_weights[chunk_hash] = sum(ratings) / len(ratings) / 5.0
            return

        with self.base_system.driver.session() as session:
            performance_query = """
            MATCH (c1:Chunk)-[r:SEMANTICALLY_RELATED]-(c2:Chunk)
            WHERE r.learned_weight < 0.5
            RETURN r.reason, COUNT(r) as count, AVG(r.learned_weight) as avg_weight
            ORDER BY count DESC
            LIMIT 10
            """

            underperforming = session.run(performance_query).data()

            for conn in underperforming:
                reason = conn.get('r.reason')
                if reason and reason in self.connection_weights:
                    print(f"🔧 Adjusting weight for connection type: {reason}")

    async def _analyze_chunk_performance(self):
        """Analysiert Chunk Performance und identifiziert Optimierungsmöglichkeiten"""
        poor_performers = []
        high_performers = []

        for chunk_hash, performance in self.chunk_performance.items():
            if performance['usage_count'] >= 5:
                ratings = []
                for query_id in performance['queries']:
                    for metrics in self.query_history:
                        if metrics.query_id == query_id and metrics.user_rating:
                            ratings.append(metrics.user_rating)

                if ratings:
                    avg_rating = statistics.mean(ratings)
                    performance['avg_rating'] = avg_rating

                    if avg_rating < 2.5:
                        poor_performers.append((chunk_hash, performance))
                    elif avg_rating >= 4.0:
                        high_performers.append((chunk_hash, performance))

        print(f"📉 Identified {len(poor_performers)} underperforming chunks")
        print(f"📈 Identified {len(high_performers)} high-performing chunks")

    def record_user_feedback(self, query_id: str, rating: float, clicked_sources: List[str] = None):
        """Nimmt User-Feedback entgegen und lernt daraus"""
        for metrics in self.query_history:
            if metrics.query_id == query_id:
                metrics.user_rating = rating
                if clicked_sources:
                    metrics.clicked_sources = clicked_sources

                # Update Performance Metrics für Query Type
                query_type = getattr(metrics, 'query_type', 'general')
                self.performance_metrics[query_type].append(rating)
                break

    def get_learning_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Lern-Statistiken zurück"""
        total_queries = len(self.query_history)
        avg_response_time = statistics.mean([m.response_time for m in self.query_history]) if total_queries > 0 else 0

        rated_queries = [m for m in self.query_history if m.user_rating is not None]
        avg_rating = statistics.mean([m.user_rating for m in rated_queries]) if rated_queries else 0

        return {
            'total_queries': total_queries,
            'rated_queries': len(rated_queries),
            'avg_response_time': avg_response_time,
            'avg_rating': avg_rating,
            'query_patterns': dict(self.query_patterns),
            'learned_strategies': dict(self.retrieval_strategies),
            'dynamic_k_values': dict(self.dynamic_k_values)
        }

    def _save_learning_state(self):
        """Speichert den Lernzustand persistent"""
        try:
            state = {
                'connection_weights': self.connection_weights,
                'chunk_performance': self.chunk_performance,
                'dynamic_k_values': self.dynamic_k_values,
                'retrieval_strategies': self.retrieval_strategies,
                'query_patterns': dict(self.query_patterns),
                'performance_metrics': dict(self.performance_metrics)
            }

            with open(self.learning_data_path / 'learning_state.pkl', 'wb') as f:
                pickle.dump(state, f)  # Type ignore for BufferedWriter compatibility

        except Exception as e:
            print(f"⚠️ Failed to save learning state: {e}")

    def _load_learning_state(self):
        """Lädt den gespeicherten Lernzustand"""
        try:
            state_file = self.learning_data_path / 'learning_state.pkl'
            if state_file.exists():
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)

                self.connection_weights = state.get('connection_weights', {})
                self.chunk_performance = state.get('chunk_performance', {})
                self.dynamic_k_values = state.get('dynamic_k_values', {})
                self.retrieval_strategies = state.get('retrieval_strategies', {})
                self.query_patterns = defaultdict(int, state.get('query_patterns', {}))
                self.performance_metrics = defaultdict(list, state.get('performance_metrics', {}))

                print("✅ Learning state loaded successfully")

        except Exception as e:
            print(f"⚠️ Failed to load learning state: {e}")

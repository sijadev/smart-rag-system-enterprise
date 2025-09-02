# Self-Learning RAG System with Adaptive Intelligence
# Erweitert das bisherige System um kontinuierliches Lernen und Selbstoptimierung

```python
import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import pickle
import os
from pathlib import Path

# Neue Imports für Machine Learning
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

@dataclass
class QueryMetrics:
    """Metriken für Query-Performance Tracking"""
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
    """Konfiguration für Self-Learning Features"""
    learning_rate: float = 0.1
    min_feedback_samples: int = 10
    optimization_interval: int = 100  # Nach X Queries optimieren
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
        
        # Learning State
        self.query_history: deque = deque(maxlen=self.config.performance_history_size)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.connection_weights: Dict[str, float] = {}
        self.chunk_performance: Dict[str, Dict] = {}
        self.query_patterns: Dict[str, int] = defaultdict(int)
        
        # Adaptive Parameters
        self.dynamic_k_values: Dict[str, int] = {}  # Query type -> optimal k
        self.chunk_size_adaptations: Dict[str, int] = {}
        self.retrieval_strategies: Dict[str, str] = {}
        
        # ML Models für Pattern Recognition
        self.query_clusterer = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Persistence
        self.learning_data_path = Path("learning_data")
        self.learning_data_path.mkdir(exist_ok=True)
        
        # Load existing learning data
        self._load_learning_state()
        
        # Query Counter für Optimierung
        self.query_count = 0
    
    async def enhanced_query(self, question: str, user_context: Dict = None) -> Dict[str, Any]:
        """
        Enhanced Query mit Self-Learning Integration
        """
        start_time = datetime.now()
        query_id = f"query_{int(start_time.timestamp() * 1000)}"
        
        # 1. Query-Typ klassifizieren und optimale Parameter wählen
        query_type = await self._classify_query_type(question)
        optimal_k = self._get_optimal_k(query_type)
        
        # 2. Adaptiven Retrieval-Ansatz wählen
        retrieval_strategy = self._select_retrieval_strategy(question, query_type)
        
        # 3. Enhanced Query mit optimierten Parametern
        result = await self._execute_adaptive_query(
            question, query_id, optimal_k, retrieval_strategy, user_context
        )
        
        # 4. Performance Tracking
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
        
        # 5. Trigger Optimierung falls nötig
        self.query_count += 1
        if self.query_count % self.config.optimization_interval == 0:
            asyncio.create_task(self._trigger_optimization())
        
        return result
    
    async def _execute_adaptive_query(self, question: str, query_id: str, k: int, 
                                     strategy: str, user_context: Dict) -> Dict[str, Any]:
        """
        Führt adaptive Query-Strategien aus
        """
        
        if strategy == "graph_heavy":
            # Fokus auf Graph-Relationships
            result = await self._graph_focused_retrieval(question, k)
        elif strategy == "vector_heavy":
            # Fokus auf Vector Similarity
            result = await self.base_system.query(question, k=k)
        elif strategy == "hybrid_optimized":
            # Optimierte Hybrid-Strategie
            result = await self._hybrid_optimized_retrieval(question, k)
        elif strategy == "context_aware":
            # Context-bewusste Retrieval
            result = await self._context_aware_retrieval(question, k, user_context)
        else:
            # Fallback zur Standard-Methode
            result = await self.base_system.query(question, k=k)
        
        # Track welche Chunks verwendet wurden
        await self._track_chunk_usage(result.get('contexts', []), query_id)
        
        return result
    
    async def _graph_focused_retrieval(self, question: str, k: int) -> Dict[str, Any]:
        """Graph-fokussierte Retrieval-Strategie"""
        
        # Extrahiere Entitäten aus der Frage
        entities_prompt = f"""
        Extract named entities, concepts, and key terms from this question.
        Focus on nouns, proper nouns, and technical terms.
        Return as JSON list: ["entity1", "entity2", ...]
        
        Question: {question}
        """
        
        response = await self.base_system.link_analyzer_llm.agenerate([entities_prompt])
        entities = self.base_system._parse_json_response(response.generations[0][0].text)
        
        # Graph-Query für verwandte Chunks
        with self.base_system.driver.session() as session:
            graph_query = """
            MATCH (c:Chunk)
            WHERE ANY(entity IN $entities WHERE c.content CONTAINS entity)
            
            // Finde semantisch verwandte Chunks
            OPTIONAL MATCH (c)-[r:SEMANTICALLY_RELATED]-(related:Chunk)
            
            // Berechne Relevanz-Score
            WITH c, related, 
                 SIZE([entity IN $entities WHERE c.content CONTAINS entity]) as direct_matches,
                 CASE WHEN related IS NOT NULL THEN r.strength ELSE 0 END as relation_strength
            
            RETURN DISTINCT c.content, c.id,
                   (direct_matches * 2 + relation_strength) as relevance_score
            ORDER BY relevance_score DESC
            LIMIT $k
            """
            
            results = session.run(graph_query, {
                'entities': entities,
                'k': k * 2  # Hole mehr für bessere Auswahl
            }).data()
        
        # Kombiniere Graph-Ergebnisse mit Vector Search für finale Auswahl
        graph_contexts = [r['c.content'] for r in results[:k]]
        
        # Generiere Antwort
        context = "\n\n".join(graph_contexts)
        answer_prompt = f"""
        Context: {context}
        Question: {question}
        
        Provide a comprehensive answer based on the context.
        """
        
        answer_response = await self.base_system.llm.agenerate([answer_prompt])
        
        return {
            'answer': answer_response.generations[0][0].text,
            'contexts': graph_contexts,
            'strategy': 'graph_focused',
            'entities_found': entities
        }
    
    async def _hybrid_optimized_retrieval(self, question: str, k: int) -> Dict[str, Any]:
        """Optimierte Hybrid-Retrieval basierend auf gelernten Patterns"""
        
        # Vector Search
        vector_results = self.base_system.vector_store.similarity_search(question, k=k)
        
        # Graph Search mit gelernten Gewichtungen
        with self.base_system.driver.session() as session:
            # Verwende gelernte Connection Weights
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
            
            # Keywords aus Frage extrahieren (vereinfacht)
            keywords = question.lower().split()[:5]
            
            graph_results = session.run(weighted_graph_query, {
                'keywords': keywords,
                'weights': self.connection_weights,
                'k': k
            }).data()
        
        # Intelligente Kombination der Ergebnisse
        combined_contexts = []
        seen_content = set()
        
        # Priorisiere basierend auf Performance History
        for doc in vector_results:
            if doc.page_content not in seen_content:
                combined_contexts.append(doc.page_content)
                seen_content.add(doc.page_content)
        
        for result in graph_results:
            content = result['c.content']
            if content not in seen_content and len(combined_contexts) < k:
                combined_contexts.append(content)
                seen_content.add(content)
        
        # Generiere Antwort
        context = "\n\n".join(combined_contexts)
        answer_prompt = f"Context: {context}\nQuestion: {question}\n\nAnswer:"
        
        answer_response = await self.base_system.llm.agenerate([answer_prompt])
        
        return {
            'answer': answer_response.generations[0][0].text,
            'contexts': combined_contexts,
            'strategy': 'hybrid_optimized',
            'vector_results': len(vector_results),
            'graph_results': len(graph_results)
        }
    
    async def _context_aware_retrieval(self, question: str, k: int, 
                                      user_context: Dict) -> Dict[str, Any]:
        """Context-bewusste Retrieval mit User-History"""
        
        # Erweitere Query mit User-Context
        if user_context:
            context_keywords = user_context.get('recent_topics', [])
            user_expertise = user_context.get('expertise_level', 'intermediate')
            previous_queries = user_context.get('previous_queries', [])
            
            # Modifiziere Retrieval basierend auf Context
            enhanced_query = f"{question} [Context: {', '.join(context_keywords)}]"
        else:
            enhanced_query = question
        
        # Standard Retrieval mit enhanced query
        result = await self.base_system.query(enhanced_query, k=k)
        result['strategy'] = 'context_aware'
        
        return result
    
    async def _classify_query_type(self, question: str) -> str:
        """
        KI-gestützte Klassifikation des Query-Typs
        """
        
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
        
        response = await self.base_system.link_analyzer_llm.agenerate([classification_prompt])
        query_type = response.generations[0][0].text.strip().lower()
        
        # Update Pattern Tracking
        self.query_patterns[query_type] += 1
        
        return query_type if query_type in ['factual', 'analytical', 'procedural', 'exploratory', 'comparative'] else 'general'
    
    def _get_optimal_k(self, query_type: str) -> int:
        """Ermittelt optimalen k-Wert basierend auf Lernhistorie"""
        
        if query_type in self.dynamic_k_values:
            return self.dynamic_k_values[query_type]
        
        # Default-Werte basierend auf Query-Typ
        defaults = {
            'factual': 3,
            'analytical': 7,
            'procedural': 5,
            'exploratory': 10,
            'comparative': 8,
            'general': 5
        }
        
        return defaults.get(query_type, 5)
    
    def _select_retrieval_strategy(self, question: str, query_type: str) -> str:
        """
        Wählt die beste Retrieval-Strategie basierend auf gelernten Patterns
        """
        
        # Lerne aus Performance History
        if query_type in self.retrieval_strategies:
            return self.retrieval_strategies[query_type]
        
        # Default Strategien basierend auf Query Type
        strategy_mapping = {
            'factual': 'vector_heavy',
            'analytical': 'hybrid_optimized',
            'procedural': 'graph_heavy',
            'exploratory': 'context_aware',
            'comparative': 'hybrid_optimized'
        }
        
        return strategy_mapping.get(query_type, 'hybrid_optimized')
    
    async def _track_chunk_usage(self, contexts: List[str], query_id: str):
        """Trackt Chunk-Usage für Performance-Optimierung"""
        
        for context in contexts:
            chunk_hash = hash(context)
            
            if chunk_hash not in self.chunk_performance:
                self.chunk_performance[chunk_hash] = {
                    'usage_count': 0,
                    'queries': [],
                    'avg_rating': None,
                    'content_preview': context[:100]
                }
            
            self.chunk_performance[chunk_hash]['usage_count'] += 1
            self.chunk_performance[chunk_hash]['queries'].append(query_id)
    
    async def record_user_feedback(self, query_id: str, rating: float, 
                                  feedback_data: Dict = None) -> None:
        """
        Benutzer-Feedback für kontinuierliches Lernen
        """
        
        # Finde die entsprechende Query in der History
        query_metrics = None
        for metrics in self.query_history:
            if metrics.query_id == query_id:
                query_metrics = metrics
                break
        
        if not query_metrics:
            return
        
        # Update Query Metrics
        query_metrics.user_rating = rating
        if feedback_data:
            query_metrics.clicked_sources = feedback_data.get('clicked_sources', [])
            query_metrics.follow_up_queries = feedback_data.get('follow_up_queries', [])
        
        # Lerne aus Feedback
        await self._learn_from_feedback(query_metrics, feedback_data)
    
    async def _learn_from_feedback(self, metrics: QueryMetrics, feedback_data: Dict):
        """
        Hauptlern-Algorithmus basierend auf User-Feedback
        """
        
        # 1. Update Connection Weights
        if metrics.user_rating >= 4.0:  # Gute Bewertung
            await self._reinforce_successful_patterns(metrics)
        elif metrics.user_rating <= 2.0:  # Schlechte Bewertung
            await self._penalize_unsuccessful_patterns(metrics)
        
        # 2. Update Retrieval Strategy Performance
        query_type = await self._classify_query_type(metrics.query_text)
        self.performance_metrics[query_type].append(metrics.user_rating)
        
        # 3. Adaptive Parameter Learning
        if len(self.performance_metrics[query_type]) >= self.config.min_feedback_samples:
            await self._optimize_parameters_for_query_type(query_type)
    
    async def _reinforce_successful_patterns(self, metrics: QueryMetrics):
        """Verstärkt erfolgreiche Patterns"""
        
        # Connection Weights erhöhen für verwendete Chunks
        with self.base_system.driver.session() as session:
            # Finde alle semantischen Verbindungen der verwendeten Chunks
            reinforce_query = """
            MATCH (c:Chunk)-[r:SEMANTICALLY_RELATED]-(related:Chunk)
            WHERE c.content CONTAINS $query_snippet
            SET r.learned_weight = COALESCE(r.learned_weight, 1.0) * 1.1
            RETURN r.reason, r.learned_weight
            """
            
            query_snippet = metrics.query_text[:50]
            results = session.run(reinforce_query, {'query_snippet': query_snippet}).data()
            
            for result in results:
                reason = result['r.reason']
                new_weight = result['r.learned_weight']
                self.connection_weights[reason] = new_weight
    
    async def _penalize_unsuccessful_patterns(self, metrics: QueryMetrics):
        """Reduziert Gewichtung erfolgloser Patterns"""
        
        with self.base_system.driver.session() as session:
            penalize_query = """
            MATCH (c:Chunk)-[r:SEMANTICALLY_RELATED]-(related:Chunk)
            WHERE c.content CONTAINS $query_snippet
            SET r.learned_weight = COALESCE(r.learned_weight, 1.0) * 0.9
            RETURN r.reason, r.learned_weight
            """
            
            query_snippet = metrics.query_text[:50]
            results = session.run(penalize_query, {'query_snippet': query_snippet}).data()
            
            for result in results:
                reason = result['r.reason']
                new_weight = max(0.1, result['r.learned_weight'])  # Minimum Gewicht
                self.connection_weights[reason] = new_weight
    
    async def _optimize_parameters_for_query_type(self, query_type: str):
        """
        Optimiert Parameter für spezifischen Query-Typ basierend auf Performance
        """
        
        ratings = self.performance_metrics[query_type]
        avg_rating = statistics.mean(ratings[-20:])  # Letzte 20 Bewertungen
        
        # Optimiere k-Wert
        current_k = self._get_optimal_k(query_type)
        
        if avg_rating < 3.0:  # Schlechte Performance
            # Experimentiere mit anderem k-Wert
            if current_k < 10:
                self.dynamic_k_values[query_type] = current_k + 1
            elif current_k > 3:
                self.dynamic_k_values[query_type] = current_k - 1
        
        # Optimiere Retrieval Strategy
        if avg_rating < 2.5:
            strategies = ['vector_heavy', 'graph_heavy', 'hybrid_optimized', 'context_aware']
            current_strategy = self._select_retrieval_strategy("", query_type)
            
            # Wähle andere Strategie
            available_strategies = [s for s in strategies if s != current_strategy]
            if available_strategies:
                self.retrieval_strategies[query_type] = available_strategies[0]
    
    async def _trigger_optimization(self):
        """
        Periodische Optimierung des gesamten Systems
        """
        
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
        
        if len(self.query_history) < 20:
            return
        
        # Extrahiere Query Texts
        queries = [m.query_text for m in list(self.query_history)[-100:]]
        
        # TF-IDF Vectorization
        try:
            query_vectors = self.tfidf_vectorizer.fit_transform(queries)
            
            # K-Means Clustering
            n_clusters = min(5, len(queries) // 10)
            if n_clusters > 1:
                self.query_clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = self.query_clusterer.fit_predict(query_vectors.toarray())
                
                # Analysiere Cluster Performance
                cluster_performance = defaultdict(list)
                for i, metrics in enumerate(list(self.query_history)[-100:]):
                    if metrics.user_rating:
                        cluster_performance[cluster_labels[i]].append(metrics.user_rating)
                
                # Optimiere basierend auf Cluster Performance
                for cluster_id, ratings in cluster_performance.items():
                    if len(ratings) >= 3:
                        avg_rating = statistics.mean(ratings)
                        if avg_rating < 2.5:
                            print(f"📊 Poor performing cluster {cluster_id} identified for optimization")
        
        except Exception as e:
            print(f"⚠️ Clustering update failed: {e}")
    
    async def _optimize_connection_weights(self):
        """Optimiert Connection Weights basierend auf Performance"""
        
        # Finde underperforming connections
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
                reason = conn['r.reason']
                if reason in self.connection_weights:
                    print(f"🔧 Adjusting weight for connection type: {reason}")
    
    async def _analyze_chunk_performance(self):
        """Analysiert Chunk Performance und identifiziert Optimierungsmöglichkeiten"""
        
        # Finde häufig verwendete aber schlecht bewertete Chunks
        poor_performers = []
        high_performers = []
        
        for chunk_hash, performance in self.chunk_performance.items():
            if performance['usage_count'] >= 5:  # Mindestens 5x verwendet
                # Berechne durchschnittliches Rating für diesen Chunk
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
        
        # Log Insights
        if poor_performers:
            print(f"📉 Identified {len(poor_performers)} underperforming chunks")
        if high_performers:
            print(f"📈 Identified {len(high_performers)} high-performing chunks")
    
    def _save_learning_state(self):
        """Speichert den Lernzustand"""
        
        learning_state = {
            'connection_weights': dict(self.connection_weights),
            'dynamic_k_values': dict(self.dynamic_k_values),
            'retrieval_strategies': dict(self.retrieval_strategies),
            'query_patterns': dict(self.query_patterns),
            'performance_metrics': dict(self.performance_metrics),
            'chunk_performance': dict(self.chunk_performance)
        }
        
        with open(self.learning_data_path / "learning_state.pkl", 'wb') as f:
            pickle.dump(learning_state, f)
        
        # JSON Export für Debugging
        json_state = {k: v for k, v in learning_state.items() 
                     if k not in ['chunk_performance']}  # Zu groß für JSON
        
        with open(self.learning_data_path / "learning_state.json", 'w') as f:
            json.dump(json_state, f, indent=2, default=str)
    
    def _load_learning_state(self):
        """Lädt den gespeicherten Lernzustand"""
        
        learning_file = self.learning_data_path / "learning_state.pkl"
        if learning_file.exists():
            try:
                with open(learning_file, 'rb') as f:
                    learning_state = pickle.load(f)
                
                self.connection_weights = learning_state.get('connection_weights', {})
                self.dynamic_k_values = learning_state.get('dynamic_k_values', {})
                self.retrieval_strategies = learning_state.get('retrieval_strategies', {})
                self.query_patterns = defaultdict(int, learning_state.get('query_patterns', {}))
                self.performance_metrics = defaultdict(list, learning_state.get('performance_metrics', {}))
                self.chunk_performance = learning_state.get('chunk_performance', {})
                
                print("✅ Learning state loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Failed to load learning state: {e}")
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Gibt detaillierte Insights über das Lernsystem zurück"""
        
        insights = {
            'total_queries': len(self.query_history),
            'query_types': dict(self.query_patterns),
            'average_response_time': 0,
            'user_satisfaction': {},
            'optimization_stats': {},
            'top_performing_chunks': [],
            'learning_progress': {}
        }
        
        if self.query_history:
            # Response Time Analysis
            response_times = [m.response_time for m in self.query_history]
            insights['average_response_time'] = statistics.mean(response_times)
            
            # User Satisfaction per Query Type
            for query_type in self.query_patterns.keys():
                ratings = [m.user_rating for m in self.query_history 
                          if m.user_rating and query_type in m.query_text.lower()]
                if ratings:
                    insights['user_satisfaction'][query_type] = {
                        'average_rating': statistics.mean(ratings),
                        'total_ratings': len(ratings)
                    }
            
            # Learning Progress
            insights['learning_progress'] = {
                'optimized_parameters': len(self.dynamic_k_values),
                'learned_strategies': len(self.retrieval_strategies),
                'connection_weights': len(self.connection_weights)
            }
        
        return insights
```

# Integration Example
```python
async def main():
    """
    Beispiel für Integration des Self-Learning Systems
    """
    
    # Basis-RAG System (aus vorheriger Implementation)
    from rag_system import AdvancedRAGSystem, RAGConfig
    
    config = RAGConfig(neo4j_password="your_password")
    base_rag = AdvancedRAGSystem(config)
    
    # Self-Learning System initialisieren
    learning_config = LearningConfig(
        learning_rate=0.15,
        optimization_interval=50  # Häufigere Optimierung für Tests
    )
    
    smart_rag = SelfLearningRAGSystem(base_rag, learning_config)
    
    # Beispiel-Queries mit Feedback Loop
    questions = [
        "What are the main benefits of renewable energy?",
        "How does machine learning work?",
        "Compare solar and wind power efficiency"
    ]
    
    for question in questions:
        # Query ausführen
        result = await smart_rag.enhanced_query(question)
        
        print(f"Q: {question}")
        print(f"A: {result['answer'][:200]}...")
        print(f"Strategy: {result['learning_metadata']['strategy']}")
        print(f"Response time: {result['learning_metadata']['response_time']:.2f}s")
        
        # Simuliere User Feedback (normalerweise von UI)
        rating = np.random.uniform(3.0, 5.0)  # Simuliert positives Feedback
        await smart_rag.record_user_feedback(
            result['query_id'], 
            rating,
            {'clicked_sources': ['source1.pdf']}
        )
        
        print(f"User rating: {rating:.1f}/5.0\n")
    
    # Learning Insights
    insights = await smart_rag.get_learning_insights()
    print("🧠 Learning Insights:")
    print(json.dumps(insights, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
```
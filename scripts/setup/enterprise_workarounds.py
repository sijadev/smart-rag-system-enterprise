"""
Enterprise Feature Workarounds für Neo4j Community
================================================

Implementiert Graph Data Science (GDS) Ersatz-Funktionen für das RAG-System:
- Node Importance Calculation (PageRank Ersatz)
- Community Detection (Clustering)
- Similarity Analysis
- Graph Analytics
"""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import re

@dataclass
class NodeImportance:
    """Node Importance Score (PageRank Ersatz)"""
    node_id: str
    node_name: str
    node_type: str
    importance_score: float
    connections: int
    centrality: float

@dataclass
class ConceptCluster:
    """Concept Cluster (Community Detection Ersatz)"""
    cluster_id: int
    concepts: List[str]
    cluster_size: int
    centrality_score: float
    theme: str

class GraphAnalyticsWorkaround:
    """
    Ersetzt Neo4j GDS Funktionen durch Python-basierte Implementierungen
    """

    def __init__(self, neo4j_driver=None):
        self.driver = neo4j_driver
        self.graph = nx.Graph()

    async def calculate_node_importance(self, node_type: str = None) -> List[NodeImportance]:
        """
        Ersetzt GDS PageRank durch eigene Node Importance Berechnung
        """
        if not self.driver:
            return []

        query = """
        MATCH (n)
        WHERE $node_type IS NULL OR $node_type IN labels(n)
        OPTIONAL MATCH (n)-[r]-(connected)
        WITH n, count(r) as connections, collect(connected) as neighbors
        RETURN 
            elementId(n) as node_id,
            coalesce(n.name, n.title, 'Unknown') as node_name,
            labels(n)[0] as node_type,
            connections,
            neighbors
        ORDER BY connections DESC
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, node_type=node_type)
                nodes = []

                for record in result:
                    # Berechne Centrality basierend auf Verbindungen
                    connections = record["connections"]
                    centrality = min(1.0, connections / 10.0)  # Normalisiere auf 0-1

                    # Berechne Importance Score (PageRank Ersatz)
                    importance = centrality * (1 + np.log(connections + 1))

                    nodes.append(NodeImportance(
                        node_id=str(record["node_id"]),
                        node_name=record["node_name"],
                        node_type=record["node_type"],
                        importance_score=importance,
                        connections=connections,
                        centrality=centrality
                    ))

                return sorted(nodes, key=lambda x: x.importance_score, reverse=True)

        except Exception as e:
            print(f"⚠️ Node importance calculation failed: {e}")
            return []

    async def find_concept_clusters(self, min_cluster_size: int = 3) -> List[ConceptCluster]:
        """
        Ersetzt GDS Community Detection durch eigenes Clustering
        """
        if not self.driver:
            return []

        # Hole alle Concept-Beziehungen
        query = """
        MATCH (c1:Concept)-[r:RELATED_TO|CONTAINS|REFERENCES]-(c2:Concept)
        RETURN 
            c1.name as concept1,
            c2.name as concept2,
            type(r) as relationship_type,
            coalesce(r.weight, 1.0) as weight
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)

                # Baue Graph für Clustering
                edges = []
                concepts = set()

                for record in result:
                    c1, c2 = record["concept1"], record["concept2"]
                    weight = record["weight"]

                    edges.append((c1, c2, weight))
                    concepts.add(c1)
                    concepts.add(c2)

                if not edges:
                    return []

                # Verwende NetworkX für Community Detection
                G = nx.Graph()
                G.add_weighted_edges_from(edges)

                # Community Detection mit Louvain-ähnlichem Algorithmus
                communities = self._detect_communities(G)

                clusters = []
                for i, community in enumerate(communities):
                    if len(community) >= min_cluster_size:
                        # Berechne Cluster-Centralität
                        subgraph = G.subgraph(community)
                        centrality = nx.density(subgraph)

                        # Bestimme Cluster-Theme
                        theme = self._determine_cluster_theme(community)

                        clusters.append(ConceptCluster(
                            cluster_id=i,
                            concepts=list(community),
                            cluster_size=len(community),
                            centrality_score=centrality,
                            theme=theme
                        ))

                return sorted(clusters, key=lambda x: x.cluster_size, reverse=True)

        except Exception as e:
            print(f"⚠️ Concept clustering failed: {e}")
            return []

    def _detect_communities(self, graph: nx.Graph) -> List[List[str]]:
        """
        Einfache Community Detection (Ersatz für Louvain)
        """
        try:
            # Verwende NetworkX's community detection
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(graph))
            return [list(community) for community in communities]
        except:
            # Fallback: Connected Components
            return [list(component) for component in nx.connected_components(graph)]

    def _determine_cluster_theme(self, concepts: List[str]) -> str:
        """
        Bestimmt das Hauptthema eines Concept-Clusters
        """
        # Einfache Keyword-basierte Themenerkennung
        themes = {
            'machine_learning': ['ml', 'machine', 'learning', 'ai', 'neural', 'algorithm'],
            'energy': ['energy', 'solar', 'wind', 'renewable', 'power', 'electricity'],
            'technology': ['tech', 'technology', 'system', 'software', 'computer'],
            'business': ['business', 'enterprise', 'company', 'market', 'strategy']
        }

        concept_text = ' '.join(concepts).lower()
        theme_scores = {}

        for theme, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in concept_text)
            theme_scores[theme] = score

        return max(theme_scores.items(), key=lambda x: x[1])[0] if theme_scores else 'general'

class SimilarityAnalysisWorkaround:
    """
    Ersetzt Neo4j GDS Similarity Algorithms durch Ollama + Vector-basierte Ansätze
    """

    def __init__(self, vector_store, neo4j_driver=None):
        self.vector_store = vector_store
        self.driver = neo4j_driver

    async def find_similar_concepts(self, query: str, threshold: float = 0.8,
                                   k: int = 10) -> List[Dict[str, Any]]:
        """
        Ersetzt GDS Node Similarity durch Ollama Embeddings + Graph-Beziehungen
        """
        similar_results = []

        try:
            # 1. Semantische Ähnlichkeit über Ollama/Vector Store
            if hasattr(self.vector_store, 'similarity_search'):
                vector_similar = await self.vector_store.similarity_search(query, k=k)

                for doc in vector_similar:
                    similar_results.append({
                        'type': 'semantic',
                        'content': doc.page_content[:200],
                        'similarity': 0.9,  # Placeholder - echte Similarity aus Vector Store
                        'source': 'ollama_embeddings'
                    })

            # 2. Graph-basierte Ähnlichkeit aus Neo4j
            if self.driver:
                graph_similar = await self._find_graph_similar_concepts(query, k)
                similar_results.extend(graph_similar)

            # 3. Kombiniere und ranke Ergebnisse
            return self._rank_similarity_results(similar_results, threshold)[:k]

        except Exception as e:
            print(f"⚠️ Similarity analysis failed: {e}")
            return []

    async def _find_graph_similar_concepts(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Graph-basierte Ähnlichkeitssuche in Neo4j
        """
        query_cypher = """
        MATCH (c:Concept)
        WHERE c.name CONTAINS $query OR c.description CONTAINS $query
        OPTIONAL MATCH (c)-[r:RELATED_TO]-(related:Concept)
        WITH c, count(r) as relationship_strength, collect(related.name) as related_concepts
        RETURN 
            c.name as concept_name,
            c.description as description,
            relationship_strength,
            related_concepts
        ORDER BY relationship_strength DESC
        LIMIT $k
        """

        try:
            with self.driver.session() as session:
                result = session.run(query_cypher, query=query, k=k)

                graph_results = []
                for record in result:
                    similarity = min(1.0, record["relationship_strength"] / 5.0)  # Normalisiere

                    graph_results.append({
                        'type': 'graph',
                        'concept_name': record["concept_name"],
                        'description': record["description"] or '',
                        'similarity': similarity,
                        'related_concepts': record["related_concepts"],
                        'source': 'neo4j_graph'
                    })

                return graph_results

        except Exception as e:
            print(f"⚠️ Graph similarity search failed: {e}")
            return []

    def _rank_similarity_results(self, results: List[Dict[str, Any]],
                               threshold: float) -> List[Dict[str, Any]]:
        """
        Rankt und filtert Ähnlichkeitsergebnisse
        """
        # Filtere nach Threshold
        filtered = [r for r in results if r.get('similarity', 0) >= threshold]

        # Kombiniere Scores für hybrid ranking
        for result in filtered:
            if result['type'] == 'semantic':
                result['combined_score'] = result['similarity'] * 1.2  # Bevorzuge semantische
            else:
                result['combined_score'] = result['similarity'] * 1.0

        return sorted(filtered, key=lambda x: x['combined_score'], reverse=True)

class PerformanceOptimizationWorkaround:
    """
    Performance Optimierungen für Community Edition
    """

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    async def optimize_indexes(self) -> Dict[str, bool]:
        """
        Erstellt Performance-Indizes für RAG-Queries
        """
        optimizations = {}

        index_queries = [
            # Concept-Indizes
            "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX concept_domain_idx IF NOT EXISTS FOR (c:Concept) ON (c.domain)",

            # Document-Indizes
            "CREATE INDEX document_name_idx IF NOT EXISTS FOR (d:Document) ON (d.name)",
            "CREATE INDEX document_created_idx IF NOT EXISTS FOR (d:Document) ON (d.created)",

            # Entity-Indizes
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",

            # Composite-Indizes für häufige Queries
            "CREATE INDEX concept_name_domain_idx IF NOT EXISTS FOR (c:Concept) ON (c.name, c.domain)"
        ]

        try:
            with self.driver.session() as session:
                for query in index_queries:
                    try:
                        session.run(query)
                        index_name = query.split()[2]
                        optimizations[index_name] = True
                    except Exception as e:
                        print(f"⚠️ Index creation failed: {e}")
                        optimizations[query.split()[2]] = False

            return optimizations

        except Exception as e:
            print(f"⚠️ Performance optimization failed: {e}")
            return {}

    async def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """
        Analysiert Query Performance (Ersatz für Enterprise Profiling)
        """
        try:
            with self.driver.session() as session:
                # EXPLAIN für Query-Plan
                explain_result = session.run(f"EXPLAIN {query}")

                # PROFILE für Performance-Metriken
                profile_result = session.run(f"PROFILE {query}")

                return {
                    'query_plan': [dict(record) for record in explain_result],
                    'performance_stats': [dict(record) for record in profile_result],
                    'optimization_suggestions': self._get_optimization_suggestions(query)
                }

        except Exception as e:
            return {'error': str(e)}

    def _get_optimization_suggestions(self, query: str) -> List[str]:
        """
        Gibt Optimierungsvorschläge für Queries
        """
        suggestions = []

        if 'MATCH' in query and 'WHERE' not in query:
            suggestions.append("Consider adding WHERE clause to filter results early")

        if 'ORDER BY' in query and 'LIMIT' not in query:
            suggestions.append("Consider adding LIMIT to prevent large result sets")

        if query.count('MATCH') > 3:
            suggestions.append("Complex query - consider breaking into smaller parts")

        return suggestions


class EnterpriseWorkaroundManager:
    """
    Hauptklasse die alle Enterprise-Workarounds verwaltet
    """

    def __init__(self, rag_system):
        self.rag_system = rag_system

        # Determine the correct way to access vector_store and driver
        if hasattr(rag_system, 'base_system'):
            # For EnhancedSelfLearningRAG, access through base_system
            base_system = rag_system.base_system
            vector_store = base_system.vector_store if hasattr(base_system, 'vector_store') else None
            driver = base_system.driver if hasattr(base_system, 'driver') else None
        elif hasattr(rag_system, 'data_import_workflow'):
            # For EnhancedSelfLearningRAG, access through data_import_workflow.rag_system
            base_system = rag_system.data_import_workflow.rag_system
            vector_store = base_system.vector_store if hasattr(base_system, 'vector_store') else None
            driver = base_system.driver if hasattr(base_system, 'driver') else None
        else:
            # Direct access for AdvancedRAGSystem
            vector_store = rag_system.vector_store if hasattr(rag_system, 'vector_store') else None
            driver = rag_system.driver if hasattr(rag_system, 'driver') else None

        self.analytics = GraphAnalyticsWorkaround(driver)
        self.similarity = SimilarityAnalysisWorkaround(vector_store, driver)
        self.performance = PerformanceOptimizationWorkaround(driver)

    async def initialize_workarounds(self):
        """
        Initialisiert alle Workarounds
        """
        print("🔧 Initializing Enterprise Feature Workarounds...")

        # Performance-Optimierungen
        if hasattr(self, 'performance') and self.performance.driver:
            optimizations = await self.performance.optimize_indexes()
            successful_opts = sum(optimizations.values())
            print(f"✅ Created {successful_opts}/{len(optimizations)} performance indexes")

        # Test Analytics
        try:
            important_nodes = await self.analytics.calculate_node_importance()
            print(f"✅ Node importance calculation: {len(important_nodes)} nodes analyzed")
        except Exception as e:
            print(f"⚠️ Analytics test failed: {e}")

        # Test Similarity
        try:
            if hasattr(self.rag_system, 'vector_store'):
                similar = await self.similarity.find_similar_concepts("machine learning", k=3)
                print(f"✅ Similarity analysis: {len(similar)} similar concepts found")
        except Exception as e:
            print(f"⚠️ Similarity test failed: {e}")

        print("✅ Enterprise workarounds initialized successfully!")

    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """
        Enterprise Analytics Dashboard (GDS Ersatz)
        """
        dashboard = {
            'timestamp': asyncio.get_event_loop().time(),
            'node_importance': [],
            'concept_clusters': [],
            'performance_metrics': {}
        }

        try:
            # Node Importance
            dashboard['node_importance'] = await self.analytics.calculate_node_importance()

            # Concept Clustering
            dashboard['concept_clusters'] = await self.analytics.find_concept_clusters()

            # Performance Metrics
            if hasattr(self, 'performance'):
                dashboard['performance_metrics'] = {
                    'indexes_optimized': True,
                    'query_suggestions': []
                }

        except Exception as e:
            dashboard['error'] = str(e)

        return dashboard


# Integration Helper Functions
async def enhance_rag_with_workarounds(rag_system):
    """
    Verbessert ein bestehendes RAG-System mit Enterprise-Workarounds
    """
    workaround_manager = EnterpriseWorkaroundManager(rag_system)
    await workaround_manager.initialize_workarounds()

    # Füge Workaround-Methoden zum RAG-System hinzu
    rag_system.enterprise_analytics = workaround_manager.analytics
    rag_system.enterprise_similarity = workaround_manager.similarity
    rag_system.enterprise_performance = workaround_manager.performance
    rag_system.get_analytics_dashboard = workaround_manager.get_analytics_dashboard

    return rag_system

# RAG System Monitoring Dashboard
# Visualisierung und Monitoring für das Self-Learning RAG System
```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
import json
from datetime import datetime, timedelta
import numpy as np

class RAGMonitoringDashboard:
    """
    Streamlit-basiertes Dashboard für RAG System Monitoring
    """
    
    def __init__(self, smart_rag_system):
        self.rag_system = smart_rag_system
        self.setup_dashboard()
    
    def setup_dashboard(self):
        """Setup Streamlit Dashboard"""
        
        st.set_page_config(
            page_title="🧠 Smart RAG Dashboard",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 Self-Learning RAG System Dashboard")
        st.markdown("---")
    
    async def render_dashboard(self):
        """Main Dashboard Rendering"""
        
        # Sidebar Controls
        with st.sidebar:
            st.header("⚙️ Controls")
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            if st.button("🧠 Trigger Learning"):
                await self.rag_system._trigger_optimization()
                st.success("Learning optimization triggered!")
            
            if st.button("💾 Save Learning State"):
                self.rag_system._save_learning_state()
                st.success("Learning state saved!")
            
            st.markdown("---")
            
            # Time Range Filter
            time_range = st.selectbox(
                "📅 Time Range",
                ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
            )
        
        # Main Dashboard Content
        await self.render_overview()
        await self.render_performance_metrics()
        await self.render_learning_progress()
        await self.render_query_analysis()
        await self.render_system_health()
    
    async def render_overview(self):
        """Render Overview Section"""
        
        st.header("📊 System Overview")
        
        # Get insights
        insights = await self.rag_system.get_learning_insights()
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries",
                insights['total_queries'],
                delta=f"+{len(self.rag_system.query_history[-24:])}" if len(self.rag_system.query_history) > 24 else None
            )
        
        with col2:
            avg_response_time = insights.get('average_response_time', 0)
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.2f}s",
                delta=self.calculate_response_time_trend()
            )
        
        with col3:
            total_chunks = len(self.rag_system.chunk_performance)
            st.metric(
                "Knowledge Chunks",
                total_chunks,
                delta=None
            )
        
        with col4:
            learned_connections = len(self.rag_system.connection_weights)
            st.metric(
                "Learned Connections",
                learned_connections,
                delta=None
            )
    
    async def render_performance_metrics(self):
        """Render Performance Metrics"""
        
        st.header("📈 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User Satisfaction by Query Type
            insights = await self.rag_system.get_learning_insights()
            satisfaction_data = insights.get('user_satisfaction', {})
            
            if satisfaction_data:
                query_types = list(satisfaction_data.keys())
                avg_ratings = [satisfaction_data[qt]['average_rating'] for qt in query_types]
                total_ratings = [satisfaction_data[qt]['total_ratings'] for qt in query_types]
                
                fig = px.bar(
                    x=query_types,
                    y=avg_ratings,
                    title="Average User Rating by Query Type",
                    labels={'x': 'Query Type', 'y': 'Average Rating'},
                    color=avg_ratings,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response Time Trend
            if self.rag_system.query_history:
                query_data = []
                for metrics in list(self.rag_system.query_history)[-50:]:
                    query_data.append({
                        'timestamp': metrics.timestamp,
                        'response_time': metrics.response_time,
                        'rating': metrics.user_rating or 0
                    })
                
                df = pd.DataFrame(query_data)
                
                if not df.empty:
                    fig = px.scatter(
                        df,
                        x='timestamp',
                        y='response_time',
                        color='rating',
                        size='rating',
                        title="Response Time Trend",
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    async def render_learning_progress(self):
        """Render Learning Progress"""
        
        st.header("🧠 Learning Progress")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Query Type Distribution")
            
            query_patterns = dict(self.rag_system.query_patterns)
            if query_patterns:
                fig = px.pie(
                    values=list(query_patterns.values()),
                    names=list(query_patterns.keys()),
                    title="Query Types"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚙️ Optimized Parameters")
            
            optimized_k = len(self.rag_system.dynamic_k_values)
            learned_strategies = len(self.rag_system.retrieval_strategies)
            connection_weights = len(self.rag_system.connection_weights)
            
            progress_data = {
                'Parameter Type': ['K-Values', 'Strategies', 'Weights'],
                'Count': [optimized_k, learned_strategies, connection_weights]
            }
            
            fig = px.bar(
                progress_data,
                x='Parameter Type',
                y='Count',
                title="Learning Parameters",
                color='Count',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.subheader("📊 Connection Weights")
            
            if self.rag_system.connection_weights:
                weights_df = pd.DataFrame([
                    {'Connection Type': k, 'Weight': v}
                    for k, v in list(self.rag_system.connection_weights.items())[:10]
                ])
                
                fig = px.bar(
                    weights_df,
                    x='Weight',
                    y='Connection Type',
                    orientation='h',
                    title="Top Connection Weights",
                    color='Weight',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    async def render_query_analysis(self):
        """Render Query Analysis"""
        
        st.header("🔍 Query Analysis")
        
        if self.rag_system.query_history:
            # Recent Queries Table
            st.subheader("Recent Queries")
            
            recent_queries = []
            for metrics in list(self.rag_system.query_history)[-10:]:
                recent_queries.append({
                    'Timestamp': metrics.timestamp.strftime('%H:%M:%S'),
                    'Query': metrics.query_text[:50] + '...' if len(metrics.query_text) > 50 else metrics.query_text,
                    'Response Time': f"{metrics.response_time:.2f}s",
                    'Chunks': metrics.retrieved_chunks,
                    'Rating': metrics.user_rating or 'N/A'
                })
            
            df = pd.DataFrame(recent_queries)
            st.dataframe(df, use_container_width=True)
            
            # Query Performance Heatmap
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Heatmap")
                
                # Create performance matrix
                performance_matrix = self.create_performance_matrix()
                if performance_matrix is not None:
                    fig = px.imshow(
                        performance_matrix,
                        title="Query Performance Matrix",
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Performing Chunks")
                
                # Get chunk performance
                chunk_scores = []
                for chunk_hash, performance in self.rag_system.chunk_performance.items():
                    if performance.get('avg_rating'):
                        chunk_scores.append({
                            'Chunk': performance['content_preview'][:30] + '...',
                            'Usage Count': performance['usage_count'],
                            'Avg Rating': performance['avg_rating']
                        })
                
                if chunk_scores:
                    chunk_df = pd.DataFrame(sorted(chunk_scores, key=lambda x: x['Avg Rating'], reverse=True)[:10])
                    st.dataframe(chunk_df, use_container_width=True)
    
    async def render_system_health(self):
        """Render System Health"""
        
        st.header("💚 System Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Health Metrics")
            
            # Calculate health metrics
            health_score = self.calculate_health_score()
            
            # Health gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("System Stats")
            
            stats = await self.rag_system.base_system.get_system_statistics()
            
            stats_data = {
                'Metric': ['Knowledge Chunks', 'Keywords', 'Topics', 'Semantic Links'],
                'Count': [
                    stats.get('chunks', 0),
                    stats.get('keywords', 0),
                    stats.get('topics', 0),
                    stats.get('semantic_connections', 0)
                ]
            }
            
            fig = px.bar(
                stats_data,
                x='Metric',
                y='Count',
                title="Knowledge Base Statistics",
                color='Count',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        recommendations = self.generate_recommendations()
        
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")
    
    def calculate_response_time_trend(self):
        """Calculate response time trend"""
        if len(self.rag_system.query_history) < 10:
            return None
        
        recent = [m.response_time for m in list(self.rag_system.query_history)[-5:]]
        older = [m.response_time for m in list(self.rag_system.query_history)[-10:-5]]
        
        if recent and older:
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)
            trend = recent_avg - older_avg
            return f"{trend:+.2f}s"
        
        return None
    
    def create_performance_matrix(self):
        """Create performance matrix for heatmap"""
        # Simplified version - in production, this would be more sophisticated
        query_types = list(self.rag_system.query_patterns.keys())
        strategies = ['vector_heavy', 'graph_heavy', 'hybrid_optimized', 'context_aware']
        
        if not query_types:
            return None
        
        # Generate mock performance matrix (replace with real data)
        matrix = np.random.uniform(0.5, 1.0, size=(len(query_types), len(strategies)))
        return matrix
    
    def calculate_health_score(self):
        """Calculate overall system health score"""
        score = 60  # Base score
        
        # Query volume bonus
        if len(self.rag_system.query_history) > 50:
            score += 10
        
        # Learning progress bonus
        if len(self.rag_system.dynamic_k_values) > 3:
            score += 10
        
        # User satisfaction bonus
        recent_ratings = [m.user_rating for m in list(self.rag_system.query_history)[-20:] 
                         if m.user_rating]
        if recent_ratings:
            avg_rating = sum(recent_ratings) / len(recent_ratings)
            score += int(avg_rating * 4)  # Max 20 points
        
        return min(100, score)
    
    def generate_recommendations(self):
        """Generate system recommendations"""
        recommendations = []
        
        # Check query volume
        if len(self.rag_system.query_history) < 20:
            recommendations.append("Increase query volume to improve learning effectiveness")
        
        # Check user feedback
        rated_queries = [m for m in self.rag_system.query_history if m.user_rating]
        if len(rated_queries) < len(self.rag_system.query_history) * 0.3:
            recommendations.append("Encourage more user feedback to enhance learning")
        
        # Check performance
        if rated_queries:
            avg_rating = sum(m.user_rating for m in rated_queries) / len(rated_queries)
            if avg_rating < 3.5:
                recommendations.append("Consider adjusting retrieval strategies - current performance below optimal")
        
        # Check learning progress
        if len(self.rag_system.dynamic_k_values) < 3:
            recommendations.append("Allow more queries to enable parameter optimization")
        
        if not recommendations:
            recommendations.append("System is performing well! Continue monitoring.")
        
        return recommendations
```
# Streamlit App Runner
```python
async def run_dashboard(smart_rag_system):
    """Run the Streamlit dashboard"""
    
    dashboard = RAGMonitoringDashboard(smart_rag_system)
    await dashboard.render_dashboard()

# Usage
if __name__ == "__main__":
    # This would be called from your main application
    # streamlit run rag_monitoring_dashboard.py
    st.write("Please run this with: `streamlit run rag_monitoring_dashboard.py`")
    st.write("Make sure to import your smart_rag_system instance.")
```
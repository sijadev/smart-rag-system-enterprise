#!/usr/bin/env python3
"""SmartRAGSystem: Bootstrapping and runtime wiring using DI and Factories

Minimal, non-invasive implementation that wires existing Factories and DIContainer.
Provides initialize() to register defaults and create services, and query() to run
retrieval using registered strategies.
"""

import time
from typing import Optional

from . import bootstrap
from .config.builders import RAGSystemConfig
from .di_container import DIContainer
from .factories import (DatabaseFactory, LLMServiceFactory,
                        RetrievalStrategyFactory)
from .interfaces import (IGraphStore, IVectorStore, QueryContext,
                         RetrievalStrategy)


class SmartRAGSystem:
    def __init__(self, config: Optional[RAGSystemConfig] = None):
        self.config = config or RAGSystemConfig()
        self.container = DIContainer()
        self.llm_service = None
        self.vector_store = None
        self.graph_store = None
        self.initialized = False

    def initialize(self):
        """Register defaults and create core services. Non-blocking, idempotent."""
        if self.initialized:
            return

        # Register default adapters/strategies
        bootstrap.register_all_defaults()

        # Create LLM service
        try:
            self.llm_service = LLMServiceFactory.create(
                self.config.llm_provider,
                {
                    "model_name": self.config.model_name,
                    "temperature": self.config.temperature,
                },
            )
            self.container.register_instance(type(self.llm_service), self.llm_service)
        except Exception:
            self.llm_service = None

        # Create Vector Store with fallback to 'mock' if registration missing
        try:
            self.vector_store = DatabaseFactory.create_vector_store(
                self.config.vector_store_type,
                {"collection_name": self.config.collection_name},
            )
            self.container.register_instance(IVectorStore, self.vector_store)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)

            logger.warning(
                f"Vector store '{self.config.vector_store_type}' creation failed: {e}. Attempting direct mock import as fallback."
            )

            # Try registry fallback first
            try:
                self.vector_store = DatabaseFactory.create_vector_store(
                    "mock", {"collection_name": self.config.collection_name}
                )
                logger.info("Using fallback mock vector store from registry")
                self.container.register_instance(IVectorStore, self.vector_store)
            except Exception:
                # If registry does not contain 'mock', import mock class
                # directly
                try:
                    from src.adapters.mock_vector_store import MockVectorStore

                    self.vector_store = MockVectorStore(
                        {"collection_name": self.config.collection_name}
                    )
                    logger.info("Using direct MockVectorStore import as fallback")
                    self.container.register_instance(IVectorStore, self.vector_store)
                except Exception as e2:
                    logger.error(f"Direct MockVectorStore import failed: {e2}")
                    self.vector_store = None

        # Create Graph Store if enabled
        if self.config.enable_graph_store:
            try:
                # prefer neo4j if configured
                self.graph_store = DatabaseFactory.create_graph_store(
                    "neo4j",
                    {
                        "neo4j_uri": self.config.neo4j_uri,
                        "neo4j_user": self.config.neo4j_user,
                        "neo4j_password": self.config.neo4j_password,
                    },
                )
                self.container.register_instance(IGraphStore, self.graph_store)
            except Exception as e:
                try:
                    import logging

                    logger = logging.getLogger(__name__)
                except Exception:
                    logger = None

                if logger:
                    logger.warning(
                        f"Neo4j store creation failed: {e}. Falling back to 'mock'."
                    )

                # try mock fallback from registry, else direct import
                try:
                    self.graph_store = DatabaseFactory.create_graph_store("mock", {})
                    self.container.register_instance(IGraphStore, self.graph_store)
                except Exception:
                    try:
                        from src.adapters.mock_graph_store import \
                            MockGraphStore

                        self.graph_store = MockGraphStore({})
                        self.container.register_instance(IGraphStore, self.graph_store)
                    except Exception as e3:
                        logger.error(f"Direct MockGraphStore import failed: {e3}")
                        self.graph_store = None

        self.initialized = True

    async def query(
        self, query_text: str, strategy: Optional[RetrievalStrategy] = None, k: int = 5
    ):
        """Run retrieval using chosen strategy. Returns RetrievalResult."""
        if not self.initialized:
            self.initialize()

        chosen = strategy or self.config.default_retrieval_strategy
        # Explicit type annotation to satisfy static type checkers
        chosen: RetrievalStrategy = strategy or self.config.default_retrieval_strategy
        # Build appropriate strategy instance
        strat = None
        try:
            if chosen == RetrievalStrategy.VECTOR_ONLY:
                strat = RetrievalStrategyFactory.create(
                    RetrievalStrategy.VECTOR_ONLY, self.vector_store
                )
            elif chosen == RetrievalStrategy.GRAPH_ONLY:
                strat = RetrievalStrategyFactory.create(
                    RetrievalStrategy.GRAPH_ONLY, self.graph_store
                )
            elif chosen == RetrievalStrategy.HYBRID:
                strat = RetrievalStrategyFactory.create(
                    RetrievalStrategy.HYBRID,
                    self.vector_store,
                    self.graph_store,
                    self.llm_service,
                )
            elif chosen == RetrievalStrategy.SEMANTIC_SEARCH:
                strat = RetrievalStrategyFactory.create(
                    RetrievalStrategy.SEMANTIC_SEARCH,
                    self.vector_store,
                    self.llm_service,
                )
            else:
                strat = RetrievalStrategyFactory.create(chosen, self.vector_store)
        except Exception as e:
            raise RuntimeError(f"Could not create retrieval strategy: {e}")

        # Create simple QueryContext
        ctx = QueryContext(query_id=f"q_{int(time.time() * 1000)}")
        return await strat.retrieve(query_text, ctx, k=k)

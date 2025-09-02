#!/usr/bin/env python3
"""
Chain of Responsibility Pattern für Query Processing
==================================================

Implementiert eine Pipeline von Query-Prozessoren die nacheinander versuchen,
eine Query zu verarbeiten
"""

from typing import List, Dict, Any, Optional
import asyncio
import logging
from abc import abstractmethod
from ..interfaces import (
    IQueryProcessor, QueryContext, RAGResponse
)
from ..monitoring.observers import EventManager

logger = logging.getLogger(__name__)


class QueryProcessorChain:
    """Chain of Responsibility für Query-Processing"""

    def __init__(self, event_manager: Optional[EventManager] = None):
        self.processors: List[IQueryProcessor] = []
        self.event_manager = event_manager

    def add_processor(self, processor: IQueryProcessor) -> 'QueryProcessorChain':
        """Fügt Prozessor zur Chain hinzu"""
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.__class__.__name__}")
        return self

    def remove_processor(self, processor: IQueryProcessor) -> 'QueryProcessorChain':
        """Entfernt Prozessor aus Chain"""
        if processor in self.processors:
            self.processors.remove(processor)
            logger.info(f"Removed processor: {processor.__class__.__name__}")
        return self

    async def process_query(self, query: str, context: QueryContext) -> RAGResponse:
        """Verarbeitet Query durch die Chain"""
        start_time = asyncio.get_event_loop().time()

        for processor in self.processors:
            try:
                if processor.can_handle(query, context):
                    logger.info(f"Processing query with: {processor.__class__.__name__}")

                    response = await processor.process(query, context)
                    processing_time = asyncio.get_event_loop().time() - start_time
                    response.processing_time = processing_time

                    # Event für erfolgreiches Processing
                    if self.event_manager:
                        await self.event_manager.notify('query_completed', {
                            'query_id': context.query_id,
                            'processor': processor.__class__.__name__,
                            'processing_time': processing_time,
                            'confidence': response.confidence
                        })

                    return response

            except Exception as e:
                logger.error(f"Processor {processor.__class__.__name__} failed: {e}")

                # Event für Fehler
                if self.event_manager:
                    await self.event_manager.notify('query_failed', {
                        'query_id': context.query_id,
                        'processor': processor.__class__.__name__,
                        'error': str(e)
                    })

                continue  # Versuche nächsten Prozessor

        # Kein Prozessor konnte Query verarbeiten
        fallback_response = RAGResponse(
            answer="Sorry, I couldn't process your query. Please try rephrasing it.",
            contexts=[],
            sources=[],
            confidence=0.0,
            processing_time=asyncio.get_event_loop().time() - start_time,
            metadata={'error': 'no_processor_available'}
        )

        if self.event_manager:
            await self.event_manager.notify('query_failed', {
                'query_id': context.query_id,
                'error': 'no_processor_available'
            })

        return fallback_response


class BaseQueryProcessor(IQueryProcessor):
    """Basis-Klasse für Query-Prozessoren"""

    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
        self.success_count = 0

    async def process(self, query: str, context: QueryContext) -> RAGResponse:
        """Template Method für Query-Processing"""
        self.processed_count += 1

        try:
            # Pre-processing
            await self._preprocess(query, context)

            # Main processing
            response = await self._do_process(query, context)

            # Post-processing
            response = await self._postprocess(response, query, context)

            self.success_count += 1
            return response

        except Exception as e:
            logger.error(f"Processing failed in {self.name}: {e}")
            raise

    async def _preprocess(self, query: str, context: QueryContext) -> None:
        """Pre-processing Hook"""
        pass

    @abstractmethod
    async def _do_process(self, query: str, context: QueryContext) -> RAGResponse:
        """Hauptverarbeitung - muss implementiert werden"""
        pass

    async def _postprocess(self, response: RAGResponse, query: str, context: QueryContext) -> RAGResponse:
        """Post-processing Hook"""
        return response

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Prozessor-Statistiken zurück"""
        success_rate = self.success_count / self.processed_count if self.processed_count > 0 else 0
        return {
            'name': self.name,
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'success_rate': success_rate
        }


class FactualQueryProcessor(BaseQueryProcessor):
    """Prozessor für faktische Fragen"""

    def __init__(self, retrieval_strategy, llm_service):
        super().__init__("FactualQueryProcessor")
        self.retrieval_strategy = retrieval_strategy
        self.llm_service = llm_service

        # Keywords die auf faktische Fragen hindeuten
        self.factual_keywords = [
            'what is', 'who is', 'when', 'where', 'how many', 'which',
            'define', 'definition', 'meaning', 'explain'
        ]

    def can_handle(self, query: str, context: QueryContext) -> bool:
        """Prüft ob Query faktisch ist"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.factual_keywords)

    async def _do_process(self, query: str, context: QueryContext) -> RAGResponse:
        """Verarbeitet faktische Query"""
        # Retrieval
        retrieval_result = await self.retrieval_strategy.retrieve(query, context, k=3)

        if not retrieval_result.contexts:
            return RAGResponse(
                answer="I couldn't find relevant information to answer your question.",
                contexts=[],
                sources=[],
                confidence=0.1,
                processing_time=0.0,
                metadata={'processor': 'factual', 'retrieval_results': 0}
            )

        # Generate Answer
        context_text = "\n\n".join(retrieval_result.contexts)
        prompt = f"""
        Based on the following context, provide a concise and accurate answer:

        Context:
        {context_text}

        Question: {query}

        Answer:
        """

        answer = await self.llm_service.generate(prompt, context)

        # Calculate confidence based on retrieval scores
        avg_confidence = sum(retrieval_result.confidence_scores) / len(retrieval_result.confidence_scores)

        return RAGResponse(
            answer=answer,
            contexts=retrieval_result.contexts,
            sources=retrieval_result.sources,
            confidence=avg_confidence,
            processing_time=0.0,  # Will be set by chain
            metadata={
                'processor': 'factual',
                'retrieval_strategy': retrieval_result.metadata.get('strategy'),
                'retrieval_results': len(retrieval_result.contexts)
            }
        )


class AnalyticalQueryProcessor(BaseQueryProcessor):
    """Prozessor für analytische Fragen"""

    def __init__(self, retrieval_strategy, llm_service):
        super().__init__("AnalyticalQueryProcessor")
        self.retrieval_strategy = retrieval_strategy
        self.llm_service = llm_service

        self.analytical_keywords = [
            'analyze', 'compare', 'contrast', 'evaluate', 'assess',
            'why', 'how does', 'relationship', 'impact', 'effect',
            'pros and cons', 'advantages', 'disadvantages'
        ]

    def can_handle(self, query: str, context: QueryContext) -> bool:
        """Prüft ob Query analytisch ist"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.analytical_keywords)

    async def _do_process(self, query: str, context: QueryContext) -> RAGResponse:
        """Verarbeitet analytische Query"""
        # Mehr Kontext für Analyse benötigt
        retrieval_result = await self.retrieval_strategy.retrieve(query, context, k=7)

        if not retrieval_result.contexts:
            return RAGResponse(
                answer="I need more context to provide a thorough analysis.",
                contexts=[],
                sources=[],
                confidence=0.1,
                processing_time=0.0,
                metadata={'processor': 'analytical', 'retrieval_results': 0}
            )

        context_text = "\n\n".join(retrieval_result.contexts)
        prompt = f"""
        Perform a thorough analysis based on the provided context:

        Context:
        {context_text}

        Question: {query}

        Please provide:
        1. Key points and evidence
        2. Analysis and reasoning
        3. Conclusions

        Analysis:
        """

        answer = await self.llm_service.generate(prompt, context)

        # Höhere Confidence für mehr Kontext
        confidence_boost = min(len(retrieval_result.contexts) / 10, 0.2)
        avg_confidence = sum(retrieval_result.confidence_scores) / len(retrieval_result.confidence_scores)
        final_confidence = min(avg_confidence + confidence_boost, 1.0)

        return RAGResponse(
            answer=answer,
            contexts=retrieval_result.contexts,
            sources=retrieval_result.sources,
            confidence=final_confidence,
            processing_time=0.0,
            metadata={
                'processor': 'analytical',
                'retrieval_strategy': retrieval_result.metadata.get('strategy'),
                'retrieval_results': len(retrieval_result.contexts),
                'confidence_boost': confidence_boost
            }
        )


class ProceduralQueryProcessor(BaseQueryProcessor):
    """Prozessor für Verfahrensfragen (How-to)"""

    def __init__(self, retrieval_strategy, llm_service):
        super().__init__("ProceduralQueryProcessor")
        self.retrieval_strategy = retrieval_strategy
        self.llm_service = llm_service

        self.procedural_keywords = [
            'how to', 'steps to', 'process', 'procedure', 'method',
            'guide', 'tutorial', 'instructions', 'setup', 'install'
        ]

    def can_handle(self, query: str, context: QueryContext) -> bool:
        """Prüft ob Query prozedural ist"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.procedural_keywords)

    async def _do_process(self, query: str, context: QueryContext) -> RAGResponse:
        """Verarbeitet prozedurale Query"""
        retrieval_result = await self.retrieval_strategy.retrieve(query, context, k=5)

        if not retrieval_result.contexts:
            return RAGResponse(
                answer="I couldn't find specific instructions for your request.",
                contexts=[],
                sources=[],
                confidence=0.1,
                processing_time=0.0,
                metadata={'processor': 'procedural', 'retrieval_results': 0}
            )

        context_text = "\n\n".join(retrieval_result.contexts)
        prompt = f"""
        Based on the context, provide step-by-step instructions:

        Context:
        {context_text}

        Question: {query}

        Please format your answer as:
        1. Step one...
        2. Step two...
        3. etc.

        Include any important notes or warnings.

        Instructions:
        """

        answer = await self.llm_service.generate(prompt, context)

        avg_confidence = sum(retrieval_result.confidence_scores) / len(retrieval_result.confidence_scores)

        return RAGResponse(
            answer=answer,
            contexts=retrieval_result.contexts,
            sources=retrieval_result.sources,
            confidence=avg_confidence,
            processing_time=0.0,
            metadata={
                'processor': 'procedural',
                'retrieval_strategy': retrieval_result.metadata.get('strategy'),
                'retrieval_results': len(retrieval_result.contexts)
            }
        )


class FallbackQueryProcessor(BaseQueryProcessor):
    """Fallback-Prozessor für alle anderen Queries"""

    def __init__(self, retrieval_strategy, llm_service):
        super().__init__("FallbackQueryProcessor")
        self.retrieval_strategy = retrieval_strategy
        self.llm_service = llm_service

    def can_handle(self, query: str, context: QueryContext) -> bool:
        """Kann alle Queries verarbeiten (Fallback)"""
        return True

    async def _do_process(self, query: str, context: QueryContext) -> RAGResponse:
        """Verarbeitet allgemeine Query"""
        retrieval_result = await self.retrieval_strategy.retrieve(query, context, k=5)

        if not retrieval_result.contexts:
            return RAGResponse(
                answer="I'm sorry, but I couldn't find relevant information to answer your question. "
                       "Could you please rephrase or provide more context?",
                contexts=[],
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={'processor': 'fallback', 'retrieval_results': 0}
            )

        context_text = "\n\n".join(retrieval_result.contexts)
        prompt = f"""
        Based on the following information, please answer the question:

        Context:
        {context_text}

        Question: {query}

        Answer:
        """

        answer = await self.llm_service.generate(prompt, context)

        # Niedrigere Confidence für Fallback
        avg_confidence = sum(retrieval_result.confidence_scores) / len(retrieval_result.confidence_scores)
        fallback_confidence = avg_confidence * 0.8  # 20% Reduktion

        return RAGResponse(
            answer=answer,
            contexts=retrieval_result.contexts,
            sources=retrieval_result.sources,
            confidence=fallback_confidence,
            processing_time=0.0,
            metadata={
                'processor': 'fallback',
                'retrieval_strategy': retrieval_result.metadata.get('strategy'),
                'retrieval_results': len(retrieval_result.contexts),
                'confidence_penalty': 0.2
            }
        )


# Utility Functions

def create_default_chain(retrieval_strategy, llm_service,
                         event_manager: Optional[EventManager] = None) -> QueryProcessorChain:
    """Erstellt Standard Query-Processing-Chain"""
    chain = QueryProcessorChain(event_manager)

    # Reihenfolge ist wichtig - spezifischste zuerst
    chain.add_processor(FactualQueryProcessor(retrieval_strategy, llm_service))
    chain.add_processor(AnalyticalQueryProcessor(retrieval_strategy, llm_service))
    chain.add_processor(ProceduralQueryProcessor(retrieval_strategy, llm_service))
    chain.add_processor(FallbackQueryProcessor(retrieval_strategy, llm_service))  # Immer zuletzt

    logger.info("Created default query processing chain")
    return chain


def create_custom_chain(processors: List[IQueryProcessor],
                        event_manager: Optional[EventManager] = None) -> QueryProcessorChain:
    """Erstellt benutzerdefinierte Chain"""
    chain = QueryProcessorChain(event_manager)

    for processor in processors:
        chain.add_processor(processor)

    return chain

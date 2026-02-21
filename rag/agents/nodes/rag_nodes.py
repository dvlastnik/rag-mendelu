from typing import cast, Literal, Dict
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import END
from flashrank import Ranker, RerankRequest
import difflib

from database.base.MyDocument import MyDocument
from rag.agents.state import AgentState, WorkerState
from rag.agents.models import MultiQuery, GradeDocumentsBatch, GradeHallucinations, ExtractionScheme
from rag.agents.enums import NodeName
from rag.agents.prompts import Prompts
from database.base.BaseDbRepository import BaseDbRepository
from text_embedding import TextEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)

class RagNodes:
    def __init__(self, llm: BaseChatModel, db_repository: BaseDbRepository, embedding_service: TextEmbeddingService):
        self.llm = llm
        self.db_repository = db_repository
        self.embedding_service = embedding_service
        self.reranker = Ranker(model_name='ms-marco-MiniLM-L-12-v2')
        
    def _validate_metadata_field(self, value: str, field_name: str) -> str | None:
        """
        Helper method to validate and optionally fuzzy-match a metadata field value.
        
        Args:
            value: The value to validate (e.g., 'pakistan', 'wmo')
            field_name: The field name in Qdrant (e.g., 'locations', 'entities')
            
        Returns:
            Validated (possibly fuzzy-matched) value, or None if not found
        """
        if not value:
            return None
            
        if field_name == 'locations' and value.lower() == 'global':
            return None
            
        validated = self.db_repository.validate_filter(value.lower(), field_name)
        if not validated:
            logger.warning(f"{field_name.title()} '{value}' not found in database")
        return validated
    
    def query_rewriter_agent(self, state: AgentState):
        logger.info("--- QUERY REWRITER ---")

        messages = state['messages']
        original_query = messages[-1].content

        # Include recent conversation history so follow-up questions resolve correctly
        history = messages[-4:-1]
        if history:
            context_lines = [f"{m.type.upper()}: {m.content[:300]}" for m in history]
            user_input = "Conversation so far:\n" + "\n".join(context_lines) + f"\n\nCurrent question: {original_query}"
        else:
            user_input = f"Current question: {original_query}"

        try:
            rewriter = self.llm.with_structured_output(MultiQuery)
            result = cast(MultiQuery, rewriter.invoke([
                SystemMessage(content=Prompts.get_query_rewriter_agent_prompt()),
                HumanMessage(content=user_input)
            ]))
            rephrasings = [q.strip() for q in result.queries if q.strip()]
        except Exception as e:
            logger.warning(f"Multi-query generation failed ({e}), falling back to original query only.")
            rephrasings = []

        # Always keep the original as first query for maximum recall
        all_queries = [original_query] + rephrasings

        # Deduplicate while preserving order
        seen: set = set()
        unique_queries = []
        for q in all_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        logger.info(f"Queries ({len(unique_queries)}): {unique_queries}")
        return {"rewritten_queries": unique_queries}
    
    def research_worker(self, state: WorkerState):
        target = state.get('target')
        search_text = state.get('query')
        logger.info(f"--- WORKER SEARCHING: {target}, {search_text} ---")
        
        logger.info(f'Search query: {search_text}')
        try:
            response_list = self.embedding_service.get_embedding_with_uuid(data=search_text)
            if not response_list:
                return {'search_results': [f"Error: Could not generate embeddings for query."]}
                
            query_data = response_list[0]
            dense_vec = query_data.embedding
            sparse_vec = query_data.sparse
            
        except Exception as e:
            logger.info(f"Worker Embedding Error: {e}")
            return {'search_results': []}
        
        strategies = self._get_strategies(target)
        raw_docs = []
        used_strategy = None
        n_results = 50
        sufficient_results = 15  # stop early when a confident strategy finds enough

        for strategy in strategies:
            logger.info(f"Trying Strategy: {strategy['name']}")
            db_result = self.db_repository.search(
                text=search_text,
                text_embedded=dense_vec,
                sparse_embedded=sparse_vec,
                filter_dict=strategy['filter'],
                n_results=n_results
            )

            if db_result.success and len(db_result.data) > 0:
                raw_docs.extend(db_result.data)
                used_strategy = strategy
                logger.info(f"- Found {len(raw_docs)} docs using {strategy['name']}")

            if len(raw_docs) >= n_results:
                logger.info("✅ Found 50 documents, stopping")
                break
            if len(raw_docs) >= sufficient_results and strategy.get('confidence', 0) >= 0.7:
                logger.info(f"✅ Sufficient confident results ({len(raw_docs)}) from '{strategy['name']}', stopping")
                break
            if strategy['name'] == 'Pure Vector Search':
                logger.info(f"✅ Tried all strategies, found {len(raw_docs)} documents")
                break
        
        
        if not raw_docs:
            return {'search_results': []}
        
        if used_strategy and used_strategy['name'] == 'Pure Vector Search' and (target.year or target.location):
            warning_doc = MyDocument(
                id='filter_warning',
                text=f"⚠️ WARNING: No exact matches found for the specified filters (Year: {target.year}, Location: {target.location}). The following results are based on semantic similarity and may be from different years or locations. Please verify the context carefully.",
                metadata={'source': 'system_warning'}
            )
            raw_docs = [warning_doc] + raw_docs
            
        return {'search_results': raw_docs}
    
    def retrieval_grader_agent_llm(self, state: AgentState):
        logger.info("--- GRADING RETRIEVED DOCS ---")
        original_question = state['messages'][-1].content
        raw_documents = state.get('search_results')
        if len(raw_documents) == 0:
            return {'filtered_results': []}
        elif len(raw_documents) <= 5:
            return {'filtered_results': raw_documents}

        unique_docs = []
        seen_ids = set()
        for doc in raw_documents:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_docs.append(doc)

        doc_texts = []
        for index, doc in enumerate(unique_docs):
            doc_texts.append(f"[{index}] --- DOCUMENT START ---\n{doc.text}\n--- DOCUMENT END ---")
        context_block = "\n\n".join(doc_texts)

        grader_llm = self.llm.with_structured_output(GradeDocumentsBatch)

        try:
            grade_result = grader_llm.invoke([
                SystemMessage(content=Prompts.get_retrieval_grader_agent_prompt()),
                HumanMessage(content=f"User Question: {original_question}\n\nCandidate Documents:\n{context_block}")
            ])
            
            result = cast(GradeDocumentsBatch, grade_result)
            relevant_indices = result.relevant_indices
            
            filtered_docs = []
            for i in relevant_indices:
                if 0 <= i < len(unique_docs):
                    filtered_docs.append(unique_docs[i])
                    logger.info(f"Keeping Doc [{i}]")
                else:
                    logger.warning(f"LLM returned invalid index {i}")

            logger.info(f"Kept {len(filtered_docs)}/{len(unique_docs)} documents.")
            return {'filtered_results': filtered_docs}

        except Exception as e:
            logger.error(f"Grading failed: {e}. Fallback: Keeping all unique docs.")
            return {'filtered_results': unique_docs}
        
    def retrieval_grader_agent(self, state: AgentState):
        logger.info("--- GRADING & RE-RANKING DOCS ---")
        original_question = state['messages'][-1].content
        raw_documents = state.get('search_results', [])
        
        if not raw_documents:
            return {'filtered_results': []}
        
        unique_docs = []
        seen_ids = set()
        for doc in raw_documents:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_docs.append(doc)
        
        passages = [
            {"id": i, "text": doc.text, "meta": doc.metadata}
            for i, doc in enumerate(unique_docs)
        ]
        
        rerank_request = RerankRequest(query=original_question, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        
        sorted_docs = [
            unique_docs[result['id']] 
            for result in sorted(reranked_results, key=lambda x: x['score'], reverse=True)
        ]
        
        top_docs = sorted_docs[:20]

        logger.info(f"Kept top {len(top_docs)} documents after reranking")
        return {'filtered_results': top_docs}
    
    def context_compressor_agent(self, state: AgentState):
        logger.info("--- COMPRESSING CONTEXT ---")
        query = state['messages'][-1].content
        docs = state.get('filtered_results', [])

        if not docs:
            return {'context_compressor_results': []}

        compressed_docs = []
        for doc in docs:
            if len(doc.text) < 200:
                compressed_docs.append(doc)
                continue
            try:
                response = self.llm.invoke([
                    SystemMessage(content=Prompts.get_context_compressor_prompt()),
                    HumanMessage(content=f"Query: {query}\n\nDocument:\n{doc.text}")
                ])
                compressed_text = response.content.strip()
                if not compressed_text or len(compressed_text) < 20:
                    compressed_docs.append(doc)
                    continue
                compressed_docs.append(MyDocument(
                    id=doc.id,
                    text=compressed_text,
                    embedding=doc.embedding,
                    sparse_embedding=doc.sparse_embedding,
                    metadata=doc.metadata,
                ))
            except Exception as e:
                logger.warning(f"Compression failed for doc {doc.id}: {e}, keeping original")
                compressed_docs.append(doc)

        logger.info(f"Compressed {len(compressed_docs)} documents")
        return {'context_compressor_results': compressed_docs}

    def synthesizer_agent(self, state: AgentState):
        logger.info("--- SYNTHESIZING FINAL ANSWER ---")
        user_query = state['messages'][-1].content
        results = state.get('context_compressor_results') or state.get('filtered_results', [])
        if not results:
            return {'messages': [AIMessage(content='I could not find specific information in the database to answer your question.')]}

        doc_texts = []
        for i, doc in enumerate(results, 1):
            doc_texts.append(f"[{i}] {doc.text}\nSource: {doc.metadata.get('source', 'unknown')}")

        context_block = "\n\n".join(doc_texts)
        human_prompt = f'User Question: "{user_query}"\n\nSources:\n{context_block}'

        is_retry = state.get("hallucination_status") == "hallucinated"
        if is_retry:
            human_prompt += "\n\nCRITICAL WARNING: Your previous answer was rejected for containing facts NOT present in the sources. Rewrite using ONLY the provided sources."

        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_synthesizer_agent_prompt()),
            HumanMessage(content=human_prompt)
        ])

        return {'messages': [response]}
    
    def hallucination_grader_agent(self, state: AgentState):
        logger.info("--- CHECKING FOR HALLUCINATIONS ---")
        current_retries = state.get('hallucination_retries', 0)
        documents = state.get('context_compressor_results', [])
        generated_answer = state['messages'][-1].content

        doc_texts = []
        for doc in documents:
            doc_texts.append(f"{doc.text}\nSource file: {doc.metadata['source']}")
        context = '\n'.join(doc_texts)


        hallucination_grader_llm = self.llm.with_structured_output(GradeHallucinations)

        grade = hallucination_grader_llm.invoke([
            SystemMessage(content=Prompts.get_hallucination_grader_agent()),
            HumanMessage(content=f'Set of facts: \n\n {context} \n\n LLM generation: {generated_answer}')
        ])
        
        score = cast(GradeHallucinations, grade)
        if score.is_relevant.lower() == 'yes':
            logger.info('--- DECISION: GENERATION IS GROUNDED ---')
            return {'hallucination_status': 'clean'}
        else:
            logger.info('--- DECISION: HALLUCINATION DETECTED ---')
            return {
                'hallucination_status': 'hallucinated',
                'hallucination_retries': current_retries + 1
            }

    def error_agent(self, state: AgentState):
        logger.info("--- ERROR ---")
        return {'messages': [AIMessage(content='I could not process your request. Please try to be more specific with your question.')]}

    @staticmethod
    def validate_and_map(state: AgentState) -> Literal[NodeName.RESEARCH_WORKER, NodeName.ERROR]:
        data = state.get('extracted_data', [])
        queries = state.get('rewritten_queries') or [state['messages'][-1].content]

        if not data:
            dummy_target = ExtractionScheme(location=None, year=None, entities=None)
            return [
                Send(NodeName.RESEARCH_WORKER, {'target': dummy_target, 'query': q})
                for q in queries
            ]

        return [
            Send(NodeName.RESEARCH_WORKER, {'target': t, 'query': q})
            for t in data
            for q in queries
        ]
    
    @staticmethod
    def route_hallucination(state: AgentState) -> Literal[NodeName.SYNTHESIZER, END]: # type: ignore
        MAX_RETRIES = 2
        status = state.get('hallucination_status')
        retries = state.get("hallucination_retries", 0)

        if status == 'hallucinated':
            if retries >= MAX_RETRIES:
                logger.warning(f"Max retries ({MAX_RETRIES}) reached. Returning error message.")
                # Replace the hallucinated message with error message
                state['messages'][-1] = AIMessage(
                    content="I couldn't generate a reliable answer from the available data. The information in the database may not fully address your question. Please try rephrasing or asking about a different aspect."
                )
                return END 
            
            return NodeName.SYNTHESIZER
        return END
    
    def _get_strategies(self, target: ExtractionScheme) -> Dict:
        strategies = []
    
        valid_year = self.db_repository.validate_filter(target.year, 'years')
        valid_location = self.db_repository.validate_filter(target.location, 'locations')
        valid_entities = None
        if target.entities:
            for entity in target.entities:
                if self.db_repository.validate_filter(entity, 'entities'):
                    valid_entities = target.entities
                    break
        
        # Strategy 1: Strict filter (highest confidence)
        strict_filter = {}
        if valid_location:
            strict_filter['locations'] = [valid_location]
        if valid_year:
            strict_filter['years'] = [valid_year]
        if valid_entities:
            strict_filter['entities'] = valid_entities
        
        if strict_filter:
            strategies.append({
                "filter": strict_filter,
                "name": f"Strict ({', '.join(strict_filter.keys())})",
                "confidence": 1.0
            })
        
        # Strategy 2: Relaxed filters (medium confidence)
        if len(strict_filter) > 1:
            if 'years' in strict_filter:
                strategies.append({
                    "filter": {'years': strict_filter['years']},
                    "name": "Year Only",
                    "confidence": 0.7
                })
            if 'locations' in strict_filter:
                strategies.append({
                    "filter": {'locations': strict_filter['locations']},
                    "name": "Location Only",
                    "confidence": 0.6
                })
        
        # Strategy 3: Semantic-only (low confidence, but add warning)
        strategies.append({
            "filter": {},
            "name": "Pure Vector Search",
            "confidence": 0.3,
            "add_warning": True if (target.year or target.location) else False
        })
        
        return strategies
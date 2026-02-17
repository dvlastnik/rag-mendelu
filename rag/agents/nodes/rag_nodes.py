from typing import cast, Literal, Dict
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import END
from flashrank import Ranker, RerankRequest
import difflib

from database.base.MyDocument import MyDocument
from rag.agents.state import AgentState, WorkerState
from rag.agents.models import MultiExtraction, GradeDocumentsBatch, GradeHallucinations, ExtractionScheme
from rag.agents.enums import NodeName
from rag.agents.prompts import Prompts
from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
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
    
        original_query = state['messages'][-1].content
        
        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_query_rewriter_agent_prompt()),
            HumanMessage(content=f'Input Query: "{original_query}"')
        ])
        
        rewritten_query = response.content.strip()
        logger.info(f"Rewritten: {rewritten_query}")

        return {"rewritten_query": rewritten_query}
    
    def query_verifier_agent(self, state: AgentState):
        logger.info('--- QUERY VERIFIER ---')
        
        original_query = state.get('original_query')
        candidate_query = state.get('candidate_query')
        
        verification_input = f"""
        [Original Query]: {original_query}
        [Rewritten Query]: {candidate_query}
        """
        
        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_query_verifier_prompt()),
            HumanMessage(content=verification_input)
        ])
        
        raw_output = response.content
        logger.info(f'Verifier Thought: {raw_output}')

        decision = 'PASS'
        reasoning = 'No reasoning provided.'
        
        if 'REASONING:' in raw_output:
            parts = raw_output.split('DECISION:')
            reasoning = parts[0].replace('REASONING:', '').strip()
        
        if 'DECISION: FAIL' in raw_output.upper():
            decision = 'FAIL'
        
        if decision == 'PASS':
            logger.info('✅ Query Verified.')
            return {
                'final_query': candidate_query, 
                "feedback": None
            }
        else:
            logger.warning(f'❌ Verification Failed. Feedback: {reasoning}')
            return {
                "feedback": reasoning,
                "final_query": None
            }
    
    def extractor_agent(self, state: AgentState):
        logger.info("--- EXTRACTING TARGETS ---")
        query = state['messages'][-1].content
        valid_data = self.db_repository.valid_metadata
        valid_years = valid_data['years']
        
        extractor = self.llm.with_structured_output(MultiExtraction)
        result = extractor.invoke([
            SystemMessage(content=Prompts.get_extractor_agent_prompt()),
            HumanMessage(content=query)
        ])
        
        raw_extraction = cast(MultiExtraction, result)
        final_targets = []
        for item in raw_extraction.targets:
            if item.year:
                if item.year in valid_years:
                    pass
                else:
                    logger.warning(f'Dropping invalid year: {item.year}')
                    item.year = None

            if item.location:
                item.location = self._validate_metadata_field(item.location, 'locations')
            
            if item.entities:
                validated_entities = []
                for entity in item.entities:
                    validated = self._validate_metadata_field(entity, 'entities')
                    if validated:
                        validated_entities.append(validated)
                item.entities = validated_entities if validated_entities else None
            
            final_targets.append(item)

        logger.info(f"Final Targets: {final_targets}")
        return {'extracted_data': final_targets}
    
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
        
        for strategy in strategies:
            logger.info(f"Trying Strategy: {strategy['name']}")
            db_result = self.db_repository.search(
                text=search_text,
                text_embedded=dense_vec,
                sparse_embedded=sparse_vec,
                filter_dict=strategy['filter'],
                n_results=10
            )

            if db_result.success and len(db_result.data) > 0:
                raw_docs = db_result.data
                used_strategy = strategy
                logger.info(f"✅ Found {len(raw_docs)} docs using {strategy['name']}")
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
    
    def retrieval_grader_agent(self, state: AgentState):
        logger.info("--- GRADING RETRIEVED DOCS ---")
        original_question = state['messages'][-1].content
        raw_documents = state.get('search_results')
        if len(raw_documents) == 0:
            return {'filtered_results': []}

        # Deduplicate by document ID
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
    
    def synthesizer_agent(self, state: AgentState):
        logger.info("--- SYNTHESIZING FINAL ANSWER ---")
        user_query = state['messages'][-1].content
        results = state.get('filtered_results')
        if len(results) == 0:
            return {'messages': [AIMessage(content='I could not find specific information in the database to answer your comparison.')]}

        doc_texts = []
        for doc in results:
            doc_texts.append(f"{doc.text}\nSource file: {doc.metadata['source']}")

        context_block = "\n\n".join(doc_texts)
        human_prompt = f"""User Question: "{user_query}"
        Context from Database:
        -----
        {context_block}
        -----
        """

        is_retry = state.get("hallucination_status") == "hallucinated"
        if is_retry:
            human_prompt += "\n\nCRITICAL WARNING: Your previous answer was rejected because it contained facts NOT present in the search results. Rewrite it using ONLY the provided text."

        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_synthesizer_agent_prompt()),
            HumanMessage(content=human_prompt)
        ])

        return {'messages': [response]}
    
    def hallucination_grader_agent(self, state: AgentState):
        logger.info("--- CHECKING FOR HALLUCINATIONS ---")
        current_retries = state.get('hallucination_retries', 0)
        documents = state.get('filtered_results', [])
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
        query = state.get('rewritten_query') or state['messages'][-1].content
        
        if not data:
            dummy_target = ExtractionScheme(location=None, year=None, entities=None)
            return [
                Send(NodeName.RESEARCH_WORKER, {'target': dummy_target, 'query': query})
            ]

        return [
            Send(NodeName.RESEARCH_WORKER, {'target': t, 'query': query}) 
            for t in data
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
        valid_year = self.db_repository.validate_filter(target.year, 'years')
        valid_location = self.db_repository.validate_filter(target.location, 'locations')
        valid_entities = None
        if target.entities:
            
            for entity in target.entities:
                if self.db_repository.validate_filter(entity, 'entities'):
                    valid_entities = target.entities
                    break

        strategies = []
        strict_filter = {}
        name_parts = []
        
        if valid_location:
            strict_filter['locations'] = [valid_location]
            name_parts.append("Location")
            
        if valid_year:
            strict_filter['years'] = [valid_year]
            name_parts.append("Year")
        
        if valid_entities:
            strict_filter['entities'] = valid_entities
            name_parts.append("Entities")
            
        if strict_filter:
            strategies.append({
                "filter": strict_filter,
                "name": f"Strict ({' + '.join(name_parts)})"
            })
        

        # backup
        if len(strict_filter) > 1:
            if 'years' in strict_filter:
                strategies.append({"filter": {'years': strict_filter['years']}, "name": "Relaxed (Year Only)"})
            if 'locations' in strict_filter:
                strategies.append({"filter": {'locations': strict_filter['locations']}, "name": "Relaxed (Location Only)"})

        strategies.append({"filter": {}, "name": "Pure Vector Search"})

        return strategies
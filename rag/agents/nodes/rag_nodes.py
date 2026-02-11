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
        
    def query_rewriter_agent(self, state: AgentState):
        logger.info("--- QUERY REWRITER ---")
    
        original_query = state.get('original_query') or state['messages'][-1].content
        feedback = state.get('feedback')
        retry_count = state.get('retry_count', 0)

        user_msg_content = f"Input Query: \"{original_query}\""
        
        if feedback:
            user_msg_content += f"\n\nPREVIOUS ATTEMPT FAILED:\nCritique: {feedback}\n\nTask: Rewrite the query again, fixing the issues mentioned above."

        print(f'user msg content: {user_msg_content}')
        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_query_rewriter_agent_prompt()),
            HumanMessage(content=user_msg_content)
        ])
        
        new_candidate = response.content.strip()
        
        logger.info(f"Draft #{retry_count + 1}: {new_candidate}")

        return {
            "original_query": original_query,
            "candidate_query": new_candidate,
            "retry_count": retry_count + 1
        }
    
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
        valid_years = [str(y) for y in valid_data['years']]
        valid_locations = valid_data['locations']

        system_prompt = Prompts.get_extractor_agent_prompt()
        
        extractor = self.llm.with_structured_output(MultiExtraction)
        result = extractor.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        
        raw_extraction = cast(MultiExtraction, result)
        final_targets = []
        for item in raw_extraction.targets:
            if item.year:
                if str(item.year) in valid_years:
                    pass
                else:
                    logger.warning(f'Dropping invalid year: {item.year}')
                    item.year = None

            item.location = item.location.lower()
            if item.location:
                if item.location == 'global':
                    item.location = None
                elif item.location in valid_locations:
                    pass
                else:
                    matches = difflib.get_close_matches(item.location, valid_locations, n=1, cutoff=0.6)
                    print(f'matches: {matches}')
                    if matches:
                        logger.info(f"Fuzzy Match: '{item.location}' mapped to '{matches[0]}'")
                        item.location = matches[0]
                    else:
                        logger.warning(f"No match found for location: '{item.location}'")
                        item.location = None
            
            final_targets.append(item)

        logger.info(f"Final Targets: {final_targets}")
        return {'extracted_data': final_targets}
    
    def research_worker_v2(self, state: WorkerState):
        target = state.get('target')
        search_text = state.get('query')
        logger.info(f"--- WORKER SEARCHING: {target} ---")

        expanded_query = search_text
        if target.topics:
            topics_str = ', '.join(target.topics)
            expanded_query = f'{expanded_query} {topics_str}'
        
        logger.info(f'Expanded query: {expanded_query}')
        try:
            response_list = self.embedding_service.get_embedding_with_uuid(data=expanded_query)
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
        for strategy in strategies:
            logger.info(f"Trying Strategy: {strategy['name']}")
            db_result = self.db_repository.search(
                text_embedded=dense_vec,
                sparse_embedded=sparse_vec,
                filter_dict=strategy["filter"],
                n_results=50 
            )

            if db_result.success and len(db_result.data) > 0:
                raw_docs = db_result.data
                logger.info(f"✅ Found {len(raw_docs)} docs using {strategy['name']}")
                break
        
        if not raw_docs:
            return {'search_results': [f"No data found for {target}."]}
        
        logger.info('--- RERANKING CANDIDATES ---')
        passages = [
            {'id': doc.id, 'text': doc.text, 'meta': doc.metadata} 
            for doc in raw_docs
        ]
        rerank_request = RerankRequest(query=expanded_query, passages=passages)
        results = self.reranker.rerank(rerank_request)

        final_docs = [MyDocument.from_dict(r) for r in results[:20]]
        return {'search_results': final_docs}

    def research_worker(self, state: WorkerState):
        target = state.get('target')
        search_text = state.get('query')
        logger.info(f"--- WORKER SEARCHING: {target} ---")

        expanded_query = search_text
        if target.topics:
            topics_str = ', '.join(target.topics)
            expanded_query = f'{expanded_query} {topics_str}'
        
        logger.info(f'Expanded query: {expanded_query}')
        try:
            response_list = self.embedding_service.get_embedding_with_uuid(data=expanded_query)
            if not response_list:
                return {'search_results': [f"Error: Could not generate embeddings for query."]}
                
            query_data = response_list[0]
            dense_vec = query_data.embedding
            sparse_vec = query_data.sparse
            
        except Exception as e:
            logger.info(f"Worker Embedding Error: {e}")
            return {'search_results': []}
        
        # search_filters = {}
        # if target.location:
        #     search_filters['locations'] = [target.location.lower()]
        # if target.year:
        #     search_filters['years'] = [target.year]

        # db_result = self.db_repository.search(
        #     text_embedded=dense_vec,
        #     sparse_embedded=sparse_vec,
        #     filter_dict=search_filters,
        #     n_results=10
        # )

        # found_docs = []
        # if db_result.success and len(db_result.data) > 0:
        #     found_docs = [doc for doc in db_result.data]
        # else:
        #     logger.warning(f"Search failed for strict filters: {search_filters}")
            
        #     broad_result = self.db_repository.search(
        #         text_embedded=dense_vec,
        #         sparse_embedded=sparse_vec,
        #         n_results=5
        #     )
            
        #     if broad_result.data:
        #         warning_header = f"""
        #         [SYSTEM WARNING]: Exact matches for {search_filters} were NOT found. 
        #         The following information is semantically relevant but might be from different years or locations.
        #         Verify dates carefully.
        #         """
        #         found_docs = [MyDocument(id='warning', text=warning_header)] + [doc for doc in broad_result.data]
        #         print(found_docs)

        # if not found_docs:
        #     return {'search_results': [f"No data found."]}
        strategies = self._get_strategies(target)
        raw_docs = []
        for strategy in strategies:
            logger.info(f"Trying Strategy: {strategy['name']}")
            db_result = self.db_repository.search(
                text_embedded=dense_vec,
                sparse_embedded=sparse_vec,
                filter_dict=strategy['filter'],
                n_results=10
            )

            if db_result.success and len(db_result.data) > 0:
                raw_docs = db_result.data
                logger.info(f"✅ Found {len(raw_docs)} docs using {strategy['name']}")
                break
        
        if not raw_docs:
            return {'search_results': [f"No data found for {target}."]}
            
        return {'search_results': raw_docs}
    
    def retrieval_grader_agent(self, state: AgentState):
        logger.info("--- GRADING RETRIEVED DOCS ---")
        question = state.get('query')
        raw_documents = state.get('search_results')
        if len(raw_documents) == 0:
            return {'filtered_results': []}

        unique_docs = []
        seen_hashes = set()
        for doc in raw_documents:
            doc_hash = hash(doc.text.strip()) 
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)

        doc_texts = []
        for index, doc in enumerate(unique_docs):
            doc_texts.append(f"[{index}] --- DOCUMENT START ---\n{doc.text}\n--- DOCUMENT END ---")
        context_block = "\n\n".join(doc_texts)

        grader_llm = self.llm.with_structured_output(GradeDocumentsBatch)

        try:
            grade_result = grader_llm.invoke([
                SystemMessage(content=Prompts.get_retrieval_grader_agent_prompt()),
                HumanMessage(content=f"User Question: {question}\n\nCandidate Documents:\n{context_block}")
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
        query = state.get('final_query') or state.get(f'original_query')
        
        if not data:
            return NodeName.ERROR

        return [
            Send(NodeName.RESEARCH_WORKER, {'target': t, 'query': query}) 
            for t in data
        ]
    
    @staticmethod
    def route_hallucination(state: AgentState) -> Literal[NodeName.SYNTHESIZER, END]: # type: ignore
        MAX_RETRIES = 3
        status = state.get('hallucination_status')
        retries = state.get("hallucination_retries", 0)

        if status == 'hallucinated':
            if retries >= MAX_RETRIES:
                logger.warning(f"Max retries ({MAX_RETRIES}) reached. Stopping infinite loop.")
                return END 
            
            return NodeName.SYNTHESIZER
        return END
    
    @staticmethod
    def should_retry(state: AgentState) -> str:
        feedback = state.get('feedback')
        retry_count = state.get('retry_count', 0)
        MAX_RETRIES = 3

        if feedback is None:
            return NodeName.EXTRACTOR
        
        if retry_count < MAX_RETRIES:
            return NodeName.QUERY_REWRITER
        
        return NodeName.EXTRACTOR
    
    def _get_strategies(self, target: ExtractionScheme) -> Dict:
        valid_year = self.db_repository.validate_filter(target.year, 'years')
        valid_location = self.db_repository.validate_filter(target.location, 'locations')

        strategies = []
        strict_filter = {}
        name_parts = []
        if valid_location:
            strict_filter['locations'] = [valid_location]
            name_parts.append("Location")
            
        if valid_year:
            strict_filter['years'] = [valid_year]
            name_parts.append("Year")
            
        if strict_filter:
            strategies.append({
                "filter": strict_filter,
                "name": f"Smart Strict ({' + '.join(name_parts)})"
            })
        

        # backup
        if len(strict_filter) > 1:
            if 'years' in strict_filter:
                strategies.append({"filter": {'years': strict_filter['years']}, "name": "Relaxed (Year Only)"})
            if 'locations' in strict_filter:
                strategies.append({"filter": {'locations': strict_filter['locations']}, "name": "Relaxed (Location Only)"})

        strategies.append({"filter": {}, "name": "Pure Vector Search"})

        return strategies
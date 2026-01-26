from typing import cast, Literal
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import END

from rag.agents.state import AgentState, WorkerState
from rag.agents.models import MultiExtraction, GradeDocumentsBatch, GradeHallucinations
from rag.agents.enums import NodeName
from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)

class RagNodes:
    def __init__(self, llm: BaseChatModel, db_repository: BaseDbRepository, embedding_service: TextEmbeddingService):
        self.llm = llm
        self.db_repository = db_repository
        self.embedding_service = embedding_service
        
    def query_rewriter_agent(self, state: AgentState):
        logger.info("--- QUERY REWRITING ---")
        original_query = state['messages'][-1].content
        system_prompt = """You are a Search Query Optimizer. 
        Your task is to refine the user's query to be better suited for a vector database search.

        CRITICAL RULES:
        1. **PRESERVE INTENT**: NEVER change the meaning of the question (e.g., do NOT change "warmest" to "coolest", "highest" to "lowest").
        2. **PRESERVE ENTITIES**: Keep specific years (2023), organizations (WMO), and proper nouns exactly as they are.
        3. **CLARIFY**: Remove conversational filler ("Can you tell me...", "I was wondering").
        4. **Target**: Focus on finding factual statements in a scientific report.

        If the original query is already specific and clear, output it exactly as is.
        """

        history_text = "\n".join([f"{m.type}: {m.content}" for m in state['messages'][:-1]])
        user_input = f"History:\n{history_text}\nInput: \"{original_query}\"\nOutput:"

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])

        logger.info(f"Original: {original_query}")
        logger.info(f"Rewritten: {response.content}")
        
        return {"rewritten_query": response.content}
    
    def extractor_agent(self, state: AgentState):
        logger.info("--- EXTRACTING TARGETS ---")
        query = state['rewritten_query']
        if not query:
            query = state['messages'][-1].content
        
        logger.info(f"extracting in: {query}")
        extractor = self.llm.with_structured_output(MultiExtraction)
        system_prompt = """You are an expert metadata extractor for climate research. 
        Extract mentioned city, country, year, and crucial topics.

        CRITICAL GEOGRAPHICAL RULES:
        1. If a natural region or landmark is mentioned, infer the COUNTRY.
           Examples:
           - Input: "Swiss Alps" -> Output Country: "Switzerland"
           - Input: "Greenland Ice Sheet" -> Output Country: "Greenland"
           - Input: "Amazon Rainforest" -> Output Country: "Brazil"
           - Input: "Horn of Africa" -> Output Country: "East Africa" (or specific countries if named)
           - Input: "Yangtze River" -> Output Country: "China"
        
        2. If the user mentions a demonym (e.g., "French droughts"), extract the Country ("France").
        """
        result = extractor.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        
        extracted_data = cast(MultiExtraction, result).targets
        valid_data = [t for t in extracted_data if t.country or t.year or t.topics or t.city]
        if not valid_data:
             return {'extracted_data': []}
        
        return {"extracted_data": extracted_data}

    def research_worker(self, state: WorkerState):
        target = state.get('target')
        search_text = state.get('query')
        logger.info(f"--- WORKER SEARCHING: {target} ---")

        expanded_query = search_text
        if target.topics:
            topics_str = ', '.join(target.topics)
            expanded_query = f'{expanded_query}\n\nImportant Keywords: {topics_str}'
        
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
        
        search_filters = {}
        if target.country:
            search_filters['countries'] = [target.country.lower()]
        if target.year:
            search_filters['years'] = [target.year]
        if target.city:
            search_filters['cities'] = [target.city.lower()]

        db_result = self.db_repository.search(
            text_embedded=dense_vec,
            sparse_embedded=sparse_vec,
            filter_dict=search_filters,
            n_results=5
        )
        if not db_result.success:
            return {'search_results': [f"Database Error: {db_result.message}"]}
        
        if len(db_result.data) == 0:
            db_result = self.db_repository.search(
                text_embedded=dense_vec,
                sparse_embedded=sparse_vec,
                n_results=5
            )
        
        found_docs = [doc.text for doc in db_result.data]
        if not found_docs:
            return {'search_results': [f"No detailed data found for {target}."]}
            
        return {'search_results': found_docs}
    
    def retrieval_grader_agent(self, state: AgentState):
        logger.info("--- GRADING RETRIEVED DOCS ---")
        question = state.get('rewritten_query')
        raw_documents = state.get('search_results')
        if len(raw_documents) == 0:
            return {'filtered_results': []}

        unique_docs = []
        seen_hashes = set()
        for doc in raw_documents:
            doc_hash = hash(doc.strip()) 
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)

        doc_texts = []
        for index, doc in enumerate(unique_docs):
            doc_texts.append(f"[{index}] --- DOCUMENT START ---\n{doc}\n--- DOCUMENT END ---")
        context_block = "\n\n".join(doc_texts)

        grader_llm = self.llm.with_structured_output(GradeDocumentsBatch)
        system_prompt = """You are a precise grader. You will be given a list of retrieved documents, each labeled with an ID (e.g., [0], [1]).
        
        Your task is to identify which documents are RELEVANT to the user's question.
        
        CRITICAL RULES:
        1. Return ONLY the list of integer IDs (e.g., [0, 2]) for documents that contain relevant facts.
        2. If a document helps answer *any part* of the comparison (e.g. only about Greenland), it is RELEVANT.
        3. Be lenient. If in doubt, include it.
        """

        try:
            grade_result = grader_llm.invoke([
                SystemMessage(content=system_prompt),
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

        context_block = "\n\n".join(results)
        prompt = f"""You are senior synthetizer.
        User Question: "{user_query}"
        
        Context from Database:
        -----
        {context_block}
        -----
        
        Instructions:
        1. Answer the question using ONLY the context provided above.
        2. If the user asked for a comparison, explicitly contrast the data points (e.g., "While Item A has X, Item B has Y").
        3. START DIRECTLY. Do not say "Based on the search results" or "Here is the answer". Just state the facts.
        4. If the context does not contain the answer, state that you do not know. Do not guess.
        """

        is_retry = state.get("hallucination_status") == "hallucinated"
        if is_retry:
            prompt += "\n\nCRITICAL WARNING: Your previous answer was rejected because it contained facts NOT present in the search results. Rewrite it using ONLY the provided text."

        response = self.llm.invoke([
            SystemMessage(content="Synthesize the search results into a cohesive answer."),
            HumanMessage(content=prompt)
        ])

        return {'messages': [response]}
    
    def hallucination_grader_agent(self, state: AgentState):
        logger.info("--- CHECKING FOR HALLUCINATIONS ---")
        current_retries = state.get('hallucination_retries', 0)
        documents = state.get('filtered_results', [])
        generated_answer = state['messages'][-1].content
        context = '\n'.join(documents)

        hallucination_grader_llm = self.llm.with_structured_output(GradeHallucinations)
        system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        
        CRITICAL GRADING RULES:
        1. If the LLM response says "I cannot answer", "I don't know", or "No information found", and the provided facts are empty or irrelevant, this is GROUNDED ('yes').
        2. If the LLM response contains specific numbers, dates, or names that are NOT present in the 'Set of facts', this is a HALLUCINATION ('no').
        3. The answer must be derived ONLY from the provided facts. Do not allow outside knowledge.
        
        Give 'yes' or 'no'. 'yes' means grounded/faithful. 'no' means hallucinated."""
        grade = hallucination_grader_llm.invoke([
            SystemMessage(content=system_prompt),
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
        return {'messages': [AIMessage(content='I could not process your request. Please mention at least a specific YEAR, CITY or COUNTRY')]}


    @staticmethod
    def validate_and_map(state: AgentState) -> Literal[NodeName.RESEARCH_WORKER, NodeName.ERROR]:
        data = state.get('extracted_data', [])
        query = state.get('rewritten_query')
        
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
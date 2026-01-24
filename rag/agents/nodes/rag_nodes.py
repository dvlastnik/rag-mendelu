from typing import cast, Literal
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import END
from logging import Logger

from rag.agents.state import AgentState, WorkerState
from rag.agents.models import MultiExtraction, GradeDocuments, GradeHallucinations
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
        system_prompt = """You are a Query Reformulator. 
                        Your task is to rewrite the user's latest message into a standalone, clear search query based on the conversation history.

                        RULES:
                        1. DO NOT answer the question.
                        2. DO NOT add facts or hallucinations.
                        3. ONLY return the rewritten query text.

                        EXAMPLES:
                        History: [User: "Tell me about droughts in France."]
                        Input: "And in Spain?"
                        Output: "Tell me about droughts in Spain."

                        History: [User: "What was the GDP of Germany in 2020?"]
                        Input: "Compare it with Italy."
                        Output: "Compare the GDP of Germany and Italy in 2020."

                        History: []
                        Input: "Floods in Prague 2002"
                        Output: "Floods in Prague 2002"
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
        documents = state.get('search_results')
        if len(documents) == 0:
            return {'filtered_results': []}

        filtered_docs = []

        grader_llm = self.llm.with_structured_output(GradeDocuments)
        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.

        CRITICAL INSTRUCTION:
        The user might ask a "Comparison" question (e.g., "Compare X and Y").
        - If a document contains information about **X**, it is **RELEVANT**.
        - If a document contains information about **Y**, it is **RELEVANT**.
        - The document does **NOT** need to contain both.
        - The document does **NOT** need to perform the comparison itself.

        If the document provides *any* facts that contribute to answering *part* of the question, grade it as 'yes'."""

        for doc in documents:
            grade = grader_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f'Retrieved document: \n\n {doc} \n\n User Question: {question}')
            ])
            
            score = cast(GradeDocuments, grade)
            logger.info(doc)
            if score.is_relevant.lower() == 'yes':
                logger.info(f"Grade: DOCUMENT RELEVANT")
                filtered_docs.append(doc)
            else:
                logger.info(f"Grade: DOCUMENT IRRELEVANT (Filtered Out)")
            logger.info('------------------------------------------------')
        
        return {'filtered_results': filtered_docs}
    
    def synthesizer_agent(self, state: AgentState):
        logger.info("--- SYNTHESIZING FINAL ANSWER ---")
        user_query = state['messages'][-1].content
        results = state.get('filtered_results')
        if len(results) == 0:
            return {'messages': [AIMessage(content='I could not find specific information in the database to answer your comparison.')]}

        context_block = "\n\n".join(results)
        prompt = f"""You are a climate research analyst. 
        The user asked: "{user_query}"
        
        Below are the search results gathered from the database:
        -----
        {context_block}
        -----
        
        Synthesize these findings into a clear, direct answer. 
        If the user asked for a comparison, explicitly contrast the data points (e.g., "While Italy had X, Czechia had Y").
        Do not make up facts not present in the search results.
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
        documents = state.get('search_results')
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
            return {'hallucination_status': 'hallucinated'}

    def error_agent(self, state: AgentState):
        logger.info("--- ERROR ---")
        return {'messages': [AIMessage(content='I could not process your request. Please mention at least a specific YEAR, CITY or COUNTRY')]}


    @staticmethod
    def validate_and_map(state: AgentState) -> Literal[NodeName.RESEARCH_WORKER, NodeName.ERROR]:
        data = state.get('extracted_data', [])
        query = state.get('rewritten_query')
            
        return [
            Send(NodeName.RESEARCH_WORKER, {'target': t, 'query': query}) 
            for t in data
        ]
    
    @staticmethod
    def route_hallucination(state: AgentState):
        status = state.get('hallucination_status')
        if status == 'hallucinated':
            return NodeName.SYNTHESIZER
        return END
import re
from difflib import SequenceMatcher
from typing import cast, Literal, List
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import END

from flashrank import Ranker, RerankRequest

from rag.agents.state import AgentState, WorkerState
from rag.agents.models import MultiQuery, GradeHallucinations, ExtractionScheme, CompletenessCheck
from rag.agents.enums import NodeName, Intent
from rag.agents.prompts import Prompts
from database.base.BaseDbRepository import BaseDbRepository
from text_embedding import TextEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)

class RagNodes:
    def __init__(
        self,
        llm: BaseChatModel,
        db_repository: BaseDbRepository,
        embedding_service: TextEmbeddingService,
        context_window: int = 8192,
    ):
        self.llm = llm
        self.db_repository = db_repository
        self.embedding_service = embedding_service
        self.reranker = Ranker(model_name='ms-marco-MiniLM-L-12-v2')

        params = self._compute_model_params(context_window)
        self.distiller_top_n = params['top_n']
        self.distiller_chars_per_doc = params['chars_per_doc']
        self.listing_chars_per_doc = params['listing_chars']
        self.max_completeness_iterations = params['max_iterations']

    @staticmethod
    def _compute_model_params(context_window: int) -> dict:
        if context_window <= 4096:
            return {'top_n': 5, 'chars_per_doc': 1200, 'listing_chars': 200, 'max_iterations': 2}
        elif context_window <= 8192:
            return {'top_n': 10, 'chars_per_doc': 2000, 'listing_chars': 500, 'max_iterations': 3}
        elif context_window <= 32768:
            return {'top_n': 15, 'chars_per_doc': 3000, 'listing_chars': 800, 'max_iterations': 3}
        else:
            return {'top_n': 20, 'chars_per_doc': 5000, 'listing_chars': 1500, 'max_iterations': 4}

    def _validate_metadata_field(self, value: str, field_name: str) -> str | None:
        if not value:
            return None
        if field_name == 'locations' and value.lower() == 'global':
            return None
        validated = self.db_repository.validate_filter(value.lower(), field_name)
        if not validated:
            logger.warning(f"{field_name.title()} '{value}' not found in database")
        return validated

    def query_decomposer_agent(self, state: AgentState):
        logger.info("--- QUERY DECOMPOSER ---")

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
                SystemMessage(content=Prompts.get_query_decomposer_agent_prompt()),
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
        # query_type = state.get('query_type', 'factual')
        logger.info(f"--- WORKER SEARCHING: {search_text} ---")

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

        db_result = self.db_repository.search(
            text=search_text,
            text_embedded=dense_vec,
            sparse_embedded=sparse_vec,
            filter_dict=None,
            n_results=50
        )

        if db_result.success:
            seen = set(state.get('seen_doc_ids', []))
            fresh_docs = [doc for doc in db_result.data if doc.id not in seen]

            if not fresh_docs:
                logger.info("No new docs after filtering already seen docs")
                return {'search_results': []}
            
            return {'search_results': fresh_docs}

        return {'search_results': []}

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

        top_docs = sorted_docs[:self.distiller_top_n]

        logger.info(f"Kept top {len(top_docs)} documents after reranking (cutoff={self.distiller_top_n})")
        return {'filtered_results': top_docs}

    def fact_extractor_agent(self, state: AgentState):
        logger.info("--- FACT EXTRACTOR ---")
        query = state['messages'][-1].content
        docs = state.get('filtered_results', [])

        if not docs:
            return {'distilled_facts': []}

        doc_blocks = []
        sources_seen = []
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in sources_seen:
                sources_seen.append(source)
            truncated = doc.text[:self.distiller_chars_per_doc]
            doc_blocks.append(f"[{source}]\n{truncated}")

        combined_docs = "\n---\n".join(doc_blocks)
        human_prompt = f"Question: {query}\n\nDocuments:\n{combined_docs}"

        try:
            response = self.llm.invoke([
                SystemMessage(content=Prompts.get_fact_extractor_prompt()),
                HumanMessage(content=human_prompt)
            ])
            extracted = response.content.strip()

            cleaned = re.sub(r'NO RELEVANT FACTS FOUND\.?', '', extracted, flags=re.IGNORECASE).strip()
            if not cleaned or len(cleaned) < 15:
                logger.info("Fact extractor returned empty list — skipping (no new relevant facts).")
                return {'distilled_facts': []}
            extracted = cleaned
        except Exception as e:
            logger.warning(f"Fact extractor failed ({e}), skipping.")
            return {'distilled_facts': []}

        sources_tag = f"[Sources: {', '.join(sources_seen)}]"
        attributed = f"{sources_tag}\n{extracted}"

        logger.info(f"Distilled facts block ({len(attributed)} chars)")
        return {'distilled_facts': [attributed]}

    def completeness_checker_agent(self, state: AgentState):
        logger.info("--- COMPLETENESS CHECKER ---")
        iterations = state.get('retrieval_iterations', 0)
        prev_query = state.get('completeness_follow_up_query', '')

        messages = state['messages']
        user_query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), '')
        generated_answer = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), '')

        try:
            facts_context = "\n\n".join(state.get('distilled_facts', []))
            human_content = (
                f"Question: {user_query}\n\n"
                f"Available facts from database:\n{facts_context}\n\n"
                f"Answer: {generated_answer}"
            )
            checker_llm = self.llm.with_structured_output(CompletenessCheck)
            result = cast(CompletenessCheck, checker_llm.invoke([
                SystemMessage(content=Prompts.get_completeness_checker_prompt()),
                HumanMessage(content=human_content),
            ]))

            follow_up = result.follow_up_query.strip() if not result.is_complete else ''

            # Semantic duplicate guard — catches near-identical queries that differ in wording
            if follow_up and prev_query:
                similarity = SequenceMatcher(None, follow_up.lower(), prev_query.lower()).ratio()
                if similarity > 0.6:
                    logger.info(f"Completeness checker repeated similar query (similarity={similarity:.2f}) — treating as complete.")
                    follow_up = ''

            logger.info(f"Completeness: complete={result.is_complete} | follow_up='{follow_up}'")
            return {
                'retrieval_iterations': iterations + 1,
                'completeness_follow_up_query': follow_up,
            }
        except Exception as e:
            logger.warning(f"Completeness check failed ({e}), treating as complete.")
            return {'retrieval_iterations': iterations + 1, 'completeness_follow_up_query': ''}

    def route_completeness_check(self, state: AgentState):
        iterations = state.get('retrieval_iterations', 0)
        follow_up = state.get('completeness_follow_up_query', '')
        # query_type = state.get('query_type', 'factual')

        if not follow_up or iterations >= self.max_completeness_iterations:
            logger.info(f"Completeness → HALLUCINATION_GRADER (iterations={iterations}, follow_up='{follow_up}')")
            return NodeName.HALLUCINATION_GRADER_AGENT

        logger.info(f"Completeness → RESEARCH_WORKER with query: '{follow_up}' (iteration {iterations})")
        dummy_target = ExtractionScheme()
        seen_ids = [doc.id for doc in state.get('search_results', []) if hasattr(doc, 'id')]
        return [Send(NodeName.RESEARCH_WORKER, {'target': dummy_target, 'query': follow_up, 'seen_doc_ids': seen_ids})]

    def synthesizer_agent(self, state: AgentState):
        logger.info("--- SYNTHESIZING FINAL ANSWER ---")
        user_query = state['messages'][-1].content
        distilled_facts = state.get('distilled_facts', [])

        if not distilled_facts:
            return {'messages': [AIMessage(content='I could not find specific information in the database to answer your question.')]}
        facts_block = "\n\n".join(distilled_facts)
        human_prompt = f'User Question: "{user_query}"\n\nFacts:\n{facts_block}'

        is_retry = state.get("hallucination_status") == "hallucinated"
        if is_retry:
            human_prompt += "\n\nCRITICAL WARNING: Your previous answer was rejected for containing facts NOT present in the sources. Rewrite using ONLY the provided facts."

        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_synthesizer_agent_prompt()),
            HumanMessage(content=human_prompt)
        ])

        return {'messages': [response]}

    def hallucination_grader_agent(self, state: AgentState):
        logger.info("--- CHECKING FOR HALLUCINATIONS ---")
        current_retries = state.get('hallucination_retries', 0)
        distilled_facts = state.get('distilled_facts', [])
        filtered_results = state.get('filtered_results', [])
        generated_answer = state['messages'][-1].content

        if distilled_facts:
            context = "\n\n".join(distilled_facts)
        else:
            # Listing fallback: use raw filtered results
            context = "\n".join(
                doc.text[:self.listing_chars_per_doc]
                for doc in filtered_results[:10]
            )

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

    def error_agent(self, state: AgentState):  # noqa: ARG002
        logger.info("--- ERROR ---")
        return {'messages': [AIMessage(content='I could not process your request. Please try to be more specific with your question.')]}

    @staticmethod
    def validate_and_map(state: AgentState) -> Literal[NodeName.RESEARCH_WORKER, NodeName.ERROR]:
        data = state.get('extracted_data', [])
        queries = state.get('rewritten_queries') or [state['messages'][-1].content]
        # query_type = state.get('query_type', 'factual')

        if not data:
            dummy_target = ExtractionScheme()
            return [
                Send(NodeName.RESEARCH_WORKER, {'target': dummy_target, 'query': q}) #, 'query_type': query_type})
                for q in queries
            ]

        return [
            Send(NodeName.RESEARCH_WORKER, {'target': t, 'query': q}) #, 'query_type': query_type})
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
                state['messages'][-1] = AIMessage(
                    content="I couldn't generate a reliable answer from the available data. The information in the database may not fully address your question. Please try rephrasing or asking about a different aspect."
                )
                return END

            return NodeName.SYNTHESIZER
        return END


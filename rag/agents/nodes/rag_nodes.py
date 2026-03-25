import re
from difflib import SequenceMatcher
from typing import cast, Literal, List
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import END
from flashrank import Ranker, RerankRequest

from rag.agents.state import AgentState, WorkerState
from rag.agents.models import MultiQuery, GradeHallucinations, CompletenessCheck, QueryPlan, QueryStrategy, SQLQueryPlan
from rag.agents.enums import NodeName, Intent
from rag.agents.prompts import Prompts
from database.base.base_db_repository import BaseDbRepository
from database.duck_db_repository import DuckDbRepository
from text_embedding import TextEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ModelParams():
    def __init__(self, top_n: int, chars_per_doc: int, listing_chars: int, max_iterations: int):
        self.top_n = top_n
        self.chars_per_doc = chars_per_doc
        self.listing_chars = listing_chars
        self.max_iterations = max_iterations

    @staticmethod
    def create_from_context_window(context_window: int) -> 'ModelParams':
        if context_window <= 4096:
            return ModelParams(5, 1200, 200, 2)
        elif context_window <= 8192:
            return ModelParams(10, 2000, 500, 3)
        elif context_window <= 32768:
            return ModelParams(15, 3000, 800, 3)
        else:
            return ModelParams(20, 5000, 1500, 4)

class RagNodes:
    def __init__(
        self,
        llm: BaseChatModel,
        db_repository: BaseDbRepository,
        embedding_service: TextEmbeddingService,
        duck_db_repo: DuckDbRepository,
        context_window: int = 8192,
        available_sources: List[str] | None = None,
    ):
        self.llm = llm
        self.db_repository = db_repository
        self.embedding_service = embedding_service
        self.reranker = Ranker(model_name='ms-marco-MiniLM-L-12-v2')
        self.duck_db_repo = duck_db_repo
        self.available_sources = available_sources or []
        self._compact_catalog = duck_db_repo.get_compact_catalog() if duck_db_repo else ""

        params = ModelParams.create_from_context_window(context_window)
        self.distiller_top_n = params.top_n
        self.distiller_chars_per_doc = params.chars_per_doc
        self.listing_chars_per_doc = params.listing_chars
        self.max_completeness_iterations = params.max_iterations

    def query_planner_agent(self, state: AgentState):
        logger.info("--- QUERY PLANNER ---")
        messages = state['messages']
        original_query = messages[-1].content
        detected_source = state.get('detected_source')
        intent = state.get('intent', Intent.RAG)

        history = messages[-4:-1]
        if history:
            context_lines = [f"{m.type.upper()}: {m.content[:300]}" for m in history]
            user_input = "Conversation so far:\n" + "\n".join(context_lines) + f"\n\nCurrent question: {original_query}"
        else:
            user_input = f"Current question: {original_query}"

        if not self.duck_db_repo or not self._compact_catalog:
            return self._decompose_vector_queries(user_input, original_query, detected_source, intent)

        try:
            planner = self.llm.with_structured_output(QueryPlan)
            plan = cast(QueryPlan, planner.invoke([
                SystemMessage(content=Prompts.get_query_planner_prompt(
                    self._compact_catalog,
                    self.available_sources,
                )),
                HumanMessage(content=user_input),
            ]))

            if intent in (Intent.RAG_SUMMARIZATION, Intent.RAG_EXHAUSTIVE) and detected_source and plan.strategy == QueryStrategy.VECTOR:
                plan = QueryPlan(strategy=QueryStrategy.SCROLL)

            if plan.strategy in (QueryStrategy.SQL, QueryStrategy.HYBRID) and plan.sql_sources:
                known_tables = self.duck_db_repo.list_tables()
                valid_sources = [t for t in plan.sql_sources if t in known_tables]
                unknown = [t for t in plan.sql_sources if t not in known_tables]
                if unknown:
                    logger.warning(f"QueryPlanner chose unknown table(s) {unknown} (known: {known_tables}), removing them")
                if not valid_sources:
                    logger.warning("No valid sql_sources remain, falling back to vector")
                    plan = QueryPlan(strategy=QueryStrategy.VECTOR, vector_queries=plan.vector_queries)
                elif valid_sources != plan.sql_sources:
                    plan = QueryPlan(strategy=plan.strategy, sql_sources=valid_sources, sql_hint=plan.sql_hint, vector_queries=plan.vector_queries)

            logger.info(f"QueryPlan: strategy={plan.strategy}, sql_sources={plan.sql_sources}, queries={plan.vector_queries}")
            return {
                "query_plan": plan,
                "rewritten_queries": plan.vector_queries or [original_query],
            }

        except Exception as e:
            logger.warning(f"QueryPlanner LLM call failed ({e}), falling back to vector decomposition")
            return self._decompose_vector_queries(user_input, original_query, detected_source, intent)

    def _decompose_vector_queries(
        self,
        user_input: str,
        original_query: str,
        detected_source: str | None,
        intent: Intent,
    ) -> dict:
        """Fallback: generate vector search queries (old query_decomposer_agent logic)."""
        try:
            rewriter = self.llm.with_structured_output(MultiQuery)
            result = cast(MultiQuery, rewriter.invoke([
                SystemMessage(content=Prompts.get_query_decomposer_agent_prompt()),
                HumanMessage(content=user_input),
            ]))
            rephrasings = [q.strip() for q in result.queries if q.strip()][:4]
        except Exception as e:
            logger.warning(f"Multi-query generation failed ({e}), using original query only")
            rephrasings = []

        # delete duplicates
        seen: set = set()
        unique_queries: List[str] = []
        for q in [original_query] + rephrasings:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        strategy = (
            QueryStrategy.SCROLL
            if intent in (Intent.RAG_SUMMARIZATION, Intent.RAG_EXHAUSTIVE) and detected_source
            else QueryStrategy.VECTOR
        )
        plan = QueryPlan(strategy=strategy, vector_queries=unique_queries)
        logger.info(f"Fallback QueryPlan: strategy={plan.strategy}, queries={unique_queries}")
        return {"query_plan": plan, "rewritten_queries": unique_queries}

    def research_worker(self, state: WorkerState):
        search_text = state.get('query')
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

    def scroll_retriever(self, state: AgentState):
        """Fetch ALL chunks from a specific source, rerank by relevance, return top-N to fact extractor."""
        source = state.get('detected_source')
        if not source:
            logger.warning("scroll_retriever called without detected_source — returning empty")
            return {'filtered_results': [], 'search_results': []}

        docs = self.db_repository.scroll_all_by_source(source, limit=500)
        logger.info(f"scroll_retriever fetched {len(docs)} docs from source '{source}'")

        if not docs:
            return {'filtered_results': [], 'search_results': []}

        query = state['messages'][-1].content
        passages = [{"id": i, "text": doc.text, "meta": doc.metadata} for i, doc in enumerate(docs)]
        reranked = self.reranker.rerank(RerankRequest(query=query, passages=passages))
        sorted_docs = [docs[r['id']] for r in sorted(reranked, key=lambda x: x['score'], reverse=True)]

        return {'filtered_results': sorted_docs, 'search_results': docs}

    def analytical_query_agent(self, state: AgentState):
        """Execute a SQL query on DuckDB and store the result as a distilled fact."""
        logger.info("--- ANALYTICAL QUERY (DuckDB) ---")
        plan = state.get('query_plan')
        original_question = state['messages'][-1].content

        if not plan or not plan.sql_sources or not self.duck_db_repo:
            logger.error("analytical_query_agent called without sql_sources or duck_db_repo — returning empty")
            return {"distilled_facts": [], "sql_result": None}

        sources_label = ", ".join(plan.sql_sources)
        combined_schema = "\n\n---\n\n".join(
            self.duck_db_repo.get_schema(t) for t in plan.sql_sources
        )
        hint = plan.sql_hint or original_question

        try:
            generator = self.llm.with_structured_output(SQLQueryPlan)
            sql_plan = cast(SQLQueryPlan, generator.invoke([
                SystemMessage(content=Prompts.get_sql_generator_prompt(combined_schema)),
                HumanMessage(content=f"Question: {original_question}\nHint: {hint}"),
            ]))
            logger.info(f"Generated SQL: {sql_plan.sql}")

            df = self.duck_db_repo.run_select(sql_plan.sql)
            if df.empty:
                logger.info("SQL returned no rows")
                return {
                    "distilled_facts": [f"[{sources_label}] SQL query returned no results."],
                    "sql_result": "",
                }

            result_text = df.to_string(index=False)
            fact = f"[{sources_label}] {sql_plan.explanation}\n{result_text}"
            logger.info(f"SQL result ({len(df)} rows):\n{result_text[:400]}")
            return {"distilled_facts": [fact], "sql_result": result_text}

        except Exception as e:
            logger.error(f"analytical_query_agent failed: {e}")
            return {
                "distilled_facts": [f"[{sources_label}] SQL query could not be executed: {e}"],
                "sql_result": None,
            }

    def retrieval_grader_agent(self, state: AgentState):
        logger.info("--- GRADING & RE-RANKING DOCS ---")
        original_question = state['messages'][-1].content
        raw_documents = state.get('search_results', [])
        intent = state.get('intent', Intent.RAG)

        if not raw_documents:
            return {'filtered_results': []}

        existing_filtered = state.get('filtered_results', [])
        existing_ids = {doc.id for doc in existing_filtered if hasattr(doc, 'id')}

        unique_docs = []
        seen_ids = set()
        for doc in raw_documents:
            if doc.id not in seen_ids and doc.id not in existing_ids:
                seen_ids.add(doc.id)
                unique_docs.append(doc)

        if not unique_docs:
            logger.info("No new unique docs to rerank after excluding already-filtered docs")
            return {'filtered_results': []}

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

        if intent == Intent.RAG_EXHAUSTIVE:
            effective_top_n = self.distiller_top_n * 3
        else:
            effective_top_n = self.distiller_top_n

        top_docs = sorted_docs[:effective_top_n]

        logger.info(f"Kept top {len(top_docs)} NEW documents after reranking (cutoff={effective_top_n}, intent={intent})")
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

        batch_size = self.distiller_top_n
        batches = [doc_blocks[i:i + batch_size] for i in range(0, len(doc_blocks), batch_size)]
        logger.info(f"Fact extractor: {len(docs)} docs → {len(batches)} batch(es) of {batch_size}")

        all_extracted = []
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch_idx: {batch_idx}...")
            combined_docs = "\n---\n".join(batch)
            human_prompt = f"Question: {query}\n\nDocuments:\n{combined_docs}"
            try:
                response = self.llm.invoke([
                    SystemMessage(content=Prompts.get_fact_extractor_prompt()),
                    HumanMessage(content=human_prompt)
                ])
                extracted = response.content.strip()
                cleaned = re.sub(r'NO RELEVANT FACTS FOUND\.?', '', extracted, flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) >= 15:
                    all_extracted.append(cleaned)
                    logger.info(f"- Facts extracted")
                else:
                    logger.info(f"- No facts found")
            except Exception as e:
                logger.warning(f"Fact extractor failed on batch {batch_idx} ({e}), skipping.")

        if not all_extracted:
            logger.info("Fact extractor returned empty list — skipping (no new relevant facts).")
            return {'distilled_facts': []}

        sources_tag = f"[Sources: {', '.join(sources_seen)}]"
        attributed = f"{sources_tag}\n" + "\n".join(all_extracted)

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

            # semantic duplicate guard
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
        intent = state.get('intent', Intent.RAG)
        detected_source = state.get('detected_source')
        plan = state.get('query_plan')

        if plan and plan.strategy == QueryStrategy.SQL:
            logger.info("Completeness → HALLUCINATION_GRADER (SQL query, gap-fill not applicable)")
            return NodeName.HALLUCINATION_GRADER_AGENT

        if plan and plan.strategy == QueryStrategy.SCROLL:
            logger.info(f"Completeness → HALLUCINATION_GRADER (SCROLL already retrieved all docs from source '{detected_source}', gap-fill not applicable)")
            return NodeName.HALLUCINATION_GRADER_AGENT

        if detected_source and intent == Intent.RAG_SUMMARIZATION:
            logger.info(f"Completeness → HALLUCINATION_GRADER (summarization scroll-based query, skipping gap-check)")
            return NodeName.HALLUCINATION_GRADER_AGENT

        if not follow_up or iterations >= self.max_completeness_iterations:
            logger.info(f"Completeness → HALLUCINATION_GRADER (iterations={iterations}, follow_up='{follow_up}')")
            return NodeName.HALLUCINATION_GRADER_AGENT

        logger.info(f"Completeness → RESEARCH_WORKER with query: '{follow_up}' (iteration {iterations})")
        seen_ids = [doc.id for doc in state.get('search_results', []) if hasattr(doc, 'id')]
        return [Send(NodeName.RESEARCH_WORKER, {'query': follow_up, 'seen_doc_ids': seen_ids})]

    def synthesizer_agent(self, state: AgentState):
        logger.info("--- SYNTHESIZING FINAL ANSWER ---")
        user_query = state['messages'][-1].content
        distilled_facts = state.get('distilled_facts', [])

        if not distilled_facts:
            filtered_results = state.get('filtered_results', [])
            if not filtered_results:
                return {'messages': [AIMessage(content='I could not find specific information in the database to answer your question.')]}
            logger.info("distilled_facts empty — falling back to raw filtered_results for synthesis")
            raw_blocks = []
            for doc in filtered_results[:10]:
                source = doc.metadata.get('source', 'unknown')
                raw_blocks.append(f"[{source}]\n{doc.text[:self.listing_chars_per_doc]}")
            facts_block = "\n---\n".join(raw_blocks)
        else:
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

    def error_agent(self, _state: AgentState):
        logger.info("--- ERROR ---")
        return {'messages': [AIMessage(content='I could not process your request. Please try to be more specific with your question.')]}

    @staticmethod
    def route_query_plan(state: AgentState):
        """Route from QUERY_PLANNER based on the QueryPlan strategy."""
        plan = state.get('query_plan')
        queries = state.get('rewritten_queries') or [state['messages'][-1].content]

        if not plan:
            return [Send(NodeName.RESEARCH_WORKER, {'query': q}) for q in queries]

        if plan.strategy == QueryStrategy.SQL:
            logger.info("route_query_plan → ANALYTICAL_QUERY (sql)")
            return NodeName.ANALYTICAL_QUERY

        if plan.strategy == QueryStrategy.HYBRID:
            logger.info("route_query_plan → ANALYTICAL_QUERY (hybrid, will continue to vector after)")
            return NodeName.ANALYTICAL_QUERY

        if plan.strategy == QueryStrategy.SCROLL:
            logger.info("route_query_plan → SCROLL_RETRIEVER")
            return NodeName.SCROLL_RETRIEVER

        logger.info(f"route_query_plan → RESEARCH_WORKER fan-out ({len(queries)} queries)")
        return [Send(NodeName.RESEARCH_WORKER, {'query': q}) for q in queries]

    @staticmethod
    def route_after_analytical(state: AgentState):
        """After ANALYTICAL_QUERY: hybrid continues to vector search, sql goes straight to synthesizer."""
        plan = state.get('query_plan')
        queries = state.get('rewritten_queries') or [state['messages'][-1].content]

        if plan and plan.strategy == QueryStrategy.HYBRID:
            logger.info("route_after_analytical → RESEARCH_WORKER fan-out (hybrid)")
            return [Send(NodeName.RESEARCH_WORKER, {'query': q}) for q in queries]

        logger.info("route_after_analytical → SYNTHESIZER (sql)")
        return NodeName.SYNTHESIZER

    @staticmethod
    def route_hallucination(state: AgentState) -> Literal[NodeName.SYNTHESIZER, END]: # type: ignore
        MAX_RETRIES = 2
        status = state.get('hallucination_status')
        retries = state.get("hallucination_retries", 0)

        if status == 'hallucinated':
            if retries >= MAX_RETRIES:
                logger.warning(f"Max retries ({MAX_RETRIES}) reached. Returning last draft with caveat.")
                last_draft = state['messages'][-1].content
                state['messages'][-1] = AIMessage(
                    content=f"Based on available context (note: information may be incomplete):\n\n{last_draft}"
                )
                return END

            return NodeName.SYNTHESIZER
        return END

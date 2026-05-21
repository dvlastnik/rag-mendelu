from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag.agents.scientific_source import ScientificDataSource
from rag.agents.models import ScientificRelevanceDecision
from rag.agents.state import AgentState
from rag.agents.prompts import Prompts
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ScientificNodes:
    def __init__(self, scientific_source: ScientificDataSource, llm: BaseChatModel):
        self.scientific_source = scientific_source
        self.llm = llm

    def scientific_retriever(self, state: AgentState) -> dict:
        """
        Search the main project's climate_data Qdrant collection.
        Uses state['rewritten_queries'] if populated, else falls back to
        the last HumanMessage content. Cap at 3 queries × 8 results each.
        Deduplicate by source_id:variable key.
        Returns: {"scientific_results": List[ScientificSearchResult]}
        """
        logger.info("--- SCIENTIFIC RETRIEVER ---")

        queries = state.get('rewritten_queries') or []
        if not queries:
            messages = state.get('messages', [])
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    queries = [msg.content]
                    break

        queries = queries[:3]
        logger.info(f"Scientific retriever using {len(queries)} queries: {queries}")

        seen: set = set()
        results = []
        for query in queries:
            try:
                raw = self.scientific_source.search(query, top_k=8)
                for r in raw:
                    key = f"{r.source_id}:{r.variable}"
                    if key not in seen:
                        seen.add(key)
                        results.append(r)
            except Exception as e:
                logger.warning(f"Scientific source search failed for query '{query}': {e}")

        logger.info(f"Scientific retriever found {len(results)} unique results")
        return {"scientific_results": results}

    def scientific_relevance_grader(self, state: AgentState) -> dict:
        """
        LLM evaluation: do the scientific results sufficiently answer the user's question?
        - If no results returned → immediately fallback to docs.
        - Otherwise ask LLM to judge relevance.
        Returns: {"data_source_scope": "scientific_sufficient" | "fallback_to_docs"}
        """
        logger.info("--- SCIENTIFIC RELEVANCE GRADER ---")
        scientific_results = state.get('scientific_results', [])

        if not scientific_results:
            logger.info("No scientific results — falling back to docs")
            return {"data_source_scope": "fallback_to_docs"}

        question = next(
            (m.content for m in reversed(state.get('messages', [])) if isinstance(m, HumanMessage)),
            ""
        )
        results_text = "\n\n".join([
            f"Source: {r.source_id} | Variable: {r.variable} | Score: {r.score:.3f}\n{r.text}"
            for r in scientific_results
        ])

        decision = self.llm.with_structured_output(ScientificRelevanceDecision).invoke([
            SystemMessage(content=Prompts.get_scientific_relevance_prompt()),
            HumanMessage(content=f"User question: {question}\n\nScientific results:\n{results_text}"),
        ])

        logger.info(f"Scientific relevance: is_relevant={decision.is_relevant} — {decision.reasoning}")
        scope = "scientific_sufficient" if decision.is_relevant else "fallback_to_docs"
        return {"data_source_scope": scope}

    def multi_source_synthesizer(self, state: AgentState) -> dict:
        """
        Merge distilled_facts (from docs RAG path) + scientific_results
        into a single labeled answer using the local LLM.
        Output format:
          ## From Climate Datasets
          [formatted scientific_results]

          ## From Documents
          [distilled_facts joined]
        Returns: {"messages": [AIMessage(content=unified_answer)]}
        """
        logger.info("--- MULTI SOURCE SYNTHESIZER ---")

        scientific_results = state.get('scientific_results', [])
        distilled_facts = state.get('distilled_facts', [])

        if scientific_results:
            scientific_lines = []
            for r in scientific_results:
                scientific_lines.append(
                    f"Source: {r.source_id} | Variable: {r.variable} | Score: {r.score:.3f}\n{r.text}"
                )
            scientific_section = "\n\n".join(scientific_lines)
        else:
            scientific_section = "No scientific dataset results available."

        if distilled_facts:
            docs_block = f"\n\nDOCUMENT FACTS:\n{chr(10).join(distilled_facts)}"
        else:
            docs_block = ""

        user_content = f"SCIENTIFIC DATASET RESULTS:\n{scientific_section}{docs_block}"

        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_multi_source_synthesis_prompt()),
            HumanMessage(content=user_content),
        ])

        return {"messages": [response]}

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag.agents.scientific_source import ScientificDataSource
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

        docs_section = "\n".join(distilled_facts) if distilled_facts else "No document facts available."

        user_content = (
            f"SCIENTIFIC DATASET RESULTS:\n{scientific_section}\n\n"
            f"DOCUMENT FACTS:\n{docs_section}"
        )

        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_multi_source_synthesis_prompt()),
            HumanMessage(content=user_content),
        ])

        return {"messages": [response]}

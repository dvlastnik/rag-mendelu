import difflib
import re
from typing import cast, Literal, List
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from rag.agents.state import AgentState
from rag.agents.models import GeneralOrRagDecision
from rag.agents.enums import NodeName, Intent
from rag.agents.prompts import Prompts
from utils.logging_config import get_logger

logger = get_logger(__name__)

_EXHAUSTIVE_PATTERNS = re.compile(
    r'\b(list\s+all|every\s+\w+\s+mentioned|all\s+\w+\s+in\b|how\s+many|name\s+all)\b',
    re.IGNORECASE,
)
_SUMMARIZATION_PATTERNS = re.compile(
    r'\b(summarize|summarise|summary\s+of|overview\s+of|what\s+is\s+.+\s+about)\b',
    re.IGNORECASE,
)


class GeneralNodes:
    def __init__(self, llm: BaseChatModel, available_sources: List[str] | None = None):
        self.llm = llm
        self.available_sources = available_sources or []

    def _match_source(self, raw_source: str | None) -> str | None:
        """Fuzzy-match a user-mentioned source against available_sources."""
        if not raw_source or not self.available_sources:
            return None
        
        lower_map = {s.lower(): s for s in self.available_sources}
        if raw_source.lower() in lower_map:
            return lower_map[raw_source.lower()]
        
        matches = difflib.get_close_matches(raw_source.lower(), lower_map.keys(), n=1, cutoff=0.5)
        if matches:
            return lower_map[matches[0]]
        return None

    @staticmethod
    def _keyword_intent_upgrade(text: str, intent: Intent) -> Intent:
        """Upgrade intent based on keyword patterns for robustness with small models."""
        if intent == Intent.GENERAL:
            return intent
        if _EXHAUSTIVE_PATTERNS.search(text):
            return Intent.RAG_EXHAUSTIVE
        if _SUMMARIZATION_PATTERNS.search(text):
            return Intent.RAG_SUMMARIZATION
        return intent

    def router_agent(self, state: AgentState):
        logger.info("--- ROUTING ---")
        messages = state['messages']
        last_message = messages[-1].content

        history_summary = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:]])
        human_msg = f'Context:\n{history_summary}\n\nCurrent User Input: {last_message}'

        decision = self.llm.with_structured_output(GeneralOrRagDecision).invoke([
            SystemMessage(content=Prompts.get_router_agent_prompt(self.available_sources)),
            HumanMessage(content=human_msg)
        ])

        decision = cast(GeneralOrRagDecision, decision)

        intent = self._keyword_intent_upgrade(last_message, decision.intent)
        if intent != decision.intent:
            logger.info(f"Keyword fallback upgraded intent: {decision.intent} -> {intent}")

        detected_source = self._match_source(decision.detected_source)
        if decision.detected_source and not detected_source:
            logger.warning(f"Source '{decision.detected_source}' not matched in available sources: {self.available_sources}")

        logger.info(f"Router decision: intent={intent}, detected_source={detected_source}")
        return {
            "intent": intent,
            "detected_source": detected_source,
        }

    def general_agent(self, state: AgentState):
        logger.info("--- GENERAL CHAT ---")
        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_general_agent_prompt()),
            *state['messages']
        ])
        return {"messages": [response]}

    @staticmethod
    def route_intent(state: AgentState) -> Literal[NodeName.QUERY_PLANNER, NodeName.GENERAL]:
        intent = state['intent']
        if intent in (Intent.RAG, Intent.RAG_EXHAUSTIVE, Intent.RAG_SUMMARIZATION):
            return NodeName.QUERY_PLANNER
        return NodeName.GENERAL

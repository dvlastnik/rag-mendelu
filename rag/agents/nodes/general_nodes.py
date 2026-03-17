from typing import cast, Literal, List
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from rag.agents.state import AgentState
from rag.agents.models import GeneralOrRagDecision, ExtractionScheme
from rag.agents.enums import NodeName, Intent
from rag.agents.prompts import Prompts
from utils.logging_config import get_logger

logger = get_logger(__name__)

class GeneralNodes:
    def __init__(self, llm: BaseChatModel, available_sources: List[str] | None = None):
        self.llm = llm
        self.available_sources = available_sources or []

    def router_agent(self, state: AgentState):
        logger.info("--- ROUTING ---")
        messages = state['messages']
        last_message = messages[-1].content

        history_summary = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:]])
        human_msg = f'Context:\n{history_summary}\n\nCurrent User Input: {last_message}'

        decision = self.llm.with_structured_output(GeneralOrRagDecision).invoke([
            SystemMessage(content=Prompts.get_router_agent_prompt()),
            HumanMessage(content=human_msg)
        ])

        decision = cast(GeneralOrRagDecision, decision)
        return {
            "intent": decision.intent,
            "detected_source": decision.detected_source,
        }

    def general_agent(self, state: AgentState):
        logger.info("--- GENERAL CHAT ---")
        response = self.llm.invoke([
            SystemMessage(content=Prompts.get_general_agent_prompt()),
            *state['messages']
        ])
        return {"messages": [response]}

    @staticmethod
    def route_intent(state: AgentState) -> Literal[NodeName.QUERY_DECOMPOSER, NodeName.GENERAL]:
        intent = state['intent']
        if intent is Intent.RAG:
            return NodeName.QUERY_DECOMPOSER
        return NodeName.GENERAL

from typing import cast, Literal
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from rag.agents.state import AgentState
from rag.agents.models import GeneralOrRagDecision
from rag.agents.enums import NodeName, Intent
from utils.logging_config import get_logger

logger = get_logger(__name__)

class GeneralNodes:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def router_agent(self, state: AgentState):
        logger.info("--- ROUTING ---")
        messages = state['messages']
        last_message = messages[-1].content

        history_summary = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:]])
        human_msg = f'Context:\n{history_summary}\n\nCurrent User Input: {last_message}'
        
        system_prompt = """You are a strict Classification Bot.
        You must choose between two options: 'rag' or 'general'.

        DEFINITIONS:
        1. 'general': ONLY for greetings (Hi, Hello), goodbyes (Bye), or polite phrases (Thanks, Cool).
        2. 'rag': For EVERYTHING else. Any question, any statement of fact, any request for comparison, any mention of weather/floods/countries.

        CRITICAL RULES:
        - If the user asks a question -> MUST be 'rag'.
        - If the user mentions a year (2022) or country -> MUST be 'rag'.
        - If the user refers to previous messages ("and Italy?") -> MUST be 'rag'.
        - "Compare floods" is NOT general conversation. It is a data request.

        EXAMPLES:
        Input: "Hi there" -> general
        Input: "Compare floods in Italy" -> rag
        Input: "And Czech Republic?" -> rag
        Input: "What about 2022?" -> rag
        Input: "Thanks" -> general
        """
        
        decision = self.llm.with_structured_output(GeneralOrRagDecision).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg)
        ])
        
        return {"intent": cast(GeneralOrRagDecision, decision).intent}

    def general_agent(self, state: AgentState):
        logger.info("--- GENERAL CHAT ---")
        response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant. Respond politely to the user."),
            *state['messages']
        ])
        return {"messages": [response]}

    @staticmethod
    def route_intent(state: AgentState) -> Literal[NodeName.EXTRACTOR, NodeName.GENERAL]:
        if state['intent'] == Intent.RAG:
            return NodeName.QUERY_REWRITER
        return NodeName.GENERAL
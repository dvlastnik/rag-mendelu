from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class Judgement(BaseModel):
    relevancy_score: int = Field(description="Score 1-5. How well does the answer address the user's question?")
    faithfulness_score: int = Field(description="Score 1-5. Is the answer fully supported by the Retrieved Context?")
    reasoning: str = Field(description="Concise explanation for the scores.")

class Judge:
    def __init__(self, model_name="llama3.1:8b"):
        self.llm = init_chat_model(
            model=model_name,
            model_provider='ollama', 
            temperature=0, 
            num_ctx=8192
        )
        self.structured_llm = self.llm.with_structured_output(Judgement)

    def evaluate(self, question: str, answer: str, context: list[str], ground_truth: str = None):
        context_text = "\n".join(context[:10])

        prompt = f"""
        You are a strict QA Auditor for a RAG system.
        Evaluate the Generated Answer based on the Context and Ground Truth.

        --- INPUT DATA ---
        QUESTION: {question}
        ANSWER: {answer}
        RETRIEVED CONTEXT:
        {context_text}
        GROUND TRUTH: {ground_truth}
        ------------------

        --- SCORING CRITERIA ---
        1. RELEVANCY (1-5):
           - 5 = Answer completely answers the question and matches Ground Truth logic.
           - 1 = Answer is irrelevant or refuses to answer.

        2. FAITHFULNESS (1-5):
           - 5 = Every fact in the answer is found in the RETRIEVED CONTEXT.
           - 1 = Answer contains hallucinations or info not in context.

        Provide a concise reasoning (2-3 sentences) and assign both scores.
        """

        return self.structured_llm.invoke(prompt)
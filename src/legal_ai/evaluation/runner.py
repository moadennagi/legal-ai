from legal_ai.interfaces import RAGInterface, RunnerInterface
from legal_ai.models.schemas import SingleTurnSample
from typing import Any


class Runner(RunnerInterface):
    
    async def run(
        self,
        rag: RAGInterface,
        user_query: str,
        similarity_threshold: float,
        history: list[dict[str, str]],
    ) -> SingleTurnSample:
        """
        Run Rag.ask() and return the response and the sources.
        """
        contexts: list[dict[str, Any]] = []
        res = await rag.ask(
            user_query=user_query, similarity_threshold=similarity_threshold, history=history
        )
        if res.get("contexts"):
            contexts = res["contexts"]
        return SingleTurnSample(
            user_input=user_query,
            response=res["answer"],
            retrieved_contexts=contexts,
        )

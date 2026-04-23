from ragas.metrics import DiscreteMetric
from ragas import experiment
from ragas.llms import llm_factory
from legal_ai.interfaces import RAGInterface, RunnerInterface
from typing import Any
from legal_ai.adapters import OllamaLLMClientAdapter

ollama_client = OllamaLLMClientAdapter()
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.

        Consider the response correct if it:
        1. Contains the key information from the expected answer
        2. Is factually accurate based on the provided context
        3. Adequately addresses the question asked

        Return 'pass' if the response is correct, 'fail' if it's incorrect.

        Question: {question}
        Expected Answer: {expected_answer}
        Model Response: {response}

        Evaluation:
        """,
    allowed_values=["pass", "fail"],
)


class RagasEvaluation:
    def __init__(self, rag_client: RAGInterface, generation_model: str, runner: RunnerInterface):
        self.rag_client = rag_client
        self.generation_model = generation_model
        self.runner = runner

    @experiment()
    async def run_experiment(
        self,
        rag: RAGInterface,
        row: dict[str, Any],
        metric: DiscreteMetric,
        similarity_threshold: float,
    ) -> dict[str, Any]:
        """Run experiment and return a result.

        Args:
            rag (RAGInterface): _description_
            row (dict[str, Any]): _description_
            metric (DiscreteMetric): _description_

        Returns:
            dict[str, Any]: _description_
        """
        response_sample = await self.runner.run(
            rag=rag,
            user_query=row["question"],
            similarity_threshold=similarity_threshold,
            history=[],
        )

        res = await metric.ascore(
            question=row["question"],
            expected_answer=row["answer"],
            response=response_sample.response,
            llm=llm_factory(model=rag.generation_model, client=ollama_client),
        )

        return {
            **row,
            "model_response": response_sample.response,
            "evaluation_type": metric.name,
            "evaluation_score": res.value,
            "evaluation_reason": res.reason,
            "retrieved_documents": [
                doc.get("content", "")[:200] + "..."
                for doc in response_sample.retrieved_contexts
                if doc.get("content")
            ],
        }

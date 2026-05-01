from ragas import evaluate, SingleTurnSample, EvaluationDataset, MultiTurnSample
from ragas.evaluation import EvaluationResult
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from legal_ai.interfaces import RAGInterface
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from legal_ai.settings import settings
from legal_ai.models.schemas import EvaluationDatasetRow
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RagasEvaluation:
    def __init__(self, llm_as_judge: str):
        self.llm_as_judge = llm_as_judge

    async def evaluate_response(
        self,
        rag: RAGInterface,
        rows: list[EvaluationDatasetRow],
        similarity_threshold: float,
        hyde: bool = False,
        rerank: bool = False,
        contextualize_query: bool = False,
    ) -> EvaluationResult:
        """Évalue le pipeline RAG sur un dataset annoté et retourne les métriques RAGAS."""
        samples: list[SingleTurnSample | MultiTurnSample] = []
        for row in rows:
            response = await rag.ask(
                user_query=row.question,
                similarity_threshold=similarity_threshold,
                history=[],
                hyde=hyde,
                rerank=rerank,
                contextualize_query=contextualize_query,
            )
            contexts = [chunk["content"] for chunk in response.context]
            sample_turn = SingleTurnSample(
                user_input=row.question,
                retrieved_contexts=contexts,
                response=response.answer,
                reference=row.ground_truth,
            )
            samples.append(sample_turn)
        evalaution_dataset = EvaluationDataset(samples=samples)
        api_key = settings.openai_api_key
        llm = LangchainLLMWrapper(ChatOpenAI(model=self.llm_as_judge, api_key=api_key))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))
        result = evaluate(
            dataset=evalaution_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )

        return result

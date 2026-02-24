from legal_ai.interfaces import LLMClientInterface, RAGInterface
from legal_ai.settings import settings
from typing import Any


class ConversationManager:
    def __init__(self, llm_client: LLMClientInterface, rag: RAGInterface):
        self.history: list[dict[str, str]] = []
        self.llm_client = llm_client
        self.rag = rag

    async def ask(self, query: str, similarity_threshold: float) -> dict[str, Any]:
        """
        Update conversation history with user quer and llm answer, manage history token
        limit by sliding window of summary + last for messages. the conversation
        summary is done by asking the llm.

        Args:
            query (str): user query
            similarity_threshold (float): similarity threshold

        Returns:
            dict[str, Any]: answer with sources
        """
        self.history.append({"role": "user", "content": query})
        # TODO: 2000 is magic number
        if self._count_tokens() >= 2000:
            self._compress()
        answer = await self.rag.ask(
            user_query=query,
            similarity_threshold=similarity_threshold,
            history=self.history[:-1],  # in order to not sent the user query twice
        )
        self.history.append({"role": "assistant", "content": answer["answer"]})
        return answer

    def _count_tokens(self) -> int:
        """
        Return approximate count of tokens in self.history.

        Returns:
            int: count of tokens
        """
        res = 0
        for data in self.history:
            for _, message in data.items():
                if not message:
                    continue
                # TODO: 4 is a magic number, make it configurable
                res += len(message) // 4
        return res

    def _compress(self):
        """
        Compress conversation history by asking the llm to summerize the conversation
        and keeping last 4 messages
        """
        prompt = {"role": "user", "content": f"Résume la conversation suivante: \n {self.history}"}
        response = self.llm_client.chat(model=settings.generation_model, messages=[prompt])
        _copy = [{"role": "system", "content": response}] + self.history[-4:]
        self.history = _copy

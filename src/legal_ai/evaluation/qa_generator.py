from legal_ai.interfaces import LLMClientInterface
from legal_ai.evaluation.utils import CSV_HEADERS
import csv
from typing import Coroutine
import asyncio
import random


class QASyntheticGenerator:
    def __init__(self, llm_client: LLMClientInterface, model: str):
        # llm has to be different from the one used for generation (rag)
        self.llm_client = llm_client
        self.model = model

    async def llm_generate_question(self, context: str) -> str:
        user_prompt = f""""Extrait du Bulletin Officiel :

        {context}

        ---

        Génère une question juridique dont la réponse complète
        est contenue dans cet extrait.
        Question :
        """
        system_prompt = """Tu es un collaborateur administratif qui cherche une information dans la réglementation.
        formule la question comme un utilisateur non-juriste la poserait dans un moteur de recherche, 
        en langage courant, sans utiliser le vocabulaire normatif de l'extrait
        """
        answer = await self.llm_client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return answer

    async def llm_generate_ground_truth(self, question: str, context: str) -> str:
        user_prompt = f"""Extrait du Bulletin Officiel :

        {context}

        ---

        Question : {question}

        Fournis une réponse de référence concise (1 à 3 phrases),
        strictement basée sur l'extrait. Cite le numéro d'article si disponible.
        Réponse :
        """
        system_prompt = """Tu es un expert en droit marocain. Réponds uniquement en te basant
        sur l'extrait fourni. N'invente aucune information absente du texte.
        Réponds UNIQUEMENT avec la réponse de référence, sans préambule.

        """
        answer = await self.llm_client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return answer

    async def generate(
        self, input_csv: str, output_csv: str, offset: int = 6, limit: int = 60
    ) -> int:
        data: list[dict[str, str]] = []
        question_tasks: list[Coroutine[None, None, str]] = []
        ground_truth_tasks: list[Coroutine[None, None, str]] = []
        # build simple dataset of questions
        target_rows: list[dict[str, str]] = []
        with open(input_csv, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            i = 0
            for row in reader:
                if i < offset:
                    i += 1
                    continue

                target_rows.append(row)
                i += 1

        random_rows = random.sample(target_rows, min(limit, len(target_rows)))
        for row in random_rows:
            obj = {
                "context": row["context"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "source_doc": row["source_doc"],
                "source_article": row["source_article"],
                "chunk_index": row["chunk_index"],
                "official_date": row["official_date"],
            }
            data.append(obj)

        # use llm to generate questions
        for row in data:
            question_tasks.append(self.llm_generate_question(row["context"]))
        questions = await asyncio.gather(*question_tasks)

        # loop over rows and update question
        i = 0
        while i < len(data):
            if data[i]["question"]:
                i += 1
                continue
            data[i]["question"] = questions[i]
            i += 1

        pending = [i for i, row in enumerate(data) if not row["ground_truth"]]
        ground_truth_tasks = [
            self.llm_generate_ground_truth(data[i]["question"], data[i]["context"]) for i in pending
        ]
        results = await asyncio.gather(*ground_truth_tasks)
        for i, gt in zip(pending, results):
            data[i]["ground_truth"] = gt

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
            writer.writeheader()

            for obj in data:
                writer.writerow(obj)
        return len(data)

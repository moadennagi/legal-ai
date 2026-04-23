from legal_ai.interfaces import ChunkResult, DocumentSplitterInterface
from legal_ai.models.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class GenericSplitter(DocumentSplitterInterface):
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
    ) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_document(self, document: Document) -> list[ChunkResult]:
        """Split the given document into chunks using RecursiveCharacterSplitter.

        Args:
            document (Document): Document instance

        Returns:
            list[ChunkResult]: ChunkResult list
        """
        if not document.text_content:
            return []
        res: list[ChunkResult] = []
        chunks = self.text_splitter.split_text(document.text_content)
        for i, chunk in enumerate(chunks):
            res.append(ChunkResult(page_content=chunk, id=str(i), metadata={}))
        return res

    def construct_enriched_content(self, chunk: ChunkResult) -> str:
        return chunk.page_content

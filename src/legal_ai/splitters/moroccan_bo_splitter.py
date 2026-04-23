from langchain_core.documents import Document as LangchainDocument

from legal_ai.interfaces import DocumentSplitterInterface
from legal_ai.models.document import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import re
from legal_ai.interfaces import ChunkResult

_BOLD_LINE_RE = re.compile(r"^\*\*(.+)\*\*$")

_ARTICLE_RE = re.compile(
    r"^(?:Article|ARTICLE|Art\.?\s*|ART\.?\s*)\s*(?:premier|PREMIER|\d+)",
    re.IGNORECASE,
)

_KEYWORD_RULES: list[tuple[int, re.Pattern]] = [
    # ── H1: top-level BO sections (always ALL-CAPS, always the same words) ──
    (1, re.compile(r"^DAHIR$", re.IGNORECASE)),
    (1, re.compile(r"^TEXTES?\s+G[ÉE]N[ÉE]RAUX$", re.IGNORECASE)),
    (1, re.compile(r"^TEXTES?\s+PARTICULIERS?$", re.IGNORECASE)),
    (1, re.compile(r"^AVIS\s+ET\s+COMMUNICATIONS?$", re.IGNORECASE)),
    (1, re.compile(r"^SOMMAIRE$", re.IGNORECASE)),
    (1, re.compile(r"^[ÉE]DITIONS?\s+DE\s+TRADUCTION", re.IGNORECASE)),
    # ── H2: legal instruments (always start with these keywords) ──
    (2, re.compile(r"^Dahir\s+n[°º]", re.IGNORECASE)),
    (2, re.compile(r"^Loi\s+n[°º]", re.IGNORECASE)),
    (2, re.compile(r"^D[ée]cret\s+n[°º]", re.IGNORECASE)),
    (2, re.compile(r"^Arr[êe]t[ée]\s+", re.IGNORECASE)),
    (2, re.compile(r"^D[ée]cisje ion\s+", re.IGNORECASE)),
    (2, re.compile(r"^Avis\s+(?:du|de\s+la|n[°º])\b", re.IGNORECASE)),
    (2, re.compile(r"^Nomination\s+", re.IGNORECASE)),
    (2, re.compile(r"^Homologation\s+", re.IGNORECASE)),
    (2, re.compile(r"^[ÉE]quivalences?\s+de\s+", re.IGNORECASE)),
    # ── H3: parts (legal convention) ──
    (3, re.compile(r"\bPARTIE\b", re.IGNORECASE)),
    # Roman numeral sections: "I. Topic...", "IV. Défis..."
    (3, re.compile(r"^[IVXLC]+[\.\)]\s+")),
    # ── H4: titles (legal convention) ──
    (4, re.compile(r"^TITRE\s+", re.IGNORECASE)),
    # Numbered subsections: "1. Topic...", "2. Topic..."
    (4, re.compile(r"^\d+[\.\)]\s+")),
    # ── H5: chapters (legal convention) ──
    (5, re.compile(r"^Chapitre\s+", re.IGNORECASE)),
    # Lettered subsections: "a. Topic...", "b. Topic..."
    (5, re.compile(r"^[a-z][\.\)]\s+")),
]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

HEADERS_TO_SPLIT_ON = [
    ("#", "division"),  # DAHIR, TEXTES GENERAUX, TEXTES PARTICULIERS
    ("##", "instrument"),  # Dahir n°, Loi n°, Décret n°, Arrêté…
    ("###", "partie"),  # PREMIÈRE PARTIE…
    ("####", "titre"),  # TITRE PREMIER…
    ("#####", "chapitre"),  # Chapitre premier…
    ("######", "section"),  # Dénomination et objet, Missions…
]


class MoroccanBulettinOfficielSplitter(DocumentSplitterInterface):
    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 500,
    ) -> None:
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _classify(self, text: str) -> int | None:
        """
        Classify a heading text.

        Returns:
            1-6:  matched a keyword rule → use this level
            None: it's an article → convert to **bold**
            -1:   free-text heading → caller will infer from context
        """
        s = text.strip()

        if _ARTICLE_RE.match(s):
            return None

        for level, pattern in _KEYWORD_RULES:
            if pattern.search(s):
                return level

        return -1

    def _fix_heading_hierarchy(
        self,
        markdown: str,
        *,
        articles_as_bold: bool = True,
    ) -> str:
        """
        Post-process a Docling markdown string: reassign heading levels.

        APPROACH: Hybrid — two kinds of headings, two strategies.

        Known keyword headings → level from the keyword rule.
        Free-text headings     → one level below the last known heading above.

        1. KEYWORD HEADINGS (pattern-matched):
        Headings that use FIXED legal vocabulary shared across ALL BO issues.
        "Chapitre", "TITRE", "Article", "PREMIÈRE PARTIE", "Arrêté du..."
        These are defined by Moroccan legal drafting conventions and won't change.
        → Assign level directly from the keyword.

        2. FREE-TEXT HEADINGS (context-inferred):
        Headings that could say anything — section titles in an Avis,
        descriptive sub-chapter names, report topics, etc.
        → Assign level based on position: one level below the last known
            keyword heading above them, capped at 6.

        In a BO document, free-text headings always appear UNDER a structural
        keyword heading. An Avis section title comes after "Avis du Conseil..."
        (H2), so it becomes H3. A descriptive sub-chapter name comes after
        "Chapitre premier" (H5), so it becomes H6. The structural keywords
        act as anchors, and everything else flows relative to them.

        Hierarchy:
            #      → DAHIR, TEXTES GENERAUX, TEXTES PARTICULIERS, AVIS ET COMMUNICATIONS
            ##     → Dahir n°, Loi n°, Décret n°, Arrêté, Avis du Conseil...
            ###    → PREMIÈRE PARTIE, roman numeral sections (I. II. III.)
            ####   → TITRE PREMIER, numbered subsections (1. 2. 3.)
            #####  → Chapitre premier, lettered subsections (a. b. c.)
            ###### → free-text headings (inferred from context)
            **bold** → Article / Art.

        Example trace:

            ## TEXTES GENERAUX          → keyword match H1  →  # TEXTES GENERAUX
            ## Arrêté du ministre...    → keyword match H2  →  ## Arrêté du ministre...
            ## Chapitre premier         → keyword match H5  →  ##### Chapitre premier
            ## Dénomination et objet    → unknown, last=5   →  ###### Dénomination et objet
            ## Article premier          → article           →  **Article premier**
            ## Article 2                → article           →  **Article 2**
            ## Chapitre II              → keyword match H5  →  ##### Chapitre II
            ## Missions                 → unknown, last=5   →  ###### Missions

            ## AVIS ET COMMUNICATIONS   → keyword match H1  →  # AVIS ET COMMUNICATIONS
            ## Avis du Conseil...       → keyword match H2  →  ## Avis du Conseil...
            ## Introduction             → unknown, last=2   →  ### Introduction
            ## I. Les micro-entreprises → keyword match H3  →  ### I. Les micro-entreprises
            ## 1. Le tissu...           → keyword match H4  →  #### 1. Le tissu...
            ## Un titre libre           → unknown, last=4   →  ##### Un titre libre

        Args:
            markdown: raw markdown from Docling's export_to_markdown().
            articles_as_bold: convert Article headings to **bold** paragraphs.
        """
        lines = markdown.split("\n")
        out: list[str] = []
        last_known_level: int = 1

        for line in lines:
            # if its not a heading add to the lines
            m = _HEADING_RE.match(line)

            if not m:
                level = self._classify(line)
                # some instrument show as plain text
                if level == 2:
                    out.append(f"{'#' * level} {line}")
                    last_known_level = level
                    continue

                bold_m = _BOLD_LINE_RE.match(line.strip())
                if bold_m:
                    text = bold_m.group(1).strip()
                    level = self._classify(text)
                    if level is not None and level != -1 and level <= 2:
                        last_known_level = level
                        out.append(f"{'#' * level} {text}")
                        continue

                out.append(line)
                continue

            # it is a heading, we need to find its level
            text = m.group(2).strip()
            level = self._classify(text)

            if level is None:
                # Article → bold
                out.append(f"\n**{text}**\n" if articles_as_bold else f"###### {text}")

            elif level == -1:
                # Free-text heading → one level below last known, capped at 6
                inferred = min(last_known_level + 1, 6)
                out.append(f"{'#' * inferred} {text}")

            else:
                # Keyword heading → exact level
                last_known_level = level
                out.append(f"{'#' * level} {text}")

        return "\n".join(out)

    def split_document(self, document: Document) -> list[ChunkResult]:
        """
        Split a document's markdown content into chunks.

        1. MarkdownHeaderTextSplitter splits on headings → semantic sections
        with metadata (division, instrument, partie, titre, chapitre, section).
        2. RecursiveCharacterTextSplitter sub-splits long sections to fit
        the embedding model's context window.

        If a document lacks some heading levels (e.g. no PARTIE or TITRE),
        those metadata keys simply won't appear — no errors, no empty splits.
        """
        if not document.text_content:
            return []
        fixed_markdown = self._fix_heading_hierarchy(document.text_content, articles_as_bold=False)
        md_sections = self.md_splitter.split_text(fixed_markdown)
        chunks = self.text_splitter.split_documents(md_sections)
        chunks = self._filter_chunks(chunks)
        res = [
            ChunkResult(page_content=chunk.page_content, id=chunk.id, metadata=chunk.metadata)
            for chunk in chunks
        ]
        return res

    def _is_table_like(self, text: str) -> bool:
        """Check if content is mostly table formatting (pipe-delimited rows)."""
        lines = text.strip().splitlines()
        if not lines:
            return True
        pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
        return pipe_lines / len(lines) > 0.5

    def _filter_chunks(self, chunks: list[LangchainDocument]) -> list[LangchainDocument]:
        """
        Filter out bad chunks: empty, too short, table-of-contents,
        and table-like formatting artifacts that are not part of a hierarchy.

        Args:
            chunks (list[LangchainDocument]): a list of langchain documents

        Returns:
            list[LangchainDocument]: of list good langchain documents
        """
        good_chunks: list[LangchainDocument] = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content:
                continue

            # skip table of content
            if chunk.metadata.get("division", "").strip().upper() == "SOMMAIRE":
                continue

            # skip very short chunks (headers, separators, etc.)
            if len(content) < 20:
                continue

            # skip table-like formatting artifacts only if it has no legal hierarchy
            if self._is_table_like(content) and not chunk.metadata.get("instrument"):
                continue

            good_chunks.append(chunk)
        return good_chunks

    def construct_enriched_content(self, chunk: ChunkResult) -> str:
        """
        Construct a string representing the chunk with its context.
        (for contextual embedding)

        Args:
            chunk (dict[str, Any]): a chunk as created by langchain text splitter
        """
        metadata_keys = ["division", "instrument", "partie", "titre", "chapitre", "section"]
        breadcrumbs: list[str] = []
        if not chunk.metadata:
            return chunk.page_content

        if not metadata_keys:
            metadata_keys = [key for key in chunk.metadata]

        for key in metadata_keys:
            if key not in chunk.metadata:
                continue
            clean = re.sub(r"^#+\s*", "", chunk.metadata.get(key))
            breadcrumbs.append(clean)
        header = " > ".join(breadcrumbs)
        return f"[{header}]\n{chunk.page_content}"

"""
Bulletin Officiel heading hierarchy fixer.

APPROACH: Hybrid — two kinds of headings, two strategies.

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

WHY THIS WORKS:
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
"""

import re

# ═══════════════════════════════════════════════════════════════════════
# ARTICLE DETECTION — these become **bold**, not headings
# ═══════════════════════════════════════════════════════════════════════

_ARTICLE_RE = re.compile(
    r"^(?:Article|ARTICLE|Art\.?\s*)\s*(?:premier|PREMIER|\d+)",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════
# KEYWORD RULES — only for truly fixed legal vocabulary
#
# These are stable across ALL Bulletin Officiel issues because they
# come from legal drafting conventions, not from document content.
#
# Rule: (level, pattern)
# Evaluated top-to-bottom, first match wins.
# ═══════════════════════════════════════════════════════════════════════

_KEYWORD_RULES: list[tuple[int, re.Pattern]] = [
    # ── H1: top-level BO sections (always ALL-CAPS, always the same words) ──
    (1, re.compile(r"^DAHIR$", re.IGNORECASE)),
    (1, re.compile(r"^TEXTES?\s+G[ÉE]N[ÉE]RAUX$", re.IGNORECASE)),
    (1, re.compile(r"^TEXTES?\s+PARTICULIERS?$", re.IGNORECASE)),
    (1, re.compile(r"^AVIS\s+ET\s+COMMUNICATIONS?$", re.IGNORECASE)),
    (1, re.compile(r"^SOMMAIRE$", re.IGNORECASE)),
    # ── H2: legal instruments (always start with these keywords) ──
    (2, re.compile(r"^Dahir\s+n[°º]", re.IGNORECASE)),
    (2, re.compile(r"^Loi\s+n[°º]", re.IGNORECASE)),
    (2, re.compile(r"^D[ée]cret\s+n[°º]", re.IGNORECASE)),
    (2, re.compile(r"^Arr[êe]t[ée]\s+", re.IGNORECASE)),
    (2, re.compile(r"^D[ée]cision\s+", re.IGNORECASE)),
    (2, re.compile(r"^Avis\s+(?:du|de\s+la|n[°º])\b", re.IGNORECASE)),
    (2, re.compile(r"^Nomination\s+", re.IGNORECASE)),
    (2, re.compile(r"^Homologation\s+", re.IGNORECASE)),
    (2, re.compile(r"^[ÉE]quivalences?\s+de\s+", re.IGNORECASE)),
    # ── H3: parts (legal convention) ──
    (3, re.compile(r"\bPARTIE\b", re.IGNORECASE)),
    (3, re.compile(r"^DISPOSITIONS?\s+G[ÉE]N[ÉE]RALES?$", re.IGNORECASE)),
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

# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _classify(text: str) -> int | None:
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


def fix_heading_hierarchy(
    markdown: str,
    *,
    articles_as_bold: bool = True,
) -> str:
    """
    Post-process a Docling markdown string: reassign heading levels.

    Known keyword headings → level from the keyword rule.
    Free-text headings     → one level below the last known heading above.

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
            out.append(line)
            continue
        # it is a heading, we need to find its level
        text = m.group(2).strip()
        level = _classify(text)

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

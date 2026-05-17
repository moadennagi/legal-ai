import pytest
from unittest.mock import MagicMock
from legal_ai.splitters.moroccan_bo_splitter import MoroccanBulettinOfficielSplitter
from langchain_core.documents import Document as LangchainDocument


@pytest.fixture
def splitter():
    bo_splitter = MoroccanBulettinOfficielSplitter(chunk_overlap=200, chunk_size=1500)
    return bo_splitter


def test_fix_heading_hierarchy_update_headings_with__correct_levels(splitter):
    input_md = "\n".join(
        [
            "## TEXTES GÉNÉRAUX",
            "## Arrêté du ministre... n° 123-24",
            "free text",
            "## Chapitre premier",
            "## Dispositions générales",
            "## Article premier",
        ]
    )
    res = splitter._fix_heading_hierarchy(input_md, articles_as_bold=False)
    lines = res.split("\n")
    assert "# TEXTES GÉNÉRAUX" in lines
    assert "## Arrêté du ministre... n° 123-24" in lines
    assert "free text" in lines
    assert "##### Chapitre premier" in lines
    assert "###### Dispositions générales" in lines
    assert "###### Article premier" in lines


@pytest.mark.parametrize(
    "text,expected_level",
    [
        ("TEXTES GÉNÉRAUX", 1),
        ("DAHIR", 1),
        ("Loi n° 12-24", 2),
        ("Décret n° 2-24-100", 2),
        ("Arrêté du ministre...", 2),
        ("PREMIÈRE PARTIE", 3),
        ("I. Introduction", 3),
        ("TITRE PREMIER", 4),
        ("1. Contexte", 4),
        ("Chapitre premier", 5),
        ("a. Définitions", 5),
        ("Article premier", None),
        ("Article 2", None),
        ("Quelque chose de libre", -1),
    ],
)
def test_classify(splitter, text, expected_level):
    assert splitter._classify(text) == expected_level


def test_filter_chunks_filter_out_bad_chunks(splitter):
    # Test cases: (content, metadata, should_pass, description)
    test_cases = [
        # PASS: normal content with hierarchy
        (
            "This is a good chunk with legal content spanning more than twenty characters.",
            {"instrument": "Dahir n° 1-24", "chapitre": "Chapitre premier"},
            True,
            "good chunk with hierarchy",
        ),
        # FAIL: empty content
        ("", {"instrument": "Dahir n° 1-24"}, False, "empty chunk"),
        # FAIL: whitespace only
        ("   \n  \t  ", {"instrument": "Dahir n° 1-24"}, False, "whitespace-only chunk"),
        # FAIL: SOMMAIRE division (table of contents)
        (
            "Long content that would normally pass but is marked as SOMMAIRE division.",
            {"division": "SOMMAIRE"},
            False,
            "SOMMAIRE division",
        ),
        # FAIL: SOMMAIRE division (case insensitive)
        (
            "Long content in sommaire table of contents section here.",
            {"division": "   sommaire   "},
            False,
            "SOMMAIRE division (case insensitive)",
        ),
        # FAIL: very short chunk (< 20 chars)
        ("Short", {"instrument": "Dahir n° 1-24"}, False, "chunk too short"),
        # FAIL: table-like without instrument
        (
            "|head|head1|\n|----|-----|\n|one |two|\n|three|four|",
            {"division": "TEXTS"},
            False,
            "table-like without instrument",
        ),
        # PASS: table-like WITH instrument (legal table, not artifact)
        (
            "|head|head1|\n|----|-----|\n|one |two|\n|three|four|",
            {"instrument": "Decree n° 1-24"},
            True,
            "table-like with instrument",
        ),
        # PASS: minimal valid content (exactly 20 chars)
        ("Exactly twenty chars", {"instrument": "Loi n° 1-24"}, True, "minimal valid length"),
    ]

    for content, metadata, should_pass, description in test_cases:
        doc = LangchainDocument(page_content=content, metadata=metadata)
        result = splitter._filter_chunks([doc])

        if should_pass:
            assert len(result) == 1, f"FAIL: {description} — should pass but was filtered"
        else:
            assert len(result) == 0, f"FAIL: {description} — should be filtered but passed"


def test_constructed_enriched_chunks_return_chunks_updated_with_metadata(splitter):
    # given a chunk assert that the method return an enriched chunk
    # enriched chunk is a chunk with its metadata
    chunk_data = {"page_content": "test chunk", "metadata": {"instrument": "Decret 12"}}
    doc = LangchainDocument(**chunk_data)
    res = splitter.construct_enriched_content(doc)
    assert res == "[Decret 12]\ntest chunk"


def test_split_document(splitter):
    input_md = "\n".join(
        [
            "## Dahir n° 1-25-64 du 22 joumada I 1447 (14 novembre 2025) portant  promulgation de la loi  n° 03-25  relative  aux organismes de placement collectif en valeurs mobilières.",
            "## LOUANGE A DIEU SEUL !",
            "(Grand Sceau de Sa Majesté Mohammed VI)",
            "## Chapitre II",
            "## Du FCP",
            "## Article 18",
            "Le FCP est une copropriété qui n'a pas la personnalité morale.",
            "## Article 19",
            "Un FCP peut comporter un ou plusieurs compartiments.",
        ]
    )
    document = MagicMock()
    document.text_content = input_md
    res = splitter.split_document(document)
    assert len(res) == 4
    chunks_contents = [chunk.page_content for chunk in res]
    assert (
        "###### Article 18\nLe FCP est une copropriété qui n'a pas la personnalité morale."
        in chunks_contents
    )

import json

import pytest

from applications.trust_rag.corpus import lexical_search
from applications.trust_rag.evaluation import reference_metrics, summarize
from applications.trust_rag.pipeline import citation_errors
from applications.trust_rag.retrieval import rrf
from applications.trust_rag.splits import split_cases
from applications.trust_rag.table import calculate, select_slot
from haystack import Document


def test_corpus_keeps_table_sources_and_excludes_eval_metadata(corpus):
    docs = lexical_search(corpus.root / "corpus.sqlite", "总资产", 4)
    assert len(docs) == 1
    assert "125" in docs[0].content
    assert "MUST_NOT_LEAK" not in json.dumps(docs[0].to_dict())
    assert docs[0].meta["location"]["cell"] == "C5"


def test_table_queries_cannot_escape_current_turn_or_inject_sql(corpus):
    with pytest.raises(ValueError, match="scope"):
        select_slot(corpus, {"doc_id": "doc1", "cell": "C5"}, {"other"})
    assert select_slot(corpus, {"doc_id": "doc1", "cell": "C5' OR 1=1 --"}, {"doc1"}) == []
    docs = select_slot(corpus, {"doc_id": "doc1", "cell": "C5"}, {"doc1"})
    assert len(docs) == 1


def test_calculation_requires_distinct_operands_and_consistent_units():
    a = Document(id="a", content="a", meta={"value": "0.1", "unit": "亿元"})
    b = Document(id="b", content="b", meta={"value": "0.2", "unit": "亿元"})
    assert calculate("sum", [a, b])["value"] == "0.3"
    assert calculate("percentage_change", [a, b])["value"] == "100"
    with pytest.raises(ValueError, match="distinct"):
        calculate("difference", [a, a])
    with pytest.raises(ValueError, match="units"):
        calculate("sum", [a, Document(id="c", meta={"value": "1", "unit": "万元"})])
    with pytest.raises(ValueError, match="zero"):
        calculate("ratio", [a, Document(id="z", meta={"value": "0", "unit": "亿元"})])


def test_verifier_rejects_fabricated_ids_and_quotes():
    docs = [Document(id="a", content="资本充足率为百分之十。")]
    answer = {"refused": False, "citations": [{"evidence_id": "yesterday", "quote": "资本充足率"}]}
    assert citation_errors(answer, docs)
    answer["citations"] = [{"evidence_id": "a", "quote": "资本充足率为百分之二十"}]
    assert citation_errors(answer, docs)
    answer["citations"] = [{"evidence_id": "a", "quote": "资本充足率为百分之十"}]
    assert citation_errors(answer, docs) == []


def test_rrf_deduplicates_and_rewards_agreement():
    a, b, c = [Document(id=i, content=i) for i in "abc"]
    result = rrf([[a, b, b], [c, b]], 3)
    assert [d.id for d in result] == ["b", "a", "c"]


def test_errors_and_refusals_remain_in_denominator():
    rows = [
        {"id": "1", "qa_type": "type", "correct": True, "status": "answered"},
        {"id": "2", "qa_type": "type", "correct": False, "status": "refused"},
        {"id": "3", "qa_type": "type", "correct": False, "status": "error"},
    ]
    assert summarize(rows)["accuracy"] == 1 / 3


def test_missing_fine_grained_gold_is_not_counted_as_failure():
    result = reference_metrics(
        {"expected_evidence_refs": [{"source_title": "办法", "cell": None}]}, [{"meta": {"source_title": "办法"}}]
    )
    assert result == {"source_hit": True, "position_hit": None}


def test_grouped_split_covers_each_type_and_detects_tampering(tmp_path):
    cases = []
    for category in range(5):
        for doc in range(6):
            cases.append(
                {
                    "id": f"{category}-{doc}",
                    "question": f"法规{chr(0x4E00 + doc * 5 + category)}如何解释？",
                    "qa_type": str(category),
                    "source_type": "word",
                    "difficulty": "easy",
                    "expected_evidence_refs": [{"source_title": f"源{category}-{doc}"}],
                }
            )
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(map(json.dumps, cases)), encoding="utf-8")
    result = split_cases(path, tmp_path / "splits", trials=1000)
    assert set(result["coverage"]["dev"]) == set(result["coverage"]["test"]) == set(map(str, range(5)))
    dev = {json.loads(line)["id"] for line in (tmp_path / "splits/dev.jsonl").read_text().splitlines()}
    test = {json.loads(line)["id"] for line in (tmp_path / "splits/test.jsonl").read_text().splitlines()}
    assert not dev & test
    (tmp_path / "splits/dev.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        split_cases(path, tmp_path / "splits")

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from applications.trust_rag.evaluation import compare, evaluate, freeze
from applications.trust_rag.pipeline import RAG, Variant
from applications.trust_rag.retrieval import build_embeddings, build_qdrant, require_complete_embeddings

from .test_pipeline import OfflineProvider


def test_partial_embedding_build_cannot_be_evaluated(corpus):
    provider = OfflineProvider(corpus)
    build_embeddings(corpus, provider, workers=1, limit=1)
    with pytest.raises(ValueError, match="not complete"):
        require_complete_embeddings(corpus, provider)


def test_error_diagnostics_preserve_categories_without_exception_payloads():
    from applications.trust_rag.evaluation import error_diagnostics

    inner = ValueError("secret document and API key")
    outer = RuntimeError("wrapped sensitive content")
    outer.__cause__ = inner
    assert error_diagnostics(outer) == [{"type": "RuntimeError"}, {"type": "ValueError"}]


def test_source_metric_recognizes_parent_title_without_inventing_position_hits():
    from applications.trust_rag.evaluation import reference_metrics

    case = {"expected_evidence_refs": [{"source_title": "资本管理办法", "cell": "C5"}]}
    docs = [
        {
            "meta": {
                "source_title": "附件：风险权重表",
                "parent_source_title": "资本管理办法",
                "location": {"cell": "B2"},
            }
        }
    ]
    assert reference_metrics(case, docs) == {"source_hit": True, "position_hit": False}


def test_test_set_rejects_changed_configuration_before_any_api(corpus, monkeypatch):
    split_dir = corpus.root / "splits"
    split_dir.mkdir()
    (split_dir / "manifest.json").write_text("{}", encoding="utf-8")
    freeze(corpus, Variant.stage("B4"))
    changed = replace(corpus, top_k=10)
    with pytest.raises(ValueError, match="frozen protocol changed"):
        evaluate(changed, Variant.stage("B4"), "test", corpus.root / "never-created")
    assert not (corpus.root / "never-created").exists()


def test_comparison_requires_identical_denominators_and_unique_ids(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    common = {
        "settings_sha256": "same",
        "corpus_sha256": "same",
        "split_sha256": "same",
        "cases_sha256": "same",
        "case_ids": ["Q1"],
        "dependencies": {},
        "code_sha256": "same",
        "platform": "test",
    }
    for path in (left, right):
        path.mkdir()
        (path / "protocol.json").write_text(json.dumps(common), encoding="utf-8")
        row = {"id": "Q1", "qa_type": "x", "status": "answered", "correct": path == right}
        (path / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare(left, right)["accuracy_delta"] == 1
    result_file = right / "results.jsonl"
    result_file.write_text(result_file.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        compare(left, right)


def test_pipeline_two_concurrent_requests_have_separate_results(corpus):
    settings = replace(corpus, qdrant_local=True)
    provider = OfflineProvider(settings)
    build_embeddings(settings, provider, workers=1)
    build_qdrant(settings, provider)
    rag = RAG(settings, Variant.stage("B4"), provider)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(rag.ask, ["问题一 A.125 B.2 C.3 D.4", "问题二 A.125 B.2 C.3 D.4"]))
        assert a is not b
        assert a["answer"] is not b["answer"]
        a["answer"]["citations"].clear()
        assert b["answer"]["citations"]
    finally:
        rag.close()

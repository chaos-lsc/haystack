import hashlib
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from applications.trust_rag.pipeline import RAG, Variant
from applications.trust_rag.retrieval import build_embeddings, build_qdrant


class OfflineProvider:
    def __init__(self, settings):
        self.settings = settings
        self.events = []
        self.local = threading.local()
        self.calls = []

    def embed(self, texts, *, cached_only=False):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def key(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def chat(self, system, user, operation="generation"):
        import json

        self.calls.append(operation)
        if operation == "table_planning":
            return {"slots": [{"name": "a", "doc_id": "doc1", "cell": "C5"}], "operations": []}
        if operation == "verification":
            return {"supported": True, "reason": "fixture"}
        evidence = json.loads(user)["evidence"][0]
        return {
            "selected_option": "A",
            "answer": "125亿元",
            "refused": False,
            "citations": [{"evidence_id": evidence["id"], "quote": "总资产为125亿元"}],
        }


@pytest.mark.parametrize("stage", ["B0", "B1", "B2", "B3", "B4"])
def test_real_haystack_pipeline_and_qdrant_local_contract(corpus, stage):
    settings = replace(corpus, qdrant_local=True)
    provider = OfflineProvider(settings)
    build_embeddings(settings, provider, workers=1)
    if stage != "B0":
        build_qdrant(settings, provider)
    rag = RAG(settings, Variant.stage(stage), provider)
    try:
        result = rag.ask("总资产是多少？A.125 B.50 C.10 D.0")
        assert result["answer"]["selected_option"] == "A"
        assert result["documents"][0]["id"] == "ev1"
        assert result["retrieval_seconds"] >= 0
        assert ("table_planning" in provider.calls) == (stage in {"B3", "B4"})
        assert ("verification" in provider.calls) == (stage == "B4")
        assert rag.ask("总资产？A.125 B.1 C.2 D.3")["answer"]["citations"][0]["evidence_id"] == "ev1"
    finally:
        rag.close()


def test_provider_rejects_partial_embedding_response(corpus, monkeypatch):
    from applications.trust_rag.provider import Provider, ProviderError

    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    provider = Provider(corpus)
    monkeypatch.setattr(
        provider, "_call", lambda *args: SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[1.0])])
    )
    with pytest.raises(ProviderError, match="indices"):
        provider.embed(["one", "two"])


def test_qdrant_lost_acknowledgement_replays_same_points(monkeypatch):
    from qdrant_client.http.exceptions import ResponseHandlingException

    from applications.trust_rag.retrieval import upsert_points

    class LostAcknowledgement:
        def __init__(self):
            self.points = {}
            self.calls = 0

        def upsert(self, collection, points, wait):
            self.calls += 1
            self.points.update({point.id: point for point in points})
            if self.calls == 1:
                raise ResponseHandlingException(TimeoutError())
            return "acknowledged"

    monkeypatch.setattr("applications.trust_rag.retrieval.time.sleep", lambda _: None)
    client = LostAcknowledgement()
    point = SimpleNamespace(id="stable-id", vector=[1.0])
    assert upsert_points(client, "collection", [point]) == "acknowledged"
    assert list(client.points) == ["stable-id"]

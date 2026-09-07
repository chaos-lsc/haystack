"""Native Haystack baseline and disk-backed incremental retrieval components."""

import json
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from haystack import Document, component
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from .config import Settings
from .corpus import batches, lexical_search
from .provider import Provider


def client_for(settings: Settings) -> QdrantClient:
    if settings.qdrant_local:
        return QdrantClient(path=str(settings.root / "qdrant-local"))
    return QdrantClient(url=settings.qdrant_url, timeout=60)


def embedding_identity(settings: Settings, provider: Provider) -> str:
    return provider.key("trust-rag-index-identity-v1")[:16]


def collection_name(settings: Settings, provider: Provider) -> str:
    corpus = json.loads((settings.root / "corpus.manifest.json").read_text(encoding="utf-8"))
    return f"{settings.collection}_{corpus['sqlite_sha256'][:12]}_{embedding_identity(settings, provider)}"


def require_complete_embeddings(settings: Settings, provider: Provider) -> dict:
    build = json.loads((settings.root / "embedding-build.json").read_text(encoding="utf-8"))
    corpus = json.loads((settings.root / "corpus.manifest.json").read_text(encoding="utf-8"))
    if build["partial"] or build["count"] != corpus["chunks"]:
        raise ValueError("Full-corpus embeddings are not complete")
    if (
        build["identity"] != embedding_identity(settings, provider)
        or build.get("corpus_sha256") != corpus["sqlite_sha256"]
    ):
        raise ValueError("Embedding manifest model/corpus mismatch")
    return build


def build_embeddings(settings: Settings, provider: Provider, workers: int = 4, limit: int | None = None) -> dict:
    start, count, expected_dimensions = time.perf_counter(), 0, None
    source = batches(settings.root / "corpus.sqlite", settings.embedding_batch, limit)

    def embed(batch):
        vectors = provider.embed([doc.content for doc in batch])
        if len({len(v) for v in vectors}) != 1:
            raise ValueError("embedding dimensions changed within batch")
        return len(batch), len(vectors[0])

    pending = deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in source:
            pending.append(pool.submit(embed, batch))
            if len(pending) >= workers * 2:
                n, dimensions = pending.popleft().result()
                if expected_dimensions is not None and dimensions != expected_dimensions:
                    raise ValueError("embedding dimensions changed across batches")
                expected_dimensions = dimensions
                count += n
                if count % (settings.embedding_batch * 20) == 0:
                    print(json.dumps({"embedded": count, "seconds": round(time.perf_counter() - start, 2)}), flush=True)
        while pending:
            n, dimensions = pending.popleft().result()
            if expected_dimensions is not None and dimensions != expected_dimensions:
                raise ValueError("embedding dimensions changed across batches")
            expected_dimensions = dimensions
            count += n
    if not count:
        raise ValueError("empty corpus")
    result = {
        "count": count,
        "dimensions": dimensions,
        "seconds": time.perf_counter() - start,
        "identity": embedding_identity(settings, provider),
        "corpus_sha256": json.loads((settings.root / "corpus.manifest.json").read_text(encoding="utf-8"))[
            "sqlite_sha256"
        ],
        "partial": limit is not None,
        "events": list(provider.events),
    }
    (settings.root / "embedding-build.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {k: v for k, v in result.items() if k != "events"}


def upsert_points(client, collection: str, points: list):
    for attempt in range(3):
        try:
            return client.upsert(collection, points, wait=True)
        except ResponseHandlingException:
            if attempt == 2:
                raise
            # Deterministic IDs make replay safe even when only the acknowledgement was lost.
            time.sleep(2**attempt)


def wait_for_optimization(client, collection: str, timeout: float = 1200) -> dict:
    deadline, stable = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        info = client.get_collection(collection)
        if info.optimizer_status != "ok":
            raise ValueError("Qdrant optimizer reported an error")
        stable = stable + 1 if info.status == models.CollectionStatus.GREEN else 0
        if stable >= 3:
            return {
                "status": info.status.value,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
            }
        time.sleep(2)
    raise TimeoutError("Qdrant optimization did not settle before the maintenance deadline")


def build_qdrant(settings: Settings, provider: Provider, limit: int | None = None) -> dict:
    client, name = client_for(settings), collection_name(settings, provider)
    try:
        build = json.loads((settings.root / "embedding-build.json").read_text(encoding="utf-8"))
        if limit is None:
            require_complete_embeddings(settings, provider)
        if build["identity"] != embedding_identity(settings, provider):
            raise ValueError("embedding cache model mismatch")
        if not client.collection_exists(name):
            client.create_collection(
                name,
                vectors_config=models.VectorParams(
                    size=build["dimensions"], distance=models.Distance.COSINE, on_disk=True
                ),
                hnsw_config=models.HnswConfigDiff(on_disk=True, max_indexing_threads=1),
                optimizers_config=models.OptimizersConfigDiff(max_optimization_threads=1),
                on_disk_payload=True,
            )
        count = 0
        for batch in batches(settings.root / "corpus.sqlite", 64, limit):
            vectors = provider.embed([doc.content for doc in batch], cached_only=True)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.id)),
                    vector=vector,
                    payload={"id": doc.id, "content": doc.content, "meta": doc.meta},
                )
                for doc, vector in zip(batch, vectors, strict=True)
            ]
            upsert_points(client, name, points)
            count += len(points)
            if count % 2048 == 0:
                print(json.dumps({"qdrant_points": count}), flush=True)
        actual = client.count(name, exact=True).count
        if actual != count:
            raise ValueError(f"Unexpected Qdrant point count {actual}, expected {count}")
        optimization = wait_for_optimization(client, name)
        result = {
            "collection": name,
            "count": count,
            "partial": limit is not None,
            "local_mode": settings.qdrant_local,
            "optimization": optimization,
        }
        (settings.root / "qdrant-build.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        client.close()


def baseline(settings: Settings, provider: Provider, limit: int | None = None) -> InMemoryEmbeddingRetriever:
    if limit is None:
        require_complete_embeddings(settings, provider)
    store = InMemoryDocumentStore(embedding_similarity_function="cosine", return_embedding=False, shared=False)
    for batch in batches(settings.root / "corpus.sqlite", 64, limit):
        vectors = provider.embed([doc.content for doc in batch], cached_only=True)
        store.write_documents([replace(doc, embedding=vector) for doc, vector in zip(batch, vectors, strict=True)])
    return InMemoryEmbeddingRetriever(store, top_k=settings.candidate_k)


@component
class QueryEmbedder:
    def __init__(self, provider: Provider):
        self.provider = provider

    @component.output_types(embedding=list[float], started_at=float)
    def run(self, query: str):
        started = time.perf_counter()
        return {
            "embedding": self.provider.embed([self.provider.settings.query_prefix + query])[0],
            "started_at": started,
        }


@component
class QdrantRetriever:
    def __init__(self, settings: Settings, provider: Provider):
        complete = require_complete_embeddings(settings, provider)
        self.settings, self.client = settings, client_for(settings)
        self.collection = collection_name(settings, provider)
        if not self.client.collection_exists(self.collection):
            self.client.close()
            raise ValueError("Qdrant collection is missing; run index-qdrant first")
        if self.client.count(self.collection, exact=True).count != complete["count"]:
            self.client.close()
            raise ValueError("Qdrant collection is incomplete")

    @component.output_types(documents=list[Document])
    def run(self, query_embedding: list[float]):
        result = self.client.query_points(
            self.collection,
            query=query_embedding,
            limit=self.settings.candidate_k,
            with_payload=True,
            with_vectors=False,
        )
        return {
            "documents": [
                Document(
                    id=point.payload["id"],
                    content=point.payload["content"],
                    meta=point.payload["meta"],
                    score=point.score,
                )
                for point in result.points
            ]
        }


def rrf(rankings: list[list[Document]], top_k: int, k: int = 60) -> list[Document]:
    scores, documents = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            if doc.id in seen:
                continue
            seen.add(doc.id)
            scores[doc.id] = scores.get(doc.id, 0.0) + 1 / (k + rank)
            documents[doc.id] = doc
    ordered = sorted(documents, key=lambda identifier: (-scores[identifier], identifier))[:top_k]
    return [Document(id=i, content=documents[i].content, meta=documents[i].meta, score=scores[i]) for i in ordered]


@component
class RetrievalSelection:
    def __init__(self, settings: Settings, hybrid: bool):
        self.settings, self.hybrid = settings, hybrid

    @component.output_types(documents=list[Document], retrieval_seconds=float)
    def run(self, query: str, documents: list[Document], started_at: float):
        if self.hybrid:
            lexical = lexical_search(self.settings.root / "corpus.sqlite", query, self.settings.candidate_k)
            documents = rrf([documents, lexical], self.settings.top_k)
        return {"documents": documents[: self.settings.top_k], "retrieval_seconds": time.perf_counter() - started_at}

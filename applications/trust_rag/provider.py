"""API transport with bounded retries, usage accounting, and redacted errors."""

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import Any

import numpy as np
from openai import OpenAI

from .config import Settings


class ProviderError(RuntimeError):
    pass


class Provider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "not-configured"),
            base_url=settings.api_base,
            timeout=settings.timeout,
            max_retries=0,
        )
        self.events = deque(maxlen=20000)
        self.run_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.local = threading.local()
        self.rate_lock = threading.Lock()
        self.cooldown_until = 0.0
        with self.cache() as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    @contextmanager
    def cache(self):
        db = sqlite3.connect(self.settings.root / "embeddings.sqlite", timeout=60)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA cache_size=-4096")
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, event: dict):
        with self.lock:
            self.events.append(event)
            with (self.settings.root / "api-usage.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**event, "run_id": self.run_id, "timestamp": time.time()}) + "\n")
        if hasattr(self.local, "events"):
            self.local.events.append(event)

    def key(self, text: str) -> str:
        descriptor = [self.settings.api_base, self.settings.embedding_model, "float", text]
        return hashlib.sha256(json.dumps(descriptor, ensure_ascii=False).encode()).hexdigest()

    def _call(self, operation: str, fn):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not configured")
        for attempt in range(3):
            with self.rate_lock:
                cooldown = max(0.0, self.cooldown_until - time.monotonic())
            if cooldown:
                time.sleep(cooldown)
            start = time.perf_counter()
            try:
                result = fn()
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                event = {
                    "operation": operation,
                    "attempt": attempt + 1,
                    "status": status,
                    "error": type(exc).__name__,
                    "seconds": time.perf_counter() - start,
                }
                self.record(event)
                if attempt < 2 and (status in (429, 500, 502, 503, 504) or status is None):
                    if status == 429:
                        headers = getattr(getattr(exc, "response", None), "headers", {})
                        try:
                            delay = max(15 * (2**attempt), float(headers.get("retry-after", 0)))
                        except (TypeError, ValueError):
                            delay = 15 * (2**attempt)
                        with self.rate_lock:
                            self.cooldown_until = max(self.cooldown_until, time.monotonic() + delay)
                    else:
                        time.sleep(2**attempt)
                    continue
                raise ProviderError(f"{operation}: {type(exc).__name__} (HTTP {status}); details redacted") from None
            usage = getattr(result, "usage", None)
            self.record(
                {
                    "operation": operation,
                    "attempt": attempt + 1,
                    "seconds": time.perf_counter() - start,
                    "model": getattr(result, "model", None),
                    "usage": usage.model_dump() if usage else {},
                }
            )
            return result
        raise ProviderError("retry budget exhausted")

    def embed(self, texts: list[str], *, cached_only: bool = False) -> list[list[float]]:
        keys = [self.key(text) for text in texts]
        cached: dict[str, list[float]] = {}
        with self.cache() as db:
            for key in dict.fromkeys(keys):
                row = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    cached[key] = np.frombuffer(row[0], dtype="<f4").tolist()
        missing = list(dict.fromkeys(text for key, text in zip(keys, texts, strict=True) if key not in cached))
        if cached_only and missing:
            raise ProviderError("Corpus embedding cache is incomplete; run embed before loading an index")
        for offset in range(0, len(missing), self.settings.embedding_batch):
            batch = missing[offset : offset + self.settings.embedding_batch]
            result = self._call(
                "embedding",
                lambda: self.client.embeddings.create(
                    model=self.settings.embedding_model, input=batch, encoding_format="float"
                ),
            )
            entries = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in entries] != list(range(len(batch))):
                raise ProviderError("embedding response has missing or duplicate indices")
            with self.cache() as db:
                for text, entry in zip(batch, entries, strict=True):
                    vector = np.asarray(entry.embedding, dtype="<f4")
                    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                        raise ProviderError("invalid embedding vector")
                    cached[self.key(text)] = vector.tolist()
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (self.key(text), vector.tobytes()))
        return [cached[key] for key in keys]

    def chat(self, system: str, user: str, *, operation: str = "generation") -> dict[str, Any]:
        result = self._call(
            operation,
            lambda: self.client.chat.completions.create(
                model=self.settings.generation_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=self.settings.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            ),
        )
        if not result.choices or result.choices[0].finish_reason != "stop":
            raise ProviderError("incomplete model output")
        try:
            payload = json.loads(result.choices[0].message.content or "")
        except (TypeError, ValueError):
            raise ProviderError("invalid JSON model output") from None
        if not isinstance(payload, dict):
            raise ProviderError("model output must be a JSON object")
        return payload

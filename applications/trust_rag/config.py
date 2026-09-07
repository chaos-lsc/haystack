"""Explicit, reproducible runtime configuration; secrets never enter manifests."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    data_dir: str = ".lab-data"
    api_base: str = "https://api.siliconflow.cn/v1"
    generation_model: str = "Qwen/Qwen3.5-9B"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    query_prefix: str = "Instruct: Given a Chinese financial regulation question, retrieve relevant evidence.\nQuery: "
    top_k: int = 12
    candidate_k: int = 32
    max_context_chars: int = 24000
    max_tokens: int = 2048
    timeout: float = 90.0
    embedding_batch: int = 32
    qdrant_url: str = "http://127.0.0.1:6333"
    collection: str = "trust_rag"
    # Local Qdrant is for tests/smoke only; resource acceptance requires the server.
    qdrant_local: bool = False

    def __post_init__(self):
        if not 1 <= self.top_k <= self.candidate_k <= 200:
            raise ValueError("Require 1 <= top_k <= candidate_k <= 200")
        if not 1 <= self.embedding_batch <= 256 or self.timeout <= 0 or self.max_tokens <= 0:
            raise ValueError("Invalid API batch/timeout/token budget")
        if not 1000 <= self.max_context_chars <= 100000:
            raise ValueError("Context budget outside supported range")
        for url in (self.api_base, self.qdrant_url):
            parsed = urlsplit(url)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Endpoints must not contain embedded credentials or query parameters")

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls()

    @property
    def root(self) -> Path:
        path = Path(self.data_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_credentials(env_file: str | None = None) -> None:
    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "false")

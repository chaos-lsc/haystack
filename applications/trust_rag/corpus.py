"""Streaming conversion of source Evidence into a bounded SQLite corpus."""

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import jieba

from haystack import Document

SCHEMA_VERSION = 2
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 160


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text: str) -> str:
    return " ".join(token.lower() for token in jieba.cut(text) if re.search(r"\w", token))


@contextmanager
def connect(path: Path):
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    try:
        with db:
            yield db
    finally:
        db.close()


def as_document(row: sqlite3.Row) -> Document:
    return Document(id=row["id"], content=row["content"], meta=json.loads(row["meta"]))


def build_corpus(source: Path, target: Path) -> dict:
    """Idempotent build, atomically published; answer-bearing filenames are excluded."""
    source_hash = file_hash(source)
    manifest_path = target.with_suffix(".manifest.json")
    if target.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_sha256"] == source_hash and manifest["schema"] == SCHEMA_VERSION:
            if file_hash(target) != manifest["sqlite_sha256"]:
                raise ValueError("Existing corpus does not match its frozen manifest")
            return manifest
        raise ValueError("Corpus exists with a different source/schema; choose a new data directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".building.sqlite")
    # Only this known, unpublished temporary file may be replaced.
    temporary.unlink(missing_ok=True)
    count, evidence_count, excluded, chars = 0, 0, 0, 0
    with connect(temporary) as db:
        db.executescript("""
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL,
                sheet TEXT, cell TEXT, metric TEXT, period TEXT, value TEXT, unit TEXT
            );
            CREATE INDEX table_scope ON evidence(doc_id, sheet, cell);
            CREATE VIRTUAL TABLE lexical USING fts5(id UNINDEXED, terms, tokenize='unicode61');
        """)
        with source.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                original = record.get("metadata", {})
                title = record["source_title"]
                filename = original.get("file_name", "")
                if any(marker in (filename + title).lower() for marker in ("qa数据", "competition_qa", "mcq_cases")):
                    excluded += 1
                    continue
                quote = str(record.get("quote", "")).strip()
                if not quote:
                    continue
                location = record.get("location", {})
                meta = {
                    "doc_id": record["doc_id"],
                    "evidence_id": record["evidence_id"],
                    "source_title": title,
                    "parent_source_title": original.get("source_title") or title,
                    "file_name": filename,
                    "location": location,
                    "page": original.get("page_number"),
                    "source": original.get("source"),
                    "metric": original.get("metric") or original.get("indicator", ""),
                    "period": original.get("period", ""),
                    "unit": original.get("unit", ""),
                    "value": original.get("cell_value", ""),
                    "row_header": original.get("row_header", ""),
                    "column_header": original.get("column_header", ""),
                }
                evidence_count += 1
                for offset in range(0, len(quote), CHUNK_CHARS - CHUNK_OVERLAP):
                    piece = quote[offset : offset + CHUNK_CHARS]
                    if offset and len(piece) <= CHUNK_OVERLAP:
                        break
                    identifier = record["evidence_id"] if offset == 0 else f"{record['evidence_id']}:{offset}"
                    content = f"来源：{title}\n{piece}"
                    metadata = {**meta, "chunk_offset": offset}
                    db.execute(
                        "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            identifier,
                            record["doc_id"],
                            content,
                            json.dumps(metadata, ensure_ascii=False),
                            location.get("sheet", "").strip(),
                            location.get("cell", ""),
                            meta["metric"],
                            meta["period"],
                            str(meta["value"]),
                            meta["unit"],
                        ),
                    )
                    db.execute("INSERT INTO lexical VALUES (?, ?)", (identifier, tokens(content)))
                    count += 1
                    chars += len(content)
                if evidence_count % 2000 == 0:
                    db.commit()
                    print(json.dumps({"corpus_records": evidence_count, "chunks": count}), flush=True)
        db.execute("INSERT INTO lexical(lexical) VALUES ('optimize')")
        db.commit()
        docs = db.execute("SELECT COUNT(DISTINCT doc_id) FROM evidence").fetchone()[0]
    temporary.replace(target)
    manifest = {
        "schema": SCHEMA_VERSION,
        "source_sha256": source_hash,
        "source_path": str(source.resolve()),
        "evidence_count": evidence_count,
        "chunks": count,
        "documents": docs,
        "excluded": excluded,
        "characters": chars,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "text_format": "source-title + original-quote-v1",
        "lexical": "jieba-0.42.1 + SQLite FTS5 BM25",
        "sqlite_sha256": file_hash(target),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def batches(path: Path, batch_size: int = 32, limit: int | None = None):
    with connect(path) as db:
        cursor = db.execute("SELECT * FROM evidence ORDER BY rowid LIMIT ?", (limit or -1,))
        while rows := cursor.fetchmany(batch_size):
            yield [as_document(row) for row in rows]


def lexical_search(path: Path, query: str, top_k: int) -> list[Document]:
    terms = list(dict.fromkeys(tokens(query).split()))[:80]
    if not terms:
        return []
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    with connect(path) as db:
        rows = db.execute(
            """SELECT e.*, hits.rank FROM
            (SELECT id, bm25(lexical) AS rank FROM lexical WHERE lexical MATCH ? ORDER BY rank LIMIT ?) hits
            JOIN evidence e ON e.id = hits.id ORDER BY hits.rank""",
            (match, top_k),
        ).fetchall()
        result = []
        for row in rows:
            doc = as_document(row)
            result.append(replace(doc, score=-row["rank"]))
        return result

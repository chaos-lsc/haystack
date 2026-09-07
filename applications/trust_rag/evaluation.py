"""Fixed-denominator evaluation with test-set freeze and aligned comparisons."""

import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .config import Settings
from .corpus import file_hash
from .pipeline import RAG, Variant
from .resources import ResourceMonitor
from .splits import normalize


def error_diagnostics(exc: Exception) -> list[dict]:
    """Keep nested failure categories without logging API payloads or credentials."""
    result, seen = [], set()
    while exc is not None and id(exc) not in seen and len(result) < 8:
        seen.add(id(exc))
        item = {"type": type(exc).__name__}
        if isinstance(exc, ValidationError):
            item["validation_types"] = [error["type"] for error in exc.errors(include_input=False)]
        result.append(item)
        exc = exc.__cause__ or exc.__context__
    return result


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol(settings: Settings) -> dict:
    return {
        "settings": asdict(settings),
        "settings_sha256": settings.fingerprint,
        "code_sha256": code_hash(),
        "corpus_sha256": file_hash(settings.root / "corpus.manifest.json"),
        "split_sha256": file_hash(settings.root / "splits" / "manifest.json"),
        "dependencies": {
            package: version(package) for package in ("haystack-ai", "qdrant-client", "jieba", "openai", "numpy")
        },
    }


def freeze(settings: Settings, variant: Variant) -> dict:
    path = settings.root / "freeze.json"
    result = {**protocol(settings), "final_variant": asdict(variant), "frozen_at": time.time()}
    if path.exists():
        raise ValueError("Already frozen; use a new experiment directory to change the protocol")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def reference_metrics(case: dict, docs: list[dict]) -> dict:
    refs = case.get("expected_evidence_refs", [])
    source_hits, position_hits, locatable = [], [], []
    for ref in refs:
        matching = [
            d
            for d in docs
            if normalize(ref["source_title"])
            in {
                normalize(d["meta"].get("source_title", "")),
                normalize(d["meta"].get("parent_source_title", "")),
            }
        ]
        source_hits.append(bool(matching))
        fields = {k: ref[k] for k in ("sheet", "cell", "page") if ref.get(k) is not None and str(ref[k]).strip()}
        if fields:
            locatable.append(ref)
            position_hits.append(
                any(
                    all(
                        normalize(
                            str(
                                doc["meta"].get("page")
                                if key == "page"
                                else doc["meta"].get("location", {}).get(key, "")
                            )
                        )
                        == normalize(str(value))
                        for key, value in fields.items()
                    )
                    for doc in matching
                )
            )
    return {"source_hit": all(source_hits) if refs else None, "position_hit": all(position_hits) if locatable else None}


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    status = Counter(row["status"] for row in rows)
    result = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "status": dict(status),
    }
    for metric in ("source_hit", "position_hit"):
        applicable = [row.get(metric) for row in rows if row.get(metric) is not None]
        result[metric] = {
            "count": len(applicable),
            "hits": sum(applicable),
            "rate": sum(applicable) / len(applicable) if applicable else None,
        }
    for metric in ("seconds", "retrieval_seconds"):
        values = [row[metric] for row in rows if metric in row]
        result[metric] = {
            "count": len(values),
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
    groups = defaultdict(list)
    for row in rows:
        groups[row["qa_type"]].append(row)
    result["by_type"] = {
        key: {
            "total": len(group),
            "correct": sum(r["correct"] for r in group),
            "accuracy": sum(r["correct"] for r in group) / len(group),
        }
        for key, group in groups.items()
    }
    for field in ("source_type", "difficulty"):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row.get(field, "unknown")].append(row)
        result["by_" + field] = {
            key: {
                "total": len(group),
                "correct": sum(r["correct"] for r in group),
                "accuracy": sum(r["correct"] for r in group) / len(group),
            }
            for key, group in buckets.items()
        }
    result["wrong_answer_rate"] = (
        sum(r["status"] == "answered" and not r["correct"] for r in rows) / total if total else None
    )
    result["refusal_rate"] = status["refused"] / total if total else None
    result["error_rate"] = status["error"] / total if total else None
    usage = Counter()
    operations = Counter()
    for row in rows:
        for event in row.get("api_events", []):
            operations[event["operation"]] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += event.get("usage", {}).get(key, 0) or 0
    result["api_attempts"] = dict(operations)
    result["token_usage"] = dict(usage)
    result["cost"] = "Token usage recorded; apply verified provider prices, not an invented amount."
    return result


def evaluate(
    settings: Settings, variant: Variant, split: str, output: Path, concurrency: int = 2, limit: int | None = None
) -> dict:
    if output.exists():
        raise ValueError("Run directory exists; use a new run ID to preserve failures and provenance")
    if concurrency not in (1, 2) or (limit is not None and limit <= 0):
        raise ValueError("concurrency must be 1 or 2 and limit must be positive")
    current = protocol(settings)
    if split == "test":
        frozen = json.loads((settings.root / "freeze.json").read_text(encoding="utf-8"))
        if any(frozen[k] != v for k, v in current.items()):
            raise ValueError("Test evaluation blocked: frozen protocol changed")
    case_path = settings.root / "splits" / f"{split}.jsonl"
    manifest = json.loads((settings.root / "splits" / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(case_path) != manifest["files"][case_path.name]:
        raise ValueError("Frozen case file changed")
    cases = [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines()][:limit]
    if not cases:
        raise ValueError("empty evaluation set")
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                **current,
                "variant": asdict(variant),
                "split": split,
                "cases_sha256": file_hash(case_path),
                "case_ids": [c["id"] for c in cases],
                "limit": limit,
                "concurrency": concurrency,
                "platform": platform.platform(),
                "started_at": time.time(),
                "cache_policy": "Corpus vectors prebuilt; warm-queries before comparative evaluation. Actual API events retained.",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = []
    with ResourceMonitor() as monitor:
        rag = RAG(settings, variant)

        def run(case):
            row = {
                "id": case["id"],
                "qa_type": case["qa_type"],
                "source_type": case["source_type"],
                "difficulty": case["difficulty"],
                "correct_option": case["correct_option"],
            }
            start = time.perf_counter()
            try:
                # Online entrypoint sees only question text, never labels or gold locations.
                result = rag.ask(case["question"])
                row.update(result)
                row["correct"] = result["answer"]["selected_option"] == case["correct_option"]
                row["status"] = "refused" if result["answer"]["refused"] else "answered"
                row.update(reference_metrics(case, result["retrieved_documents"]))
                row["context_metrics"] = reference_metrics(case, result["documents"])
            except Exception as exc:
                row.update(
                    {
                        "correct": False,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_diagnostics": error_diagnostics(exc),
                        "seconds": time.perf_counter() - start,
                        "source_hit": False,
                        "position_hit": False
                        if any(r.get("cell") or r.get("page") for r in case["expected_evidence_refs"])
                        else None,
                        "api_events": list(getattr(rag.provider.local, "events", [])),
                    }
                )
            return row

        try:
            with (
                (output / "results.jsonl").open("w", encoding="utf-8") as handle,
                ThreadPoolExecutor(max_workers=concurrency) as pool,
            ):
                futures = [pool.submit(run, case) for case in cases]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "done": len(rows),
                                "total": len(cases),
                                "id": row["id"],
                                "correct": row["correct"],
                                "status": row["status"],
                            }
                        ),
                        flush=True,
                    )
        finally:
            rag.close()
    report = {**summarize(rows), "resources": monitor.result, "variant": asdict(variant), "split": split}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def compare(left: Path, right: Path) -> dict:
    a_protocol, b_protocol = [json.loads((p / "protocol.json").read_text(encoding="utf-8")) for p in (left, right)]
    for field in ("settings_sha256", "corpus_sha256", "split_sha256", "cases_sha256", "case_ids", "dependencies"):
        if a_protocol[field] != b_protocol[field]:
            raise ValueError(f"Incomparable runs: {field} differs")
    runs = [
        list(map(json.loads, (p / "results.jsonl").read_text(encoding="utf-8").splitlines())) for p in (left, right)
    ]
    if any(len({r["id"] for r in rows}) != len(rows) for rows in runs):
        raise ValueError("Duplicate result IDs")
    a, b = [{r["id"]: r for r in rows} for rows in runs]
    if not a or set(a) != set(b) or set(a) != set(a_protocol["case_ids"]):
        raise ValueError("Incomplete or misaligned runs")
    improvements = [i for i in a if not a[i]["correct"] and b[i]["correct"]]
    regressions = [i for i in a if a[i]["correct"] and not b[i]["correct"]]
    return {
        "n": len(a),
        "accuracy_delta": (len(improvements) - len(regressions)) / len(a),
        "improvements": improvements,
        "regressions": regressions,
        "left": summarize(list(a.values())),
        "right": summarize(list(b.values())),
        "same_code": a_protocol["code_sha256"] == b_protocol["code_sha256"],
        "same_platform": a_protocol["platform"] == b_protocol["platform"],
    }

"""Evidence-grouped, stratified split with a frozen leakage audit."""

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .corpus import file_hash


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", value).lower())


def split_cases(source: Path, destination: Path, seed: int = 20260907, trials: int = 12000) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_file = destination / "manifest.json"
    if manifest_file.exists():
        result = json.loads(manifest_file.read_text(encoding="utf-8"))
        if result["source_sha256"] != file_hash(source):
            raise ValueError("Frozen split source changed")
        for name, digest in result["files"].items():
            if file_hash(destination / name) != digest:
                raise ValueError("Frozen split was modified")
        return result
    cases = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    parent = list(range(len(cases)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[root(a)] = root(b)

    owners, normalized_questions, similar_pairs = {}, [], []
    for i, case in enumerate(cases):
        for reference in case["expected_evidence_refs"]:
            key = normalize(reference.get("source_title") or Path(reference["file_name"]).stem)
            if not key:
                raise ValueError("Missing grouping evidence")
            if key in owners:
                union(i, owners[key])
            owners[key] = i
        question = normalize(case["question"])
        for j, other in enumerate(normalized_questions):
            if min(len(question), len(other)) / max(len(question), len(other), 1) < 0.98:
                continue
            if SequenceMatcher(None, question, other, autojunk=False).ratio() >= 0.98:
                union(i, j)
                similar_pairs.append([cases[j]["id"], case["id"]])
        normalized_questions.append(question)
    groups = defaultdict(list)
    for i, case in enumerate(cases):
        groups[root(i)].append(case)
    grouped = list(groups.values())
    totals = Counter(case["qa_type"] for case in cases)
    strata = Counter((c["source_type"], c["qa_type"], c["difficulty"]) for c in cases)
    rng, best, best_score = random.Random(seed), None, float("inf")
    for _ in range(trials):
        selected = {i for i in range(len(grouped)) if rng.random() < 0.3}
        test = [c for i in selected for c in grouped[i]]
        counts = Counter(c["qa_type"] for c in test)
        if any(not 0 < counts[k] < n for k, n in totals.items()):
            continue
        cross = Counter((c["source_type"], c["qa_type"], c["difficulty"]) for c in test)
        missing = sum(not 0 < cross[k] < n for k, n in strata.items())
        score = missing * 100 + abs(len(test) / len(cases) - 0.3) * 10
        score += sum(abs(counts[k] / n - 0.3) for k, n in totals.items())
        if score < best_score:
            best, best_score = selected, score
    if best is None:
        raise ValueError("Cannot cover every qa_type without group leakage; inspect data, do not duplicate cases")
    partitions = {
        "dev": [c for i, g in enumerate(grouped) if i not in best for c in g],
        "test": [c for i, g in enumerate(grouped) if i in best for c in g],
    }
    files, coverage, cross_coverage = {}, {}, {}
    for name, values in partitions.items():
        path = destination / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in sorted(values, key=lambda c: c["id"])),
            encoding="utf-8",
        )
        files[path.name] = file_hash(path)
        coverage[name] = dict(Counter(c["qa_type"] for c in values))
        cross_coverage[name] = dict(
            Counter(" / ".join((c["source_type"], c["qa_type"], c["difficulty"])) for c in values)
        )
    result = {
        "source_sha256": file_hash(source),
        "seed": seed,
        "groups": len(grouped),
        "counts": {k: len(v) for k, v in partitions.items()},
        "coverage": coverage,
        "cross_coverage": cross_coverage,
        "near_duplicate_pairs": similar_pairs,
        "group_case_ids": [[c["id"] for c in g] for g in grouped],
        "files": files,
        "policy": "source-title groups + >=0.98 question similarity; five types must cover both sets",
        "prior_exposure": "Cases may have been used in the old project; this is not a novel blind test.",
    }
    manifest_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

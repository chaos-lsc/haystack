"""Combine original Evidence and raw legacy Word exports without consulting evaluation labels."""

import argparse
import json
import re
from pathlib import Path

from applications.trust_rag.corpus import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--legacy-word", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file to preserve source provenance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files, count = set(), 0
    exports = []
    with args.output.open("w", encoding="utf-8") as output:
        with args.evidence.open(encoding="utf-8-sig") as source:
            for line in source:
                record = json.loads(line)
                files.add(record.get("metadata", {}).get("file_name"))
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        for path in sorted(args.legacy_word.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if raw["file_name"] in files:
                continue
            if file_hash(Path(raw["source_path"])) != raw["source_sha256"]:
                raise ValueError("Legacy source changed since export")
            stem = re.sub(r"^\d{3}_", "", Path(raw["file_name"]).stem)
            parent, separator, attachment = stem.rpartition("_")
            parent = parent.replace("_", " ").strip() if separator else stem
            title = attachment if separator else stem
            text = raw["text"].replace("\r", "\n").replace("\x07", "\t").strip()
            if not text:
                raise ValueError("Empty legacy export")
            record = {
                "doc_id": "legacy_" + raw["source_sha256"][:16],
                "evidence_id": "legacy_" + raw["source_sha256"],
                "source_title": title,
                "quote": text,
                "location": {},
                "metadata": {
                    "source": "word",
                    "file_name": raw["file_name"],
                    "source_title": parent,
                    "source_sha256": raw["source_sha256"],
                },
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            exports.append({"file_name": raw["file_name"], "sha256": raw["source_sha256"], "characters": len(text)})
    manifest = {
        "original_evidence_sha256": file_hash(args.evidence),
        "merged_evidence_sha256": file_hash(args.output),
        "records": count,
        "legacy_documents": exports,
        "scope": "Original Evidence plus all available legacy .doc sources; no QA workbook or evaluation labels.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"records": count, "legacy_documents": len(exports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

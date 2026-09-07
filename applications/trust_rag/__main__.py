"""Run with `hatch -e trust-rag run lab --help`."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .config import Settings, load_credentials
from .corpus import build_corpus, connect
from .evaluation import compare, evaluate, freeze
from .pipeline import RAG, Variant
from .provider import Provider
from .resources import ResourceMonitor
from .retrieval import build_embeddings, build_qdrant
from .splits import split_cases


def main():
    parser = argparse.ArgumentParser(description="Trust RAG reproducible experiment runner")
    parser.add_argument("--config")
    parser.add_argument("--env-file", help="Local secret file; never copied to outputs")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--evidence", type=Path, required=True)
    split = commands.add_parser("split")
    split.add_argument("--cases", type=Path, required=True)
    commands.add_parser("smoke-api")
    commands.add_parser("estimate")
    warm = commands.add_parser("warm-queries")
    warm.add_argument("--split", choices=["dev", "test"], default="dev")
    embedding = commands.add_parser("embed")
    embedding.add_argument("--workers", type=int, default=4)
    embedding.add_argument("--limit", type=int)
    index = commands.add_parser("index-qdrant")
    index.add_argument("--limit", type=int)
    for name in ("ask", "evaluate", "freeze", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--stage", choices=["B0", "B1", "B2", "B3", "B4"], default="B0")
        command.add_argument("--without", action="append", choices=["bm25", "tables", "verifier"], default=[])
        if name == "ask":
            command.add_argument("question")
            command.add_argument("--baseline-limit", type=int)
        elif name == "evaluate":
            command.add_argument("--split", choices=["dev", "test"], default="dev")
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--concurrency", type=int, choices=[1, 2], default=2)
            command.add_argument("--limit", type=int)
        elif name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
    comparison = commands.add_parser("compare")
    comparison.add_argument("left", type=Path)
    comparison.add_argument("right", type=Path)
    comparison.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings.load(args.config)
    load_credentials(args.env_file)
    if hasattr(args, "stage"):
        variant = Variant.stage(args.stage)
        if args.without:
            variant = replace(
                variant,
                name=args.stage + "-without-" + "-".join(sorted(args.without)),
                **dict.fromkeys(args.without, False),
            )
    if args.command == "prepare":
        with ResourceMonitor() as monitor:
            result = build_corpus(args.evidence, settings.root / "corpus.sqlite")
        (settings.root / "corpus-resources.json").write_text(json.dumps(monitor.result, indent=2), encoding="utf-8")
    elif args.command == "split":
        result = split_cases(args.cases, settings.root / "splits")
    elif args.command == "smoke-api":
        provider = Provider(settings)
        vector = provider.embed(["银行资本充足率监管要求"])[0]
        answer = provider.chat('返回JSON {"ok":true}。', "这是接口兼容性检查。")
        result = {
            "embedding_model": settings.embedding_model,
            "dimensions": len(vector),
            "generation_model": settings.generation_model,
            "json_response": answer,
            "events": list(provider.events),
        }
        (settings.root / "api-smoke.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif args.command == "estimate":
        corpus = json.loads((settings.root / "corpus.manifest.json").read_text(encoding="utf-8"))
        sample = json.loads((settings.root / "embedding-build.json").read_text(encoding="utf-8"))
        with connect(settings.root / "corpus.sqlite") as db:
            chars = db.execute(
                "SELECT SUM(length(content)) FROM (SELECT content FROM evidence ORDER BY rowid LIMIT ?)",
                (sample["count"],),
            ).fetchone()[0]
        used = sum(event.get("usage", {}).get("prompt_tokens", 0) for event in sample["events"])
        result = {
            "sample_chunks": sample["count"],
            "sample_chars": chars,
            "sample_api_tokens": used,
            "estimated_full_embedding_tokens": round(used / chars * corpus["characters"]),
            "raw_vector_bytes": corpus["chunks"] * sample["dimensions"] * 4,
            "pricing": "Account-specific rates must be applied to token counts; no amount invented.",
            "caveat": "Estimate uses initial sample; cached inputs reduce charged usage, actual usage logged.",
        }
        (settings.root / "cost-estimate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "warm-queries":
        if args.split == "test" and not (settings.root / "freeze.json").exists():
            raise ValueError("Freeze the protocol before warming held-out test queries")
        cases = [
            json.loads(line)
            for line in (settings.root / "splits" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        provider = Provider(settings)
        for offset in range(0, len(cases), settings.embedding_batch):
            provider.embed(
                [settings.query_prefix + case["question"] for case in cases[offset : offset + settings.embedding_batch]]
            )
        result = {"split": args.split, "queries": len(cases), "events": list(provider.events)}
        (settings.root / f"warm-queries-{args.split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command in ("embed", "index-qdrant"):
        provider = Provider(settings)
        with ResourceMonitor() as monitor:
            result = (
                build_embeddings(settings, provider, args.workers, args.limit)
                if args.command == "embed"
                else build_qdrant(settings, provider, args.limit)
            )
        result["resources"] = monitor.result
        (settings.root / f"{args.command}-resources.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "ask":
        rag = RAG(settings, variant, baseline_limit=args.baseline_limit)
        try:
            result = rag.ask(args.question)
        finally:
            rag.close()
    elif args.command == "evaluate":
        result = evaluate(settings, variant, args.split, args.output, args.concurrency, args.limit)
    elif args.command == "freeze":
        result = freeze(settings, variant)
    elif args.command == "compare":
        result = compare(args.left, args.right)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.command == "serve":
        import uvicorn

        from .serving import create_app

        uvicorn.run(create_app(settings, variant), host=args.host, port=args.port, workers=1)
        return
    else:
        result = asdict(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

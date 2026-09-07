"""Exercise a running final service with two requests and record whole-host resources."""

import argparse
import json
import platform
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import psutil

from applications.trust_rag.corpus import file_hash
from applications.trust_rag.resources import ResourceMonitor


def oom_count():
    path = Path("/proc/vmstat")
    if not path.exists():
        return None
    values = dict(line.split() for line in path.read_text().splitlines())
    return int(values["oom_kill"]) if "oom_kill" in values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_type < 1 or args.rounds < 1 or args.output.exists():
        parser.error("Positive counts and a new output directory are required")
    groups, selected = Counter(), []
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if groups[case["qa_type"]] < args.per_type:
            selected.append(case)
            groups[case["qa_type"]] += 1
    if len(groups) != 5 or min(groups.values()) != args.per_type:
        parser.error("Workload must cover all five question types with the requested count")
    with urlopen(args.url.rstrip("/") + "/health", timeout=30) as response:
        health = json.load(response)
    if health.get("variant") == "B0":
        parser.error("Resource acceptance is for the final improved service; baseline is local only")
    args.output.mkdir(parents=True)
    initial_oom = oom_count()

    def ask(item):
        repetition, case = item
        start = time.perf_counter()
        row = {"id": case["id"], "qa_type": case["qa_type"], "round": repetition}
        try:
            request = Request(
                args.url.rstrip("/") + "/ask",
                data=json.dumps({"question": case["question"]}, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=600) as response:
                result = json.load(response)
                row["http_status"] = response.status
            row.update(
                status="refused" if result["answer"]["refused"] else "answered",
                correct=result["answer"]["selected_option"] == case["correct_option"],
                api_events=result.get("api_events", []),
            )
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, http_status=getattr(exc, "code", None))
        row["seconds"] = time.perf_counter() - start
        return row

    rows = []
    workload = [(repetition, case) for repetition in range(args.rounds) for case in selected]
    with ResourceMonitor() as monitor, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ask, item) for item in workload]
        with (args.output / "results.jsonl").open("w", encoding="utf-8") as handle:
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"done": len(rows), "total": len(workload), "status": row["status"]}), flush=True)
    final_oom = oom_count()
    report = {
        "health": health,
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "disk": psutil.disk_usage(str(args.output.resolve()))._asdict(),
        "case_file_sha256": file_hash(args.cases),
        "coverage": dict(groups),
        "rounds": args.rounds,
        "concurrency": 2,
        "requests": len(rows),
        "status": dict(Counter(row["status"] for row in rows)),
        "seconds_p50": float(np.percentile([r["seconds"] for r in rows], 50)),
        "seconds_p95": float(np.percentile([r["seconds"] for r in rows], 95)),
        "resources": monitor.result,
        "oom_kills_during_run": None if initial_oom is None or final_oom is None else final_oom - initial_oom,
        "acceptance_claim": False,
        "scope": "Warm two-request workload only. Review together with import, optimization and cold-start records.",
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Render completed, aligned runs without changing the frozen evaluation code."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for directory in args.runs:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        rows = list(map(json.loads, (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()))
        if len({row["id"] for row in rows}) != len(rows) or {r["id"] for r in rows} != set(protocol["case_ids"]):
            raise ValueError(f"Incomplete or duplicate results: {directory}")
        if entries:
            for key in ("settings_sha256", "corpus_sha256", "split_sha256", "cases_sha256", "case_ids"):
                if entries[0][2][key] != protocol[key]:
                    raise ValueError(f"Runs are not aligned: {key}")
        entries.append((directory, report, protocol, {r["id"]: r for r in rows}))
    first = entries[0]
    lines = [
        "# 分阶段实验结果",
        "",
        f"评测集合：{first[1]['split']}；题数：{first[1]['total']}。",
        "正确率分母保留拒答与运行错误；本报告仅覆盖列出的运行。",
        "",
        "| 配置 | 正确/总数 | 正确率 | 拒答 | 错误 | 检索 P95(s) | 问答 P95(s) | 进程峰值 MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, report, _, _ in entries:
        lines.append(
            f"| {report['variant']['name']} | {report['correct']}/{report['total']} | "
            f"{report['accuracy']:.2%} | {report['status'].get('refused', 0)} | "
            f"{report['status'].get('error', 0)} | {report['retrieval_seconds']['p95'] or 0:.3f} | "
            f"{report['seconds']['p95'] or 0:.3f} | {report['resources']['process_rss_peak'] / 2**20:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 分类表现",
            "",
            "| 题型 | " + " | ".join(e[1]["variant"]["name"] for e in entries) + " |",
            "|---|" + "---:|" * len(entries),
        ]
    )
    types = sorted({kind for e in entries for kind in e[1]["by_type"]})
    for kind in types:
        values = []
        for _, report, _, _ in entries:
            row = report["by_type"].get(kind)
            values.append(f"{row['correct']}/{row['total']} ({row['accuracy']:.1%})" if row else "无样本")
        lines.append("| " + kind + " | " + " | ".join(values) + " |")
    lines.extend(["", "## 相邻阶段差异", ""])
    for earlier, later in zip(entries, entries[1:]):
        improved = [i for i in earlier[3] if not earlier[3][i]["correct"] and later[3][i]["correct"]]
        regressed = [i for i in earlier[3] if earlier[3][i]["correct"] and not later[3][i]["correct"]]
        lines.append(
            f"- {earlier[1]['variant']['name']} → {later[1]['variant']['name']}："
            f"改善 {len(improved)} 题，退步 {len(regressed)} 题；"
            f"净变化 {(len(improved) - len(regressed)) / len(earlier[3]):+.2%}。"
        )
        if improved:
            lines.append("  - 改善题目：" + ", ".join(improved))
        if regressed:
            lines.append("  - 退步题目：" + ", ".join(regressed))
    lines.extend(
        [
            "",
            "## 证据与限制",
            "",
            "进程峰值不包括独立 Qdrant 进程，不能用它证明整机 2 GiB 通过。整机指标与运行环境在各报告中单独保存。",
            "检索延迟包含查询向量缓存读取；比较前统一预热开发问题。生成 API 的服务端缓存与远程负载不完全可控。",
            "API 费用按实际 token 与已核实单价计算；未核实的金额不填零。",
            "开发集用于组件选择，不能据此宣称保留测试集提升或统计显著。",
            "",
        ]
    )
    for path, _, protocol, _ in entries:
        lines.append(f"- {path.as_posix()}：代码 {protocol['code_sha256']}；{protocol['platform']}。")
    if len({e[2]["code_sha256"] for e in entries}) != 1:
        lines.append("- 注意：这些运行的代码哈希不同，应检查变更原因；不能直接视为仅组件配置不同的严格消融。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

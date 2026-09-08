"""从 pytest junit xml 生成 Markdown 测报（对位 JD：输出详细准确的测试报告）。

用法: python scripts/gen_report.py reports/junit.xml docs/test-report.md
"""
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse(junit_path: str) -> dict:
    tree = ET.parse(junit_path)
    root = tree.getroot()
    suites = root.iter("testsuite") if root.tag == "testsuites" else [root]
    by_module: dict[str, dict] = defaultdict(lambda: {"total": 0, "failed": 0, "skipped": 0, "time": 0.0})
    for suite in suites:
        for case in suite.iter("testcase"):
            classname = case.get("classname", "unknown")
            module = ".".join(classname.split(".")[:3]) or classname
            row = by_module[module]
            row["total"] += 1
            row["time"] += float(case.get("time", 0))
            if case.find("failure") is not None or case.find("error") is not None:
                row["failed"] += 1
            elif case.find("skipped") is not None:
                row["skipped"] += 1
    return by_module


def render(by_module: dict) -> str:
    total = sum(r["total"] for r in by_module.values())
    failed = sum(r["failed"] for r in by_module.values())
    skipped = sum(r["skipped"] for r in by_module.values())
    passed = total - failed - skipped
    duration = sum(r["time"] for r in by_module.values())
    lines = [
        "# 测试报告",
        "",
        f"- 生成时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        (
            f"- 结论: {'✅ 通过' if failed == 0 else '❌ 存在失败'}"
            f"（通过 {passed} / 失败 {failed} / 跳过 {skipped}，合计 {total}，耗时 {duration:.1f}s）"
        ),
        "",
        "| 模块 | 用例数 | 通过 | 失败 | 跳过 | 耗时(s) |",
        "|---|---|---|---|---|---|",
    ]
    for module in sorted(by_module):
        r = by_module[module]
        p = r["total"] - r["failed"] - r["skipped"]
        lines.append(
            f"| {module} | {r['total']} | {p} | {r['failed']} | {r['skipped']} | {r['time']:.2f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    junit, out = sys.argv[1], sys.argv[2]
    report = render(parse(junit))
    Path(out).write_text(report, encoding="utf-8")
    print(report)
#!/usr/bin/env bash
# 一键运行全量测试并生成测报（Linux CI / 本地 Linux 均可）
# 用法: bash scripts/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== 1/4 构建 C++ 原生 kernel =="
bash scripts/build_native.sh

echo "== 2/4 静态检查 =="
ruff check sut tests scripts

echo "== 3/4 全量测试（生成 junit） =="
python -m pytest tests -v --junitxml=reports/junit.xml

echo "== 4/4 生成 Markdown 测报 =="
python scripts/gen_report.py reports/junit.xml docs/test-report.md
echo "测报: docs/test-report.md"
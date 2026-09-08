#!/usr/bin/env bash
# 构建 C++ 原生算子库 (Linux/g++)。用法: bash scripts/build_native.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
g++ -O2 -shared -fPIC -o "$ROOT/sut/native/libmm.so" "$ROOT/sut/native/matmul.cpp"
echo "built: $ROOT/sut/native/libmm.so"
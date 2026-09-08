"""YAML 用例加载器。"""
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


def load_yaml_cases(file_name: str) -> list[dict]:
    with open(DATA_DIR / file_name, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise TypeError(f"用例文件 {file_name} 顶层必须为数组")
    return data


def case_ids(cases: list[dict]) -> list[str]:
    return [c.get("id", f"case-{i}") for i, c in enumerate(cases)]
"""pytest 根 conftest：保证项目根在 sys.path（使 sut / tests 包可导入）。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
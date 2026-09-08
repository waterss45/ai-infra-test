"""pytest 根 conftest：保证项目根在 sys.path 并集中暴露 fixtures。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.core.native_loader import native_lib  # noqa: E402,F401
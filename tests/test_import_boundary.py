"""邊界測試：契約層與核心層不得依賴子系統，整個 echo_stream 不得 import ``uep``。

守的是「ES 不變成第六個孤島」與「uep -> echo_stream 單向」兩條線。
以 AST 掃描（含函式內的延遲 import），不實際 import 任何模組。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "echo_stream"
STDLIB = set(sys.stdlib_module_names)
"""contracts/ 與 core/ 現況只用標準庫；白名單即標準庫，新增任何第三方或子系統 import 都會失敗。"""
UPPER_LAYERS = {"adapters", "web", "cli", "sinks"}
"""contracts/、core/ 不得匯入這些兄弟套件（adapters 才是子系統的邊界）。"""


def _py_files(sub: str | None = None) -> list[Path]:
    root = PKG / sub if sub else PKG
    return sorted(root.rglob("*.py"))


def _imports(path: Path) -> list[tuple[int, int, str | None]]:
    """回傳 ``(行號, level, 模組名)``；``level>0`` 為相對 import。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, int, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, 0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.lineno, node.level, node.module))
            if node.level > 0 and node.module is None:
                # ``from .. import adapters`` 這種：被匯入的名稱本身就是套件
                found.extend((node.lineno, node.level, alias.name) for alias in node.names)
    return found


def _rel(path: Path) -> str:
    return path.relative_to(PKG.parent).as_posix()


def test_掃描範圍不是空的():
    assert len(_py_files("contracts")) >= 5
    assert len(_py_files("core")) >= 5


@pytest.mark.parametrize("sub", ["contracts", "core"])
def test_契約與核心只依賴標準庫(sub: str):
    violations = []
    for path in _py_files(sub):
        for lineno, level, module in _imports(path):
            if level > 0:
                continue
            top = (module or "").split(".")[0]
            if top == "echo_stream":
                continue
            if top not in STDLIB:
                violations.append(f"{_rel(path)}:{lineno} import {module}")
    msg = "contracts/、core/ 不得 import 子系統或第三方：\n" + "\n".join(violations)
    assert not violations, msg


@pytest.mark.parametrize("sub", ["contracts", "core"])
def test_契約與核心不匯入上層套件(sub: str):
    violations = []
    for path in _py_files(sub):
        for lineno, level, module in _imports(path):
            parts = (module or "").split(".")
            if level == 0:
                if parts[0] == "echo_stream" and len(parts) > 1 and parts[1] in UPPER_LAYERS:
                    violations.append(f"{_rel(path)}:{lineno} import {module}")
            elif parts[0] in UPPER_LAYERS:
                violations.append(f"{_rel(path)}:{lineno} from {'.' * level}{module}")
    assert not violations, "contracts/、core/ 不得匯入 adapters 等上層：\n" + "\n".join(violations)


def test_整個echo_stream不import_uep():
    violations = []
    for path in _py_files():
        for lineno, level, module in _imports(path):
            if level == 0 and (module or "").split(".")[0] == "uep":
                violations.append(f"{_rel(path)}:{lineno} import {module}")
    msg = "ES 不得 import uep（uep -> echo_stream 單向）：\n" + "\n".join(violations)
    assert not violations, msg

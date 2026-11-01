"""config 的 .env parser 測試——重點是多行引號值（SYSTEM_PROMPT 人設）。"""

from __future__ import annotations

import pytest

from echo_stream import config


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """把 _project_root 指到 tmp，寫入指定內容的 .env。"""

    def write(content: str):
        (tmp_path / ".env").write_text(content, encoding="utf-8")
        monkeypatch.setattr(config, "_project_root", lambda: tmp_path)
        config._dotenv_values.cache_clear()
        return tmp_path

    yield write
    config._dotenv_values.cache_clear()


def test_基本鍵值與引號(env_file):
    env_file('A=1\nB="hello"\nC=\'x\'\n# 註解\n\nexport D=4\n')
    assert config.get("A") == "1"
    assert config.get("B") == "hello"
    assert config.get("C") == "x"
    assert config.get("D") == "4"


def test_多行引號值(env_file):
    env_file('SYSTEM_PROMPT="你是 U.E.P。\n  個性活潑。\n說話簡短。"\nAFTER=ok\n')
    prompt = config.get("SYSTEM_PROMPT")
    assert prompt == "你是 U.E.P。\n  個性活潑。\n說話簡短。"
    # 縮排要保留、多行值之後的鍵不受影響
    assert config.get("AFTER") == "ok"


def test_未閉合引號吃到檔尾不炸(env_file):
    env_file('X="沒有閉合\n第二行\nY=1\n')
    # 收不到閉引號就一路收到底——寬鬆處理，別讓整個 .env 掛掉
    assert "第二行" in (config.get("X") or "")


def test_空值不炸(env_file):
    env_file("EMPTY=\nA=1\n")
    assert config.get("EMPTY") is None  # 空字串視同未設定
    assert config.get("A") == "1"


# --- ensure_subsystem：環境已裝的版本優先於同層猜測 ---


def _fake_package(root, name):
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return root


def test_ensure_subsystem_環境已裝就不插同層_repo(tmp_path, monkeypatch):
    import sys

    installed = _fake_package(tmp_path / "installed", "fake_sub_pkg")
    sibling = _fake_package(tmp_path / "sibling", "fake_sub_pkg")
    monkeypatch.setattr(sys, "path", [str(installed), *sys.path])
    monkeypatch.delenv("ECHO_STREAM_FAKE_REPO", raising=False)
    monkeypatch.setattr(config, "_dotenv_values", lambda: {})
    monkeypatch.setattr(config, "subsystem_path", lambda kind: sibling)

    repo = config.ensure_subsystem("FAKE", "fake_sub_pkg")
    assert repo == installed
    assert str(sibling) not in sys.path


def test_ensure_subsystem_明確指定的_repo_優先(tmp_path, monkeypatch):
    import sys

    installed = _fake_package(tmp_path / "installed", "fake_sub_pkg2")
    explicit = _fake_package(tmp_path / "explicit", "fake_sub_pkg2")
    monkeypatch.setattr(sys, "path", [str(installed), *sys.path])
    monkeypatch.setattr(config, "_dotenv_values", lambda: {})

    repo = config.ensure_subsystem("FAKE", "fake_sub_pkg2", explicit)
    assert repo == explicit.resolve()
    assert sys.path[0] == str(explicit.resolve())


def test_ensure_subsystem_沒裝才猜同層(tmp_path, monkeypatch):
    import sys

    sibling = _fake_package(tmp_path / "sibling", "fake_sub_pkg3")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("ECHO_STREAM_FAKE_REPO", raising=False)
    monkeypatch.setattr(config, "_dotenv_values", lambda: {})
    monkeypatch.setattr(config, "subsystem_path", lambda kind: sibling)

    assert config.ensure_subsystem("FAKE", "fake_sub_pkg3") == sibling
    assert sys.path[0] == str(sibling)

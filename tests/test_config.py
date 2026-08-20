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

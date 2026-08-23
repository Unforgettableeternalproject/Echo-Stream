"""`echo_stream.toml`——可隨手調的預設參數。

與 :mod:`config`（``.env``）的分工：``.env`` 放密鑰、路徑、模型名稱這種
「裝機一次」的東西；這裡放語速、切句權重、靜音閾值、墊片文案這種
「每次測試都可能想改」的數值。艾斯維爾 2026-08-23 要求——面板改的值只活
在一次執行，他要有個檔案可以固定自己的預設。

讀檔用標準庫 ``tomllib``（3.11+）；3.10 退回 ``tomli``；都沒有就回空 dict
並 log 一次——設定檔是便利品，沒有它管線照跑。**不寫檔**：面板改值不會
回寫，避免「改了面板結果把檔案覆蓋掉」這種意外。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "echo_stream.toml"


def config_path() -> Path:
    override = os.environ.get("ECHO_STREAM_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / CONFIG_FILENAME


def _parse_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.warning("沒有 tomllib / tomli，%s 不會被讀取", CONFIG_FILENAME)
            return {}
    return tomllib.loads(text)


@lru_cache(maxsize=1)
def load_defaults() -> dict[str, Any]:
    """讀 ``echo_stream.toml``。檔案不存在 → 空 dict（全部走程式碼預設）。

    快取一次：serve 期間不會重讀，改了檔要重啟（跟 .env 一樣）。
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = _parse_toml(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 設定檔壞了不該擋啟動
        logger.warning("%s 解析失敗，忽略：%s: %s", path, type(exc).__name__, exc)
        return {}
    logger.info("載入預設參數：%s（%s）", path, "、".join(data) or "空")
    return data


def section(name: str) -> dict[str, Any]:
    value = load_defaults().get(name)
    return dict(value) if isinstance(value, dict) else {}


__all__ = ["load_defaults", "section", "config_path", "CONFIG_FILENAME"]

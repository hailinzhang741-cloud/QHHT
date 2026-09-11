"""原油(SC) 配置 — 独立于沪铜。"""
from __future__ import annotations

from config import DATA_DIR

OIL_MAIN_CODE = "SC.INE"
OIL_CONT_CODE = "SCL.INE"
OIL_EXCHANGE = "INE"
OIL_SYMBOL = "SC"
OIL_DATA_DIR = DATA_DIR / "oil"
OIL_DATA_DIR.mkdir(parents=True, exist_ok=True)

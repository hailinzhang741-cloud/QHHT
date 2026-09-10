"""模型阈值配置 — 由校准脚本自动优化写入。"""
from __future__ import annotations

import json
from pathlib import Path

from config import DATA_DIR

CONFIG_PATH = DATA_DIR / "results" / "model_config.json"
CONFIG_V5_PATH = DATA_DIR / "results" / "model_config_v5.json"

# 默认阈值（校准前 fallback）
DEFAULT_CONFIG = {
    "version": 4,
    "long_only": True,
    "filter": {
        "1d": {"th_up": 0.62, "th_down": 0.38, "min_agree": 3},
        "5d": {"th_up": 0.54, "th_down": 0.45, "min_agree": 3},
    },
    "calibration_stats": {},
}

DEFAULT_CONFIG_V5 = {
    "version": 5,
    "long_only": True,
    "filter": {
        "1d": {"th_up": 0.62, "th_down": 0.38, "min_agree": 3},
        "5d": {"th_up": 0.57, "th_down": 0.44, "min_agree": 4},
    },
    "quality_filter": {"max_rsi14": 76, "max_price_pos20": 0.92, "min_ret5": -0.08},
    "calibration_stats": {},
}


def load_config(version: int | None = None) -> dict:
    if version == 5:
        if CONFIG_V5_PATH.exists():
            return json.loads(CONFIG_V5_PATH.read_text(encoding="utf-8"))
        return DEFAULT_CONFIG_V5.copy()

    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if version is None or cfg.get("version", 4) == version:
            return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict, version: int | None = None) -> None:
    v = version or cfg.get("version", 4)
    path = CONFIG_V5_PATH if v >= 5 else CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

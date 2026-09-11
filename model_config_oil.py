"""原油模型阈值配置。"""
from __future__ import annotations

import json

from config_oil import OIL_DATA_DIR

CONFIG_PATH = OIL_DATA_DIR / "results" / "model_config_v5.json"

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


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return DEFAULT_CONFIG_V5.copy()


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

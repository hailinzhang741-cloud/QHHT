"""原油(SC) 预测引擎 — 独立于沪铜 engine。"""
from __future__ import annotations

from model_config_oil import load_config


def get_oil_engine():
    import forecast_oil_v5 as engine

    return engine, load_config(), 5

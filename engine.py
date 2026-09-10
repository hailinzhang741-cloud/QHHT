"""预测引擎路由 — v4 / v5 可切换，默认 v4。"""
from __future__ import annotations

from typing import Any

from config import load_env
from model_config import load_config


def resolve_version(version: int | None = None) -> int:
    if version is not None:
        return version
    env = load_env("MODEL_VERSION", "").strip()
    if env.isdigit():
        return int(env)
    cfg = load_config()
    return int(cfg.get("version", 4))


def get_engine(version: int | None = None) -> tuple[Any, dict, int]:
    """返回 (engine_module, config, version)。"""
    v = resolve_version(version)
    if v >= 5:
        import forecast_v5 as engine

        cfg = load_config(version=5)
        return engine, cfg, 5

    import forecast as engine

    cfg = load_config(version=4)
    return engine, cfg, 4


def fetch_macro_data(start: str, use_cache: bool, version: int) -> tuple[Any, Any]:
    if version < 5:
        return None, None
    from macro_features import fetch_cn_pmi_series, fetch_lme_inventory_series

    inv = fetch_lme_inventory_series(start, use_cache=use_cache)
    pmi = fetch_cn_pmi_series(start, use_cache=use_cache)
    return inv, pmi

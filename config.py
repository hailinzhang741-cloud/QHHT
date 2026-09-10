"""Tushare 配置 — 优先从环境变量读取 token。"""
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def load_env(key: str, default: str = "") -> str:
    """从项目 .env 读取配置；文件未设置时再回退系统环境变量。"""
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(f"{key}="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return os.getenv(key, default).strip() or default


def _load_token() -> str:
    """优先读取项目 .env，避免被系统里旧的 TUSHARE_TOKEN 环境变量覆盖。"""
    return load_env("TUSHARE_TOKEN", "")


TUSHARE_TOKEN = _load_token()

# 沪铜合约常量
CU_MULTIPLIER = 5          # 5 吨/手
CU_TICK_SIZE = 10          # 最小变动 10 元/吨
CU_TICK_VALUE = 50         # 1 跳 = 50 元/手
CU_EXCHANGE = "SHFE"
CU_MAIN_CODE = "CU.SHF"    # 主力合约
CU_CONT_CODE = "CUL.SHF"   # 连续合约（主连）

# 推送配置（从 .env 读取，见 notify.py）
# PUSHPLUS_TOKEN / SERVERCHAN_KEY / SMTP_HOST / SMTP_USER / SMTP_PASS / NOTIFY_EMAIL

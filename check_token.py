"""检查 Tushare Token 是否可用。"""
from __future__ import annotations

import os
import sys

import tushare as ts

from config import TUSHARE_TOKEN, ROOT_DIR


def main() -> int:
    print("=" * 50)
    print("  Tushare Token 诊断")
    print("=" * 50)

    env_token = os.getenv("TUSHARE_TOKEN", "").strip()
    file_token = TUSHARE_TOKEN.strip()

    print(f"项目 .env 路径 : {ROOT_DIR / '.env'}")
    print(f".env token 长度 : {len(file_token)}")
    if file_token:
        print(f".env token 前缀 : {file_token[:6]}...{file_token[-6:]}")
    else:
        print(".env token      : (空)")

    if env_token:
        same = env_token == file_token
        print(f"系统环境变量    : 已设置，长度 {len(env_token)}，与 .env {'相同' if same else '不同'}")
        if not same:
            print("  提示: 已修复 config.py，现在优先使用 .env 中的 token。")
    else:
        print("系统环境变量    : 未设置")

    if not file_token or file_token == "your_tushare_token_here":
        print("\n[失败] 请在 .env 中填入真实 token：")
        print("  TUSHARE_TOKEN=从 https://tushare.pro/user/token 复制")
        return 1

    if len(file_token) < 32:
        print(f"\n[警告] token 长度 = {len(file_token)}，可能复制不完整。")
    elif len(file_token) not in (56, 64):
        print(f"\n[提示] token 长度 = {len(file_token)}（常见为 56 或 64 位）。")

    if env_token and env_token != file_token:
        print("\n[提示] 系统环境变量里的 token 与 .env 不一致。")
        print("  当前脚本已优先使用 .env；建议删除系统变量避免混淆：")
        print("  setx TUSHARE_TOKEN \"\"   （或在「系统环境变量」里删除该项）")

    ts.set_token(file_token)
    pro = ts.pro_api(file_token)

    try:
        df = pro.trade_cal(exchange="SHFE", start_date="20250901", end_date="20250905")
        print(f"\n[成功] token 有效，测试接口 trade_cal 返回 {len(df)} 条。")

        df2 = pro.fut_daily(ts_code="CU.SHF", start_date="20250901", end_date="20250905")
        print(f"[成功] 期货接口 fut_daily 返回 {len(df2)} 条。")
        return 0
    except Exception as exc:
        msg = str(exc)
        print(f"\n[失败] Tushare 返回: {msg}")
        print("\n请按以下步骤排查：")
        print("  1. 登录 https://tushare.pro/user/token")
        print("  2. 点击复制完整 token（不要多空格、不要换行）")
        print("  3. 用 notepad .env 粘贴，格式: TUSHARE_TOKEN=xxxx")
        print("  4. 保存后重新运行: python check_token.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())

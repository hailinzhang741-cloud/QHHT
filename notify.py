"""邮件 / 微信推送 — 仅在有效 5 日信号时通知。"""
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import requests

from config import ROOT_DIR, load_env


def _load_env(key: str) -> str:
    return load_env(key, "")


def _push_pushplus_one(token: str, title: str, content: str, to: str = "") -> bool:
    payload: dict = {"token": token, "title": title, "content": content, "template": "txt"}
    if to:
        payload["to"] = to
    r = requests.post("http://www.pushplus.plus/send", json=payload, timeout=15)
    data = r.json() if r.ok else {}
    ok = r.ok and data.get("code") == 200
    if ok:
        print(f"[notify] PushPlus 成功 (token ...{token[-6:]})")
    else:
        msg = data.get("msg", r.text[:120])
        print(f"[notify] PushPlus 失败 (token ...{token[-6:]}): {msg}")
        if data.get("code") == 905:
            print("[notify] 请到 https://verify.pushplus.plus 完成实名，或改用 Server酱/邮件")
    return ok


def _push_pushplus(title: str, content: str) -> bool:
    """
    支持两种多微信号配置（可组合）：
    1. PUSHPLUS_TOKEN=token1,token2  — 各自 PushPlus 账号，各收一条
    2. PUSHPLUS_TOKEN=主token + PUSHPLUS_TO=好友令牌1,好友令牌2 — 主账号推送给好友
    """
    tokens = [t.strip() for t in _load_env("PUSHPLUS_TOKEN").split(",") if t.strip()]
    if not tokens:
        return False
    to = _load_env("PUSHPLUS_TO").strip()
    sent_any = False
    for i, token in enumerate(tokens):
        # 仅第一个 token 使用 PUSHPLUS_TO（好友消息由发送方账号发出）
        friend_to = to if i == 0 and to else ""
        if _push_pushplus_one(token, title, content, friend_to):
            sent_any = True
    return sent_any


def _push_serverchan(title: str, content: str) -> bool:
    key = _load_env("SERVERCHAN_KEY")
    if not key:
        return False
    r = requests.post(
        f"https://sctapi.ftqq.com/{key}.send",
        data={"title": title, "desp": content},
        timeout=15,
    )
    ok = r.ok and r.json().get("code") == 0
    print(f"[notify] Server酱 {'成功' if ok else '失败'}: {r.text[:120]}")
    return ok


def _send_email(title: str, content: str) -> bool:
    host = _load_env("SMTP_HOST")
    user = _load_env("SMTP_USER")
    password = _load_env("SMTP_PASS")
    to_addr = _load_env("NOTIFY_EMAIL")
    port = int(_load_env("SMTP_PORT") or "465")
    if not all([host, user, password, to_addr]):
        return False

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = title
    msg["From"] = user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP_SSL(host, port, timeout=20) as server:
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
        print(f"[notify] 邮件已发送至 {to_addr}")
        return True
    except Exception as exc:
        print(f"[notify] 邮件失败: {exc}")
        return False


def _prob_line(prob_up: float, prob_down: float, signal: str) -> str:
    up = prob_up * 100
    down = prob_down * 100
    line = f"涨{up:.1f}%  跌{down:.1f}%"
    if signal == "观望":
        line += "  观望"
    return line


def _format_references(sig: dict) -> list[str]:
    """提炼推送用的参考依据（简明）。"""
    tech = f"RSI{sig.get('RSI14', '-')} {sig.get('MA5/MA20', '-')} 20日位{sig.get('20日区间位置', '-')}%"
    fund = f"仓单{sig.get('仓单信号', '中性')} 持仓{sig.get('持仓信号', '中性')}"
    ratio = f"沪伦比{sig.get('沪伦比', '-')} {sig.get('沪伦比信号', '')}"
    macro = f"LME库存{sig.get('LME库存5日变化_%', '-')}% {sig.get('LME库存信号', '')} PMI{sig.get('制造业PMI', '-')} {sig.get('PMI信号', '')}"
    subs = (
        f"技术{sig.get('蒙特卡洛_5日', '-')}% "
        f"基本面{sig.get('基本面_上涨率', '-')}% "
        f"沪伦比{sig.get('沪伦比_上涨率', '-')}% "
        f"宏观{sig.get('宏观_上涨率', '-')}% "
        f"({sig.get('子模型看多_5日', '-')})"
    )
    return [
        f"技术面 {tech}",
        f"基本面 {fund}",
        f"比价 {ratio.strip()}",
        f"宏观 {macro.strip()}",
        f"子模型涨率 {subs}",
    ]


def format_notify_content(result) -> tuple[str, str]:
    """推送正文：涨跌概率 + 参考依据；不确定时备注「观望」。"""
    sig = result.signals
    p1 = result.prob_up_1d * 100
    p5 = result.prob_up_5d * 100
    title = f"沪铜v5 | {result.as_of_date} | 明日涨{p1:.0f}% 周涨{p5:.0f}%"

    lines = [
        f"沪铜 v5  {result.as_of_date}",
        f"收盘 {result.close:,.0f} 元/吨",
        "",
        f"明日: {_prob_line(result.prob_up_1d, result.prob_down_1d, result.signal_1d)}",
        f"一周: {_prob_line(result.prob_up_5d, result.prob_down_5d, result.signal_5d)}",
        "",
        "参考依据:",
    ]
    lines.extend(_format_references(sig))
    return title, "\n".join(lines)


def daily_notify_enabled() -> bool:
    """DAILY_NOTIFY=1 时每个交易日都推送概率报告（适合云端定时）。"""
    return load_env("DAILY_NOTIFY", "").strip().lower() in ("1", "true", "yes")


def notify_if_signal(result, force: bool = False) -> bool:
    """
    默认仅当 5 日信号为「做多」时推送。
    force=True 或 DAILY_NOTIFY=1 时每日推送概率报告。
    """
    force = force or daily_notify_enabled()
    if result.signal_5d != "做多" and not force:
        print(f"[notify] 5日信号={result.signal_5d}，跳过推送")
        return False

    title, content = format_notify_content(result)

    sent = any([
        _push_pushplus(title, content),
        _push_serverchan(title, content),
        _send_email(title, content),
    ])
    if not sent:
        print("[notify] 未配置推送渠道（PUSHPLUS_TOKEN / SERVERCHAN_KEY / SMTP）")
    return sent


def test_notify() -> None:
    """测试推送配置。"""
    title = "沪铜预测推送测试"
    content = "如果你收到这条消息，说明推送配置成功。"
    _push_pushplus(title, content)
    _push_serverchan(title, content)
    _send_email(title, content)

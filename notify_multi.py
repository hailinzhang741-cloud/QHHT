"""沪铜 + 原油 联合推送 — 不修改 notify.py 沪铜逻辑。"""
from __future__ import annotations

from config import load_env
from notify import (
    _format_references,
    _prob_line,
    _push_pushplus,
    _push_serverchan,
    _send_email,
    daily_notify_enabled,
)
from notify_state import already_pushed, dedupe_run_date, resolve_run_slot, save_state
from intraday_snapshot import IntradayBundle
from trading_labels import format_1d_horizon_label, format_5d_horizon_label, format_push_header


def _format_oil_references(sig: dict) -> list[str]:
    tech = f"RSI{sig.get('RSI14', '-')} {sig.get('MA5/MA20', '-')} 20日位{sig.get('20日区间位置', '-')}%"
    fund = f"仓单{sig.get('仓单信号', '中性')} 持仓{sig.get('持仓信号', '中性')}"
    ratio = f"SC/WTI比{sig.get('SC/WTI比', '-')} z={sig.get('ratio_z60', '-')}"
    macro = (
        f"WTI{sig.get('WTI前日_%', '-')}% "
        f"美库存{sig.get('美库存5日_%', '-')}% "
        f"PMI{sig.get('制造业PMI', '-')}"
    )
    subs = (
        f"技术{sig.get('蒙特卡洛_5日', '-')}% "
        f"基本面{sig.get('基本面_上涨率', '-')}% "
        f"SC/WTI{sig.get('SC/WTI比_上涨率', '-')}% "
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


def _intraday_lines(snap, unit: str) -> list[str]:
    if snap is None:
        return []
    return [snap.format_line(unit=unit)]


def _horizon_block(result, as_of_date: str) -> list[str]:
    label_1d, note_1d = format_1d_horizon_label(as_of_date)
    label_5d, note_5d = format_5d_horizon_label(as_of_date)
    block = [
        f"{label_1d}: {_prob_line(result.prob_up_1d, result.prob_down_1d, result.signal_1d)}",
        f"{label_5d}: {_prob_line(result.prob_up_5d, result.prob_down_5d, result.signal_5d)}",
    ]
    notes = [n for n in (note_1d, note_5d) if n]
    if notes:
        block.append("※ " + "；".join(dict.fromkeys(notes)))
    return block


def format_multi_content(
    cu_result,
    oil_result,
    slot: str,
    intraday: IntradayBundle | None = None,
) -> tuple[str, str]:
    cu_sig = cu_result.signals
    oil_sig = oil_result.signals
    slot_label = f"{slot[:2]}:{slot[2:]}" if len(slot) == 4 else slot
    header = format_push_header(cu_result.as_of_date, slot)
    title = (
        f"期货v5 {slot_label} | 铜1d{cu_result.prob_up_1d*100:.0f}% 油1d{oil_result.prob_up_1d*100:.0f}%"
    )
    show_intraday = intraday is not None and (intraday.cu or intraday.oil)
    lines = [
        header,
        "（概率=纯v5日K模型；盘中现价仅供参考，不修正概率）",
        "",
        f"沪铜 v5  日K截至 {cu_result.as_of_date}",
        f"日K收盘 {cu_result.close:,.0f} 元/吨",
    ]
    if show_intraday and intraday.cu:
        lines.extend(_intraday_lines(intraday.cu, "元/吨"))
    lines.append("")
    lines.extend(_horizon_block(cu_result, cu_result.as_of_date))
    lines.extend(["", "参考依据:"])
    lines.extend(_format_references(cu_sig))
    lines.extend([
        "",
        f"原油 v5  日K截至 {oil_result.as_of_date}",
        f"日K收盘 {oil_result.close:,.1f} 元/桶",
    ])
    if intraday and intraday.oil:
        lines.extend(_intraday_lines(intraday.oil, "元/桶"))
    if intraday and intraday.wti:
        lines.append(
            f"WTI实时 {intraday.wti.last:.2f} USD  "
            f"今开→现价 {intraday.wti.session_ret_open_pct:+.2f}%  ({intraday.wti.source})"
        )
    lines.append("")
    lines.extend(_horizon_block(oil_result, oil_result.as_of_date))
    lines.extend(["", "参考依据:"])
    lines.extend(_format_oil_references(oil_sig))
    return title, "\n".join(lines)


def notify_multi_if_signal(
    cu_result,
    oil_result,
    force: bool = False,
    source: str = "local",
    check_dedupe: bool = True,
    slot: str | None = None,
    intraday: IntradayBundle | None = None,
) -> bool:
    slot = slot or resolve_run_slot()
    daily = daily_notify_enabled()
    any_long = cu_result.signal_5d == "做多" or oil_result.signal_5d == "做多"
    if not any_long and not force and not daily:
        print(f"[notify] 铜5日={cu_result.signal_5d} 油5日={oil_result.signal_5d}，跳过推送")
        return False

    run_date = dedupe_run_date()
    dedupe_key = f"{run_date}_{slot}"
    if check_dedupe and not force and already_pushed(cu_result.as_of_date, slot=slot, run_date=run_date):
        print(f"[notify] {dedupe_key} 已推送，跳过 ({source})")
        return False

    title, content = format_multi_content(cu_result, oil_result, slot, intraday=intraday)
    sent = any([
        _push_pushplus(title, content),
        _push_serverchan(title, content),
        _send_email(title, content),
    ])
    if not sent:
        print("[notify] 未配置推送渠道（PUSHPLUS_TOKEN / SERVERCHAN_KEY / SMTP）")
        return False

    save_state(cu_result.as_of_date, source, slot=slot, run_date=run_date)
    return sent

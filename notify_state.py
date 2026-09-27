"""推送去重 — 本地/云端共享 notify_state.json，避免重复推送。"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import requests

from config import BJT, DATA_DIR, load_env, now_bjt

# 各时段有效推送窗口（北京时间）；窗口外写入的状态不阻挡定时推送
SLOT_WINDOWS_BJT: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "0840": ((8, 0), (9, 14)),
    "1030": ((10, 0), (11, 14)),
    "1415": ((14, 0), (15, 14)),
    "2050": ((20, 30), (21, 14)),
}

STATE_REL_PATH = "data/results/notify_state.json"
STATE_LOCAL = DATA_DIR / "results" / "notify_state.json"
DEFAULT_REPO = "hailinzhang741-cloud/QHHT"


def _repo() -> str:
    return load_env("GITHUB_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO


def _branch() -> str:
    return load_env("GITHUB_BRANCH", "main").strip() or "main"


def _github_token() -> str:
    return load_env("GITHUB_PAT", "") or load_env("GITHUB_TOKEN", "")


def fetch_remote_state() -> dict:
    owner, name = _repo().split("/", 1)
    url = f"https://raw.githubusercontent.com/{owner}/{name}/{_branch()}/{STATE_REL_PATH}"
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            return json.loads(r.text)
    except Exception:
        pass
    return {}


def resolve_run_slot() -> str:
    """时段标识: 0840 / 1030 / 1415 / 2050；未设置时按当前北京时间推断。"""
    slot = load_env("RUN_SLOT", "").strip()
    if slot:
        return slot.replace(":", "")
    now = now_bjt()
    hhmm = now.hour * 100 + now.minute
    if hhmm < 930:
        return "0840"
    if hhmm < 1200:
        return "1030"
    if hhmm < 1700:
        return "1415"
    return "2050"


def dedupe_run_date() -> str:
    """去重按「推送运行日 + 时段」，不按数据 as_of_date（跨日同一 as_of 仍要推）。"""
    return now_bjt().strftime("%Y%m%d")


def _state_key(slot: str | None = None, run_date: str | None = None) -> str:
    slot = slot or resolve_run_slot()
    run_date = run_date or dedupe_run_date()
    return f"{run_date}_{slot}"


def _parse_pushed_at(iso: str) -> datetime | None:
    try:
        if iso.endswith("Z"):
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(BJT)
    except Exception:
        return None


def _in_slot_window(slot: str, dt_bjt: datetime) -> bool:
    window = SLOT_WINDOWS_BJT.get(slot)
    if not window:
        return True
    (sh, sm), (eh, em) = window
    start = dt_bjt.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = dt_bjt.replace(hour=eh, minute=em, second=59, microsecond=999999)
    return start <= dt_bjt <= end


def _slot_push_counts(entry: dict, slot: str, run_date: str | None) -> bool:
    """该时段记录是否算「已在正确时间窗口内推送过」。"""
    if not entry.get("pushed"):
        return False
    pushed_at = entry.get("pushed_at")
    if not pushed_at:
        return True
    dt = _parse_pushed_at(pushed_at)
    if dt is None:
        return True
    if run_date and dt.strftime("%Y%m%d") != run_date:
        return False
    return _in_slot_window(slot, dt)


def load_state() -> dict:
    remote = fetch_remote_state()
    if remote.get("as_of_date") or remote.get("slots"):
        return remote
    if STATE_LOCAL.exists():
        try:
            return json.loads(STATE_LOCAL.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _slot_window_end_bjt(slot: str, run_date: str) -> datetime | None:
    window = SLOT_WINDOWS_BJT.get(slot)
    if not window:
        return None
    (_, _), (eh, em) = window
    day = datetime.strptime(run_date, "%Y%m%d")
    return day.replace(hour=eh, minute=em, second=59, tzinfo=BJT)


def missed_slots_today(reference: datetime | None = None) -> list[str]:
    """今日已过窗口结束时间、但窗口内尚未成功推送的时段（用于开机补推）。"""
    now = reference or now_bjt()
    if now.weekday() >= 5:
        return []
    run_date = now.strftime("%Y%m%d")
    state = load_state()
    slots_map = state.get("slots") or {}
    missed: list[str] = []
    for slot in ("0840", "1030", "1415", "2050"):
        end = _slot_window_end_bjt(slot, run_date)
        if end is None or now <= end:
            continue
        key = f"{run_date}_{slot}"
        entry = slots_map.get(key) or {}
        if not _slot_push_counts(entry, slot, run_date):
            missed.append(slot)
    return missed


def already_pushed(as_of_date: str, slot: str | None = None, run_date: str | None = None) -> bool:
    state = load_state()
    slot = slot or resolve_run_slot()
    key = _state_key(slot, run_date)
    slots = state.get("slots") or {}
    if key in slots and _slot_push_counts(slots[key], slot, run_date):
        return True
    # 兼容旧版 as_of_date 键（20260911_0840）
    legacy_key = f"{as_of_date}_{slot}"
    if legacy_key != key and legacy_key in slots and _slot_push_counts(slots[legacy_key], slot, run_date):
        return True
    if not slots and str(state.get("as_of_date")) == str(as_of_date) and bool(state.get("pushed")):
        legacy_slot = state.get("slot") or "0840"
        if legacy_slot == slot and str(as_of_date) == dedupe_run_date():
            return True
    return False


def save_state(as_of_date: str, source: str, slot: str | None = None, run_date: str | None = None) -> bool:
    slot = slot or resolve_run_slot()
    run_date = run_date or dedupe_run_date()
    key = _state_key(slot, run_date)
    prev = load_state()
    slots = dict(prev.get("slots") or {})
    slots[key] = {
        "pushed": True,
        "source": source,
        "as_of_date": str(as_of_date),
        "pushed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    payload = {
        "as_of_date": str(as_of_date),
        "run_date": run_date,
        "slot": slot,
        "pushed": True,
        "source": source,
        "pushed_at": slots[key]["pushed_at"],
        "slots": slots,
    }
    STATE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    STATE_LOCAL.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    token = _github_token()
    if not token:
        print(f"[notify] 未配置 GITHUB_PAT，仅写本地状态 ({source})")
        return False

    owner, name = _repo().split("/", 1)
    api = f"https://api.github.com/repos/{owner}/{name}/contents/{STATE_REL_PATH}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    sha = None
    try:
        meta = requests.get(api, headers=headers, params={"ref": _branch()}, timeout=15)
        if meta.ok:
            sha = meta.json().get("sha")
    except Exception:
        pass

    body = {
        "message": f"notify state {key} via {source}",
        "content": base64.b64encode(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")).decode(
            "ascii"
        ),
        "branch": _branch(),
    }
    if sha:
        body["sha"] = sha

    try:
        r = requests.put(api, headers=headers, json=body, timeout=20)
        if r.ok:
            print(f"[notify] 已记录推送状态 -> GitHub ({source}, {as_of_date})")
            return True
        print(f"[notify] GitHub 状态写入失败: {r.status_code} {r.text[:120]}")
    except Exception as exc:
        print(f"[notify] GitHub 状态写入异常: {exc}")
    return False

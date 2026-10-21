"""日→週→月のルーティン用 計算部品（Streamlit から切り離した純粋関数）

    毎日（3分）/nisshi : 1件書く → 機械がタグを提案 → 人が OK／直し        … 事実を記録する
    毎週（10分）/shukei: 今週の候補を原文で読み、本当に困りごとかを判定   … 意味を確かめる
    毎月（30分）/tsuki : 月の推定と傾向を見て、準備度を答え、手を打つ順番を決める … 行動を決める

時間の粒度が上がるほど「機械の仕事」が減り「人の判断」が増える、という順に並べている。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

import diary_core as C

# ---------------------------------------------------------------- 毎日：タグ提案
TAG_CAND = "困りごと候補"
TAG_NEG = "否定の可能性"
TAG_NONE = "一致なし"
CONFIRM_OK = "OK"
CONFIRM_TO_CAND = "直し→候補にする"
CONFIRM_TO_NONE = "直し→該当なし"
CONFIRM_OPTS = [CONFIRM_OK, CONFIRM_TO_CAND, CONFIRM_TO_NONE]

TAG_CHECK_COLS = ["key", "語群", "機械の提案", "人の確認", "確定", "記録日時"]


def propose_tags(text: str, groups: list[C.KwGroup], use_syn: bool, neg_re) -> pd.DataFrame:
    """1件の日誌に、語群ごとのタグ案を出す（機械がやるのは「一致」と「否定の可能性」まで）"""
    rows = []
    for g in groups:
        if not g.base.strip():
            continue
        occ = C.occurrences(text, g.terms(use_syn), neg_re)
        if not occ:
            prop, ctx, words = TAG_NONE, "", ""
        else:
            prop = TAG_NEG if any(o.neg for o in occ) else TAG_CAND
            a, k, b = C.kwic(text, occ[0], 8)
            ctx = f"{a}【{k}】{b}"
            words = "・".join(dict.fromkeys(o.t for o in occ))
        rows.append({"語群": g.key, "語": g.base, "一致した語": words, "文脈": ctx,
                     "機械の提案": prop, "人の確認": CONFIRM_OK})
    return pd.DataFrame(rows, columns=["語群", "語", "一致した語", "文脈", "機械の提案", "人の確認"])


def confirmed_flag(proposal: str, confirm: str) -> int:
    """人の確認を反映した確定タグ（1＝困りごと候補として週次へ回す）"""
    if confirm == CONFIRM_TO_CAND:
        return 1
    if confirm == CONFIRM_TO_NONE:
        return 0
    return 1 if proposal == TAG_CAND else 0


def candidate_masks(df: pd.DataFrame, groups: list[C.KwGroup], tag_checks: pd.DataFrame):
    """週次に回す候補。日次で人が確認したタグがあればそれを、なければルール（一致かつ否定なし）を使う

    戻り値: ({語群: bool Series}, {語群: 出どころ Series「日次確認」/「ルール」/""})
    """
    tc = {}
    if tag_checks is not None and len(tag_checks):
        last = tag_checks.drop_duplicates(["key", "語群"], keep="last")
        tc = {(r.key, r.語群): int(r.確定) for r in last.itertuples()}
    masks, srcs = {}, {}
    for g in groups:
        if not g.base.strip():
            continue
        m, s = [], []
        for _, r in df.iterrows():
            if not isinstance(r["本文"], str):
                m.append(False)
                s.append("")
                continue
            k = (r["key"], g.key)
            if k in tc:
                m.append(bool(tc[k]))
                s.append("日次確認")
            else:
                m.append(bool(r[f"hit_{g.key}"] and not r[f"neg_{g.key}"]))
                s.append("ルール" if r[f"hit_{g.key}"] else "")
        masks[g.key] = pd.Series(m, index=df.index, dtype=bool)
        srcs[g.key] = pd.Series(s, index=df.index, dtype="string")
    return masks, srcs


# ---------------------------------------------------------------- 時間の軸（日・週・月）
def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def week_start(d) -> date:
    """月曜はじまりの週の初日"""
    d = pd.Timestamp(d).date()
    return d - timedelta(days=d.weekday())


def add_period_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = to_date(out["日付"])
    ws = (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    out["週"] = ws.fillna("不明").astype("string")
    if "月" not in out:
        out["月"] = out["日付"].str.slice(0, 7).fillna("不明")
    return out


def week_label(ws: str) -> str:
    if ws == "不明":
        return "日付不明"
    s = pd.Timestamp(ws)
    e = s + pd.Timedelta(days=6)
    return f"{s.month}/{s.day}〜{e.month}/{e.day}"


def entry_days(df: pd.DataFrame) -> set[date]:
    return set(to_date(df["日付"]).dropna().dt.date)


def streak(days: set[date], ref: date) -> int:
    """ref（今日）から遡って毎日書けている日数。今日がまだなら昨日から数える"""
    d = ref if ref in days else ref - timedelta(days=1)
    n = 0
    while d in days:
        n += 1
        d -= timedelta(days=1)
    return n


def rhythm_table(df: pd.DataFrame, ref: date, weeks: int = 12) -> pd.DataFrame:
    """記入リズム：直近 weeks 週の 曜日×週 の記入件数（カレンダー型ヒートマップ用）"""
    start = week_start(ref) - timedelta(weeks=weeks - 1)
    d = to_date(df["日付"]).dt.date
    cnt = d.value_counts()
    rows = []
    for i in range((ref - start).days + 1):
        day = start + timedelta(days=i)
        rows.append({"日": pd.Timestamp(day), "週": pd.Timestamp(week_start(day)), "曜日": "月火水木金土日"[day.weekday()],
                     "曜日番号": day.weekday(), "件数": int(cnt.get(day, 0))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 毎週：候補と判定
def week_candidates(df: pd.DataFrame, masks: dict, labels: dict, ws: str, carry_over: bool = False) -> pd.DataFrame:
    """その週（＋任意で未判定の持ち越し）の候補を、日誌×語群の行で返す"""
    rows = []
    for i, r in df.iterrows():
        in_week = r["週"] == ws
        older = r["週"] != "不明" and r["週"] < ws
        if not (in_week or (carry_over and older)):
            continue
        for g, m in masks.items():
            if not m[i]:
                continue
            lab = labels.get(f"{r['key']}:{g}")
            if older and lab is not None:
                continue
            rows.append({"idx": i, "語群": g, "持ち越し": bool(older), "判定": lab})
    return pd.DataFrame(rows, columns=["idx", "語群", "持ち越し", "判定"])


def period_counts(df: pd.DataFrame, masks: dict, col: str) -> pd.DataFrame:
    """週（または月）× 語群 の候補件数と、本文ありの件数"""
    base = df[df["本文"].notna()]
    out = base.groupby(col).size().rename("本文あり").to_frame()
    for g, m in masks.items():
        out[g] = m[base.index].groupby(base[col]).sum().astype(int)
    return out.drop(index="不明", errors="ignore").sort_index()


# ---------------------------------------------------------------- 毎月：推定・傾向・準備度・順番
READINESS = {0: "0 手がない（誰も決めていない）", 1: "1 担当は決まっている",
             2: "2 手順が書いてある", 3: "3 試して効いた"}


def month_stat(df: pd.DataFrame, mask: pd.Series, month: str, prev: str | None, seed: int = 0) -> dict:
    """その月の困りごと割合を Beta(1,1) で推定し、前月より高い事後確率を出す（分母は本文あり）"""
    rng = np.random.default_rng(seed)
    base = df["本文"].notna()

    def kn(mo):
        sel = base & (df["月"] == mo)
        return int((mask & sel).sum()), int(sel.sum())

    k, n = kn(month)
    a = rng.beta(1 + k, 1 + n - k, 4000)
    res = {"k": k, "n": n, "推定": (1 + k) / (2 + n),
           "下限90": float(np.quantile(a, .05)), "上限90": float(np.quantile(a, .95)),
           "前月k": None, "前月n": None, "前月推定": None, "P(前月より高い)": None}
    if prev:
        k0, n0 = kn(prev)
        b = rng.beta(1 + k0, 1 + n0 - k0, 4000)
        res.update({"前月k": k0, "前月n": n0, "前月推定": (1 + k0) / (2 + n0), "P(前月より高い)": float((a > b).mean())})
    return res


def priority(likelihood: float, readiness: int, rising: float | None = None, w_trend: float = 0.5) -> float:
    """手を打つ順番の目安 = 起きやすさ ×（1 − 準備度/3）×（1 + w×増える見込み）

    増える見込み = P(前月より高い) − 0.5 を 0 以上に切ったもの。重みは慣習で、正解ではない。
    """
    trend = 0.0 if rising is None else max(0.0, rising - 0.5)
    return round(likelihood * (1 - readiness / 3) * (1 + w_trend * trend), 4)


def priority_table(stats: dict, readiness: dict, names: dict, w_trend: float = 0.5) -> pd.DataFrame:
    rows = []
    for g, s in stats.items():
        r = int(readiness.get(g, 0))
        rows.append({"語群": g, "困りごと": names.get(g, g), "今月": f"{s['k']}/{s['n']}", "起きやすさ": round(s["推定"], 3),
                     "P(前月より高い)": None if s["P(前月より高い)"] is None else round(s["P(前月より高い)"], 2),
                     "準備度": r, "優先度（目安）": priority(s["推定"], r, s["P(前月より高い)"], w_trend)})
    t = pd.DataFrame(rows).sort_values("優先度（目安）", ascending=False).reset_index(drop=True)
    t.insert(0, "提案順", range(1, len(t) + 1))
    return t


MONTH_COLS = ["月", "語群", "準備度", "人が決めた順番", "打つ手", "担当", "期限", "記録日時"]


def review_decisions(df: pd.DataFrame, masks: dict, decisions: pd.DataFrame, month: str) -> pd.DataFrame:
    """前月に決めた手と、今月の割合を並べる（前後関係であって、効いたかどうかの証明ではない）"""
    if decisions is None or not len(decisions):
        return pd.DataFrame()
    prev = (pd.Period(month, "M") - 1).strftime("%Y-%m")
    dec = decisions[decisions["月"] == prev].drop_duplicates(["語群"], keep="last")
    rows = []
    for r in dec.itertuples():
        if r.語群 not in masks:
            continue
        a = month_stat(df, masks[r.語群], prev, None)
        b = month_stat(df, masks[r.語群], month, None)
        rows.append({"語群": r.語群, "前月に決めた手": r.打つ手, "担当": r.担当, "準備度（前月）": r.準備度,
                     "前月": f"{a['k']}/{a['n']}", "今月": f"{b['k']}/{b['n']}",
                     "変化(ポイント)": round((b["推定"] - a["推定"]) * 100, 1)})
    return pd.DataFrame(rows)

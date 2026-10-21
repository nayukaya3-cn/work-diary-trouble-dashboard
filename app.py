"""作業日誌 困りごと分析ダッシュボード — 日→週→月のルーティン版

    毎日（3分） /nisshi  書く → 機械のタグ案に OK／直し
    毎週（10分）/shukei  候補の日誌が本当に困りごとかを判定する
    毎月（30分）/tsuki   準備度を答え、手を打つ順番を決める

起動:  streamlit run app.py   → http://localhost:8501/nisshi のように、ページを URL で直接開ける
データ: 既定は架空サンプル50件。実データは手元（data/raw/、.gitignore 済み）でのみ扱う（README参照）。
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
import streamlit as st

import diary_core as C
import routine_core as R
import store as S
from views import deep, learn, routine
from views.common import BASIS_JA, Ctx

st.set_page_config(page_title="作業日誌 困りごと分析", page_icon="🌾", layout="wide")

SRC_SAMPLE = "架空サンプル（50件）"
SRC_FILE = "手元の日誌ファイル（data/raw/diary.csv）"
SRC_UP = "CSV/TSV をアップロード（一時）"


# ============================================================ サイドバー：設定
def sidebar():
    sb = st.sidebar
    with sb.expander("⚙️ データ", expanded=False):
        src = st.radio("読み込むデータ", [SRC_SAMPLE, SRC_FILE, SRC_UP], key="src",
                       help="毎日書き足すなら「手元の日誌ファイル」。記録は data/raw/ に保存され、リポジトリには入りません。")
        up = None
        if src == SRC_UP:
            up = st.file_uploader("列：日付・作業・記入者・本文（作業時間は任意）", type=["csv", "tsv", "txt"])
            st.caption("⚠️ 公開サーバー上のアプリに実データを上げない。実データは手元で `streamlit run` して使う。")
        if src != SRC_FILE:
            st.caption("この画面で足した日誌・判定は**このセッション内だけ**に残ります（各ページからCSVで保存できます）。")
        names = st.text_area("置換する氏名（1行1つ）", "\n".join(C.SAMPLE_NAMES), height=90)
        orgs = st.text_area("置換する取引先（1行1つ）", "\n".join(C.SAMPLE_ORGS), height=70)
        auto = st.checkbox("辞書にない「〇〇さん」「〇〇青果」・電話・メールも自動置換", True)
        ref = st.date_input("基準日（今日）", date.today(), help="毎日・毎週・毎月の「今」をどこに置くか")
    with sb.expander("⚙️ キーワード", expanded=False):
        groups = []
        for i, g in enumerate(C.DEFAULT_GROUPS):
            c1, c2 = st.columns([1, 2])
            base = c1.text_input(f"語{i+1}", g.base, key=f"kb{i}")
            syn = c2.text_input("言い換え", "、".join(g.syn), key=f"ks{i}")
            groups.append(C.KwGroup(g.key, base, [s.strip() for s in re.split(r"[,、，]", syn) if s.strip()]))
        use_syn = st.checkbox("言い換え（同義語）も数える", False)
        neg = st.text_input("否定・解消の表現（語の直後12文字・句点まで）", C.DEFAULT_NEG)
    with sb.expander("⚙️ 数える基準（学ぶ・深掘り）", expanded=False):
        basis = st.radio("何を「困りごと」と数えるか", list(BASIS_JA), format_func=BASIS_JA.get, key="basis")
    return (src, up, [n.strip() for n in names.splitlines() if n.strip()],
            [o.strip() for o in orgs.splitlines() if o.strip()], auto, ref, groups, use_syn, neg, basis)


src, up, names, orgs, auto, ref, groups, use_syn, neg_pat, basis = sidebar()
neg_re = C.compile_neg(neg_pat)
if neg_re is None and neg_pat.strip():
    st.sidebar.error("否定表現の正規表現に誤りがあります（否定チェックは無効になっています）。")

# ============================================================ データ準備
store = S.Store(persist=(src == SRC_FILE))
if src == SRC_FILE:
    S.ensure_diary_file()
    base_df = C.read_diary(S.DIARY_PATH)
    is_sample, sig = False, ("file",)
elif src == SRC_UP and up is not None:
    base_df = C.read_diary(up)
    is_sample, sig = False, ("upload", up.name, up.size)
else:
    if src == SRC_UP:
        st.sidebar.info("ファイル未選択のため、サンプルを表示しています。")
    base_df = C.load_sample()
    is_sample, sig = True, ("sample",)

if st.session_state.get("data_sig") != sig:          # データの出どころが変わったら記録を入れ替える
    st.session_state.data_sig = sig
    for k in ("_store", "_photos", "_added"):
        st.session_state.pop(k, None)
    store = S.Store(persist=(src == SRC_FILE))
    st.session_state.labels_init = True
    st.session_state.ed_ver = 0

added = store.added_rows()
if len(added):                                       # セッション内で足した日誌を後ろにつなぐ
    add = added.copy()
    add["作業時間"] = pd.to_numeric(add["作業時間"], errors="coerce")
    raw_df = pd.concat([base_df.drop(columns="ID"), add], ignore_index=True).astype(C.DTYPE_MAP)
    raw_df["作業"] = raw_df["作業"].fillna("（未記入）")
    raw_df.insert(0, "ID", [f"D{i:03d}" for i in range(1, len(raw_df) + 1)])
else:
    raw_df = base_df

df_raw = raw_df.copy(deep=True)                       # 第3回 E02：入力直後の控え
anon, anon_log, _ = C.anonymize(raw_df, names, orgs, auto)
df = R.add_period_cols(C.analyze(anon, groups, use_syn, neg_re))
G = {g.key: g for g in groups if g.base.strip()}
if not G:
    st.error("キーワードを1つ以上入れてください。")
    st.stop()

if st.session_state.pop("labels_init", False) or "labels" not in st.session_state:
    st.session_state.labels = store.load_labels() if store.persist else (C.sample_labels(df) if is_sample else {})
    st.session_state.labels_saved = dict(st.session_state.labels)
labels: dict = st.session_state.labels

tag_checks = store.table("tag_checks", R.TAG_CHECK_COLS)
cand, cand_src = R.candidate_masks(df, list(G.values()), tag_checks)

pages: dict = {}
ctx = Ctx(pages=pages, raw_df=raw_df, df_raw=df_raw, df=df, anon_log=anon_log, groups=groups, G=G, labels=labels,
          basis=basis, use_syn=use_syn, neg_re=neg_re, is_sample=is_sample, names=names, orgs=orgs, auto=auto,
          ref=ref, store=store, tag_checks=tag_checks, cand=cand, cand_src=cand_src)


# ============================================================ ページ（日→週→月を軸に）
def _p(fn, title, url, icon, default=False):
    def run():
        fn(ctx)
    run.__name__ = url or "home"
    page = st.Page(run, title=title, url_path=url or None, icon=icon, default=default)
    pages[url or "home"] = page
    return page


nav = st.navigation({
    "ルーティン（日→週→月）": [
        _p(routine.page_home, "今日のルーティン", "", "🗓️", default=True),
        _p(routine.page_nisshi, "毎日 /nisshi（3分）", "nisshi", "✍️"),
        _p(routine.page_shukei, "毎週 /shukei（10分）", "shukei", "🔍"),
        _p(routine.page_tsuki, "毎月 /tsuki（30分）", "tsuki", "🧭"),
    ],
    "学ぶ（第1〜3回）": [
        _p(learn.page_data, "0 データと匿名化", "data", "🔒"),
        _p(learn.page_kai1, "① 1件を読む", "kai1", "1️⃣"),
        _p(learn.page_kai2, "② for・ifで数える", "kai2", "2️⃣"),
        _p(learn.page_kai3, "③ pandasで表にする", "kai3", "3️⃣"),
    ],
    "深掘り（発展）": [
        _p(deep.page_judge, "④ 原文を読む・判定", "hantei", "📖"),
        _p(deep.page_insight, "⑤ インサイト", "insight", "💡"),
        _p(deep.page_model, "⑥ 判定モデル", "model", "🤖"),
    ],
})
nav.run()

# 手元ファイルのときは、どのページで判定を変えても最後にまとめて保存する
if store.persist and st.session_state.labels != st.session_state.get("labels_saved"):
    store.save_labels(st.session_state.labels)
    st.session_state.labels_saved = dict(st.session_state.labels)

st.divider()
st.caption("サンプルは架空の作業日誌（小規模農業法人を想定、50件・2026年4〜9月、本文欠損2件・作業時間欠損5件）。"
           "氏名・取引先・電話番号も架空。否定表現・共起語・判定モデル・優先度はいずれも簡易な目安で、決めるのは人です。")

"""ページ間で共有する文脈（Ctx）と表示部品"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

import diary_core as C

BASIS_JA = {"all": "一致した日誌", "noneg": "否定の可能性を除いた日誌", "human": "人が困りごとと判定した日誌"}
JUDGE = {None: "未確認", 1: "困りごと", 0: "該当せず"}
JUDGE_INV = {v: k for k, v in JUDGE.items()}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


@dataclass
class Ctx:
    raw_df: pd.DataFrame
    df_raw: pd.DataFrame
    df: pd.DataFrame
    anon_log: pd.DataFrame
    groups: list
    G: dict
    labels: dict
    basis: str
    use_syn: bool
    neg_re: object
    is_sample: bool
    names: list
    orgs: list
    auto: bool
    ref: date
    store: object
    tag_checks: pd.DataFrame
    cand: dict = field(default_factory=dict)       # 語群 → 週次に回す候補（bool Series）
    cand_src: dict = field(default_factory=dict)   # 語群 → 「日次確認」/「ルール」
    pages: dict = field(default_factory=dict)      # url_path → st.Page（page_link 用）

    @property
    def n_all(self) -> int:
        return len(self.df)

    @property
    def n_txt(self) -> int:
        return int(self.df["本文"].count())

    @property
    def hit_any(self) -> pd.Series:
        return self.df[[f"hit_{g}" for g in self.G]].astype(bool).any(axis=1)

    def lab(self, row, g):
        return self.labels.get(f"{row['key']}:{g}")

    def mask_of(self, g, b=None):
        return C.basis_mask(self.df, g, b or self.basis, self.labels)

    def any_mask(self, b=None):
        m = pd.Series(False, index=self.df.index)
        for g in self.G:
            m |= self.mask_of(g, b)
        return m


def show_df(d: pd.DataFrame, **kw):
    """表示用：内部列（occ_ など）を除く"""
    d = d[[c for c in d.columns if not str(c).startswith(("occ_", "key"))]]
    st.dataframe(d, **kw)


def header(ctx: Ctx):
    """学ぶ・深掘りページ共通の見出しと指標"""
    if ctx.n_all == 0:
        st.info("まだ日誌がありません。「毎日 /nisshi」で1件書くと、ここで分析できます。")
        st.stop()
    n_all, n_txt, hit_any = ctx.n_all, ctx.n_txt, ctx.hit_any
    reviewed = sum(1 for _, r in ctx.df.iterrows() for g in ctx.G if r[f"hit_{g}"] and ctx.lab(r, g) is not None)
    total_pairs = int(sum(ctx.df[f"hit_{g}"].sum() for g in ctx.G))
    c = st.columns(5)
    c[0].metric("全レコード", f"{n_all} 件", f"本文欠損 {n_all - n_txt} 件", delta_color="off", delta_arrow="off")
    c[1].metric("本文あり（分母）", f"{n_txt} 件")
    c[2].metric("キーワードに一致", f"{int(hit_any.sum())} 件", f"本文ありの {hit_any.sum()/max(1,n_txt)*100:.1f}%",
                delta_color="off", delta_arrow="off")
    c[3].metric("原文を確認済み", f"{reviewed} / {total_pairs}", "日誌×語", delta_color="off", delta_arrow="off")
    c[4].metric("困りごとと判定", f"{int(ctx.any_mask('human').sum())} 件")

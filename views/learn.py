"""学ぶ（第1回〜第3回）：同じ日誌を、道具を1段ずつ増やして分析する"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import streamlit as st

import diary_core as C
from views.common import BASIS_JA, Ctx, header, show_df


def page_data(ctx: Ctx):
    """0 データと匿名化"""
    df, raw_df, df_raw, G, labels = ctx.df, ctx.raw_df, ctx.df_raw, ctx.G, ctx.labels  # noqa: F841
    basis, use_syn, neg_re, is_sample = ctx.basis, ctx.use_syn, ctx.neg_re, ctx.is_sample  # noqa: F841
    n_all, n_txt, hit_any, anon_log = ctx.n_all, ctx.n_txt, ctx.hit_any, ctx.anon_log  # noqa: F841
    lab, mask_of, any_mask = ctx.lab, ctx.mask_of, ctx.any_mask  # noqa: F841
    header(ctx)
    l, r = st.columns([3, 2])
    with l:
        st.subheader("匿名化の結果")
        st.caption("この「本文」列（匿名化後）だけが以降の分析に使われます。元の文は画面表示のみで保存しません。")
        show_raw = st.toggle("元の文も並べて表示", False)
        view = df[["ID", "日付", "作業", "記入者", "作業時間", "本文"]].copy()
        if show_raw:
            view.insert(6, "元の文", raw_df["本文"])
        show_df(view, hide_index=True, height=420, width="stretch")
    with r:
        st.subheader("置換の記録")
        st.caption("「自動」は機械的な候補。誤置換・取りこぼしを目で確認し、必要なら辞書に足します。**この対応表は共有しません。**")
        st.dataframe(anon_log, hide_index=True, width="stretch")
        st.subheader("外部AIに入力しないための確認")
        for i, t in enumerate([
            "分析・集計はこのアプリ内で実行（外部AIに本文を送らない）",
            "生データは data/raw/ に分け、リポジトリに含めない",
            "置換の記録（対応表）は共有しない・提出物に含めない",
            "「自動」の置換と取りこぼしを目で確認した",
            "公開サーバーには架空サンプルだけを置く"]):
            st.checkbox(t, key=f"safe{i}")
        with st.expander("Claude Code に生データを読ませない設定例（人の注意ではなく設定で防ぐ）"):
            st.code('{\n  "permissions": {\n    "deny": [\n      "Read(./data/raw/**)",\n'
                    '      "Read(./data/name_map.csv)"\n    ]\n  }\n}', language="json")
        out = df[["ID", "日付", "作業", "記入者", "作業時間", "本文"]]
        st.download_button("⬇️ 匿名化済みCSV（diary_anon.csv）", out.to_csv(index=False).encode("utf-8-sig"),
                           "diary_anon.csv", "text/csv", width="stretch")


def page_kai1(ctx: Ctx):
    """① 第1回 1件を読む"""
    df, raw_df, df_raw, G, labels = ctx.df, ctx.raw_df, ctx.df_raw, ctx.G, ctx.labels  # noqa: F841
    basis, use_syn, neg_re, is_sample = ctx.basis, ctx.use_syn, ctx.neg_re, ctx.is_sample  # noqa: F841
    n_all, n_txt, hit_any, anon_log = ctx.n_all, ctx.n_txt, ctx.hit_any, ctx.anon_log  # noqa: F841
    lab, mask_of, any_mask = ctx.lab, ctx.mask_of, ctx.any_mask  # noqa: F841
    header(ctx)
    st.subheader("第1回｜1件の日誌を「データ」として読む")
    st.markdown("**問い → データ → 表現 → 計算 → 検証 → 解釈**。まず1件で、`len`・`count`・`in` の違いを確かめます。")
    txt_rows = df[df["本文"].notna()]
    l, r = st.columns([2, 3])
    with l:
        pick = st.selectbox("日誌を選ぶ", txt_rows["ID"], index=int((txt_rows["ID"] == "D010").idxmax())
                            if (txt_rows["ID"] == "D010").any() else 0,
                            format_func=lambda i: f"{i}｜{df.loc[df['ID']==i,'本文'].iloc[0][:18]}…")
        kw1 = st.text_input("調べる語", "遅れ", key="kw1")
        text = df.loc[df["ID"] == pick, "本文"].iloc[0]
        st.info(text)
        if kw1:
            m1, m2, m3 = st.columns(3)
            m1.metric("len(text)", C.char_len(text), help="文字数（Unicodeコードポイント数）")
            m2.metric(f'text.count("{kw1}")', text.count(kw1), help="出現回数")
            m3.metric(f'"{kw1}" in text', str(kw1 in text), help="含むか（件数に足すかどうか）")
            st.code(f'text = "{text}"\nprint(len(text))          # {len(text)}\n'
                    f'print(text.count("{kw1}"))  # {text.count(kw1)}\nprint("{kw1}" in text)    # {kw1 in text}',
                    language="python")
    with r:
        st.markdown("**同じ数でも、意味は同じとは限らない**")
        demo = pd.DataFrame({"文": ["故障した。", "故障なし。", "人手が足りた。", "人手が足りず。"]})
        demo["len"] = demo["文"].map(len)
        demo["count(故障/人手)"] = demo["文"].map(lambda s: s.count("故障") + s.count("人手"))
        demo["人が読むと"] = ["困りごと", "困りごとではない", "困りごとではない", "困りごと"]
        st.dataframe(demo, hide_index=True, width="stretch")
        st.caption("文字数も出現回数も同じなのに意味は逆。→ 数えた後に必ず**原文へ戻る**（KWIC）。")
        if kw1:
            st.markdown(f"**KWIC：「{kw1}」の前後14文字（全日誌）**")
            rows = []
            for _, rr in txt_rows.iterrows():
                for o in C.occurrences(rr["本文"], [kw1], neg_re):
                    a, k, b = C.kwic(rr["本文"], o)
                    rows.append({"ID": rr["ID"], "前": a, "語": k, "後": b, "直後に否定?": o.neg_word or ""})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=260)
    with st.expander("発展（第1回）：全角・半角のゆれ — NFKC正規化"):
        s1, s2 = "選別機が２時間遅れ", "選別機が2時間遅れ"
        st.code(f's1 = "{s1}"\ns2 = "{s2}"\nprint(s1 == s2)                                   # {s1 == s2}\n'
                f'print(unicodedata.normalize("NFKC", s1) == s2)   # {C.nfkc(s1) == s2}', language="python")
        st.caption("見た目が近くても別の文字。正規化は原文を残したまま「処理用の列」に対して行う（第4回の前処理へ）。")
    with st.expander("発展（第1回）：次の1文字を当てる小さな言語モデル（文字バイグラム）"):
        corpus = "".join(txt_rows["本文"])
        ch = st.text_input("この文字の次に来やすい文字は？", "故", max_chars=1)
        if ch:
            nxt = Counter(corpus[i + 1] for i in range(len(corpus) - 1) if corpus[i] == ch)
            tot = sum(nxt.values())
            if tot:
                st.dataframe(pd.DataFrame([{"次の文字": k, "回数": v, "確率": round(v / tot, 2)}
                                           for k, v in nxt.most_common(6)]), hide_index=True)
            else:
                st.write("この文字はコーパスに出てきません。")
            st.caption("数えた頻度を確率にしただけでも「予測」になる。⑥の判定モデルは、この考えに人の判定を加えたもの。")


def page_kai2(ctx: Ctx):
    """② 第2回 for・ifで数える"""
    df, raw_df, df_raw, G, labels = ctx.df, ctx.raw_df, ctx.df_raw, ctx.G, ctx.labels  # noqa: F841
    basis, use_syn, neg_re, is_sample = ctx.basis, ctx.use_syn, ctx.neg_re, ctx.is_sample  # noqa: F841
    n_all, n_txt, hit_any, anon_log = ctx.n_all, ctx.n_txt, ctx.hit_any, ctx.anon_log  # noqa: F841
    lab, mask_of, any_mask = ctx.lab, ctx.mask_of, ctx.any_mask  # noqa: F841
    header(ctx)
    st.subheader("第2回｜for・if で全日誌を数える：件数と回数を分ける")
    kw2 = st.selectbox("キーワード", [g.base for g in G.values()], key="kw2")
    texts = [None if pd.isna(t) else t for t in df["本文"]]
    trace = C.loop_trace(texts, kw2, neg_re)
    l, r = st.columns(2)
    with l:
        st.markdown("**Python**")
        st.code(C.python_loop_code(kw2), language="python")
        st.markdown("**DNCL（共通テスト用プログラム表記）**")
        st.code(C.dncl_code(kw2), language="text")
    with r:
        st.markdown("**トレース表：1件ずつ実行して変数を追う**")
        step = st.slider("何件目まで実行するか（i）", 0, len(texts), min(10, len(texts)))
        tv = trace.iloc[: step + 1].copy()
        tv["c"] = tv["c"].map(lambda v: "—" if pd.isna(v) else str(int(v)))
        st.dataframe(tv.style.apply(lambda s: ["background-color: rgba(61,107,58,.18)" if s["c"] not in ("—", "0")
                                               else "" for _ in s], axis=1),
                     hide_index=True, width="stretch", height=360)
        last = trace.iloc[step]
        k1, k2, k3 = st.columns(3)
        k1.metric("件数（日誌）", int(last["件数"]))
        k2.metric("回数（出現）", int(last["回数"]))
        k3.metric("要確認（否定の可能性）", int(last["要確認"]))
    st.markdown("**小さな例で検証：人が先に期待値を決める**")
    tst = C.loop_trace(["故障", "故障、故障", "故障はなかった"], "故障", neg_re).iloc[-1]
    ok = (tst["件数"], tst["回数"], tst["要確認"]) == (3, 4, 1)
    (st.success if ok else st.error)(
        f'["故障", "故障、故障", "故障はなかった"] → 件数 {tst["件数"]}・回数 {tst["回数"]}・要確認 {tst["要確認"]}'
        + ("　✓ 期待値（3・4・1）と一致" if ok else "　✕ 期待値（3・4・1）と不一致：否定表現の設定を確認"))

    st.markdown("---\n**確認問題（共通テスト「情報Ⅰ」形式）**")
    i_q = min(10, len(texts) - 1)
    ans_doc = int(trace.iloc[i_q + 1]["件数"])
    q1 = st.radio(f"問1　上のDNCLで i = {i_q} の処理が終わった直後、変数「件数」の値は？",
                  sorted({ans_doc, ans_doc + 1, max(0, ans_doc - 1), int(trace.iloc[i_q + 1]['回数'])}),
                  index=None, key="q1", horizontal=True)
    if q1 is not None:
        (st.success if q1 == ans_doc else st.warning)(
            f"正解は {ans_doc}。トレース表の i={i_q} の行で確かめよう（「回数」と取り違えていないか）。")
    q2 = st.radio("問2　「件数」と「回数」が違う値になるのは、どんな日誌があるときか。",
                  ["本文が欠損している日誌", "1件の日誌に同じ語が2回以上出てくる日誌", "否定表現を含む日誌"],
                  index=None, key="q2")
    if q2:
        (st.success if q2.startswith("1件") else st.warning)(
            "正解：1件に2回以上出る日誌（例：D010「雨で田植えが遅れ。遅れ、遅れで…」）。件数は +1、回数は +c。")
    q3 = st.radio("問3　「否定の可能性あり」を件数から除くには、DNCLのどこを変えるとよいか。",
                  ["「件数 = 件数 + 1」を「もし 否定の可能性(…) でないならば」の中に入れる",
                   "「回数 = 回数 + c」を削除する", "繰り返しの範囲を 1 から始める"], index=None, key="q3")
    if q3:
        (st.success if q3.startswith("「件数") else st.warning)(
            "正解は1つ目。ただし否定ルールは目安。「かもしれない」「追いつかない」に誤反応する例を④で確かめる。")


def page_kai3(ctx: Ctx):
    """③ 第3回 pandasで表にする"""
    df, raw_df, df_raw, G, labels = ctx.df, ctx.raw_df, ctx.df_raw, ctx.G, ctx.labels  # noqa: F841
    basis, use_syn, neg_re, is_sample = ctx.basis, ctx.use_syn, ctx.neg_re, ctx.is_sample  # noqa: F841
    n_all, n_txt, hit_any, anon_log = ctx.n_all, ctx.n_txt, ctx.hit_any, ctx.anon_log  # noqa: F841
    lab, mask_of, any_mask = ctx.lab, ctx.mask_of, ctx.any_mask  # noqa: F841
    header(ctx)
    st.subheader("第3回｜pandas で表にする：選ぶ・数える・分母を決める・原文に戻る")
    st.caption("1行＝1日誌レコード（1人ではない）。左端の index は pandas の行ラベルで、ID 列とは別物。")

    st.markdown("#### E03・E05｜構造と欠損を見る")
    a, b = st.columns([3, 2])
    with a:
        st.code('df = pd.read_csv("diary_anon.csv", encoding="utf-8-sig", dtype=dtype_map)\n'
                'df_raw = df.copy(deep=True)\nprint(df.shape)\nprint(df.isna().sum())\n'
                'print(len(df), df["本文"].count(), df["作業時間"].count())', language="python")
        st.write(f"`df.shape` = **{df_raw.shape}**　`len(df)` = **{len(df)}**　"
                 f'`df["本文"].count()` = **{n_txt}**　`df["作業時間"].count()` = **{int(df["作業時間"].count())}**')
    with b:
        miss = df_raw.drop(columns="ID").isna().sum().rename("欠損数").to_frame()
        miss["型"] = df_raw.drop(columns="ID").dtypes.astype(str)
        st.dataframe(miss, width="stretch")
    st.warning("本文欠損を「困りごとなし」、作業時間の欠損を「0時間」と読み替えない。欠損の理由はこのデータにはない。")
    show_df(df.loc[df["本文"].isna() | df["作業時間"].isna(), ["ID", "日付", "作業", "作業時間", "本文"]],
            hide_index=True, width="stretch")

    st.markdown("#### E06｜数値の条件で選ぶ（マスク）")
    th = st.slider("作業時間のしきい値（以上）", 1.0, 8.0, 6.0, 0.5)
    hm = (df["作業時間"] >= th).fillna(False).astype(bool)
    st.code(f'long_mask = (df["作業時間"] >= {th}).fillna(False)   # 欠損は「抽出しない」\n'
            'df.loc[long_mask, ["ID", "作業", "作業時間", "本文"]]', language="python")
    st.write(f"該当 **{int(hm.sum())} 件**（作業時間あり {int(df['作業時間'].count())} 件中）。"
             "`fillna(False)` は「抽出しない」指定で、欠損を短時間だったと判断したのではない。")

    st.markdown("#### E07・E08｜文字列の条件と、条件の組み合わせ")
    a, b, c3 = st.columns(3)
    kw3 = a.text_input("検索語", "故障", key="kw3")
    task3 = b.selectbox("作業で絞る", ["（すべて）"] + sorted(df["作業"].dropna().unique().tolist()))
    rx = c3.toggle("regex=True にしてみる", False, help="「.」などが特別な意味になる")
    if kw3:
        try:
            tm = C.contains_mask(df["本文"], kw3, regex=rx)
        except re.error:
            st.error("正規表現として解釈できません（regex=False に戻してください）。")
            tm = pd.Series(False, index=df.index)
        cm = tm & (df["作業"] == task3) if task3 != "（すべて）" else tm
        st.code(f'text_mask = df["本文"].str.contains("{kw3}", na=False, regex={rx})\n'
                + (f'combined = (df["作業"] == "{task3}") & (text_mask)   # and ではなく &\n' if task3 != "（すべて）" else "")
                + f'df.loc[{"combined" if task3 != "（すべて）" else "text_mask"}, ["ID", "作業", "本文"]]', language="python")
        st.write(f"該当 **{int(cm.sum())} 件**")
        show_df(df.loc[cm, ["ID", "日付", "作業", "本文"]], hide_index=True, width="stretch")
        with st.expander("regex の違いを小さな例で（A1）"):
            toy = pd.Series(["ver.2", "故障", None], dtype="string")
            st.code(f'toy = pd.Series(["ver.2", "故障", None], dtype="string")\n'
                    f'toy.str.contains(".", na=False, regex=True).tolist()   # {toy.str.contains(".", na=False, regex=True).tolist()}\n'
                    f'toy.str.contains(".", na=False, regex=False).tolist()  # {toy.str.contains(".", na=False, regex=False).tolist()}',
                    language="python")

    st.markdown("#### E10・E11｜グループごとに集計し、分母を決める")
    a, b, c3 = st.columns(3)
    gk = a.selectbox("キーワード", list(G), format_func=lambda k: G[k].base, key="g3")
    by = b.radio("まとめる単位", ["作業", "月"], horizontal=True)
    den = c3.radio("全体の割合の分母", ["本文あり", "全レコード"], horizontal=True)
    m3 = mask_of(gk)
    k = int(m3.sum())
    d_n = n_txt if den == "本文あり" else n_all
    st.write(f"「{G[gk].base}」（基準：{BASIS_JA[basis]}）：**{k} / {d_n} = {k/d_n*100:.1f}%**"
             f"（{'本文がある日誌のうちの割合' if den=='本文あり' else '全レコード（本文欠損を含む）のうちの割合'}）")
    tbl = C.denominator_table(df, m3, by)
    st.dataframe(tbl, width="stretch")
    st.code(f'df.assign(hit=mask).groupby("{by}").agg(\n    全件数=("ID", "size"),     # 行数\n'
            '    本文あり=("本文", "count"),  # 非欠損の件数\n    該当=("hit", "sum"))         # True の数',
            language="python")
    hrs = df.groupby("作業")["作業時間"].agg(["count", "mean", "median"]).round(2)
    with st.expander("作業時間を作業ごとに要約（平均の前に有効件数を見る）"):
        st.dataframe(hrs, width="stretch")
    st.info("件数の多い作業＝困りごとが多い作業、とは言えない。毎日書く作業ほど件数は増える。分母（本文あり）と原文をあわせて読む。")

    st.markdown("#### E12｜動くコードにも誤集計はある：`count()` と `sum()`")
    st.code(f'mask = df["本文"].str.contains("{G[gk].base}", na=False, regex=False)\n'
            f'mask.count()   # {int(C.contains_mask(df["本文"], G[gk].base).count())}  ← 欠損でない値の数（全行）\n'
            f'mask.sum()     # {int(C.contains_mask(df["本文"], G[gk].base).sum())}  ← True の数（該当件数）',
            language="python")

    st.markdown("#### E13｜結果を原本とは別名で保存する")
    exp = df.loc[m3, ["ID", "日付", "作業", "作業時間", "本文"]]
    a, b = st.columns(2)
    a.download_button(f"⬇️ selected_{G[gk].base}.csv（{len(exp)}行）", exp.to_csv(index=False).encode("utf-8-sig"),
                      f"selected_{G[gk].base}.csv", "text/csv")
    unchanged = raw_df.equals(df_raw)
    b.write(f"`df_raw` と入力データは変わっていない： **{unchanged}**")

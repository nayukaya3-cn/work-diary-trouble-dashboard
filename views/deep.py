"""深掘り（発展）：原文を読む・インサイト・判定モデル"""
from __future__ import annotations

import re
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

import diary_core as C
from views.common import BASIS_JA, COLORS, JUDGE, JUDGE_INV, Ctx, header


def page_judge(ctx: Ctx):
    """④ 原文を読む・判定"""
    df, raw_df, df_raw, G, labels = ctx.df, ctx.raw_df, ctx.df_raw, ctx.G, ctx.labels  # noqa: F841
    basis, use_syn, neg_re, is_sample = ctx.basis, ctx.use_syn, ctx.neg_re, ctx.is_sample  # noqa: F841
    n_all, n_txt, hit_any, anon_log = ctx.n_all, ctx.n_txt, ctx.hit_any, ctx.anon_log  # noqa: F841
    lab, mask_of, any_mask = ctx.lab, ctx.mask_of, ctx.any_mask  # noqa: F841
    header(ctx)
    st.subheader("原文を読む：KWIC と人の判定")
    st.caption("機械がやるのは「一致」まで。困りごとかどうかは原文を読んで人が決めます（日誌×語ごとに1つ）。")
    a, b, c3 = st.columns([1, 1, 2])
    fk = a.selectbox("キーワード", ["*"] + list(G), format_func=lambda k: "すべて" if k == "*" else G[k].base)
    ff = b.selectbox("表示", ["すべて", "否定の可能性あり", "未確認のみ", "ルールと人の判定がずれる"])
    with c3:
        cc1, cc2 = st.columns(2)
        cc1.download_button("⬇️ 判定を保存（CSV）",
                            pd.DataFrame([{"key": k_.split(":")[0], "語群": k_.split(":")[1], "判定": v}
                                          for k_, v in labels.items()]).to_csv(index=False).encode("utf-8-sig"),
                            "labels.csv", "text/csv")
        lu = cc2.file_uploader("判定を読み込む", type="csv", label_visibility="collapsed")
        if lu is not None and st.session_state.get("lu_name") != lu.name:
            ld = pd.read_csv(lu, dtype=str)
            for _, x in ld.iterrows():
                labels[f"{x['key']}:{x['語群']}"] = int(x["判定"])
            st.session_state.lu_name = lu.name
            st.session_state.ed_ver += 1
            st.rerun()

    rows, keys = [], []
    for i, rr in df.iterrows():
        for g in G:
            if (fk != "*" and fk != g) or not rr[f"hit_{g}"]:
                continue
            L = lab(rr, g)
            ng = rr[f"neg_{g}"]
            if ff == "否定の可能性あり" and not ng:
                continue
            if ff == "未確認のみ" and L is not None:
                continue
            if ff == "ルールと人の判定がずれる" and not (L is not None and ((ng and L == 1) or (not ng and L == 0))):
                continue
            o = rr[f"occ_{g}"][0]
            pre, w, post = C.kwic(rr["本文"], o, 18)
            rows.append({"ID": rr["ID"], "日付": rr["日付"], "作業": rr["作業"], "前の文脈": pre, "語": w,
                         "後ろの文脈": post, "回数": rr[f"cnt_{g}"],
                         "ルール": f"否定の可能性：{o.neg_word or next(x.neg_word for x in rr[f'occ_{g}'] if x.neg)}" if ng else "一致",
                         "人の判定": JUDGE[L]})
            keys.append(f"{rr['key']}:{g}")
    st.session_state.ed_keys = keys

    def _apply_edits():
        ed = st.session_state.get(f"ed_{st.session_state.ed_ver}", {})
        for idx, ch in ed.get("edited_rows", {}).items():
            if "人の判定" in ch:
                k_ = st.session_state.ed_keys[int(idx)]
                v = JUDGE_INV.get(ch["人の判定"])
                if v is None:
                    st.session_state.labels.pop(k_, None)
                else:
                    st.session_state.labels[k_] = v
        st.session_state.ed_ver += 1

    if rows:
        st.data_editor(pd.DataFrame(rows), key=f"ed_{st.session_state.ed_ver}", on_change=_apply_edits,
                       hide_index=True, width="stretch", height=420,
                       disabled=["ID", "日付", "作業", "前の文脈", "語", "後ろの文脈", "回数", "ルール"],
                       column_config={"人の判定": st.column_config.SelectboxColumn(options=list(JUDGE_INV), required=True),
                                      "前の文脈": st.column_config.TextColumn(width="medium"),
                                      "後ろの文脈": st.column_config.TextColumn(width="medium")})
    else:
        st.info("条件に合う一致はありません。")
    if is_sample:
        st.caption("サンプルには判定例が入っています。自分で読み直して変えてかまいません。")
        if st.button("判定例をすべて消して自分で判定する"):
            labels.clear()
            st.session_state.ed_ver += 1
            st.rerun()

    l, r = st.columns(2)
    with l:
        st.markdown("**ルールと人の判定のずれ**")
        D = []
        for _, rr in df.iterrows():
            for g in G:
                L = lab(rr, g)
                if not rr[f"hit_{g}"] or L is None:
                    continue
                if rr[f"neg_{g}"] and L == 1:
                    D.append({"ID": rr["ID"], "語": G[g].base, "ずれ方": "ルールは否定・人は困りごと",
                              "反応した表現": next(x.neg_word for x in rr[f"occ_{g}"] if x.neg), "原文": rr["本文"]})
                elif not rr[f"neg_{g}"] and L == 0:
                    D.append({"ID": rr["ID"], "語": G[g].base, "ずれ方": "ルールは一致・人は該当せず",
                              "反応した表現": "", "原文": rr["本文"]})
        st.dataframe(pd.DataFrame(D), hide_index=True, width="stretch")
        st.caption("例：「かもしれない」の「ない」、「追いつかない」に否定ルールが誤反応する。ルールの限界としてメモに残す。")
    with r:
        st.markdown(f"**どのような言葉で書かれているか（共起語・{BASIS_JA[basis]}）**")
        inG = any_mask() if fk == "*" else mask_of(fk)
        excl = {t for g in G.values() for t in g.terms(True)}

        def toks(t):
            t = re.sub(r"(作業者|取引先)[A-Z]+|［[^］]*］", " ", t)
            return set(re.findall(r"[一-龥]{2,}|[ァ-ヶー]{2,}", t))
        A = df.loc[inG & df["本文"].notna(), "本文"]
        B = df.loc[~inG & df["本文"].notna(), "本文"]
        ca = Counter(w for t in A for w in toks(t))
        cb = Counter(w for t in B for w in toks(t))
        co = [{"語": w, "該当日誌": n, "該当での割合%": round(n / max(1, len(A)) * 100),
               "それ以外での割合%": round(cb[w] / max(1, len(B)) * 100)}
              for w, n in ca.items() if n >= 2 and w not in excl and w != "作業"]
        co = sorted(co, key=lambda x: (x["該当での割合%"] - x["それ以外での割合%"], x["該当日誌"]), reverse=True)[:12]
        st.dataframe(pd.DataFrame(co), hide_index=True, width="stretch")
        st.caption(f"該当 {len(A)} 件 / それ以外 {len(B)} 件。漢字・カタカナの連続を語とみなす簡易な分かち書き（形態素解析は第5回以降）。")


def page_insight(ctx: Ctx):
    """⑤ インサイト"""
    df, raw_df, df_raw, G, labels = ctx.df, ctx.raw_df, ctx.df_raw, ctx.G, ctx.labels  # noqa: F841
    basis, use_syn, neg_re, is_sample = ctx.basis, ctx.use_syn, ctx.neg_re, ctx.is_sample  # noqa: F841
    n_all, n_txt, hit_any, anon_log = ctx.n_all, ctx.n_txt, ctx.hit_any, ctx.anon_log  # noqa: F841
    lab, mask_of, any_mask = ctx.lab, ctx.mask_of, ctx.any_mask  # noqa: F841
    header(ctx)
    st.subheader("インサイト：気づきは仮説。IDの原文で確かめてから使う")
    a, b = st.columns([1, 3])
    ib = a.selectbox("数える基準", ["human", "noneg", "all"], format_func=BASIS_JA.get, key="ib")
    W = a.number_input("連鎖：何日以内", 1, 30, 7)
    tm_any = any_mask(ib)
    ms = {g: mask_of(g, ib) for g in G}
    idl = lambda idx, mx=6: ", ".join(df.loc[list(idx)[:mx], "ID"]) + (f" ほか{len(idx)-mx}件" if len(idx) > mx else "")

    INS = []
    ts = C.beta_task_stats(df, tm_any)
    cand = ts[ts["n"] >= 3].sort_values("P(他より高い)", ascending=False)
    if len(cand) and cand.iloc[0]["k"]:
        t0 = cand.iloc[0]
        INS.append((t0["P(他より高い)"], f"「{t0['作業']}」の日誌に困りごとが集まっている可能性",
                    f"{t0['n']}件中{t0['k']}件（推定 {t0['推定']*100:.0f}%、90%区間 {t0['下限90']*100:.0f}〜{t0['上限90']*100:.0f}%）。"
                    f"他の作業より高い事後確率 {t0['P(他より高い)']*100:.0f}%。",
                    t0["IDs"], "日誌数が少なく区間が広い。記入頻度の多い作業ほど書かれやすい。",
                    "強" if t0["P(他より高い)"] >= .9 and t0["n"] >= 6 else "中" if t0["P(他より高い)"] >= .75 else "弱"))
    months = sorted(df["月"].unique())
    for g, m in ms.items():
        cnt = [(mo, int(m[df["月"] == mo].sum()), int(((df["月"] == mo) & df["本文"].notna()).sum())) for mo in months]
        tot = sum(x[1] for x in cnt)
        if tot < 3:
            continue
        avg = tot / max(1, sum(x[2] for x in cnt))
        pk = max(cnt, key=lambda x: x[1] / max(1, x[2]))
        ratio = pk[1] / max(1, pk[2]) / avg if avg else 0
        if ratio >= 1.4 and pk[1] >= 2:
            INS.append((min(.95, .5 + (ratio - 1) / 3), f"「{G[g].base}」は{int(pk[0][5:])}月に多い",
                        f"{int(pk[0][5:])}月は本文あり{pk[2]}件中{pk[1]}件（{pk[1]/pk[2]*100:.0f}%）。期間平均 {avg*100:.0f}% の約{ratio:.1f}倍。",
                        list(df.index[m & (df["月"] == pk[0])]), "1か月の件数が少なく、1〜2件で順位が変わる。前年と比べたい。",
                        "中" if pk[1] >= 4 and ratio >= 1.8 else "弱"))
    best = None
    for ga in G:
        for gb in G:
            if ga != gb:
                L = C.lag_stat(df, ms[ga], ms[gb], W)
                if L["nA"] >= 3 and (best is None or L["lift"] > best[2]["lift"]):
                    best = (ga, gb, L)
    if best and best[2]["lift"] >= 1.3:
        ga, gb, L = best
        INS.append((min(.9, .45 + (L["lift"] - 1) / 2), f"「{G[ga].base}」の後に「{G[gb].base}」が続きやすい",
                    f"「{G[ga].base}」{L['nA']}件のうち{len(L['pairs'])}件で{W}日以内に「{G[gb].base}」（{L['pA']*100:.0f}%）。"
                    f"任意の日誌起点では{L['base']*100:.0f}%（約{L['lift']:.1f}倍）。",
                    [p[0] for p in L["pairs"]], "前後関係であって因果ではない。繁忙期など共通の原因の可能性。", "弱"))
    rp = C.repairs(df, ms["f"]) if "f" in ms else pd.DataFrame()
    if len(rp):
        fixed = rp[rp["復旧ID"].notna()]
        med = fixed["日数"].median() if len(fixed) else None
        INS.append((.7 + min(.2, len(rp) / 20),
                    f"故障から復旧まで{'中央値%d日' % med if med is not None else 'の記録なし'}、{len(rp)-len(fixed)}件は復旧の記載なし",
                    f"対象：{'・'.join(rp['対象'])}。復旧の記載あり{len(fixed)}件（{'・'.join(str(int(x)) for x in fixed['日数'])}日）。",
                    [i for ids in rp["記載ID"] for i in df.index[df["ID"].isin(ids)]],
                    "復旧を書く欄がない。記載なし＝未復旧とは限らない。", "中" if len(rp) >= 4 else "弱"))
    if not use_syn:
        extra = [i for i, rr in df.iterrows() if isinstance(rr["本文"], str) and not hit_any[i]
                 and any(C.occurrences(rr["本文"], g.terms(True), neg_re) for g in G.values())]
        if extra:
            INS.append((.6, f"言い換えを数えると、さらに{len(extra)}件が対象になる",
                        "基本の語では拾えない例：" + "、".join(f"「{df.at[i,'本文'][:20]}…」" for i in extra[:3]),
                        extra, "言い換えは誤一致も増やす。④で原文を確認する。", "中"))
    INS.sort(key=lambda x: -x[0])
    with b:
        cols = st.columns(2)
        for n, (sc, title, ev, ids, cav, strength) in enumerate(INS):
            with cols[n % 2].container(border=True):
                st.markdown(f"**#{n+1} {title}**　`根拠：{strength}`")
                st.write(ev)
                st.caption(f"原文：{idl(ids)}　｜　注意：{cav}")
        if not INS:
            st.info("この基準では気づきを出せるだけの件数がありません。")

    l, r = st.columns(2)
    with l:
        st.markdown("**どの作業で多いか（ベイズ推定・Beta(1,1)事前分布・90%信用区間）**")
        tgt = st.selectbox("対象", ["*"] + list(G), format_func=lambda k: "困りごと（いずれか）" if k == "*" else G[k].base)
        T = C.beta_task_stats(df, tm_any if tgt == "*" else ms[tgt])
        overall = (tm_any if tgt == "*" else ms[tgt]).sum() / n_txt
        T["ラベル"] = T.apply(lambda x: f"{x['k']}/{x['n']}  P={x['P(他より高い)']:.2f}", axis=1)
        T["強調"] = T["P(他より高い)"] >= .9
        base_ch = alt.Chart(T.drop(columns="IDs")).encode(y=alt.Y("作業:N", sort=list(T["作業"]), title=None))
        ch = (base_ch.mark_rule(strokeWidth=2).encode(x=alt.X("下限90:Q", title="困りごと日誌の割合", axis=alt.Axis(format="%"),
                                                                 scale=alt.Scale(domain=[0, 1])), x2="上限90:Q",
                                                   color=alt.condition("datum.強調", alt.value("#d03b3b"), alt.value("#2a78d6")))
              + base_ch.mark_circle(size=90, opacity=1).encode(x="推定:Q",
                                                                color=alt.condition("datum.強調", alt.value("#d03b3b"), alt.value("#2a78d6")),
                                                                tooltip=["作業", "k", "n", alt.Tooltip("推定:Q", format=".0%"),
                                                                         alt.Tooltip("下限90:Q", format=".0%"),
                                                                         alt.Tooltip("上限90:Q", format=".0%"),
                                                                         alt.Tooltip("P(他より高い):Q", format=".2f")])
              + base_ch.mark_text(align="left", dx=8, fontSize=11).encode(x=alt.value(330), text="ラベル:N")
              + alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q"))
        st.altair_chart(ch.properties(height=34 * len(T) + 20), width="stretch")
        st.caption(f"分母は本文ありの日誌。点線＝全体 {overall*100:.0f}%。件数が少ない作業ほど区間が広い。")
        with st.expander("同じ推定を Python で"):
            st.code("import numpy as np\nrng = np.random.default_rng(0)\nk, n = 3, 3            # その作業：困りごと k / 本文あり n\n"
                    "post = rng.beta(1 + k, 1 + n - k, 4000)  # 事前 Beta(1,1) × 二項尤度\n"
                    "print(post.mean(), np.quantile(post, [.05, .95]))", language="python")
    with r:
        st.markdown("**いつ起きたか（タイムライン）**")
        tl = []
        for i, rr in df.iterrows():
            for n, g in enumerate(G):
                if rr[f"hit_{g}"]:
                    tl.append({"日付": rr["日付"], "語": G[g].base, "基準に入る": bool(ms[g][i]),
                               "ID": rr["ID"], "作業": rr["作業"], "本文": rr["本文"][:30]})
        tl = pd.DataFrame(tl)
        if len(tl):
            tl["日付"] = pd.to_datetime(tl["日付"])
            enc = dict(x=alt.X("日付:T", title=None, axis=alt.Axis(format="%-m月")), y=alt.Y("語:N", title=None),
                       color=alt.Color("語:N", scale=alt.Scale(range=COLORS), legend=None),
                       tooltip=["ID", "日付:T", "作業", "本文", "基準に入る"])
            p = (alt.Chart(tl[tl["基準に入る"]]).mark_point(filled=True, size=90).encode(**enc)
                 + alt.Chart(tl[~tl["基準に入る"]]).mark_point(filled=False, size=90).encode(**enc))
            st.altair_chart(p.properties(height=200), width="stretch")
            st.caption("塗り＝基準に入った日誌、白抜き＝一致したが基準外（否定・該当せず）。")
        st.markdown(f"**困りごとの連鎖（A の後 {W} 日以内に B）**")
        cA, cB = st.columns(2)
        la = cA.selectbox("A（先）", list(G), index=min(1, len(G) - 1), format_func=lambda k: G[k].base)
        lb = cB.selectbox("B（後）", list(G), index=0, format_func=lambda k: G[k].base)
        if la != lb:
            L = C.lag_stat(df, ms[la], ms[lb], W)
            st.write(f"「{G[la].base}」{L['nA']}件のうち **{len(L['pairs'])}件（{L['pA']*100:.0f}%）** で、{W}日以内に「{G[lb].base}」。"
                     f"任意の日誌起点では {L['base']*100:.0f}%（{L['lift']:.1f}倍）。**時間の前後関係であって因果ではない。**")
            if L["pairs"]:
                st.dataframe(pd.DataFrame([{"A": df.at[i, "ID"] + " " + df.at[i, "本文"][:16], "B": df.at[j, "ID"] + " " + df.at[j, "本文"][:16],
                                            "日数": (pd.to_datetime(df.at[j, "日付"]) - pd.to_datetime(df.at[i, "日付"])).days}
                                           for i, j in L["pairs"]]), hide_index=True, width="stretch")

    l, r = st.columns(2)
    with l:
        st.markdown("**故障の対象と復旧までの日数**")
        if len(rp):
            rv = rp.assign(開始=rp["開始"].dt.strftime("%m/%d"), 記載ID=rp["記載ID"].map(", ".join))
            st.dataframe(rv, hide_index=True, width="stretch")
        st.caption("「〜機」「〜庫」「軽トラ」を手がかりにした簡易抽出。")
    with r:
        st.markdown("**作業 × キーワード（件数 / 本文あり）**")
        hm_rows = [{"作業": t, "語": G[g].base, "件数": int(ms[g][sub.index].sum()), "本文あり": len(sub),
                    "割合": ms[g][sub.index].sum() / len(sub)}
                   for t, sub in df[df["本文"].notna()].groupby("作業") for g in G]
        H = pd.DataFrame(hm_rows)
        heat = alt.Chart(H).encode(x=alt.X("語:N", title=None), y=alt.Y("作業:N", title=None))
        st.altair_chart((heat.mark_rect().encode(color=alt.Color("割合:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                                 tooltip=["作業", "語", "件数", "本文あり", alt.Tooltip("割合:Q", format=".0%")])
                         + heat.mark_text(fontSize=11).encode(text=alt.Text("件数:Q"),
                                                              color=alt.condition("datum.割合 > 0.45", alt.value("white"), alt.value("black"))))
                        .properties(height=320), width="stretch")

    st.markdown("**AIエージェント導入の示唆**（送信・発注・シフト確定は人が承認。扱うのは匿名化済み・社内データのみ）")
    AG = []
    if len(rp):
        AG.append(["故障の受付と復旧の追跡", "日誌の「故障」記載から対象・日付を抜き出して故障票を作り、業者連絡・部品待ち・復旧を状態で追う。",
                   f"故障{len(rp)}件、復旧の記載なし{int(rp['復旧ID'].isna().sum())}件", "故障票の項目を現場と決める。業者への連絡文は下書きまで。"])
    if "d" in ms:
        td = C.beta_task_stats(df, ms["d"]).sort_values(["k", "推定"], ascending=False).iloc[0]
        AG.append(["作業遅れの早期通知と予定の再計算", f"「{td['作業']}」など遅れの多い作業で、遅れの記載が出たら後工程（出荷・集荷）を再計算し連絡文を下書き。",
                   f"「遅れ」は{td['作業']}で{td['k']}/{td['n']}件", "取引先への送信は人が確認。遅れの原因を分けて記録できるか試す。"])
    if "h" in ms:
        hc = df.loc[ms["h"]].groupby("月").size().sort_values(ascending=False)
        if len(hc):
            AG.append(["繁忙期の人員計画と応援要請の下書き", f"「人手」が多い時期（{'・'.join(str(int(m_[5:]))+'月' for m_ in hc.index[hc == hc.max()])}）の前に必要人数を見積もる。",
                       f"「人手」最多月 {hc.max()}件", "前年の日誌でも同じ時期に増えるか確かめる。シフト確定は人。"])
    AG.append(["日誌入力の補助", "記入時に「困りごと あり/なし」と分類（遅れ・故障・人手・その他）を選ぶ欄を付け、否定表現の誤判定を減らす。",
               "否定ルールの誤反応（④のずれ表）", "入力の手間が増えすぎないか現場で2週間試す。"])
    st.dataframe(pd.DataFrame(AG, columns=["候補となるエージェントの仕事", "何をするか", "根拠（このデータ）", "次に確かめること・人が決めること"]),
                 hide_index=True, width="stretch")
    md = f"## 作業日誌のインサイト（基準：{BASIS_JA[ib]}、本文あり{n_txt}件／全{n_all}件）\n\n" + "\n".join(
        f"{n+1}. **{t}**（根拠：{s}）\n   - 根拠：{e}\n   - 原文：{idl(ids, 10)}\n   - 注意：{cv}"
        for n, (_, t, e, ids, cv, s) in enumerate(INS))
    st.download_button("⬇️ インサイトを Markdown で保存", md.encode("utf-8"), "insights.md", "text/markdown")


def page_model(ctx: Ctx):
    """⑥ 判定モデル（発展）"""
    df, raw_df, df_raw, G, labels = ctx.df, ctx.raw_df, ctx.df_raw, ctx.G, ctx.labels  # noqa: F841
    basis, use_syn, neg_re, is_sample = ctx.basis, ctx.use_syn, ctx.neg_re, ctx.is_sample  # noqa: F841
    n_all, n_txt, hit_any, anon_log = ctx.n_all, ctx.n_txt, ctx.hit_any, ctx.anon_log  # noqa: F841
    lab, mask_of, any_mask = ctx.lab, ctx.mask_of, ctx.any_mask  # noqa: F841
    header(ctx)
    st.subheader("発展：困りごと判定モデル（ナイーブベイズ）")
    st.caption("④の人の判定を教師データに、キーワード前後（前6・後10文字）の文字2-gramから「困りごとを含む日誌か」を推定。"
               "学習データにない並びは使わない。第1回の「次の文字予測」と同じく、数えた頻度を確率にする考え方。")
    docs = []
    for _, rr in df.iterrows():
        if not isinstance(rr["本文"], str):
            continue
        y = C.doc_label(rr, list(G.values()), labels)
        if y is not None:
            docs.append((rr, C.context(rr["本文"], list(G.values()), use_syn, neg_re), y))
    n1 = sum(d[2] for d in docs)
    n0 = len(docs) - n1
    if n1 < 3 or n0 < 3:
        st.warning(f"「困りごと」と「該当せず」がそれぞれ3件以上必要です（いま {n1} / {n0}）。④で判定を付けてください。")
    else:
        nb = C.NaiveBayes([(t, y) for _, t, y in docs])
        hit = sum((C.NaiveBayes([(t, y) for j, (_, t, y) in enumerate(docs) if j != i]).prob(t) >= .5) == y
                  for i, (_, t, y) in enumerate(docs))
        k1, k2, k3 = st.columns(3)
        k1.metric("学習データ", f"{len(docs)} 件", f"困りごと {n1}・該当せず {n0}", delta_color="off", delta_arrow="off")
        k2.metric("1件抜き交差検証", f"{hit/len(docs)*100:.0f}%", f"{hit}/{len(docs)}件", delta_color="off", delta_arrow="off")
        k3.metric("常に多数派と答えた場合", f"{max(n1,n0)/len(docs)*100:.0f}%", "比べる基準（多数派）", delta_color="off", delta_arrow="off")
        l, r = st.columns([2, 3])
        with l:
            tin = st.text_area("新しい日誌（匿名化してから入力）", "コンバインが動かず、稲刈りが半日遅れた。", height=90)
            p = nb.prob(C.context(tin, list(G.values()), use_syn, neg_re))
            st.metric("困りごとを含む確率", f"{p*100:.0f}%")
            st.progress(p)
            ws = sorted({w for w in C.bigrams(C.context(tin, list(G.values()), use_syn, neg_re)) if w in nb.V},
                        key=lambda w: -abs(nb.lp(w, 1) - nb.lp(w, 0)))[:4]
            st.caption("効いた並び：" + " ".join(f"「{w.replace('◇','○')}」{'＋' if nb.lp(w,1)>nb.lp(w,0) else '−'}" for w in ws)
                       if ws else "学習データにない文字ばかりで、判断材料がありません。")
            if re.search(r"[一-龥]{1,3}(さん|様)", tin):
                st.error("氏名らしき語があります。匿名化してから入力してください。")
        with r:
            F = [{"2-gram": w.replace("◇", "○"), "困りごと": nb.cnt[1].get(w, 0), "該当せず": nb.cnt[0].get(w, 0),
                  "差": round(nb.lp(w, 1) - nb.lp(w, 0), 2)}
                 for w in nb.V if nb.cnt[0].get(w, 0) + nb.cnt[1].get(w, 0) >= 2]
            F = pd.DataFrame(F)
            c1, c2 = st.columns(2)
            c1.markdown("**困りごと寄り**")
            c1.dataframe(F.nlargest(8, "差"), hide_index=True, width="stretch")
            c2.markdown("**該当せず寄り**")
            c2.dataframe(F.nsmallest(8, "差"), hide_index=True, width="stretch")
        st.markdown("**まだ人が確認していない一致日誌への予測**（判定の順番づけに使い、最終判断は人が原文を読んで行う）")
        todo = [{"ID": rr["ID"], "作業": rr["作業"], "本文": rr["本文"],
                 "困りごとの確率": round(nb.prob(C.context(rr["本文"], list(G.values()), use_syn, neg_re)), 2)}
                for _, rr in df.iterrows() if isinstance(rr["本文"], str) and hit_any[_]
                and C.doc_label(rr, list(G.values()), labels) is None]
        if todo:
            st.dataframe(pd.DataFrame(todo).sort_values("困りごとの確率", ascending=False), hide_index=True,
                         width="stretch",
                         column_config={"困りごとの確率": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        else:
            st.info("一致した日誌はすべて確認済みです。サイドバーで「言い換えも数える」をオンにすると未確認が増えます。")

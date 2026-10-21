"""ルーティン（日→週→月）：今日のルーティン・/nisshi・/shukei・/tsuki

時間の粒度が上がるほど、機械の仕事が減り、人の判断が増える。
    毎日  ① 計算する      : 機械がタグを出す → 人は OK／直し
    毎週  ② 確かめる      : 機械が候補を並べる → 人が原文を読んで判定
    毎月  ③ 自分の条件で判断: 機械は順番の目安まで → 人が準備度を答え、順番を決める
"""
from __future__ import annotations

from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

import diary_core as C
import routine_core as R
from views.common import JUDGE, JUDGE_INV, Ctx

ACTION_TEMPLATES = {
    "d": "遅れの記載が出たら後工程（出荷・集荷）を再計算し、連絡文を下書き（送信は人）",
    "f": "故障票をつくり、業者連絡・部品待ち・復旧を状態で追う",
    "h": "繁忙期の2週間前に必要人数を見積もり、応援要請を下書き",
}


def _link(ctx: Ctx, name: str, label: str, icon: str | None = None):
    if name in ctx.pages:
        st.page_link(ctx.pages[name], label=label, icon=icon)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _human_mask(ctx: Ctx, g: str) -> pd.Series:
    """人が「困りごと」と判定した日誌（語群 g）"""
    return pd.Series([ctx.labels.get(f"{k}:{g}") == 1 for k in ctx.df["key"]], index=ctx.df.index, dtype=bool)


def _ctx_text(row, g: str) -> str:
    occ = row[f"occ_{g}"]
    if occ:
        a, k, b = C.kwic(row["本文"], occ[0], 12)
        return f"{a}【{k}】{b}"
    return row["本文"][:30]


def _train_nb(ctx: Ctx):
    docs = []
    for _, r in ctx.df.iterrows():
        if isinstance(r["本文"], str):
            y = C.doc_label(r, list(ctx.G.values()), ctx.labels)
            if y is not None:
                docs.append((C.context(r["本文"], list(ctx.G.values()), ctx.use_syn, ctx.neg_re), y))
    n1 = sum(y for _, y in docs)
    if n1 < 3 or len(docs) - n1 < 3:
        return None
    return C.NaiveBayes(docs)


def _week_status(ctx: Ctx, ws: str) -> dict:
    c = R.week_candidates(ctx.df, ctx.cand, ctx.labels, ws)
    return {"候補": len(c), "未判定": int(c["判定"].isna().sum()), "困りごと": int((c["判定"] == 1).sum()),
            "該当せず": int((c["判定"] == 0).sum())}


# ============================================================ 今日のルーティン（ホーム）
def page_home(ctx: Ctx):
    st.title("🗓️ 今日のルーティン")
    st.markdown("**毎日3分 書く → 毎週10分 確かめる → 毎月30分 決める。** "
                "時間の粒度が上がるほど、機械の仕事が減り、人の判断が増えます。")
    df, ref = ctx.df, ctx.ref
    days = R.entry_days(df)
    ws = R.week_start(ref).isoformat()
    mo = ref.strftime("%Y-%m")
    today_n = int((R.to_date(df["日付"]).dt.date == ref).sum())
    wk = _week_status(ctx, ws)
    back = R.week_candidates(df, ctx.cand, ctx.labels, ws, carry_over=True)
    back_n = int(back["持ち越し"].sum())
    monthly = ctx.store.table("monthly", R.MONTH_COLS)
    done_m = bool((monthly["月"] == mo).any())

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("#### ✍️ 毎日（3分）")
        st.caption(f"今日：{ref.month}/{ref.day}")
        st.metric("今日の記入", f"{today_n} 件", f"連続 {R.streak(days, ref)} 日", delta_color="off", delta_arrow="off")
        st.caption("書く・写真を添える → タグ案に OK／直し")
        _link(ctx, "nisshi", "書く /nisshi", "✍️")
    with c2.container(border=True):
        st.markdown("#### 🔍 毎週（10分）")
        st.caption(f"今週：{R.week_label(ws)}")
        st.metric("今週の候補（日誌×語）", f"{wk['候補']} 件", f"未判定 {wk['未判定']}・持ち越し {back_n}",
                  delta_color="off", delta_arrow="off")
        st.caption("候補の原文を読み、本当に困りごとかを判定")
        _link(ctx, "shukei", "確かめる /shukei", "🔍")
    with c3.container(border=True):
        st.markdown("#### 🧭 毎月（30分）")
        st.caption(f"今月：{ref.month}月（終わった月のふりかえりもここで）")
        st.metric("今月の順番", "決定済み" if done_m else "まだ", f"これまでの決定 {monthly['月'].nunique()} か月",
                  delta_color="off", delta_arrow="off")
        st.caption("準備度を答え、手を打つ順番を決める")
        _link(ctx, "tsuki", "決める /tsuki", "🧭")

    if ctx.n_all == 0:
        st.info("まだ日誌がありません。まず「毎日 /nisshi」で1件書いてください。")
        return
    last = R.to_date(df["日付"]).max()
    if pd.notna(last) and (pd.Timestamp(ref) - last).days > 14:
        st.warning(f"最後の日誌は {last:%Y-%m-%d}（{(pd.Timestamp(ref) - last).days}日前）。"
                   + ("サンプルは9月末で終わっています。" if ctx.is_sample else ""))

    st.markdown("#### 時系列のリズム：書いた日（毎日）と、確かめた週（毎週）")
    rt = R.rhythm_table(df, ref, 16)
    rt["週ラベル"] = rt["週"].dt.strftime("%-m/%-d")
    wk_order = list(dict.fromkeys(rt["週ラベル"]))
    x = alt.X("週ラベル:O", sort=wk_order, title="週のはじまり（月曜）", axis=alt.Axis(labelAngle=0))
    heat = alt.Chart(rt).mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        x=x, y=alt.Y("曜日:N", sort=list("月火水木金土日"), title=None),
        color=alt.Color("件数:Q", scale=alt.Scale(domain=[0, max(2, int(rt["件数"].max()))], range=["#eef1ee", "#3d6b3a"]),
                        legend=None),
        tooltip=[alt.Tooltip("日:T", format="%Y-%m-%d"), "曜日", "件数"])
    wrows = []
    for w in sorted(rt["週"].unique()):
        s = _week_status(ctx, pd.Timestamp(w).date().isoformat())
        for k in ("困りごと", "該当せず", "未判定"):
            wrows.append({"週ラベル": pd.Timestamp(w).strftime("%-m/%-d"), "状態": k, "件数": s[k]})
    bars = alt.Chart(pd.DataFrame(wrows)).mark_bar().encode(
        x=x, y=alt.Y("件数:Q", title="候補", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("状態:N", scale=alt.Scale(domain=["困りごと", "該当せず", "未判定"],
                                                 range=["#d03b3b", "#9aa5b1", "#f2c14e"]),
                        legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("状態:N"), tooltip=["週ラベル", "状態", "件数"])
    st.altair_chart(alt.vconcat(heat.properties(height=150), bars.properties(height=110)).resolve_scale(color="independent"),
                    width="stretch")
    st.caption("上：日ごとの記入件数（濃いほど多い）。下：週ごとの候補を、判定の状態で色分け（黄＝未判定が残っている週）。")

    st.markdown("#### 日・週・月で、機械と人の分担を変える")
    st.dataframe(pd.DataFrame([
        ["毎日（3分）", "/nisshi", "① 現象を計算する", "匿名化・語の一致・否定の可能性 → タグ案", "写真を入れ、タグ表に OK／直し",
         "日誌1行・タグの確認"],
        ["毎週（10分）", "/shukei", "② 可視化して確かめる", "候補を集め、判定モデルで確からしい順に並べる", "原文を読み「困りごと／該当せず」",
         "判定（教師データ）"],
        ["毎月（30分）", "/tsuki", "③ 自分の条件で判断する", "月の割合の推定・前月比・優先度の目安", "準備度を答え、手を打つ順番を決める",
         "順番・打つ手・担当・期限"],
    ], columns=["いつ", "コマンド", "3段構え", "機械がすること", "人がすること", "残るもの"]), hide_index=True, width="stretch")


# ============================================================ 毎日 /nisshi
def page_nisshi(ctx: Ctx):
    st.title("✍️ 毎日 /nisshi（3分）")
    st.caption("書く → 機械のタグ案を見る → OK／直し → 保存。機械がやるのは「一致」と「否定の可能性」まで。")
    if msg := st.session_state.pop("nisshi_msg", None):
        st.success(msg)
    df, ref = ctx.df, ctx.ref
    days = R.entry_days(df)
    m = st.columns(3)
    m[0].metric("今日の記入", f"{int((R.to_date(df['日付']).dt.date == ref).sum())} 件")
    m[1].metric("連続記入", f"{R.streak(days, ref)} 日")
    m[2].metric("直近7日に書いた日", f"{sum((ref - timedelta(days=d)) in days for d in range(7))} / 7")

    ver = st.session_state.setdefault("nisshi_ver", 0)
    l, r = st.columns([2, 3])
    with l:
        d = st.date_input("日付", ref, key=f"n_date{ver}")
        tasks = sorted(df["作業"].dropna().unique().tolist())
        t_sel = st.selectbox("作業", tasks + ["＋新しい作業"], key=f"n_task{ver}",
                             index=len(tasks) if ctx.is_sample or not tasks else 0)
        task = st.text_input("新しい作業の名前", "果樹" if ctx.is_sample else "", key=f"n_newtask{ver}") \
            if t_sel == "＋新しい作業" else t_sel
        authors = sorted(ctx.raw_df["記入者"].dropna().unique().tolist())
        a_sel = st.selectbox("記入者", authors + ["＋新しい記入者"], key=f"n_auth{ver}")
        author = st.text_input("記入者の名前（表示・保存は手元のみ。分析では記号に置換）", key=f"n_newauth{ver}") \
            if a_sel == "＋新しい記入者" else a_sel
        hrs = st.number_input("作業時間（任意・空欄可）", 0.0, 24.0, None, 0.5, key=f"n_hrs{ver}")
        text = st.text_area("本文", "柿の木を観察。落葉の片づけが遅れている。" if ctx.is_sample else "",
                            height=110, key=f"n_text{ver}", placeholder="例：柿の木を観察。落葉の片づけが遅れている。")
        photos = st.file_uploader("写真（任意・複数可）", type=["jpg", "jpeg", "png", "webp"],
                                  accept_multiple_files=True, key=f"n_photo{ver}")
        st.caption("写真は位置情報（EXIF）を消して保存。人の顔・車のナンバー・伝票が写っていないかを確認。")
        if photos:
            st.image([p.getvalue() for p in photos], width=110)

    with r:
        if not text.strip():
            st.info("本文を書くと、ここにタグ案が出ます。")
        else:
            row = {"日付": d.isoformat(), "作業": (task or "").strip() or None, "記入者": (author or "").strip() or None,
                   "作業時間": hrs, "本文": text.strip()}
            preview = pd.concat([ctx.raw_df.drop(columns="ID"), pd.DataFrame([row])], ignore_index=True).astype(C.DTYPE_MAP)
            anon, _, _ = C.anonymize(preview, ctx.names, ctx.orgs, ctx.auto)
            anon_text = anon["本文"].iloc[-1]
            key = C.text_key(anon_text)
            st.markdown("**匿名化後の本文**（分析・判定に使うのはこちら）")
            st.info(anon_text)
            if (ctx.df["key"] == key).any():
                st.warning("同じ本文の日誌がすでにあります。二重登録でないか確認してください。")

            st.markdown("**タグ表**　機械の提案に、人が OK／直し を付ける")
            tags = R.propose_tags(anon_text, list(ctx.G.values()), ctx.use_syn, ctx.neg_re)
            ed = st.data_editor(
                tags, key=f"n_tags{ver}_{key}", hide_index=True, width="stretch",
                disabled=["語群", "語", "一致した語", "文脈", "機械の提案"],
                column_order=["語", "一致した語", "文脈", "機械の提案", "人の確認"],
                column_config={"人の確認": st.column_config.SelectboxColumn(options=R.CONFIRM_OPTS, required=True),
                               "文脈": st.column_config.TextColumn(width="medium")})
            ed["確定"] = [R.confirmed_flag(p, c) for p, c in zip(ed["機械の提案"], ed["人の確認"])]
            go = "・".join(ed.loc[ed["確定"] == 1, "語"]) or "なし"
            st.markdown(f"→ 週次（/shukei）に回す候補：**{go}**")
            if (ed["機械の提案"] == R.TAG_NEG).any():
                st.caption("「否定の可能性」は目安。「かもしれない」の「ない」などに誤反応します。困りごとなら「直し→候補にする」。")
            if st.button("保存する", type="primary", width="stretch"):
                ctx.store.add_entry(row)
                n = ctx.store.save_photos(key, photos)
                tc = ed.assign(key=key, 記録日時=_now())[R.TAG_CHECK_COLS]
                ctx.store.append("tag_checks", tc, R.TAG_CHECK_COLS)
                st.session_state.nisshi_ver = ver + 1
                st.session_state.nisshi_msg = (f"保存しました（{row['日付']}・写真{n}枚・週次へ：{go}）。"
                                               + ("" if ctx.store.persist else "　※サンプル／アップロード中はこのセッション内のみ。"))
                st.rerun()

    st.markdown("#### 直近の日誌とタグ")
    rec = df[df["本文"].notna()].copy()
    rec["_d"] = R.to_date(rec["日付"])
    rec = rec.sort_values(["_d", "ID"], ascending=False).head(8)
    pk = ctx.store.photo_keys()
    st.dataframe(pd.DataFrame([{
        "ID": x["ID"], "日付": x["日付"], "作業": x["作業"], "本文": x["本文"],
        "週次へ": "・".join(ctx.G[g].base for g in ctx.G if ctx.cand[g][i]) or "—",
        "出どころ": "・".join(dict.fromkeys(ctx.cand_src[g][i] for g in ctx.G if ctx.cand[g][i])) or "—",
        "写真": "📷" if x["key"] in pk else ""} for i, x in rec.iterrows()]), hide_index=True, width="stretch")
    if not ctx.store.persist and len(ctx.store.added_rows()):
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ 足した日誌（氏名を含む・手元保存用）", ctx.store.added_rows().to_csv(index=False).encode("utf-8-sig"),
                           "diary_added.csv", "text/csv", width="stretch")
        c2.download_button("⬇️ タグの確認記録", ctx.tag_checks.to_csv(index=False).encode("utf-8-sig"),
                           "tag_checks.csv", "text/csv", width="stretch")


# ============================================================ 毎週 /shukei
def page_shukei(ctx: Ctx):
    st.title("🔍 毎週 /shukei（10分）")
    st.caption("今週の候補（日次で確認したタグ、なければルールの一致）を原文で読み、本当に困りごとかを判定する。"
               "判定は、月次の数字と判定モデルの教師データになる。")
    df = ctx.df
    if ctx.n_all == 0:
        st.info("まだ日誌がありません。")
        return
    weeks = sorted(set(df["週"]) - {"不明"}, reverse=True)
    ref_ws = R.week_start(ctx.ref).isoformat()
    opts = [ref_ws] + [w for w in weeks if w != ref_ws]
    n_by_w = df["週"].value_counts()
    a, b = st.columns([2, 1])
    if ref_ws in weeks or not weeks:
        w_idx = 0
    else:                                             # 今週の日誌がなければ、候補のある直近の週から
        w_idx = next((opts.index(w) for w in weeks if len(R.week_candidates(df, ctx.cand, ctx.labels, w))), 1)
    ws = a.selectbox("週", opts, index=w_idx,
                     format_func=lambda w: f"{R.week_label(w)}（日誌 {int(n_by_w.get(w, 0))} 件）"
                     + ("・今週" if w == ref_ws else ""), key="w_sel")
    carry = b.toggle("先週以前の未判定も並べる", False, key="w_carry")
    cands = R.week_candidates(df, ctx.cand, ctx.labels, ws, carry)
    in_w = df["週"] == ws
    m = st.columns(4)
    m[0].metric("この週の日誌", f"{int(in_w.sum())} 件", f"本文あり {int((in_w & df['本文'].notna()).sum())}",
                delta_color="off", delta_arrow="off")
    m[1].metric("候補（日誌×語）", f"{len(cands)} 件", f"うち持ち越し {int(cands['持ち越し'].sum())}" if carry else None,
                delta_color="off", delta_arrow="off")
    m[2].metric("判定済み", f"{int(cands['判定'].notna().sum())} 件")
    m[3].metric("未判定", f"{int(cands['判定'].isna().sum())} 件")

    if not len(cands):
        st.info("この週の候補はありません。" + ("" if carry else "持ち越しを並べると、過去の未判定が出ます。"))
    else:
        nb = _train_nb(ctx)
        pk = ctx.store.photo_keys()
        rows, keys = [], []
        for c in cands.itertuples():
            r_ = df.loc[c.idx]
            p = nb.prob(C.context(r_["本文"], list(ctx.G.values()), ctx.use_syn, ctx.neg_re)) if nb else None
            rows.append({"ID": r_["ID"], "日付": r_["日付"], "作業": r_["作業"], "語": ctx.G[c.語群].base,
                         "出どころ": ctx.cand_src[c.語群][c.idx], "文脈": _ctx_text(r_, c.語群),
                         "写真": "📷" if r_["key"] in pk else "", "持ち越し": "↩" if c.持ち越し else "",
                         "困りごとの確率": p, "人の判定": JUDGE[None if pd.isna(c.判定) else int(c.判定)]})
            keys.append(f"{r_['key']}:{c.語群}")
        tbl = pd.DataFrame(rows)
        tbl["_k"] = keys
        if nb:
            tbl = tbl.sort_values("困りごとの確率", ascending=False, kind="stable")
            st.caption("判定モデル（⑥）の確率が高い順。確率は**読む順番**のためで、判定そのものではない。")
        else:
            st.caption("判定がまだ少ないため日付順（困りごと・該当せずが各3件以上で、確率の高い順に並びます）。")
        st.session_state.w_keys = tbl["_k"].tolist()
        st.session_state.setdefault("w_ver", 0)

        def _apply():
            ed = st.session_state.get(f"w_ed{st.session_state.w_ver}", {})
            for idx, ch in ed.get("edited_rows", {}).items():
                if "人の判定" in ch:
                    k_ = st.session_state.w_keys[int(idx)]
                    v = JUDGE_INV.get(ch["人の判定"])
                    if v is None:
                        st.session_state.labels.pop(k_, None)
                    else:
                        st.session_state.labels[k_] = v
            st.session_state.w_ver += 1

        st.data_editor(tbl.drop(columns="_k"), key=f"w_ed{st.session_state.w_ver}", on_change=_apply,
                       hide_index=True, width="stretch",
                       disabled=[c for c in tbl.columns if c != "人の判定"],
                       column_config={"人の判定": st.column_config.SelectboxColumn(options=list(JUDGE_INV), required=True),
                                      "文脈": st.column_config.TextColumn(width="large"),
                                      "困りごとの確率": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        if len(tbl) > 15:
            st.caption("10分で見切れないときは上から。残りは来週の持ち越しになります（未判定のまま数えない）。")
        ph = [(df.loc[c.idx, "ID"], ctx.store.photos(df.loc[c.idx, "key"])) for c in cands.drop_duplicates("idx").itertuples()]
        ph = [(i, p) for i, p in ph if p]
        if ph:
            with st.expander(f"📷 写真つきの候補（{len(ph)}件）", expanded=False):
                for i, p in ph:
                    st.image(p, width=180, caption=[i] * len(p))

    st.markdown("#### 週ごとの推移（直近12週・人が「困りごと」と判定した日誌×語）")
    end = pd.Timestamp(max(ws, ref_ws))
    span = [(end - pd.Timedelta(weeks=k)).date().isoformat() for k in range(11, -1, -1)]
    hm = {g: _human_mask(ctx, g) for g in ctx.G}
    tr = [{"週": pd.Timestamp(w).strftime("%-m/%-d"), "語": ctx.G[g].base, "件数": int((hm[g] & (df["週"] == w)).sum()), "選択": w == ws}
          for w in span for g in ctx.G]
    ch = alt.Chart(pd.DataFrame(tr)).mark_bar().encode(
        x=alt.X("週:O", sort=[pd.Timestamp(w).strftime("%-m/%-d") for w in span], title="週のはじまり（月曜）",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("件数:Q", title="困りごと", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("語:N", scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a"]), legend=alt.Legend(orient="top", title=None)),
        opacity=alt.condition("datum.選択", alt.value(1), alt.value(0.45)), tooltip=["週", "語", "件数"])
    st.altair_chart(ch.properties(height=200), width="stretch")
    st.caption("濃い棒が選んだ週。未判定の候補は入っていない（判定すると増える）。週の件数は記入の多さにも左右される。")

    tr_w = [f"- {df.loc[c.idx, 'ID']}（{ctx.G[c.語群].base}）{df.loc[c.idx, '本文']}" for c in cands.itertuples()
            if ctx.labels.get(f"{df.loc[c.idx, 'key']}:{c.語群}") == 1]
    md = (f"## 週のまとめ {R.week_label(ws)}\n\n- 日誌 {int(in_w.sum())} 件、候補 {len(cands)} 件、"
          f"未判定 {int(cands['判定'].isna().sum()) if len(cands) else 0} 件\n\n### 困りごとと判定したもの\n"
          + ("\n".join(tr_w) or "- なし") + "\n")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ 週のまとめ（Markdown・匿名化済み）", md.encode("utf-8"), f"week_{ws}.md", "text/markdown", width="stretch")
    if not ctx.store.persist:
        c2.download_button("⬇️ 判定を保存（labels.csv）",
                           pd.DataFrame([{"key": k.split(":")[0], "語群": k.split(":")[1], "判定": v}
                                         for k, v in ctx.labels.items()]).to_csv(index=False).encode("utf-8-sig"),
                           "labels.csv", "text/csv", width="stretch")


# ============================================================ 毎月 /tsuki
def page_tsuki(ctx: Ctx):
    st.title("🧭 毎月 /tsuki（30分）")
    st.caption("月の数字を見て、準備度を答え、手を打つ順番を決める。機械が出すのは順番の**目安**まで。")
    df = ctx.df
    months = sorted(set(df["月"]) - {"不明"}, reverse=True)
    if not months:
        st.info("まだ日誌がありません。")
        return
    done_m = (pd.Period(ctx.ref, "M") - 1).strftime("%Y-%m")   # 既定は「終わったばかりの月」をふりかえる
    a, b = st.columns([1, 2])
    mo = a.selectbox("月", months, index=months.index(done_m) if done_m in months else 0,
                     format_func=lambda x: f"{x[:4]}年{int(x[5:])}月", key="m_sel")
    prev = (pd.Period(mo, "M") - 1).strftime("%Y-%m")
    prev = prev if prev in months else None
    bs = b.radio("数える基準", ["人が困りごとと判定（確定）", "候補（判定待ちを含む）"], horizontal=True, key="m_basis")
    masks = {g: _human_mask(ctx, g) if bs.startswith("人") else ctx.cand[g] for g in ctx.G}
    unj = sum(int((ctx.cand[g] & (df["月"] == mo)).sum() - sum(
        ctx.labels.get(f"{k}:{g}") is not None for k in df.loc[ctx.cand[g] & (df["月"] == mo), "key"])) for g in ctx.G)
    if unj and bs.startswith("人"):
        st.warning(f"この月の候補のうち **{unj} 件が未判定** です。判定すると数字が変わります。")
        _link(ctx, "shukei", "先に /shukei で判定する", "🔍")

    decisions = ctx.store.table("monthly", R.MONTH_COLS)

    # ---- 1. ふりかえり
    st.markdown("#### 1｜先月決めた手のふりかえり")
    rv = R.review_decisions(df, masks, decisions, mo)
    if len(rv):
        rv["語群"] = rv["語群"].map(lambda g: ctx.G[g].base if g in ctx.G else g)
        st.dataframe(rv, hide_index=True, width="stretch")
        st.caption("決めた後に減った／増えたは**前後関係**。季節や作業量の変化でも動くので、効いた証明ではない。")
    else:
        st.caption("先月の決定記録はありません（この月が最初の /tsuki なら正常）。")

    # ---- 2. 今月の数字
    st.markdown(f"#### 2｜{int(mo[5:])}月の数字（分母＝本文のある日誌・Beta(1,1)・90%信用区間）")
    stats = {g: R.month_stat(df, masks[g], mo, prev) for g in ctx.G}
    cols = st.columns(len(stats))
    for col, (g, s) in zip(cols, stats.items()):
        p = s["P(前月より高い)"]
        col.metric(ctx.G[g].base, f"{s['推定']*100:.0f}%", f"{s['k']}/{s['n']}件・区間 {s['下限90']*100:.0f}〜{s['上限90']*100:.0f}%",
                   delta_color="off", delta_arrow="off")
        col.caption("前月なし" if p is None else f"前月 {s['前月k']}/{s['前月n']} → 前月より高い確率 **{p*100:.0f}%**")
    tr = []
    for m_ in sorted(months):
        for g in ctx.G:
            s = R.month_stat(df, masks[g], m_, None)
            tr.append({"月": pd.Timestamp(m_ + "-01"), "語": ctx.G[g].base, "推定": s["推定"], "下限": s["下限90"],
                       "上限": s["上限90"], "件": f"{s['k']}/{s['n']}", "選択": m_ == mo})
    T = pd.DataFrame(tr)
    T["月ラベル"] = T["月"].dt.strftime("%-m月")
    enc_x = alt.X("月ラベル:O", sort=list(dict.fromkeys(T["月ラベル"])), title=None, axis=alt.Axis(labelAngle=0))
    col_ = alt.Color("語:N", scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a"]), legend=alt.Legend(orient="top", title=None))
    band = alt.Chart(T).mark_area(opacity=0.12).encode(x=enc_x, y=alt.Y("下限:Q", axis=alt.Axis(format="%"), title="困りごとの割合"),
                                                       y2="上限:Q", color=col_)
    line = alt.Chart(T).mark_line(point=True).encode(x=enc_x, y="推定:Q", color=col_,
                                                     tooltip=["語", alt.Tooltip("月:T", format="%Y-%m"), "件",
                                                              alt.Tooltip("推定:Q", format=".0%"),
                                                              alt.Tooltip("下限:Q", format=".0%"), alt.Tooltip("上限:Q", format=".0%")])
    rule = alt.Chart(pd.DataFrame({"月ラベル": [f"{int(mo[5:])}月"]})).mark_rule(strokeDash=[4, 3], color="gray").encode(
        x=alt.X("月ラベル:O", sort=list(dict.fromkeys(T["月ラベル"]))))
    st.altair_chart((band + line + rule).properties(height=230), width="stretch")
    st.caption("帯＝90%信用区間。日誌の少ない月ほど帯が広い（1〜2件で上下する）。点線が選んだ月。")

    # ---- 3. 準備度
    st.markdown("#### 3｜準備度を答える：「来月これが起きたら、すぐ手を打てるか」")
    saved = decisions[decisions["月"] == mo].drop_duplicates("語群", keep="last").set_index("語群")
    before = decisions[decisions["月"] < mo].drop_duplicates("語群", keep="last").set_index("語群")
    ready = {}
    rc = st.columns(len(ctx.G))
    for col, g in zip(rc, ctx.G):
        src = saved if g in saved.index else before if g in before.index else None
        dv = int(src.at[g, "準備度"]) if src is not None and str(src.at[g, "準備度"]).isdigit() else 0
        ready[g] = col.radio(ctx.G[g].base, list(R.READINESS), index=dv, format_func=R.READINESS.get, key=f"m_r_{mo}_{g}")

    # ---- 4. 提案順
    st.markdown("#### 4｜機械の提案順（目安）")
    with st.expander("優先度の式と重み", expanded=False):
        st.latex(r"\text{優先度} = \text{起きやすさ} \times \left(1-\frac{\text{準備度}}{3}\right) \times "
                 r"\left(1 + w \cdot \max(0,\ P(\text{前月より高い}) - 0.5)\right)")
        w = st.slider("増える見込みの重み w", 0.0, 2.0, 0.5, 0.1, key="m_w")
        st.caption("起きやすさ＝今月の事後平均。準備度3（試して効いた）は0になる。重みは慣習で、正解はない。")
    w = st.session_state.get("m_w", 0.5)
    P = R.priority_table(stats, ready, {g: ctx.G[g].base for g in ctx.G}, w)
    st.dataframe(P.drop(columns="語群"), hide_index=True, width="stretch",
                 column_config={"起きやすさ": st.column_config.NumberColumn(format="percent"),
                                "優先度（目安）": st.column_config.ProgressColumn(min_value=0, max_value=max(0.01, P["優先度（目安）"].max()),
                                                                             format="%.3f")})

    # ---- 5. 人が決める
    st.markdown("#### 5｜人が決める：手を打つ順番・打つ手・担当・期限")
    st.caption("担当は名前ではなく役割（例：機械担当）か作業者記号で。送信・発注・シフト確定は人が行う。")
    plan = []
    for row in P.itertuples():
        g = row.語群
        s = saved.loc[g] if g in saved.index else None
        plan.append({"語群": g, "困りごと": row.困りごと, "提案順": row.提案順,
                     "人が決めた順番": int(s["人が決めた順番"]) if s is not None and str(s["人が決めた順番"]).isdigit() else row.提案順,
                     "打つ手": s["打つ手"] if s is not None else ACTION_TEMPLATES.get(g, ""),
                     "担当": s["担当"] if s is not None else "",
                     "期限": pd.to_datetime(s["期限"], errors="coerce") if s is not None
                     else pd.Timestamp(pd.Period(mo, "M").end_time.date()) + pd.Timedelta(days=30)})
    plan = pd.DataFrame(plan)
    plan["期限"] = pd.to_datetime(plan["期限"]).dt.date
    ed = st.data_editor(plan, key=f"m_plan_{mo}", hide_index=True, width="stretch",
                        disabled=["語群", "困りごと", "提案順"], column_order=["困りごと", "提案順", "人が決めた順番", "打つ手", "担当", "期限"],
                        column_config={"人が決めた順番": st.column_config.NumberColumn(min_value=1, max_value=len(plan), step=1),
                                       "打つ手": st.column_config.TextColumn(width="large"),
                                       "期限": st.column_config.DateColumn(format="YYYY-MM-DD")})
    ed["人が決めた順番"] = pd.to_numeric(ed["人が決めた順番"], errors="coerce").fillna(ed["提案順"]).astype(int)
    order = ed.sort_values(["人が決めた順番", "提案順"])
    if ed["人が決めた順番"].duplicated().any():
        st.warning("同じ順番が重複しています。")
    if (ed["人が決めた順番"] != ed["提案順"]).any():
        st.info("機械の提案順を入れ替えました。理由をメモ（打つ手の欄など）に残すと、来月のふりかえりで使えます。")
    c1, c2 = st.columns(2)
    if c1.button("この月の決定を保存", type="primary", width="stretch"):
        rows = order.assign(月=mo, 準備度=[ready[g] for g in order["語群"]], 記録日時=_now(),
                            期限=order["期限"].astype(str))[R.MONTH_COLS]
        ctx.store.append("monthly", rows, R.MONTH_COLS)
        st.success("保存しました。" + ("" if ctx.store.persist else "（このセッション内のみ。右のボタンでCSVに）"))
    md = (f"## {mo[:4]}年{int(mo[5:])}月の決定（基準：{bs}）\n\n"
          + "\n".join(f"{i}. **{r.困りごと}**（準備度 {ready[r.語群]}・今月 {stats[r.語群]['k']}/{stats[r.語群]['n']}）"
                      f"\n   - 打つ手：{r.打つ手}\n   - 担当：{r.担当 or '未定'}　期限：{r.期限}"
                      for i, r in enumerate(order.itertuples(), 1)) + "\n")
    c2.download_button("⬇️ 月の決定（Markdown）", md.encode("utf-8"), f"month_{mo}.md", "text/markdown", width="stretch")

"""日→週→月ルーティンの計算部品を、人が先に決めた期待値で検証する（pytest -q）"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diary_core as C  # noqa: E402
import routine_core as R  # noqa: E402

NEG = C.compile_neg(C.DEFAULT_NEG)
G = C.DEFAULT_GROUPS


def _prepared(extra_rows=None):
    df = C.load_sample()
    if extra_rows:
        df = pd.concat([df.drop(columns="ID"), pd.DataFrame(extra_rows)], ignore_index=True).astype(C.DTYPE_MAP)
        df.insert(0, "ID", [f"D{i:03d}" for i in range(1, len(df) + 1)])
    a, _, _ = C.anonymize(df, C.SAMPLE_NAMES, C.SAMPLE_ORGS, True)
    return R.add_period_cols(C.analyze(a, G, False, NEG))


# ---------------------------------------------------------------- 毎日
def test_daily_example_is_candidate():
    # 例：「柿の木を観察。落葉の片づけが遅れている」→ 遅れ＝候補、故障・人手＝一致なし
    t = R.propose_tags("柿の木を観察。落葉の片づけが遅れている。", G, False, NEG).set_index("語群")
    assert t.loc["d", "機械の提案"] == R.TAG_CAND
    assert t.loc["f", "機械の提案"] == R.TAG_NONE
    assert t.loc["h", "機械の提案"] == R.TAG_NONE
    assert "【遅れ】" in t.loc["d", "文脈"]


def test_daily_negation_and_human_fix():
    t = R.propose_tags("草刈り。特に遅れはなかった。", G, False, NEG).set_index("語群")
    assert t.loc["d", "機械の提案"] == R.TAG_NEG
    # 人が直したらそちらが優先
    assert R.confirmed_flag(R.TAG_NEG, R.CONFIRM_OK) == 0
    assert R.confirmed_flag(R.TAG_NEG, R.CONFIRM_TO_CAND) == 1
    assert R.confirmed_flag(R.TAG_CAND, R.CONFIRM_TO_NONE) == 0


def test_appending_keeps_existing_keys():
    # 日誌を後ろに足しても、既存の判定キー（匿名化後本文のハッシュ）は変わらない
    before = _prepared()["key"].tolist()
    after = _prepared([{"日付": "2026-10-22", "作業": "果樹", "記入者": "田中", "作業時間": "1.0",
                        "本文": "柿の木を観察。落葉の片づけが遅れている。"}])["key"].tolist()
    assert after[:len(before)] == before


def test_tag_check_overrides_rule():
    df = _prepared()
    i = df.index[df["ID"] == "D005"][0]          # 「故障かもしれない」：ルールは否定の可能性
    m0, _ = R.candidate_masks(df, G, pd.DataFrame(columns=R.TAG_CHECK_COLS))
    assert not m0["f"][i]
    tc = pd.DataFrame([{"key": df.at[i, "key"], "語群": "f", "機械の提案": R.TAG_NEG,
                        "人の確認": R.CONFIRM_TO_CAND, "確定": 1, "記録日時": ""}])
    m1, src = R.candidate_masks(df, G, tc)
    assert m1["f"][i] and src["f"][i] == "日次確認"


def test_streak():
    days = {date(2026, 10, d) for d in (18, 19, 20, 21)}
    assert R.streak(days, date(2026, 10, 21)) == 4
    assert R.streak(days, date(2026, 10, 22)) == 4     # 今日はまだ → 昨日から
    assert R.streak(days, date(2026, 10, 24)) == 0


# ---------------------------------------------------------------- 毎週
def test_week_is_monday_start():
    assert R.week_start(date(2026, 10, 22)) == date(2026, 10, 19)
    assert R.week_label("2026-10-19") == "10/19〜10/25"


def test_week_candidates_and_carry_over():
    df = _prepared()
    masks, _ = R.candidate_masks(df, G, None)
    ws = df.loc[df["ID"] == "D019", "週"].iloc[0]     # 選別機が故障・2時間遅れ（6/11）
    c = R.week_candidates(df, masks, {}, ws)
    assert set(c["語群"]) == {"d", "f"} and not c["持ち越し"].any()
    c2 = R.week_candidates(df, masks, {}, ws, carry_over=True)
    assert c2["持ち越し"].sum() > 0                   # 未判定の過去分が乗る


# ---------------------------------------------------------------- 毎月
def test_priority_order():
    # 起きやすさが同じなら、準備度が低いほど先
    assert R.priority(0.4, 0) > R.priority(0.4, 2) > R.priority(0.4, 3) == 0
    # 増える見込みは加点のみ（減る見込みで減点はしない）
    assert R.priority(0.4, 1, 0.9) > R.priority(0.4, 1, 0.5) == R.priority(0.4, 1, 0.1)


def test_month_stat_denominator_and_prev():
    df = _prepared()
    masks, _ = R.candidate_masks(df, G, None)
    s = R.month_stat(df, masks["d"], "2026-09", "2026-08")
    assert s["n"] == int(((df["月"] == "2026-09") & df["本文"].notna()).sum())   # 本文欠損は分母に入れない
    assert 0 <= s["下限90"] <= s["推定"] <= s["上限90"] <= 1
    assert s["P(前月より高い)"] is not None


def test_photo_exif_is_removed():
    import io

    from PIL import Image

    import store as S
    im = Image.new("RGB", (8, 8), "green")
    exif = Image.Exif()
    exif[0x8825] = {2: (34.0, 24.0, 0.0)}       # GPSInfo（撮影位置）
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    assert Image.open(io.BytesIO(buf.getvalue())).getexif()
    out, ext = S.strip_exif(buf.getvalue())
    assert ext == ".jpg" and not Image.open(io.BytesIO(out)).getexif()

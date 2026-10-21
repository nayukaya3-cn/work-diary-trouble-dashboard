"""人が先に決めた期待値で、計算部品を検証する（pytest -q）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diary_core as C  # noqa: E402

NEG = C.compile_neg(C.DEFAULT_NEG)


def test_small_example_doc_vs_occurrence():
    # 第2回：件数3・回数4・要確認1
    last = C.loop_trace(["故障", "故障、故障", "故障はなかった"], "故障", NEG).iloc[-1]
    assert (last["件数"], last["回数"], last["要確認"]) == (3, 4, 1)


def test_missing_is_not_counted():
    last = C.loop_trace([None, "遅れ"], "遅れ", NEG).iloc[-1]
    assert last["件数"] == 1


def test_sample_structure_and_missing():
    df = C.load_sample()
    assert df.shape == (50, 6)
    assert int(df["本文"].count()) == 48          # 本文あり
    assert int(df["作業時間"].isna().sum()) == 5


def test_count_vs_sum():
    # 第3回 E12：count は非欠損の数、sum は True の数
    df = C.load_sample()
    m = C.contains_mask(df["本文"], "故障")
    assert m.count() == 50
    assert m.sum() == 12


def test_anonymize_removes_names_and_phone():
    df = C.load_sample()
    a, log, _ = C.anonymize(df, C.SAMPLE_NAMES, C.SAMPLE_ORGS, True)
    body = "".join(a["本文"].dropna())
    for w in C.SAMPLE_NAMES + C.SAMPLE_ORGS + ["090-1234-5678"]:
        assert w not in body


def test_denominator_uses_text_present():
    df = C.load_sample()
    a, _, _ = C.anonymize(df, C.SAMPLE_NAMES, C.SAMPLE_ORGS, True)
    t = C.denominator_table(a, C.contains_mask(a["本文"], "故障"), "月")
    assert t["本文あり"].sum() == 48 and t["全件数"].sum() == 50

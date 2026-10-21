"""作業日誌 困りごと分析 — 計算部品（Streamlit から切り離して単体テストできる純粋関数）

積み上げの対応
  第1回: len / count / in / KWIC            → char_len, occurrences, kwic
  第2回: for・if で件数と回数を分けて数える  → loop_trace, dncl_code
  第3回: pandas（欠損・マスク・分母・groupby）→ to_frame, contains_mask, denominator_table
  発展 : 否定表現・人の判定・ベイズ推定・判定モデル
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PATH = DATA_DIR / "sample_diary.csv"

SAMPLE_NAMES = ["佐藤", "鈴木", "高橋", "山本"]
SAMPLE_ORGS = ["ひがし青果", "やまびこ食品", "みなと運送"]

DTYPE_MAP = {"日付": "string", "作業": "string", "記入者": "string",
             "作業時間": "Float64", "本文": "string"}

DEFAULT_NEG = "ない|なかった|なし|無し|ません|ではなく|足りた|足りている|解消|直った|問題なく"


@dataclass
class KwGroup:
    key: str
    base: str
    syn: list[str] = field(default_factory=list)

    def terms(self, use_syn: bool) -> list[str]:
        t = [self.base.strip()] if self.base.strip() else []
        if use_syn:
            t += [s for s in self.syn if s and s not in t]
        return t


DEFAULT_GROUPS = [
    KwGroup("d", "遅れ", ["遅延", "間に合わ", "延期", "押した"]),
    KwGroup("f", "故障", ["壊れ", "直らない", "動かな", "不調", "異音", "詰まり", "破れ"]),
    KwGroup("h", "人手", ["人が足り", "手が回らない", "手が足り", "応援", "残業"]),
]

# サンプル48件に付けた「人の判定例」（行番号 → 語群: 1=困りごと, 0=該当せず）
SAMPLE_LABELS = {1: {"f": 0}, 2: {"d": 1, "h": 1}, 4: {"d": 0}, 5: {"f": 1}, 7: {"f": 1}, 8: {"d": 1},
                 9: {"f": 0}, 10: {"d": 1}, 11: {"h": 1}, 12: {"h": 0}, 14: {"h": 1}, 16: {"d": 1},
                 17: {"d": 0}, 18: {"h": 1}, 19: {"f": 1, "d": 1}, 20: {"f": 1}, 21: {"h": 1}, 23: {"f": 0},
                 24: {"d": 0}, 25: {"h": 1}, 26: {"d": 1}, 28: {"h": 0}, 29: {"d": 1}, 30: {"f": 1},
                 31: {"d": 0}, 32: {"d": 1}, 33: {"h": 1, "d": 1}, 34: {"d": 1}, 35: {"h": 1}, 36: {"d": 0},
                 37: {"f": 0}, 38: {"f": 1}, 39: {"h": 0}, 40: {"f": 1}, 43: {"d": 1}, 44: {"d": 1, "h": 1},
                 45: {"h": 1}, 46: {"f": 1}, 47: {"d": 1}}


def text_key(text: str) -> str:
    """匿名化後の本文から作る判定用キー（生データは保存しない）"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- 読み込み（第3回 E02）
def read_diary(src) -> pd.DataFrame:
    """CSV/TSV を読み、列名をそろえて string / Float64 型にする"""
    raw = src.read() if hasattr(src, "read") else Path(src).read_bytes()
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    sep = "\t" if "\t" in text.splitlines()[0] else ","
    from io import StringIO
    df = pd.read_csv(StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    alias = {"date": "日付", "task": "作業", "author": "記入者", "name": "記入者", "氏名": "記入者",
             "text": "本文", "body": "本文", "内容": "本文", "メモ": "本文", "hours": "作業時間"}
    df = df.rename(columns={c: alias.get(c.strip().lower(), c.strip()) for c in df.columns})
    for col in DTYPE_MAP:
        if col not in df.columns:
            df[col] = ""
    df = df[list(DTYPE_MAP)].copy()
    # 空欄・空白だけ → 欠損（第3回 A2：空欄は理由を推測せず欠損として残す）
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df["作業時間"] = pd.to_numeric(df["作業時間"], errors="coerce").astype("Float64")
    df["作業"] = df["作業"].fillna("（未記入）")
    df.insert(0, "ID", [f"D{i:03d}" for i in range(1, len(df) + 1)])
    return df


def load_sample() -> pd.DataFrame:
    return read_diary(SAMPLE_PATH)


# ---------------------------------------------------------------- 匿名化
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _code(prefix: str, i: int) -> str:
    return prefix + (_LETTERS[i] if i < 26 else _LETTERS[i // 26 - 1] + _LETTERS[i % 26])


_NOTNAME = re.compile(r"^(皆|お客|奥|業者|担当|社長|みな|皆様|お子)$")


def anonymize(df: pd.DataFrame, names: list[str], orgs: list[str], auto: bool = True):
    """氏名・取引先・電話・メールを置換。戻り値: (匿名化後df, 置換ログdf)"""
    person: dict[str, str] = {}
    org: dict[str, str] = {}
    log: dict[tuple, dict] = {}

    def pcode(n):
        person.setdefault(n, _code("作業者", len(person)))
        return person[n]

    def ocode(n):
        org.setdefault(n, _code("取引先", len(org)))
        return org[n]

    def note(kind, src, dst, how):
        e = log.setdefault((kind, src), {"種類": kind, "元の語": src, "置換後": dst, "回数": 0, "方法": how})
        e["回数"] += 1

    for w in df["記入者"].dropna():
        pcode(w)
    for n in names:
        pcode(n)
    for o in orgs:
        ocode(o)

    dict_terms = sorted([(n, "氏名") for n in person] + [(o, "取引先") for o in orgs], key=lambda x: -len(x[0]))
    dict_re = re.compile("|".join(re.escape(t) for t, _ in dict_terms)) if dict_terms else None
    rules = [
        (re.compile(r"0\d{1,4}-\d{1,4}-\d{3,4}"), lambda m: ("電話番号", m.group(0), "［電話番号］")),
        (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), lambda m: ("メール", m.group(0), "［メール］")),
        (re.compile(r"(株式会社|有限会社|（株）|\(株\))[ァ-ヶー一-龥ぁ-んA-Za-z]{1,10}|"
                    r"[ァ-ヶー一-龥A-Za-z]{1,8}(青果|食品|商店|運送|商事|農協|物産)"),
         lambda m: ("取引先", m.group(0), ocode(m.group(0)))),
    ]
    hon_re = re.compile(r"([一-龥]{1,3})(さん|様|くん|氏)")

    def sub_segments(segs, pattern, fn):
        out = []
        for s, masked in segs:
            if masked:
                out.append((s, True))
                continue
            last = 0
            for m in pattern.finditer(s):
                rep = fn(m)
                if rep is None:
                    continue
                out.append((s[last:m.start()], False))
                out.append((rep, True))
                last = m.end()
            out.append((s[last:], False))
        return [x for x in out if x[0]]

    anon_texts, masked_html = [], []
    for raw in df["本文"]:
        if pd.isna(raw):
            anon_texts.append(pd.NA)
            masked_html.append(None)
            continue
        segs = [(raw, False)]
        if dict_re:
            def fdict(m):
                w = m.group(0)
                is_org = w in org and w not in person
                dst = ocode(w) if is_org else pcode(w)
                note("取引先" if is_org else "氏名", w, dst, "辞書")
                return dst
            segs = sub_segments(segs, dict_re, fdict)
        if auto:
            for pat, fn in rules:
                def f(m, fn=fn):
                    kind, src, dst = fn(m)
                    note(kind, src, dst, "自動")
                    return dst
                segs = sub_segments(segs, pat, f)

            def fhon(m):
                nm = m.group(1)
                if _NOTNAME.match(nm):
                    return None
                dst = pcode(nm)
                note("氏名", nm, dst, "自動")
                return dst + m.group(2)
            segs = sub_segments(segs, hon_re, fhon)
        anon_texts.append("".join(s for s, _ in segs))
        masked_html.append(segs)

    out = df.copy()
    out["記入者"] = df["記入者"].map(lambda w: person.get(w, pd.NA) if pd.notna(w) else pd.NA).astype("string")
    out["本文"] = pd.array(anon_texts, dtype="string")
    out["月"] = out["日付"].str.slice(0, 7).fillna("不明")
    logdf = pd.DataFrame(list(log.values()), columns=["種類", "元の語", "置換後", "回数", "方法"])
    return out, logdf, masked_html


# ---------------------------------------------------------------- 第1回：1件を読む
def char_len(text: str) -> int:
    """len() はUnicodeコードポイント数。単語数・トークン数ではない"""
    return len(text)


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


@dataclass
class Occ:
    s: int
    e: int
    t: str
    after: str
    neg: bool
    neg_word: str


def occurrences(text, terms: list[str], neg_re: re.Pattern | None) -> list[Occ]:
    """重ならない出現位置。長い語を優先。否定は直後12文字（句点まで）を見る"""
    if text is None or (not isinstance(text, str)) or not terms:
        return []
    occ: list[Occ] = []
    base = terms[0]
    order = sorted(terms, key=lambda t: (t != base, -len(t)))
    for t in order:
        i = 0
        while (i := text.find(t, i)) != -1:
            e = i + len(t)
            if not any(i < o.e and e > o.s for o in occ):
                after = text[e:].split("。")[0][:12]
                m = neg_re.search(after) if neg_re else None
                occ.append(Occ(i, e, t, after, bool(m), m.group(0) if m else ""))
            i = e
    return sorted(occ, key=lambda o: o.s)


def kwic(text: str, o: Occ, width: int = 14) -> tuple[str, str, str]:
    return text[max(0, o.s - width):o.s], text[o.s:o.e], text[o.e:o.e + width]


def compile_neg(pattern: str):
    try:
        return re.compile(pattern) if pattern.strip() else None
    except re.error:
        return None


# ---------------------------------------------------------------- 第2回：for・if とトレース
def loop_trace(texts: list, keyword: str, neg_re) -> pd.DataFrame:
    """DNCL と同じ手順を1件ずつ実行し、変数の変化を記録する"""
    rows = [{"i": "—", "本文[i]（先頭）": "（初期化）", "c": None, "件数": 0, "回数": 0, "要確認": 0}]
    n_doc = n_occ = n_chk = 0
    for i, t in enumerate(texts):
        if t is None or (isinstance(t, float) and math.isnan(t)) or t is pd.NA:
            rows.append({"i": str(i), "本文[i]（先頭）": "＜欠損＞", "c": None, "件数": n_doc, "回数": n_occ, "要確認": n_chk})
            continue
        oc = occurrences(t, [keyword], neg_re)
        c = len(oc)
        if c > 0:
            n_doc += 1
            n_occ += c
            if any(o.neg for o in oc):
                n_chk += 1
        rows.append({"i": str(i), "本文[i]（先頭）": t[:16] + ("…" if len(t) > 16 else ""), "c": c,
                     "件数": n_doc, "回数": n_occ, "要確認": n_chk})
    return pd.DataFrame(rows)


def dncl_code(keyword: str) -> str:
    return f"""件数 = 0
回数 = 0
要確認 = 0
i を 0 から 要素数(本文) - 1 まで 1 ずつ増やしながら繰り返す:
│ もし 本文[i] が 欠損 ならば:
│ └ 次の i へ（数えない）
│ c = 出現回数(本文[i], "{keyword}")
│ もし c > 0 ならば:
│ │ 件数 = 件数 + 1
│ │ 回数 = 回数 + c
│ │ もし 否定の可能性(本文[i], "{keyword}") ならば:
│ └ └ 要確認 = 要確認 + 1
表示する(件数, 回数, 要確認)"""


def python_loop_code(keyword: str) -> str:
    return f'''texts = list(df["本文"])        # 匿名化済みの本文
n_doc = 0      # 件数（日誌の数）
n_occ = 0      # 回数（語の出現数）
for text in texts:
    if pd.isna(text):          # 欠損は数えない（0件扱いにしない）
        continue
    c = text.count("{keyword}")
    if c > 0:                  # "{keyword}" in text と同じ判定
        n_doc = n_doc + 1
        n_occ = n_occ + c
print(n_doc, n_occ)'''


# ---------------------------------------------------------------- 第3回：pandas
def contains_mask(s: pd.Series, kw: str, regex: bool = False, case: bool = True) -> pd.Series:
    if not kw:
        raise ValueError("検索語には1文字以上を入れてください。")
    return s.str.contains(kw, na=False, regex=regex, case=case).astype(bool)


def analyze(df: pd.DataFrame, groups: list[KwGroup], use_syn: bool, neg_re) -> pd.DataFrame:
    out = df.copy()
    out["key"] = out["本文"].map(lambda t: text_key(t) if isinstance(t, str) else None)
    for g in groups:
        occ = out["本文"].map(lambda t: occurrences(t, g.terms(use_syn), neg_re))
        out[f"occ_{g.key}"] = occ
        out[f"hit_{g.key}"] = occ.map(len) > 0
        out[f"cnt_{g.key}"] = occ.map(len)
        out[f"neg_{g.key}"] = occ.map(lambda o: any(x.neg for x in o))
    return out


def denominator_table(df: pd.DataFrame, mask: pd.Series, by: str) -> pd.DataFrame:
    """第3回 E11：size（行数）/ count（本文あり）/ sum（True）を並べ、本文ありを分母に割合を出す"""
    t = df.assign(_hit=mask).groupby(by, dropna=False).agg(
        全件数=("ID", "size"), 本文あり=("本文", "count"), 該当=("_hit", "sum"))
    t["本文欠損"] = t["全件数"] - t["本文あり"]
    den = t["本文あり"].where(t["本文あり"] > 0)
    t["本文あり中の割合(%)"] = (t["該当"] / den * 100).round(1)
    return t


# ---------------------------------------------------------------- 人の判定
def sample_labels(df: pd.DataFrame) -> dict[str, int]:
    lab = {}
    for i, row in enumerate(df.itertuples(), 1):
        L = SAMPLE_LABELS.get(i)
        if L and isinstance(row.本文, str):
            for g, v in L.items():
                lab[f"{text_key(row.本文)}:{g}"] = v
    return lab


def counted(row, g: str, basis: str, labels: dict) -> bool:
    if not row[f"hit_{g}"]:
        return False
    if basis == "noneg":
        return not row[f"neg_{g}"]
    if basis == "human":
        return labels.get(f"{row['key']}:{g}") == 1
    return True


def basis_mask(df, g, basis, labels) -> pd.Series:
    return df.apply(lambda r: counted(r, g, basis, labels), axis=1).astype(bool)


def doc_label(row, groups, labels):
    ls = [labels.get(f"{row['key']}:{g.key}") for g in groups if row[f"hit_{g.key}"]]
    ls = [x for x in ls if x is not None]
    if not ls:
        return None
    return 1 if 1 in ls else 0


# ---------------------------------------------------------------- 発展：ベイズ推定
def beta_task_stats(df, target_mask: pd.Series, seed: int = 0) -> pd.DataFrame:
    """作業ごとの困りごと割合を Beta(1,1) 事前分布で推定。分母は本文ありの日誌数"""
    rng = np.random.default_rng(seed)
    base = df[df["本文"].notna()]
    m = target_mask[base.index]
    K, N = int(m.sum()), len(base)
    rows = []
    for task, sub in base.groupby("作業"):
        k, n = int(m[sub.index].sum()), len(sub)
        a = rng.beta(1 + k, 1 + n - k, 4000)
        b = rng.beta(1 + K - k, 1 + (N - n) - (K - k), 4000)
        rows.append({"作業": task, "k": k, "n": n, "推定": (1 + k) / (2 + n),
                     "下限90": float(np.quantile(a, .05)), "上限90": float(np.quantile(a, .95)),
                     "P(他より高い)": float((a > b).mean()), "IDs": list(sub.index[m[sub.index]])})
    return pd.DataFrame(rows).sort_values("推定", ascending=False).reset_index(drop=True)


def lag_stat(df, mask_a: pd.Series, mask_b: pd.Series, window: int):
    """Aの日誌の後、window日以内にBの日誌があるか（前後関係であって因果ではない）"""
    d = pd.to_datetime(df["日付"], errors="coerce")
    valid = d.notna() & df["本文"].notna()
    A = df.index[mask_a & valid]
    B = df.index[mask_b & valid]

    def follow(i):
        for j in B:
            if j != i and 0 <= (d[j] - d[i]).days <= window:
                return j
        return None
    pairs = [(i, follow(i)) for i in A]
    pairs = [(i, j) for i, j in pairs if j is not None]
    allv = df.index[valid]
    base = sum(follow(i) is not None for i in allv) / max(1, len(allv))
    pA = len(pairs) / len(A) if len(A) else 0.0
    return {"nA": len(A), "pairs": pairs, "pA": pA, "base": base, "lift": pA / base if base else 0.0}


_FIX = re.compile(r"(直っ|直り|戻り|戻っ|修理から|復旧|再開)")
_MACH = re.compile(r"軽トラ|[一-龥ァ-ヶー]{1,4}機|[一-龥]{1,3}庫")


def repairs(df, fmask: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(df["日付"], errors="coerce")
    eps: list[dict] = []
    for i in df.index[fmask]:
        t = df.at[i, "本文"]
        for o in df.at[i, "occ_f"]:
            mach = _MACH.findall(t[:o.s])
            if not mach:
                continue
            obj = mach[-1]
            e = next((x for x in eps if x["対象"] == obj and x["復旧ID"] is None), None)
            if e is None:
                eps.append({"対象": obj, "開始": d[i], "記載ID": [df.at[i, "ID"]], "復旧ID": None, "日数": None})
            elif df.at[i, "ID"] not in e["記載ID"]:
                e["記載ID"].append(df.at[i, "ID"])
    for e in eps:
        for j in df.index[(d > e["開始"]) & df["本文"].notna()].sort_values():
            sent = "。".join(s for s in df.at[j, "本文"].split("。") if e["対象"] in s)
            if sent and _FIX.search(sent):
                e["復旧ID"] = df.at[j, "ID"]
                e["日数"] = int((d[j] - e["開始"]).days)
                break
    return pd.DataFrame(eps, columns=["対象", "開始", "記載ID", "復旧ID", "日数"])


# ---------------------------------------------------------------- 発展：ナイーブベイズ
def bigrams(t: str) -> list[str]:
    t = re.sub(r"(作業者|取引先)[A-Z]+", "◇", t)
    t = re.sub(r"[\s、。，．,.！？!?（）()「」［］]", "", t)
    return [t[i:i + 2] for i in range(len(t) - 1)]


def context(t: str, groups, use_syn, neg_re) -> str:
    occ = [o for g in groups for o in occurrences(t, g.terms(use_syn), neg_re)]
    if not occ:
        return t
    return " ".join(t[max(0, o.s - 6):o.e + 10].replace("。", " ") for o in occ)


class NaiveBayes:
    def __init__(self, docs: list[tuple[str, int]]):
        self.cnt = [dict(), dict()]
        self.tot = [0, 0]
        self.nd = [0, 0]
        self.V: set[str] = set()
        for t, y in docs:
            self.nd[y] += 1
            for w in bigrams(t):
                self.cnt[y][w] = self.cnt[y].get(w, 0) + 1
                self.tot[y] += 1
                self.V.add(w)

    def lp(self, w, c):
        return math.log((self.cnt[c].get(w, 0) + 1) / (self.tot[c] + len(self.V)))

    def prob(self, t: str) -> float:
        ws = [w for w in bigrams(t) if w in self.V]
        s = [math.log((self.nd[c] + 1) / (sum(self.nd) + 2)) + sum(self.lp(w, c) for w in ws) for c in (0, 1)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        return e[1] / (e[0] + e[1])

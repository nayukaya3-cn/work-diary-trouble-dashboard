"""ルーティンの記録の置き場所

- 手元ファイル（persist=True）: data/raw/ 以下に保存（.gitignore 済み・リポジトリに入らない）
    data/raw/diary.csv            日誌の原本（毎日ここに1行ずつ足す）
    data/raw/routine/tag_checks.csv  毎日：タグへの OK／直し
    data/raw/routine/labels.csv      毎週：困りごとかどうかの判定
    data/raw/routine/monthly.csv     毎月：準備度と、人が決めた順番
    data/raw/photos/                 写真（位置情報などの EXIF は消して保存）
- サンプル・アップロード（persist=False）: ブラウザのセッション内だけ。閉じると消える（CSVで保存可）
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import diary_core as C

RAW_DIR = C.DATA_DIR / "raw"
DIARY_PATH = RAW_DIR / "diary.csv"
DIARY_COLS = ["日付", "作業", "記入者", "作業時間", "本文"]


def strip_exif(data: bytes) -> tuple[bytes, str]:
    """写真の EXIF（撮影位置など）を落として JPEG/PNG で返す。読めなければそのまま"""
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        buf = io.BytesIO()
        if im.mode in ("RGBA", "LA", "P"):
            im.save(buf, "PNG")
            return buf.getvalue(), ".png"
        im.convert("RGB").save(buf, "JPEG", quality=85)
        return buf.getvalue(), ".jpg"
    except Exception:
        return data, ".bin"


class Store:
    def __init__(self, persist: bool, root: Path = RAW_DIR):
        self.persist = persist
        self.root = root
        self.rdir = root / "routine"
        self.pdir = root / "photos"
        if persist:
            self.rdir.mkdir(parents=True, exist_ok=True)
            self.pdir.mkdir(parents=True, exist_ok=True)
        st.session_state.setdefault("_store", {})
        st.session_state.setdefault("_photos", {})

    # ------------------------------------------------------------ 表
    def table(self, name: str, cols: list[str]) -> pd.DataFrame:
        if self.persist:
            p = self.rdir / f"{name}.csv"
            if p.exists():
                t = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                return t.reindex(columns=cols, fill_value="")
            return pd.DataFrame(columns=cols)
        return st.session_state["_store"].get(name, pd.DataFrame(columns=cols)).copy()

    def save(self, name: str, t: pd.DataFrame):
        if self.persist:
            t.to_csv(self.rdir / f"{name}.csv", index=False, encoding="utf-8-sig")
        else:
            st.session_state["_store"][name] = t.copy()

    def append(self, name: str, rows: pd.DataFrame, cols: list[str]):
        t = pd.concat([self.table(name, cols), rows.reindex(columns=cols)], ignore_index=True)
        self.save(name, t)

    # ------------------------------------------------------------ 判定（labels）
    def load_labels(self) -> dict:
        t = self.table("labels", ["key", "語群", "判定"])
        return {f"{r.key}:{r.語群}": int(r.判定) for r in t.itertuples() if str(r.判定) in ("0", "1")}

    def save_labels(self, labels: dict):
        self.save("labels", pd.DataFrame([{"key": k.split(":")[0], "語群": k.split(":")[1], "判定": v}
                                          for k, v in labels.items()], columns=["key", "語群", "判定"]))

    # ------------------------------------------------------------ 日誌の追記（毎日）
    def added_rows(self) -> pd.DataFrame:
        """セッション内で足した日誌（手元ファイルのときは diary.csv に直接足すので空）"""
        return st.session_state.get("_added", pd.DataFrame(columns=DIARY_COLS))

    def add_entry(self, row: dict):
        r = pd.DataFrame([row], columns=DIARY_COLS)
        if self.persist:
            ensure_diary_file()
            if not DIARY_PATH.read_bytes().endswith(b"\n"):
                with DIARY_PATH.open("a", encoding="utf-8") as fh:
                    fh.write("\n")
            r.to_csv(DIARY_PATH, mode="a", header=False, index=False, encoding="utf-8")
        else:
            st.session_state["_added"] = pd.concat([self.added_rows(), r], ignore_index=True)

    # ------------------------------------------------------------ 写真
    def save_photos(self, key: str, files) -> int:
        n = 0
        for i, f in enumerate(files or []):
            data, ext = strip_exif(f.getvalue())
            if self.persist:
                (self.pdir / f"{key}_{i}{ext}").write_bytes(data)
            else:
                st.session_state["_photos"].setdefault(key, []).append(data)
            n += 1
        return n

    def photos(self, key: str) -> list[bytes]:
        if not isinstance(key, str):
            return []
        if self.persist:
            return [p.read_bytes() for p in sorted(self.pdir.glob(f"{key}_*"))]
        return list(st.session_state["_photos"].get(key, []))

    def photo_keys(self) -> set[str]:
        if self.persist:
            return {p.name.split("_")[0] for p in self.pdir.glob("*_*")}
        return set(st.session_state["_photos"])


def ensure_diary_file():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DIARY_PATH.exists():
        pd.DataFrame(columns=DIARY_COLS).to_csv(DIARY_PATH, index=False, encoding="utf-8-sig")

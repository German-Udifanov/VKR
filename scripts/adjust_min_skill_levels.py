"""
Сдвиг min_skill_level в Excel на заданную дельту (по умолчанию -1).

Обновляет листы machines и operations — оба участвуют в данных; решатель
использует порог из маршрута (operations), лист machines для согласованности
справочника станков.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Сдвиг min_skill_level в xlsx (machines + operations).")
    p.add_argument("xlsx", type=Path, help="Путь к .xlsx")
    p.add_argument("--delta", type=int, default=-1, help="На сколько изменить уровень (обычно -1).")
    p.add_argument("--floor", type=int, default=1, help="Минимально допустимый уровень после сдвига.")
    p.add_argument("--cap", type=int, default=6, help="Максимально допустимый уровень после сдвига.")
    p.add_argument("--no-backup", action="store_true", help="Не создавать .bak.xlsx рядом с файлом.")
    args = p.parse_args()

    path = args.xlsx.resolve()
    if not path.is_file():
        raise SystemExit(f"Файл не найден: {path}")

    if not args.no_backup:
        bak = path.with_name(path.stem + ".bak" + path.suffix)
        shutil.copy2(path, bak)
        print(f"Резервная копия: {bak}")

    sheets: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(path) as xls:
        for name in xls.sheet_names:
            sheets[name] = pd.read_excel(xls, sheet_name=name)

    for name in ("machines", "operations"):
        if name not in sheets:
            continue
        df = sheets[name]
        if "min_skill_level" not in df.columns:
            continue
        s = pd.to_numeric(df["min_skill_level"], errors="coerce").fillna(args.floor).astype(int)
        adj = (s + args.delta).clip(lower=args.floor, upper=args.cap)
        df = df.copy()
        df["min_skill_level"] = adj.astype(int)
        sheets[name] = df

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"Обновлено: {path} (delta={args.delta}, floor={args.floor}, cap={args.cap})")


if __name__ == "__main__":
    main()

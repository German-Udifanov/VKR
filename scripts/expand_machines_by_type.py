"""
Добавляет строки на лист machines: для каждого machine_type — N клонов
первой строки этого типа (required_skill, min_skill_level, cycle_time_min,
operators_required), с новыми machine_id.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from app.data_io import load_all_tables, validate_tables


DEFAULT_ADD: dict[str, int] = {
    "ASSEMBLY_BENCH": 1,
    "CNC": 2,
    "DRILL_DOWEL": 1,
    "EDGE_BANDER": 1,
    "PACKING_TABLE": 2,
    "PANEL_SAW": 1,
    "QC_STATION": 2,
    "SANDER": 2,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("xlsx", type=Path, nargs="?", default=Path("data_out/synthetic_medium_150_new.xlsx"))
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    path = (_ROOT / args.xlsx).resolve() if not args.xlsx.is_absolute() else args.xlsx.resolve()
    if not path.is_file():
        raise SystemExit(f"Не найден: {path}")

    if not args.no_backup:
        bak = path.with_name(path.stem + "_machines.bak" + path.suffix)
        shutil.copy2(path, bak)
        print(f"Резервная копия: {bak}")

    sheets: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(path) as xls:
        for name in xls.sheet_names:
            sheets[name] = pd.read_excel(xls, sheet_name=name)

    m = sheets["machines"].copy()
    next_id = int(m["machine_id"].max()) + 1
    new_rows: list[dict] = []

    for mtype, n_add in DEFAULT_ADD.items():
        if n_add <= 0:
            continue
        sub = m[m["machine_type"].astype(str) == mtype]
        if sub.empty:
            raise SystemExit(f"Нет ни одного станка типа {mtype!r} — нечем шаблониться.")
        tpl = sub.iloc[0]
        for _ in range(n_add):
            new_rows.append(
                {
                    "machine_id": next_id,
                    "machine_type": mtype,
                    "required_skill": tpl["required_skill"],
                    "min_skill_level": int(tpl["min_skill_level"]),
                    "cycle_time_min": int(tpl["cycle_time_min"]),
                    "operators_required": int(tpl["operators_required"]),
                }
            )
            next_id += 1

    sheets["machines"] = pd.concat([m, pd.DataFrame(new_rows)], ignore_index=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    t = load_all_tables(path)
    validate_tables(t)
    print(f"Добавлено станков: {len(new_rows)}. Всего: {len(sheets['machines'])}")
    print(sheets["machines"].groupby("machine_type").size().sort_index().to_string())


if __name__ == "__main__":
    main()

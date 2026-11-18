"""
Добавляет каждому сотруднику один навык из маршрутов (operations.required_skill),
которого у него ещё нет: выбирается самый «узкий» (у кого меньше всего владельцев в штате).
Уровень нового навыка: 1 при нечётном employee_id, 2 при чётном.
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

from app.data_io import parse_skills


def _serialize_skills(d: dict[str, int]) -> str:
    parts = [f"{k}:{int(v)}" for k, v in sorted(d.items())]
    return "; ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("xlsx", type=Path, help="Путь к входному .xlsx")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    path = args.xlsx.resolve()
    if not path.is_file():
        raise SystemExit(f"Не найден: {path}")

    if not args.no_backup:
        bak = path.with_name(path.stem + "_skills.bak" + path.suffix)
        shutil.copy2(path, bak)
        print(f"Резервная копия: {bak}")

    sheets: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(path) as xls:
        for name in xls.sheet_names:
            sheets[name] = pd.read_excel(xls, sheet_name=name)

    emp = sheets["employees"].copy()
    ops = sheets["operations"]
    required = sorted({str(x).strip().upper() for x in ops["required_skill"].tolist() if str(x).strip()})

    # сколько сотрудников уже имеют навык (любой уровень)
    def coverage() -> dict[str, int]:
        cnt: dict[str, int] = {s: 0 for s in required}
        for raw in emp["skills"].astype(str):
            for sk in parse_skills(raw):
                if sk in cnt:
                    cnt[sk] += 1
        return cnt

    cov = coverage()
    narrow_order = sorted(required, key=lambda s: (cov[s], s))

    choices: list[tuple[int, str, int]] = []

    for _, row in emp.sort_values("employee_id").iterrows():
        eid = int(row["employee_id"])
        cur = parse_skills(row["skills"])
        pick = None
        for sk in narrow_order:
            if sk not in cur:
                pick = sk
                break
        if pick is None:
            raise SystemExit(f"Сотрудник {eid} уже имеет все навыки из маршрутов — добавление не требуется.")
        lvl = 1 if (eid % 2 == 1) else 2
        cur[pick] = lvl
        choices.append((eid, pick, lvl))

    # preserve original row order in sheet
    id_to_skill = {eid: s for eid, s, _ in choices}
    id_to_lvl = {eid: lv for eid, _, lv in choices}
    emp = sheets["employees"].copy()
    out_skills: list[str] = []
    for _, row in emp.iterrows():
        eid = int(row["employee_id"])
        cur = parse_skills(row["skills"])
        pick = id_to_skill[eid]
        if pick in cur:
            out_skills.append(_serialize_skills(cur))
            continue
        cur[pick] = id_to_lvl[eid]
        out_skills.append(_serialize_skills(cur))
    emp["skills"] = out_skills
    sheets["employees"] = emp

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"Обновлено: {path}")
    for eid, sk, lv in choices[:8]:
        print(f"  employee_id={eid} -> +{sk}:{lv}")
    if len(choices) > 8:
        print(f"  ... всего {len(choices)} сотрудников")


if __name__ == "__main__":
    main()

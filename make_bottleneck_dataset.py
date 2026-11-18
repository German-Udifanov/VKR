from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _parse_skills(s: str) -> dict[str, int]:
    text = str(s or "").strip()
    out: dict[str, int] = {}
    if not text:
        return out
    for chunk in text.split(";"):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        k = k.strip().upper()
        try:
            out[k] = int(v.strip())
        except ValueError:
            continue
    return out


def _format_skills(d: dict[str, int]) -> str:
    parts = [f"{k}:{int(v)}" for k, v in sorted(d.items()) if int(v) > 0]
    return ";".join(parts)


def make_bottleneck(inp: Path, out: Path, *, seed: int = 7) -> None:
    rng = random.Random(seed)
    required = ["employees", "machines", "orders", "vacations", "operations", "calendar", "shifts", "downtime"]
    tables = {s: pd.read_excel(inp, sheet_name=s) for s in required}

    emp = tables["employees"].copy()
    mac = tables["machines"].copy()
    down = tables["downtime"].copy()
    cal = tables["calendar"].copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.date

    # 1) Skill bottleneck: make CNC_ROUTER skill rare (only ~15% keep >=4)
    keep_ids = set(rng.sample(emp["employee_id"].astype(int).tolist(), k=max(1, int(len(emp) * 0.15))))
    new_skills = []
    for _, r in emp.iterrows():
        skills = _parse_skills(r.get("skills", ""))
        if int(r["employee_id"]) in keep_ids:
            if "CNC_ROUTER" in skills:
                skills["CNC_ROUTER"] = max(int(skills["CNC_ROUTER"]), rng.randint(4, 6))
            else:
                skills["CNC_ROUTER"] = rng.randint(4, 6)
        else:
            if "CNC_ROUTER" in skills:
                skills["CNC_ROUTER"] = min(int(skills["CNC_ROUTER"]), rng.choice([0, 1, 2]))
        new_skills.append(_format_skills(skills))
    emp["skills"] = new_skills
    tables["employees"] = emp

    # 2) Machine bottleneck: extra downtime for CNC machines (weekly 6h block)
    mac["machine_type"] = mac["machine_type"].astype(str).str.upper()
    cnc_ids = mac.loc[mac["machine_type"] == "CNC", "machine_id"].astype(int).tolist()
    if cnc_ids:
        start = min(cal["date"])
        end = max(cal["date"])
        d = start
        extra = []
        while d <= end:
            # Monday-ish: just every 7 days from start
            from_dt = datetime(d.year, d.month, d.day, 8, 0, 0)
            to_dt = from_dt + timedelta(hours=6)
            for mid in cnc_ids:
                extra.append({"machine_id": mid, "from_dt": from_dt, "to_dt": to_dt, "reason": "BOTTLE"})
            d = d + timedelta(days=7)
        down2 = pd.concat([down, pd.DataFrame(extra)], ignore_index=True)
        down2 = down2.sort_values(["machine_id", "from_dt"]).reset_index(drop=True)
        tables["downtime"] = down2

    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name in required:
            tables[name].to_excel(writer, sheet_name=name, index=False)

    print(f"Wrote: {out}")
    print(f"Employees: CNC_ROUTER keep_ids={len(keep_ids)}/{len(emp)}")
    print(f"Machines: CNC count={len(cnc_ids)}; downtime rows={len(tables['downtime'])}")


def main() -> int:
    inp = Path("data_out/synthetic_medium_150.xlsx")
    out = Path("data_out/synthetic_medium_150_bottleneck.xlsx")
    if not inp.exists():
        raise FileNotFoundError(f"Missing input: {inp}")
    make_bottleneck(inp, out, seed=7)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


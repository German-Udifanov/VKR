from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip()
        ans_u = ans.upper()
        # Accept Cyrillic, Latin and common Windows mojibake for CP1251/CP866 issues.
        yes = {"ДА", "Д", "YES", "Y", "DA", "D", "ÄÀ"}
        no = {"НЕТ", "Н", "NO", "N", "NET", "ÍÅÒ"}
        if ans_u in yes:
            return True
        if ans_u in no:
            return False
        print("Введите ДА/НЕТ (или DA/NET).")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def shift_type_by_time(dt: datetime) -> str:
    h = dt.hour
    return "NIGHT" if (h >= 20 or h < 8) else "DAY"


def to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    return date.fromisoformat(str(x))


def to_datetime(x) -> datetime:
    if isinstance(x, datetime):
        return x
    return datetime.fromisoformat(str(x))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class RuntimeConfig:
    input_xlsx: Path
    output_dir: Path
    step_min: int = 30
    time_limit_sec: int = 60
    workers: int = 8
    use_ml: bool = False
    # objective weights (passed through to SolveConfig)
    w_tardiness: int = 80
    w_balance: int = 30
    w_idle_day: int = 2000
    w_low_work_day: int = 15
    min_work_hours_per_day: float = 6.0
    w_emp_idle_day: int = 120
    # two-pass (phase 2) tuning
    two_pass: bool = True
    tardiness_cap_margin: float = 0.08
    two_pass_phase1_time_frac: float = 0.55
    phase2_idle_mult: float = 2.0
    phase2_low_work_mult: float = 2.0
    phase2_emp_idle_mult: float = 1.5


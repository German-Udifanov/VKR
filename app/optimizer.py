from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import pandas as pd
from ortools.sat.python import cp_model

from app.calendar_mod import _is_work_day_for_pattern
from app.data_io import parse_skills
from app.planning import build_precedence_pairs


@dataclass(frozen=True)
class SolveConfig:
    step_min: int
    time_limit_sec: int
    workers: int
    w_tardiness: int
    w_balance: int
    w_idle_day: int = 20
    w_low_work_day: int = 5
    min_work_hours_per_day: float = 6.0
    # (C) штраф: «рабочий по графику день», глобально ещё есть работа, а у сотрудника нет старта операции в этот день
    w_emp_idle_day: int = 0
    # (D) второй проход: не ухудшать взвешенную просрочку больше чем на margin, усилить штрафы простоя
    two_pass: bool = True
    tardiness_cap_margin: float = 0.08
    two_pass_phase1_time_frac: float = 0.55
    phase2_idle_mult: float = 2.0
    phase2_low_work_mult: float = 2.0
    phase2_emp_idle_mult: float = 1.5


def _to_step(ts: pd.Timestamp, horizon_start: pd.Timestamp, step_min: int) -> int:
    delta = ts.to_pydatetime() - horizon_start.to_pydatetime()
    minutes = int(delta.total_seconds() // 60)
    return max(0, minutes // step_min)


def _merge_blocks_to_steps(
    blocks: list[tuple[datetime, datetime]],
    *,
    horizon_start: pd.Timestamp,
    step_min: int,
    horizon_steps: int,
) -> list[tuple[int, int]]:
    """
    Converts datetime blocks to (start_step, end_step) and merges overlaps.
    end_step is exclusive.
    """
    items: list[tuple[int, int]] = []
    for sdt, edt in blocks:
        s = _to_step(pd.Timestamp(sdt), horizon_start, step_min)
        e = _to_step(pd.Timestamp(edt), horizon_start, step_min)
        s = max(0, min(horizon_steps, s))
        e = max(0, min(horizon_steps, e))
        if e <= s:
            continue
        items.append((s, e))
    items.sort()
    merged: list[tuple[int, int]] = []
    for s, e in items:
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
    return merged


def _interval_conflicts(intervals: list[tuple[int, int]], s: int, e: int) -> bool:
    for a, b in intervals:
        if a < e and b > s:
            return True
    return False


def _add_busy(busy: list[tuple[int, int]], s: int, e: int) -> None:
    busy.append((s, e))
    busy.sort()
    # merge
    merged: list[tuple[int, int]] = []
    for a, b in busy:
        if not merged:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            if a <= pb:
                merged[-1] = (pa, max(pb, b))
            else:
                merged.append((a, b))
    busy[:] = merged


def build_heuristic_hints(
    ops_df: pd.DataFrame,
    *,
    orders_df: pd.DataFrame,
    employees_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    employee_unavailability: dict[int, list[tuple[datetime, datetime]]],
    downtime_df: pd.DataFrame,
    horizon_start: pd.Timestamp,
    horizon_steps: int,
    step_min: int,
) -> dict[int, dict[str, int]]:
    """
    Greedy heuristic that assigns each operation to one employee+machine and a start time.
    Returns hints dict: op_id -> {start_step, end_step, employee_id, machine_id}.
    """
    machine_ids_by_type = (
        machines_df.groupby("machine_type")["machine_id"].apply(lambda s: [int(v) for v in s]).to_dict()
    )
    employee_skill_map = {int(r["employee_id"]): parse_skills(str(r["skills"])) for _, r in employees_df.iterrows()}

    # Precompute unavailability in steps
    emp_block_steps: dict[int, list[tuple[int, int]]] = {}
    for eid, blocks in employee_unavailability.items():
        emp_block_steps[int(eid)] = _merge_blocks_to_steps(
            blocks, horizon_start=horizon_start, step_min=step_min, horizon_steps=horizon_steps
        )

    mach_block_steps: dict[int, list[tuple[int, int]]] = {}
    for mid, grp in downtime_df.groupby("machine_id"):
        blocks = [(pd.Timestamp(r["from_dt"]).to_pydatetime(), pd.Timestamp(r["to_dt"]).to_pydatetime()) for _, r in grp.iterrows()]
        mach_block_steps[int(mid)] = _merge_blocks_to_steps(
            blocks, horizon_start=horizon_start, step_min=step_min, horizon_steps=horizon_steps
        )

    emp_busy: dict[int, list[tuple[int, int]]] = {int(eid): [] for eid in employees_df["employee_id"].astype(int).tolist()}
    mach_busy: dict[int, list[tuple[int, int]]] = {int(mid): [] for mid in machines_df["machine_id"].astype(int).tolist()}

    # precedence tracking: prev end per order
    prev_end_by_order: dict[int, int] = {}

    hints: dict[int, dict[str, int]] = {}

    # Schedule operations in a reasonable order
    ops = ops_df.copy().sort_values(["due_at", "order_id", "op_seq"])
    for _, op in ops.iterrows():
        op_id = int(op["op_id"])
        order_id = int(op["order_id"])
        dur = int(op["duration_steps"])
        release = _to_step(pd.Timestamp(op["created_at"]), horizon_start, step_min)
        earliest = max(release, int(prev_end_by_order.get(order_id, 0)))

        req_skill = str(op["required_skill"]).upper()
        req_lvl = int(op["min_skill_level"])
        e_cand = [eid for eid, sk in employee_skill_map.items() if sk.get(req_skill, 0) >= req_lvl]
        m_cand = machine_ids_by_type.get(str(op["machine_type"]), [])
        if not e_cand or not m_cand:
            continue

        best = None  # (end, start, eid, mid)
        for eid in e_cand:
            e_blocks = emp_block_steps.get(eid, [])
            e_busy = emp_busy.get(eid, [])
            for mid in m_cand:
                m_blocks = mach_block_steps.get(mid, [])
                m_busy = mach_busy.get(mid, [])
                s = earliest
                # scan forward to find feasible slot
                # cap scanning to horizon_steps - dur
                cap = max(0, horizon_steps - dur - 1)
                while s <= cap:
                    e = s + dur
                    if _interval_conflicts(e_blocks, s, e) or _interval_conflicts(e_busy, s, e):
                        s += 1
                        continue
                    if _interval_conflicts(m_blocks, s, e) or _interval_conflicts(m_busy, s, e):
                        s += 1
                        continue
                    # feasible
                    cand = (e, s, eid, mid)
                    if best is None or cand < best:
                        best = cand
                    break

        if best is None:
            continue
        end_step, start_step, eid, mid = best
        hints[op_id] = {
            "start_step": int(start_step),
            "end_step": int(end_step),
            "employee_id": int(eid),
            "machine_id": int(mid),
        }
        _add_busy(emp_busy[eid], start_step, end_step)
        _add_busy(mach_busy[mid], start_step, end_step)
        prev_end_by_order[order_id] = end_step

    return hints


def solve_hybrid(
    ops_df: pd.DataFrame,
    *,
    orders_df: pd.DataFrame,
    employees_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    employee_unavailability: dict[int, list[tuple[datetime, datetime]]],
    downtime_df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
    vacations_df: pd.DataFrame | None = None,
    cfg: SolveConfig,
) -> pd.DataFrame:
    """
    Hybrid: greedy heuristic (hints) -> CP-SAT.
    При two_pass: фаза 1 — текущая цель; фаза 2 — ограничение на взвешенную просрочку + усиленные штрафы простоя (п. D).
    """
    min_created = pd.to_datetime(orders_df["created_at"]).min().floor("D")
    max_due = pd.to_datetime(orders_df["due_at"]).max() + pd.Timedelta(days=120)
    horizon_start = pd.Timestamp(min_created)
    horizon_steps = int(((max_due - horizon_start).total_seconds() // 60) // cfg.step_min) + 1
    steps_per_day = int(1440 // cfg.step_min)
    idle_end = pd.to_datetime(orders_df["due_at"]).max().ceil("D")
    horizon_days = int(((idle_end - horizon_start).total_seconds() // 60) // 1440) + 1

    machine_ids_by_type = machines_df.groupby("machine_type")["machine_id"].apply(lambda s: [int(v) for v in s]).to_dict()
    employee_skill_map = {int(r["employee_id"]): parse_skills(str(r["skills"])) for _, r in employees_df.iterrows()}
    employee_ids = [int(v) for v in employees_df["employee_id"].tolist()]

    eligible_emp_day: set[tuple[int, int]] = set()
    if calendar_df is not None and not calendar_df.empty:
        cal = calendar_df.copy()
        cal["date"] = pd.to_datetime(cal["date"]).dt.date
        base_date = cal["date"].min()
        rf_map = {row["date"]: int(row["is_workday_rf"]) for _, row in cal.iterrows()}
        vac = vacations_df if vacations_df is not None else pd.DataFrame()
        for d in range(horizon_days):
            cur = (horizon_start + pd.Timedelta(days=d)).date()
            irf = int(rf_map.get(cur, 1))
            for _, er in employees_df.iterrows():
                eid = int(er["employee_id"])
                pat = str(er.get("shift_pattern", "2/2")).strip()
                anchor = int(er.get("cycle_anchor", 0))
                if not _is_work_day_for_pattern(
                    cur, pattern=pat, cycle_anchor=anchor, is_workday_rf=irf, base_date=base_date
                ):
                    continue
                on_vac = False
                if not vac.empty and "employee_id" in vac.columns:
                    sub = vac[vac["employee_id"].astype(int) == eid]
                    for _, vr in sub.iterrows():
                        d0 = pd.to_datetime(vr["date_from"]).date()
                        d1 = pd.to_datetime(vr["date_to"]).date()
                        if d0 <= cur <= d1:
                            on_vac = True
                            break
                if on_vac:
                    continue
                eligible_emp_day.add((eid, d))

    heuristic_hints = build_heuristic_hints(
        ops_df,
        orders_df=orders_df,
        employees_df=employees_df,
        machines_df=machines_df,
        employee_unavailability=employee_unavailability,
        downtime_df=downtime_df,
        horizon_start=horizon_start,
        horizon_steps=horizon_steps,
        step_min=cfg.step_min,
    )

    def build_model(
        *,
        include_idle_days: bool,
        solve_cfg: SolveConfig,
        weighted_tard_ub: int | None,
    ) -> tuple[
        cp_model.CpModel,
        dict[int, cp_model.IntVar],
        dict[int, cp_model.IntVar],
        dict[tuple[int, int], cp_model.BoolVar],
        dict[tuple[int, int], cp_model.BoolVar],
        cp_model.IntVar,
    ]:
        sc = solve_cfg
        model = cp_model.CpModel()
        op_start: dict[int, cp_model.IntVar] = {}
        op_end: dict[int, cp_model.IntVar] = {}
        op_machine_present: dict[tuple[int, int], cp_model.BoolVar] = {}
        op_employee_present: dict[tuple[int, int], cp_model.BoolVar] = {}

        machine_intervals: dict[int, list[cp_model.IntervalVar]] = {
            int(mid): [] for mid in machines_df["machine_id"].tolist()
        }
        employee_intervals: dict[int, list[cp_model.IntervalVar]] = {int(eid): [] for eid in employee_ids}

        for _, op in ops_df.iterrows():
            oid = int(op["op_id"])
            dur = int(op["duration_steps"])
            s = model.new_int_var(0, horizon_steps, f"s_{oid}")
            e = model.new_int_var(0, horizon_steps, f"e_{oid}")
            model.add(e == s + dur)
            op_start[oid] = s
            op_end[oid] = e

            release = _to_step(pd.Timestamp(op["created_at"]), horizon_start, sc.step_min)
            model.add(s >= release)

            m_cand = machine_ids_by_type.get(str(op["machine_type"]), [])
            if not m_cand:
                raise ValueError(f"Нет станков типа {op['machine_type']}")
            m_bools = []
            for mid in m_cand:
                b = model.new_bool_var(f"om_{oid}_{mid}")
                iv = model.new_optional_interval_var(s, dur, e, b, f"ivm_{oid}_{mid}")
                machine_intervals[mid].append(iv)
                op_machine_present[(oid, mid)] = b
                m_bools.append(b)
            model.add_exactly_one(m_bools)

            req_skill = str(op["required_skill"]).upper()
            req_lvl = int(op["min_skill_level"])
            e_cand = [eid for eid, sk in employee_skill_map.items() if sk.get(req_skill, 0) >= req_lvl]
            if not e_cand:
                raise ValueError(f"Нет сотрудников для {req_skill} lvl>={req_lvl}")
            e_bools = []
            for eid in e_cand:
                b = model.new_bool_var(f"oe_{oid}_{eid}")
                iv = model.new_optional_interval_var(s, dur, e, b, f"ive_{oid}_{eid}")
                employee_intervals[eid].append(iv)
                op_employee_present[(oid, eid)] = b
                e_bools.append(b)
            model.add_exactly_one(e_bools)

        for mid, ivs in machine_intervals.items():
            fixed = []
            md = downtime_df[downtime_df["machine_id"].astype(int) == int(mid)]
            for idx, r in md.iterrows():
                ds = _to_step(pd.Timestamp(r["from_dt"]), horizon_start, sc.step_min)
                de = _to_step(pd.Timestamp(r["to_dt"]), horizon_start, sc.step_min)
                ds = max(0, ds)
                de = min(horizon_steps, de)
                if de <= ds:
                    continue
                fixed.append(model.new_fixed_size_interval_var(ds, max(1, de - ds), f"down_{mid}_{idx}"))
            model.add_no_overlap(ivs + fixed)

        for eid, ivs in employee_intervals.items():
            fixed = []
            blocks = employee_unavailability.get(int(eid), [])
            merged: list[tuple[datetime, datetime]] = []
            for sdt, edt in sorted(blocks, key=lambda x: x[0]):
                if not merged:
                    merged.append((sdt, edt))
                else:
                    ps, pe = merged[-1]
                    if sdt <= pe:
                        merged[-1] = (ps, max(pe, edt))
                    else:
                        merged.append((sdt, edt))
            for i, (sdt, edt) in enumerate(merged):
                ds = _to_step(pd.Timestamp(sdt), horizon_start, sc.step_min)
                de = _to_step(pd.Timestamp(edt), horizon_start, sc.step_min)
                ds = max(0, ds)
                de = min(horizon_steps, de)
                if de <= ds:
                    continue
                fixed.append(model.new_fixed_size_interval_var(ds, max(1, de - ds), f"un_{eid}_{i}"))
            model.add_no_overlap(ivs + fixed)

        for a, b in build_precedence_pairs(ops_df):
            model.add(op_start[b] >= op_end[a])

        tard_terms: list = []
        for _, order in orders_df.iterrows():
            oid = int(order["order_id"])
            due_step = _to_step(pd.Timestamp(order["due_at"]), horizon_start, sc.step_min)
            penalty = int(round(float(order.get("late_penalty", 1.0)) * 10.0))
            last = ops_df[ops_df["order_id"] == oid].sort_values("op_seq").tail(1)
            if last.empty:
                continue
            last_op_id = int(last.iloc[0]["op_id"])
            t = model.new_int_var(0, horizon_steps, f"t_{oid}")
            model.add(t >= op_end[last_op_id] - due_step)
            model.add(t >= 0)
            tard_terms.append(t * penalty)

        tard_upper = horizon_steps * max(1, len(tard_terms)) * 500
        if tard_terms:
            tard_sum = model.new_int_var(0, tard_upper, "weighted_tardiness")
            model.add(tard_sum == sum(tard_terms))
        else:
            tard_sum = model.new_int_var(0, 0, "weighted_tardiness")
            model.add(tard_sum == 0)
        if weighted_tard_ub is not None:
            model.add(tard_sum <= int(weighted_tard_ub))

        workloads: dict[int, cp_model.IntVar] = {}
        for eid in employee_ids:
            terms = []
            for (op_id, ee), b in op_employee_present.items():
                if ee == eid:
                    terms.append(b * int(ops_df.loc[ops_df["op_id"] == op_id, "duration_steps"].iloc[0]))
            wv = model.new_int_var(0, horizon_steps * 2, f"work_{eid}")
            if terms:
                model.add(wv == sum(terms))
            else:
                model.add(wv == 0)
            workloads[eid] = wv

        max_w = model.new_int_var(0, horizon_steps * 2, "max_work")
        min_w = model.new_int_var(0, horizon_steps * 2, "min_work")
        model.add_max_equality(max_w, list(workloads.values()))
        model.add_min_equality(min_w, list(workloads.values()))
        imbalance = model.new_int_var(0, horizon_steps * 2, "imbalance")
        model.add(imbalance == max_w - min_w)

        idle_terms: list = []
        low_work_slacks: list = []
        emp_idle_terms: list = []
        op_in_day: dict[tuple[int, int], cp_model.BoolVar] = {}
        day_work_rem: dict[int, cp_model.BoolVar] = {}

        need_day_layer = include_idle_days and (
            sc.w_idle_day > 0 or sc.w_low_work_day > 0 or (sc.w_emp_idle_day > 0 and len(eligible_emp_day) > 0)
        )
        if need_day_layer:
            makespan = model.new_int_var(0, horizon_steps, "makespan")
            model.add_max_equality(makespan, list(op_end.values()))

            for d in range(horizon_days):
                day_start = d * steps_per_day
                day_end = min(horizon_steps, (d + 1) * steps_per_day)

                w_d = model.new_bool_var(f"work_remaining_{d}")
                model.add(makespan >= day_start + 1).only_enforce_if(w_d)
                model.add(makespan <= day_start).only_enforce_if(w_d.Not())
                day_work_rem[d] = w_d

                in_day_vars: list = []
                for _, op in ops_df.iterrows():
                    op_id = int(op["op_id"])
                    b1 = model.new_bool_var(f"op{op_id}_start_before_day{d}_end")
                    b2 = model.new_bool_var(f"op{op_id}_end_after_day{d}_start")
                    in_day = model.new_bool_var(f"op{op_id}_in_day{d}")

                    model.add(op_start[op_id] <= day_end - 1).only_enforce_if(b1)
                    model.add(op_start[op_id] >= day_end).only_enforce_if(b1.Not())

                    model.add(op_end[op_id] >= day_start + 1).only_enforce_if(b2)
                    model.add(op_end[op_id] <= day_start).only_enforce_if(b2.Not())

                    model.add(in_day <= b1)
                    model.add(in_day <= b2)
                    model.add(in_day >= b1 + b2 - 1)
                    in_day_vars.append(in_day)
                    op_in_day[(op_id, d)] = in_day

                u_d = model.new_bool_var(f"day_used_{d}")
                model.add_max_equality(u_d, in_day_vars)

                if include_idle_days and sc.w_idle_day > 0:
                    idle_d = model.new_bool_var(f"idle_day_{d}")
                    model.add(idle_d >= w_d - u_d)
                    model.add(idle_d <= w_d)
                    idle_terms.append(idle_d)

                if sc.w_low_work_day > 0 and sc.min_work_hours_per_day > 0:
                    target_steps = int(
                        max(0, round(float(sc.min_work_hours_per_day) * 60.0 / float(sc.step_min)))
                    )
                    if target_steps > 0:
                        started_terms = []
                        for _, op in ops_df.iterrows():
                            op_id = int(op["op_id"])
                            dur = int(op["duration_steps"])
                            b_ge = model.new_bool_var(f"op{op_id}_start_ge_day{d}")
                            b_le = model.new_bool_var(f"op{op_id}_start_le_day{d}")
                            b = model.new_bool_var(f"op{op_id}_starts_in_day{d}")

                            model.add(op_start[op_id] >= day_start).only_enforce_if(b_ge)
                            model.add(op_start[op_id] <= day_start - 1).only_enforce_if(b_ge.Not())

                            model.add(op_start[op_id] <= day_end - 1).only_enforce_if(b_le)
                            model.add(op_start[op_id] >= day_end).only_enforce_if(b_le.Not())

                            model.add(b <= b_ge)
                            model.add(b <= b_le)
                            model.add(b >= b_ge + b_le - 1)
                            started_terms.append(b * dur)

                        work_started = model.new_int_var(0, horizon_steps * 2, f"work_started_{d}")
                        if started_terms:
                            model.add(work_started == sum(started_terms))
                        else:
                            model.add(work_started == 0)

                        slack = model.new_int_var(0, target_steps, f"low_work_slack_{d}")
                        model.add(work_started + slack >= target_steps).only_enforce_if(w_d)
                        model.add(slack == 0).only_enforce_if(w_d.Not())
                        low_work_slacks.append(slack)

        if (
            include_idle_days
            and sc.w_emp_idle_day > 0
            and len(eligible_emp_day) > 0
            and day_work_rem
            and op_in_day
        ):
            for (eid, d) in sorted(eligible_emp_day):
                w_d = day_work_rem.get(d)
                if w_d is None:
                    continue
                z_terms: list = []
                for _, op in ops_df.iterrows():
                    op_id = int(op["op_id"])
                    y = op_employee_present.get((op_id, eid))
                    ind = op_in_day.get((op_id, d))
                    if y is None or ind is None:
                        continue
                    z = model.new_bool_var(f"z_op{op_id}_e{eid}_d{d}")
                    model.add(z <= y)
                    model.add(z <= ind)
                    model.add(z >= y + ind - 1)
                    z_terms.append(z)
                if not z_terms:
                    continue
                u_ed = model.new_bool_var(f"emp_used_day_e{eid}_d{d}")
                model.add_max_equality(u_ed, z_terms)
                idle_ed = model.new_bool_var(f"emp_idle_e{eid}_d{d}")
                model.add(idle_ed >= w_d - u_ed)
                model.add(idle_ed <= w_d)
                emp_idle_terms.append(idle_ed)

        obj = sc.w_tardiness * tard_sum + sc.w_balance * imbalance
        if idle_terms:
            obj += sc.w_idle_day * sum(idle_terms)
        if low_work_slacks:
            obj += sc.w_low_work_day * sum(low_work_slacks)
        if emp_idle_terms:
            obj += sc.w_emp_idle_day * sum(emp_idle_terms)
        model.minimize(obj)

        return model, op_start, op_end, op_machine_present, op_employee_present, tard_sum

    def apply_hints(m: cp_model.CpModel, os, oend, omp, oep) -> None:
        for op_id, hint in heuristic_hints.items():
            if op_id in os:
                m.add_hint(os[op_id], int(hint["start_step"]))
            if op_id in oend:
                m.add_hint(oend[op_id], int(hint["end_step"]))
            mid = int(hint["machine_id"])
            eid = int(hint["employee_id"])
            if (op_id, mid) in omp:
                m.add_hint(omp[(op_id, mid)], 1)
            if (op_id, eid) in oep:
                m.add_hint(oep[(op_id, eid)], 1)

    def solve_once(m: cp_model.CpModel, *, time_limit_sec: float | None = None) -> tuple[cp_model.CpSolver, int]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_sec if time_limit_sec is not None else cfg.time_limit_sec)
        solver.parameters.num_search_workers = int(cfg.workers)
        status = solver.solve(m)
        return solver, int(status)

    p1_time = (
        float(cfg.time_limit_sec) * float(cfg.two_pass_phase1_time_frac)
        if cfg.two_pass
        else float(cfg.time_limit_sec)
    )
    p2_time = max(1.0, float(cfg.time_limit_sec) - p1_time) if cfg.two_pass else 0.0

    model2, op_start2, op_end2, om2, oe2, tard_sum2 = build_model(
        include_idle_days=True, solve_cfg=cfg, weighted_tard_ub=None
    )
    apply_hints(model2, op_start2, op_end2, om2, oe2)
    solver2, status2 = solve_once(model2, time_limit_sec=p1_time)

    if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solver, op_start, op_end, op_machine_present, op_employee_present = (
            solver2,
            op_start2,
            op_end2,
            om2,
            oe2,
        )
    else:
        model1, op_start1, op_end1, om1, oe1, _tard1 = build_model(
            include_idle_days=False, solve_cfg=cfg, weighted_tard_ub=None
        )
        apply_hints(model1, op_start1, op_end1, om1, oe1)
        solver1, status1 = solve_once(model1, time_limit_sec=float(cfg.time_limit_sec))
        if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return pd.DataFrame()
        solver, op_start, op_end, op_machine_present, op_employee_present = (
            solver1,
            op_start1,
            op_end1,
            om1,
            oe1,
        )

    if (
        cfg.two_pass
        and status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        and p2_time >= 1.0
    ):
        t1 = int(solver2.Value(tard_sum2))
        ub = int(t1 * (1.0 + float(cfg.tardiness_cap_margin))) + 1
        cfg2 = replace(
            cfg,
            w_idle_day=max(1, int(cfg.w_idle_day * float(cfg.phase2_idle_mult))),
            w_low_work_day=max(1, int(cfg.w_low_work_day * float(cfg.phase2_low_work_mult))),
            w_emp_idle_day=int(cfg.w_emp_idle_day * float(cfg.phase2_emp_idle_mult)) if cfg.w_emp_idle_day else 0,
        )
        model3, op_start3, op_end3, om3, oe3, _tard3 = build_model(
            include_idle_days=True, solve_cfg=cfg2, weighted_tard_ub=ub
        )
        apply_hints(model3, op_start3, op_end3, om3, oe3)
        for op_id in op_start2:
            if op_id in op_start3:
                model3.add_hint(op_start3[op_id], int(solver2.value(op_start2[op_id])))
            if op_id in op_end3:
                model3.add_hint(op_end3[op_id], int(solver2.value(op_end2[op_id])))
        for (oid, mid), b in om2.items():
            if (oid, mid) in om3 and int(solver2.value(b)) == 1:
                model3.add_hint(om3[(oid, mid)], 1)
        for (oid, eid), b in oe2.items():
            if (oid, eid) in oe3 and int(solver2.value(b)) == 1:
                model3.add_hint(oe3[(oid, eid)], 1)
        solver3, status3 = solve_once(model3, time_limit_sec=p2_time)
        if status3 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solver, op_start, op_end, op_machine_present, op_employee_present = (
                solver3,
                op_start3,
                op_end3,
                om3,
                oe3,
            )

    # Extract schedule
    rows = []
    for _, op in ops_df.iterrows():
        op_id = int(op["op_id"])
        s = int(solver.value(op_start[op_id]))
        e = int(solver.value(op_end[op_id]))
        start_dt = horizon_start + pd.Timedelta(minutes=s * cfg.step_min)
        end_dt = horizon_start + pd.Timedelta(minutes=e * cfg.step_min)
        machine_id = next(mid for (oid, mid), b in op_machine_present.items() if oid == op_id and solver.value(b) == 1)
        employee_id = next(eid for (oid, eid), b in op_employee_present.items() if oid == op_id and solver.value(b) == 1)
        rows.append(
            {
                "op_id": op_id,
                "order_id": int(op["order_id"]),
                "op_seq": int(op["op_seq"]),
                "machine_type": str(op["machine_type"]),
                "required_skill": str(op["required_skill"]),
                "machine_id": int(machine_id),
                "employee_id": int(employee_id),
                "start_dt": start_dt,
                "end_dt": end_dt,
                "duration_steps": int(op["duration_steps"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["start_dt", "order_id", "op_seq"]).reset_index(drop=True)


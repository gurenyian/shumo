"""Uniformly repair five-core plans with a long, overloaded task or core.

The incumbent remains authoritative: every alternative is checked and scored,
and it replaces the incumbent only when the evaluator reports an improvement.
"""

import argparse
import csv
from collections import Counter

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from fast_solver import GraphModel, list_schedule, make_plan
from hybrid_solver import groups_from_plan, official_score, signature
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order
from rescue_fivecore import OUTPUT, key, rows_by_case, compare_table, write_chart, write_final_results


def split_heavy_groups(gm, plan, cores, cfg, waits, target_factor):
    """Slice oversized incumbent tasks in operation topological order.

    Splitting a vertex of an acyclic task graph into ordered pieces keeps the
    task graph acyclic; unlike a free-form reassignment it cannot create a
    dependency cycle via another task. The evaluator decides whether the extra
    task boundaries cost more than the parallelism they expose.
    """
    if target_factor <= 0:
        raise ValueError("target_factor must be positive")
    groups, _ = groups_from_plan(plan)
    position = {v: i for i, v in enumerate(gm.order)}
    target = max(max(gm.pipes.values(), default=0) / cores * target_factor,
                 max((op["cycles"] for op in gm.ops.values()), default=1))
    pieces = []
    split_groups = 0
    for group in groups:
        work = Counter()
        for v in group:
            work[gm.ops[v]["pipe"]] += gm.ops[v]["cycles"]
        if max(work.values(), default=0) <= target:
            pieces.append(group)
            continue
        split_groups += 1
        current, current_work = [], Counter()
        for v in sorted(group, key=position.__getitem__):
            pipe, cycles = gm.ops[v]["pipe"], gm.ops[v]["cycles"]
            if current and current_work[pipe] + cycles > target:
                pieces.append(current)
                current, current_work = [], Counter()
            current.append(v)
            current_work[pipe] += cycles
        if current:
            pieces.append(current)
    schedules, _ = list_schedule(gm, pieces, cores, cfg, waits)
    candidate = make_plan(pieces, schedules)
    validate_task_order(derive_multicore_plan(gm.graph, candidate))
    return candidate, {"target_pipe_cycles": target, "split_groups": split_groups,
                       "tasks_before": len(groups), "tasks_after": len(pieces),
                       "core_tasks": [len(order) for order in schedules]}


def branch_chain_plan(gm, cores, cfg, waits, target_factor):
    """Keep fork/join boundaries visible while contracting exclusive chains.

    An edge p->v is contracted only when p has one successor and v one
    predecessor. Thus no other task can enter or leave the contracted edge,
    and the resulting quotient remains acyclic. A work cap prevents a long
    serial chain from swallowing most of the graph.
    """
    target = max(max(gm.pipes.values(), default=0) / cores * target_factor,
                 max((op["cycles"] for op in gm.ops.values()), default=1))
    groups, owner, work = [], {}, []
    for v in gm.order:
        pipe, cycles = gm.ops[v]["pipe"], gm.ops[v]["cycles"]
        p = next(iter(gm.pred[v])) if len(gm.pred[v]) == 1 else None
        previous = owner.get(p) if p is not None and len(gm.succ[p]) == 1 else None
        if previous is not None and work[previous][pipe] + cycles <= target:
            i = previous
        else:
            i = len(groups)
            groups.append([])
            work.append(Counter())
        groups[i].append(v)
        work[i][pipe] += cycles
        owner[v] = i
    schedules, _ = list_schedule(gm, groups, cores, cfg, waits)
    candidate = make_plan(groups, schedules)
    validate_task_order(derive_multicore_plan(gm.graph, candidate))
    return candidate, {"target_pipe_cycles": target, "tasks": len(groups),
                       "core_tasks": [len(order) for order in schedules]}


def run_case(case, gm, cfg, waits, timeout, factors, branches=False,
             branch_min_fraction=0.15):
    folder = OUTPUT / case
    summary = _read_json(folder / "summary.json")
    incumbent = _read_json(folder / "best_result.json")
    incumbent_plan = _read_json(folder / "best_plan.json")
    checked = {item.get("candidate") for item in summary["candidate_history"]}
    for factor in factors:
        label = "heavy_split_" + str(factor).replace(".", "p")
        if label in checked:
            continue
        try:
            candidate, detail = split_heavy_groups(gm, incumbent_plan, 5, cfg, waits, factor)
            if signature(candidate) == signature(incumbent_plan):
                result, info = None, {"status": "duplicate"}
            else:
                probe = OUTPUT / "heavy_probes" / case
                cached = probe / f"{label}_result.json"
                if cached.exists():
                    result = _read_json(cached)
                    info = {"status": "valid", "makespan": result["makespan"], "cached": True}
                else:
                    result, info = official_score(case, candidate, probe, label, timeout,
                                                  fast_evaluator=True)
            info["construction"] = detail
        except (ValueError, RuntimeError) as error:
            result, info = None, {"status": "invalid", "message": str(error)[:300]}
        summary["candidate_history"].append({"candidate": label, **info})
        if result is not None and key(result) < key(incumbent):
            incumbent, incumbent_plan = result, candidate
            summary["selected"] = label
            summary["best_makespan"] = result["makespan"]
        save(folder / "best_result.json", incumbent)
        save(folder / "best_plan.json", incumbent_plan)
        save(folder / "summary.json", summary)
        print(case, label, info["status"], "best", incumbent["makespan"], flush=True)
    largest_task = max((item["local_makespan"] for item in
                        incumbent["step3_by_task"].values()), default=0)
    if (not branches or "branch_chains" in checked or
            largest_task / max(incumbent["makespan"], 1) < branch_min_fraction):
        return
    try:
        candidate, detail = branch_chain_plan(gm, 5, cfg, waits, 1.0)
        if signature(candidate) == signature(incumbent_plan):
            result, info = None, {"status": "duplicate"}
        else:
            probe = OUTPUT / "heavy_probes" / case
            cached = probe / "branch_chains_result.json"
            if cached.exists():
                result = _read_json(cached)
                info = {"status": "valid", "makespan": result["makespan"], "cached": True}
            else:
                result, info = official_score(case, candidate, probe, "branch_chains", timeout,
                                              fast_evaluator=True)
        info["construction"] = detail
    except (ValueError, RuntimeError) as error:
        result, info = None, {"status": "invalid", "message": str(error)[:300]}
    summary["candidate_history"].append({"candidate": "branch_chains", **info})
    if result is not None and key(result) < key(incumbent):
        incumbent, incumbent_plan = result, candidate
        summary["selected"] = "branch_chains"
        summary["best_makespan"] = result["makespan"]
    save(folder / "best_result.json", incumbent)
    save(folder / "best_plan.json", incumbent_plan)
    save(folder / "summary.json", summary)
    print(case, "branch_chains", info["status"], "best", incumbent["makespan"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--factors", nargs="+", type=float, default=[1.0, 0.5])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--branch-chains", action="store_true")
    parser.add_argument("--branch-min-fraction", type=float, default=0.15)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Optional case names for a focused diagnostic run")
    args = parser.parse_args()
    if (args.threshold <= 0 or args.timeout <= 0 or args.limit < 0 or
            args.branch_min_fraction <= 0 or any(x <= 0 for x in args.factors)):
        parser.error("threshold, timeout and factors must be positive")
    cfg = read_evaluation_config(str(ROOT / "official/data/config.txt"))
    waits = read_scene_a_config(str(ROOT / "official/data/config.txt"))
    source = rows_by_case()
    with (OUTPUT / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        comparison = list(csv.DictReader(stream))
    selected = sorted((row for row in comparison if row["new_best_speedup"] and
                       float(row["new_best_speedup"]) < args.threshold and
                       (args.cases is None or row["case"] in args.cases)),
                      key=lambda row: (float(row["new_best_speedup"]), row["case"]))
    if args.limit:
        selected = selected[:args.limit]
    print("Selected cases:", len(selected), flush=True)
    for row in selected:
        case = row["case"]
        graph = _read_json(ROOT / "official/data" / f"{case}.json")
        run_case(case, GraphModel(graph), cfg, waits, args.timeout, args.factors,
                 args.branch_chains, args.branch_min_fraction)
        compare_table(source)
    compare_table(source)
    write_final_results()
    write_chart(source)


if __name__ == "__main__":
    main()

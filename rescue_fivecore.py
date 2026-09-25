"""Officially re-score cheap parallel alternatives for weak 5-core cases.

Existing results/experiments/results_all100 scores are never overwritten. The
script keeps the better official score in results/experiments/results_opt5/
comparison.csv and checkpoints each case.
Example: python rescue_fivecore.py --threshold 2 --timeout 30
"""

import argparse
import csv
import json
from pathlib import Path

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from fast_solver import GraphModel, construct, estimate, independent_component_plan
from hybrid_solver import groups_from_plan, official_score, signature
from run_problem1_all import newest_results_table
from region_solver import region_partition


ORIGINAL = ROOT / "results" / "experiments" / "results_all100"
OUTPUT = ROOT / "results" / "experiments" / "results_opt5"


def rows_by_case():
    with newest_results_table(ORIGINAL).open(encoding="utf-8-sig", newline="") as stream:
        return {row["case"]: row for row in csv.DictReader(stream) if row["cores"] == "5"}


def key(result):
    return result["makespan"], result["data_movement_bytes"]["added_copy_bytes"]


def compare_table(source):
    rows = []
    for case, row in sorted(source.items()):
        old = int(row["makespan_cycles"]) if row["makespan_cycles"] else None
        base_file = ORIGINAL / "singlecore" / f"{case}_result.json"
        baseline = _read_json(base_file)["makespan"] if base_file.exists() else None
        folder = OUTPUT / case
        summary = _read_json(folder / "summary.json") if (folder / "summary.json").exists() else {}
        best = summary.get("best_makespan", old)
        selected_folder = folder if summary else ORIGINAL / "multicore" / f"{case}_5cores"
        selected_result = _read_json(selected_folder / "best_result.json") if best else None
        rows.append({"case": case, "singlecore_makespan": baseline or "",
                     "old_5core_makespan": old or "", "new_best_5core_makespan": best or "",
                     "old_added_copy_bytes": row["added_copy_bytes"],
                     "new_best_added_copy_bytes": (
                         selected_result["data_movement_bytes"]["added_copy_bytes"]
                         if selected_result else ""),
                     "old_speedup": f"{baseline / old:.9f}" if baseline and old else "",
                     "new_best_speedup": f"{baseline / best:.9f}" if baseline and best else "",
                     "improved": bool(old and best and best < old),
                     "rescue_status": summary.get("status", "not_attempted"),
                     "best_plan_path": str(selected_folder / "best_plan.json") if best else "",
                     "best_result_path": str(selected_folder / "best_result.json") if best else ""})
    with (OUTPUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = [r for r in rows if r["old_speedup"] and r["new_best_speedup"]]
    print("All currently comparable 5-core cases:", len(complete),
          "old mean", round(sum(float(r["old_speedup"]) for r in complete) / len(complete), 6),
          "new mean", round(sum(float(r["new_best_speedup"]) for r in complete) / len(complete), 6),
          flush=True)


def write_final_results():
    with (OUTPUT / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        comparison = {row["case"]: row for row in csv.DictReader(stream)}
    with newest_results_table(ORIGINAL).open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    rows = []
    for row in original:
        case, cores = row["case"], int(row["cores"])
        result = {field: row[field] for field in (
            "case", "cores", "status", "makespan_cycles", "added_copy_bytes", "speedup")}
        result["method"] = row["method"]
        result["plan_file"] = row["plan_file"]
        result["result_file"] = row["result_file"]
        if cores == 5 and comparison[case]["new_best_5core_makespan"]:
            best = comparison[case]
            result.update({"makespan_cycles": best["new_best_5core_makespan"],
                           "added_copy_bytes": best["new_best_added_copy_bytes"],
                           "speedup": best["new_best_speedup"],
                           "method": "generic_multi_candidate_official_score",
                           "plan_file": best["best_plan_path"],
                           "result_file": best["best_result_path"]})
        rows.append(result)
    with (OUTPUT / "final_results_100cases.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    averages = []
    for cores in range(1, 6):
        values = [float(row["speedup"]) for row in rows
                  if int(row["cores"]) == cores and row["speedup"]]
        averages.append({"cores": cores, "scored_cases": len(values),
                         "average_speedup": f"{sum(values) / len(values):.9f}" if values else ""})
    with (OUTPUT / "final_average_speedup.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(averages[0]))
        writer.writeheader()
        writer.writerows(averages)


def write_chart(source):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with (OUTPUT / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        comparison = {row["case"]: row for row in csv.DictReader(stream)}
    with newest_results_table(ORIGINAL).open(encoding="utf-8-sig", newline="") as stream:
        scores = {}
        for row in csv.DictReader(stream):
            scores.setdefault(row["case"], {})[int(row["cores"])] = row
    common = [case for case in source if all(scores[case][n]["makespan_cycles"] for n in range(1, 6))]
    old = [sum(float(scores[c][n]["speedup"]) for c in common) / len(common)
           for n in range(1, 6)]
    new = old[:4] + [sum(float(comparison[c]["new_best_speedup"]) for c in common) / len(common)]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(range(1, 6), old, marker="o", label="Original solver", linewidth=2)
    ax.plot(range(1, 6), new, marker="o", label="Unified optimized candidates", linewidth=2)
    ax.annotate(f"{new[-1]:.3f}x", (5, new[-1]), xytext=(-8, 11),
                textcoords="offset points", ha="right")
    ax.set(xlabel="Number of cores", ylabel="Mean speedup vs. 1 core",
           title=f"Problem 1: {len(common)} cases scored at every core count")
    ax.set_xticks(range(1, 6))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "average_speedup_before_after.png", dpi=180)
    plt.close(fig)


def run_case(case, row, cfg, waits, timeout):
    folder = OUTPUT / case
    if (folder / "summary.json").exists():
        return
    old_file = ORIGINAL / "multicore" / f"{case}_5cores" / "best_result.json"
    old_plan_file = ORIGINAL / "multicore" / f"{case}_5cores" / "best_plan.json"
    old_result, old_plan = _read_json(old_file), _read_json(old_plan_file)
    best_result, best_plan, best_label = old_result, old_plan, "original"
    graph = _read_json(ROOT / "official" / "data" / f"{case}.json")
    gm = GraphModel(graph)
    candidates = []
    if len(gm.components) > 1:
        plan = independent_component_plan(gm, 5)
        candidates.append(("components", plan))
    for scale, label in ((1.25, "scale_1_25"), (0.75, "scale_0_75")):
        plan, _ = construct(graph, 5, cfg, waits, scale)
        candidates.append((label, plan))
    seen = {signature(old_plan)}
    history = []
    for label, plan in candidates:
        token = signature(plan)
        if token in seen:
            continue
        seen.add(token)
        probe = OUTPUT / "probes" / case
        cached = probe / f"{label}_result.json"
        if cached.exists():
            result = _read_json(cached)
            info = {"status": "valid", "makespan": result["makespan"], "cached": True}
        else:
            result, info = official_score(case, plan, probe, label, timeout,
                                          fast_evaluator=True)
        history.append({"candidate": label, **info})
        if result is not None and key(result) < key(best_result):
            best_result, best_plan, best_label = result, plan, label
    save(folder / "best_result.json", best_result)
    save(folder / "best_plan.json", best_plan)
    save(folder / "summary.json", {"status": "scored", "case": case,
                                   "old_makespan": old_result["makespan"],
                                   "best_makespan": best_result["makespan"],
                                   "selected": best_label, "candidate_history": history})
    print(case, "old", old_result["makespan"], "best", best_result["makespan"],
          "selected", best_label, flush=True)


def carry_lower_core_winners(source, timeout):
    """A previously scored plan remains legal with unused extra cores."""
    for case, row in sorted(source.items()):
        if not row["makespan_cycles"]:
            continue
        folder = OUTPUT / case
        summary_file = folder / "summary.json"
        if summary_file.exists():
            summary = _read_json(summary_file)
            best_result = _read_json(folder / "best_result.json")
            best_plan = _read_json(folder / "best_plan.json")
        else:
            original = ORIGINAL / "multicore" / f"{case}_5cores"
            best_result = _read_json(original / "best_result.json")
            best_plan = _read_json(original / "best_plan.json")
            summary = {"status": "scored", "case": case,
                       "old_makespan": best_result["makespan"],
                       "best_makespan": best_result["makespan"],
                       "selected": "original", "candidate_history": []}
        lower = []
        for cores in (2, 3, 4):
            previous = ORIGINAL / "multicore" / f"{case}_{cores}cores"
            result_file, plan_file = previous / "best_result.json", previous / "best_plan.json"
            if result_file.exists() and plan_file.exists():
                result = _read_json(result_file)
                if result["makespan"] < best_result["makespan"]:
                    lower.append((result["makespan"], cores, plan_file))
        if not lower:
            continue
        _, cores, plan_file = min(lower)
        label = f"carry_{cores}core"
        if any(item.get("candidate") == label for item in summary["candidate_history"]):
            continue
        plan = _read_json(plan_file)
        plan["core_schedules"] += [[] for _ in range(5 - len(plan["core_schedules"]))]
        result, info = official_score(case, plan, OUTPUT / "probes" / case, label, timeout,
                                      fast_evaluator=True)
        summary["candidate_history"].append({"candidate": label, **info})
        if result is not None and key(result) < key(best_result):
            best_result, best_plan = result, plan
            summary["selected"] = label
        summary["best_makespan"] = best_result["makespan"]
        save(folder / "best_result.json", best_result)
        save(folder / "best_plan.json", best_plan)
        save(summary_file, summary)
        print(case, label, info["status"], "best", best_result["makespan"], flush=True)


def refine_dependency_regions(source, cfg, waits, timeout, threshold, label_mode="both"):
    """One structural candidate per case, gated by current speedup."""
    candidate_name = "dependency_regions" if label_mode == "both" else f"dependency_regions_{label_mode}"
    result_name = "region4" if label_mode == "both" else f"region_{label_mode}"
    for case in sorted(source):
        original = ORIGINAL / "multicore" / f"{case}_5cores"
        if not (original / "best_result.json").exists():
            continue
        base_file = ORIGINAL / "singlecore" / f"{case}_result.json"
        if not base_file.exists():
            continue
        folder = OUTPUT / case
        summary_file = folder / "summary.json"
        if summary_file.exists():
            summary = _read_json(summary_file)
            best_result = _read_json(folder / "best_result.json")
            best_plan = _read_json(folder / "best_plan.json")
        else:
            best_result = _read_json(original / "best_result.json")
            best_plan = _read_json(original / "best_plan.json")
            summary = {"status": "scored", "case": case,
                       "old_makespan": best_result["makespan"],
                       "best_makespan": best_result["makespan"],
                       "selected": "original", "candidate_history": []}
        speedup = _read_json(base_file)["makespan"] / best_result["makespan"]
        if speedup >= threshold or any(
                x.get("candidate") == candidate_name for x in summary["candidate_history"]):
            continue
        try:
            graph = _read_json(ROOT / "official/data" / f"{case}.json")
            plan, detail = region_partition(GraphModel(graph), 5, cfg, waits,
                                            label_mode=label_mode)
            if signature(plan) == signature(best_plan):
                info, result = {"status": "duplicate"}, None
            else:
                probe = OUTPUT / "region_probes" / case
                cached = probe / f"{result_name}_result.json"
                if cached.exists():
                    result = _read_json(cached)
                    info = {"status": "valid", "makespan": result["makespan"], "cached": True}
                else:
                    result, info = official_score(case, plan, probe, result_name, timeout,
                                                  fast_evaluator=True)
            info["construction"] = detail
        except (RuntimeError, ValueError) as error:
            result, info = None, {"status": "invalid", "message": str(error)[:300]}
        summary["candidate_history"].append({"candidate": candidate_name, **info})
        if result is not None and key(result) < key(best_result):
            best_result, best_plan = result, plan
            summary["selected"] = candidate_name
        summary["best_makespan"] = best_result["makespan"]
        save(folder / "best_result.json", best_result)
        save(folder / "best_plan.json", best_plan)
        save(summary_file, summary)
        print(case, label_mode, "regions", info["status"], "speedup", round(speedup, 3),
              "best", best_result["makespan"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=2,
                        help="Only re-score cases with current 5-core speedup below this value")
    parser.add_argument("--timeout", type=float, default=30,
                        help="Official score timeout per candidate in seconds")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum new cases to process; zero means all selected")
    parser.add_argument("--region-threshold", type=float, default=4.5,
                        help="Score one dependency-region candidate below this current speedup")
    parser.add_argument("--sink-threshold", type=float, default=2.0,
                        help="Score a lower-communication sink-region candidate below this speedup")
    parser.add_argument("--source-threshold", type=float, default=0.0,
                        help="Also score source-region candidates below this speedup; zero disables")
    args = parser.parse_args()
    if (args.threshold <= 0 or args.timeout <= 0 or args.limit < 0 or
            args.region_threshold <= 0 or args.sink_threshold <= 0 or
            args.source_threshold < 0):
        parser.error("threshold and timeout must be positive; limit must be nonnegative")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = read_evaluation_config(str(ROOT / "official/data/config.txt"))
    waits = read_scene_a_config(str(ROOT / "official/data/config.txt"))
    source = rows_by_case()
    eligible = [(case, row) for case, row in source.items()
                if row["speedup"] and float(row["speedup"]) < args.threshold]
    eligible.sort(key=lambda item: (float(item[1]["speedup"]), item[0]))
    processed = 0
    for case, row in eligible:
        if (OUTPUT / case / "summary.json").exists():
            continue
        run_case(case, row, cfg, waits, args.timeout)
        processed += 1
        compare_table(source)
        if args.limit and processed >= args.limit:
            break
    carry_lower_core_winners(source, args.timeout)
    refine_dependency_regions(source, cfg, waits, args.timeout, args.region_threshold)
    refine_dependency_regions(source, cfg, waits, args.timeout, args.sink_threshold, "sinks")
    if args.source_threshold:
        refine_dependency_regions(source, cfg, waits, args.timeout,
                                  args.source_threshold, "sources")
    compare_table(source)
    write_final_results()
    write_chart(source)


if __name__ == "__main__":
    main()

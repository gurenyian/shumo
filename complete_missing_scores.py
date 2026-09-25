"""Complete missing problem-1 official scores using the validated fast evaluator.

Existing scores are preserved. Each missing configuration is scored with the
saved candidate first, then generic reusable candidates if needed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from fast_solver import GraphModel, construct, independent_component_plan
from hybrid_solver import official_score, signature
from run_problem1_all import CASES, refresh_tables, scored_multicore


def score_single(case, output, timeout):
    path = output / "singlecore" / f"{case}_result.json"
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "run_problem1_all.py"),
           "--single-worker", case, "--worker-output", str(path), "--fast-evaluator"]
    try:
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        print(case, "singlecore timeout", flush=True)
        return False
    if done.returncode:
        print(case, "singlecore error", done.stderr[-500:], flush=True)
        return False
    print(case, "singlecore", _read_json(path)["makespan"], flush=True)
    return True


def candidates(case, cores, output, graph, cfg, waits):
    folder = output / "multicore" / f"{case}_{cores}cores"
    existing = folder / "best_plan.json"
    if existing.exists():
        yield "saved_initial", _read_json(existing)
    for lower in range(2, cores):
        previous = output / "multicore" / f"{case}_{lower}cores"
        if not scored_multicore(previous):
            continue
        plan = _read_json(previous / "best_plan.json")
        plan["core_schedules"] += [[] for _ in range(cores - lower)]
        yield f"carry_{lower}core", plan
    gm = GraphModel(graph)
    if len(gm.components) > 1:
        yield "independent_components", independent_component_plan(gm, cores)
    for scale in (1.0, 1.25, 0.75):
        plan, _ = construct(graph, cores, cfg, waits, scale)
        yield f"scale_{str(scale).replace('.', '_')}", plan


def score_multi(case, cores, output, cfg, waits, timeout, max_candidates):
    folder = output / "multicore" / f"{case}_{cores}cores"
    if scored_multicore(folder):
        return True
    graph = _read_json(ROOT / "official" / "data" / f"{case}.json")
    seen, history = set(), []
    best = None
    for label, plan in candidates(case, cores, output, graph, cfg, waits):
        token = signature(plan)
        if token in seen:
            continue
        seen.add(token)
        result, info = official_score(case, plan, folder, f"completion_{label}", timeout,
                                      fast_evaluator=True)
        history.append({"candidate": label, **info})
        if result is not None:
            best = (plan, result, label)
            break
        if len(history) >= max_candidates:
            break
    save(folder / "completion_history.json", history)
    if best is None:
        print(case, cores, "no valid score", history, flush=True)
        return False
    plan, result, label = best
    save(folder / "best_plan.json", plan)
    save(folder / "best_result.json", result)
    save(folder / "summary.json", {
        "status": "scored", "case": case, "cores": cores,
        "best_makespan": result["makespan"],
        "best_added_copy_bytes": result["data_movement_bytes"]["added_copy_bytes"],
        "official_score_attempts": len(history),
        "selected": label, "completed_with_fast_evaluator": True,
    })
    print(case, cores, result["makespan"], label, flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", type=int, default=None,
                        help="Case numbers to process; default is all 100")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "experiments" / "results_all100")
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_candidates < 1:
        parser.error("timeout and max-candidates must be positive")
    output = args.output_root.resolve()
    errors_file = output / "errors.json"
    errors = _read_json(errors_file) if errors_file.exists() else {}
    cfg = read_evaluation_config(str(ROOT / "official/data/config.txt"))
    waits = read_scene_a_config(str(ROOT / "official/data/config.txt"))
    cases = [f"case_{i:03d}" for i in args.cases] if args.cases is not None else CASES
    for case in cases:
        if score_single(case, output, args.timeout):
            errors.pop(f"{case}:1", None)
        for cores in range(2, 6):
            if score_multi(case, cores, output, cfg, waits, args.timeout, args.max_candidates):
                errors.pop(f"{case}:{cores}", None)
        refresh_tables(output, errors)
    print("Completed missing-score pass:", output / "results.csv", flush=True)


if __name__ == "__main__":
    main()

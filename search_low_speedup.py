"""Apply the same bounded hybrid search to every weak five-core case.

This is a uniform second-stage search. It retains the incumbent whenever the
new official score is worse or unavailable.
"""

import argparse
import csv

from experiment import ROOT, _read_json, save
from hybrid_solver import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=2)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--max-evals", type=int, default=8)
    args = parser.parse_args()
    if args.threshold <= 0 or args.seconds <= 0 or args.max_evals < 1:
        parser.error("parameters must be positive")
    result_root = ROOT / "results_opt5"
    with (result_root / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["new_best_speedup"]
                and float(row["new_best_speedup"]) < args.threshold]
    print("Uniform search cases:", len(selected), flush=True)
    for row in selected:
        case = row["case"]
        trial = ROOT / "results_generic_search" / f"{case}_5cores"
        if not (trial / "summary.json").exists():
            try:
                run(case, 5, args.max_evals, args.seconds, 3,
                    ROOT / "results_generic_search", min(20, args.seconds), True)
            except (RuntimeError, ValueError) as error:
                print(case, "search error", str(error)[:300], flush=True)
        if not (trial / "best_result.json").exists():
            print(case, "no new score", flush=True)
            continue
        new_result = _read_json(trial / "best_result.json")
        folder = result_root / case
        incumbent = _read_json(folder / "best_result.json")
        new_key = (new_result["makespan"], new_result["data_movement_bytes"]["added_copy_bytes"])
        old_key = (incumbent["makespan"], incumbent["data_movement_bytes"]["added_copy_bytes"])
        if new_key < old_key:
            summary = _read_json(folder / "summary.json")
            summary["candidate_history"].append({"candidate": "bounded_hybrid_search",
                                                  "status": "valid", "makespan": new_result["makespan"]})
            summary["best_makespan"] = new_result["makespan"]
            summary["selected"] = "bounded_hybrid_search"
            save(folder / "best_result.json", new_result)
            save(folder / "best_plan.json", _read_json(trial / "best_plan.json"))
            save(folder / "summary.json", summary)
            print(case, "improved", incumbent["makespan"], "->", new_result["makespan"], flush=True)
        else:
            print(case, "retained", incumbent["makespan"], flush=True)


if __name__ == "__main__":
    main()

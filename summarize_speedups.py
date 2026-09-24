"""Summarize the official scores collected in results_all100/results.csv.

Usage: python summarize_speedups.py
Missing scores stay empty; they are never imputed from another core count.
"""

import csv
import statistics
from pathlib import Path
from run_problem1_all import newest_results_table


ROOT = Path(__file__).resolve().parent / "results_all100"
SOURCE = newest_results_table(ROOT)
DETAIL = ROOT / "speedup_by_case.csv"
AVERAGES = ROOT / "speedup_averages.csv"
CHART = ROOT / "average_speedup_92_complete_cases.png"


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    by_case = {}
    for row in source:
        case = row["case"]
        core = int(row["cores"])
        if core in by_case.setdefault(case, {}):
            raise ValueError(f"Duplicate score: {case}, {core} core(s)")
        by_case[case][core] = row

    detail = []
    for case in sorted(by_case):
        rows = by_case[case]
        if set(rows) != set(range(1, 6)):
            raise ValueError(f"Expected five configurations for {case}")
        times = {core: int(row["makespan_cycles"]) if row["makespan_cycles"] else None
                 for core, row in rows.items()}
        base = times[1]
        record = {"case": case, "complete_1_to_5": int(all(times.values()))}
        for core in range(1, 6):
            record[f"makespan_{core}core"] = times[core] or ""
            record[f"speedup_{core}core"] = (
                f"{base / times[core]:.9f}" if base and times[core] else ""
            )
            record[f"status_{core}core"] = rows[core]["status"]
        detail.append(record)

    with DETAIL.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)

    complete = [row for row in detail if row["complete_1_to_5"]]
    average_rows = []
    for core in range(1, 6):
        field = f"speedup_{core}core"
        available = [float(row[field]) for row in detail if row[field]]
        comparable = [float(row[field]) for row in complete]
        average_rows.append({
            "cores": core,
            "available_case_count": len(available),
            "available_case_mean": f"{statistics.mean(available):.9f}",
            "complete_case_count": len(comparable),
            "complete_case_mean": f"{statistics.mean(comparable):.9f}",
            "complete_case_median": f"{statistics.median(comparable):.9f}",
        })
    with AVERAGES.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(average_rows[0]))
        writer.writeheader()
        writer.writerows(average_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [row["cores"] for row in average_rows]
        y = [float(row["complete_case_mean"]) for row in average_rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, marker="o", linewidth=2.4, color="#176baf")
        for core, value in zip(x, y):
            ax.annotate(f"{value:.3f}x", (core, value), xytext=(0, 10),
                        textcoords="offset points", ha="center")
        ax.set(xlabel="Number of cores", ylabel="Mean speedup vs. 1 core",
               title=f"Problem 1: mean speedup on {len(complete)} fully scored cases")
        ax.set_xticks(x)
        ax.set_ylim(0, 3.5)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(CHART, dpi=180)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable: chart skipped")

    print(f"Cases: {len(detail)}; fully scored: {len(complete)}")
    print("Incomplete:", ", ".join(row["case"] for row in detail if not row["complete_1_to_5"]))
    for row in average_rows:
        print(f"{row['cores']} core(s): {row['complete_case_mean']}x on {row['complete_case_count']} common cases; "
              f"{row['available_case_mean']}x on {row['available_case_count']} available cases")
    print(f"Wrote {DETAIL} and {AVERAGES}")
    if CHART.exists():
        print(f"Wrote {CHART}")


if __name__ == "__main__":
    main()

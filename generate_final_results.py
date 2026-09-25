"""Build the canonical 1--5 core report from the packaged and latest scores.

The one-core row comes from ``results/final/final_results_100cases.csv``.
The 2--4 core rows come from ``results/experiments/results_fm_allcores`` and
the 5-core rows come from ``results/experiments/results_fm_5core``.  When the
raw five-core CSV is absent, the shipped canonical five-core rows are used
after validating their FM provenance.  The script also copies selected FM
plans into the final plan directory.
"""

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "results" / "final"
SOURCE_FINAL = FINAL / "final_results_100cases.csv"
SOURCE_FM = ROOT / "results" / "experiments" / "results_fm_allcores" / "results.csv"
SOURCE_5 = ROOT / "results" / "experiments" / "results_fm_5core" / "results.csv"
PLANS = FINAL / "plans"
AVERAGES = FINAL / "final_average_speedup.csv"
CHART = FINAL / "reports" / "average_speedup.svg"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_plan(row):
    """Resolve a recorded plan path without guessing a replacement path."""
    plan_file = row.get("plan_file", "")
    if not plan_file:
        raise ValueError(f"Missing FM plan path for {row.get('case')}")
    for candidate in (ROOT / plan_file, ROOT / "results" / "experiments" / plan_file,
                      FINAL / plan_file):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(plan_file)


def validate_fm_rows(rows, expected, label):
    by_key = {}
    for row in rows:
        key = expected(row)
        if key in by_key:
            raise ValueError(f"Duplicate {label} FM row for {key}")
        if row.get("status") != "scored" or not row.get("method", "").startswith("fm"):
            raise ValueError(f"Invalid FM provenance for {label} {key}")
        try:
            int(row["makespan_cycles"])
            int(row["added_copy_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid scored values for {label} {key}") from error
        by_key[key] = row
    return by_key


def main():
    packaged = read_csv(SOURCE_FINAL)
    fm_rows = validate_fm_rows(read_csv(SOURCE_FM),
                               lambda row: (row["case"], int(row["cores"])), "2-4 core")
    if SOURCE_5.exists():
        latest_5 = validate_fm_rows(read_csv(SOURCE_5), lambda row: row["case"], "5-core")
    else:
        five_rows = [row for row in packaged if int(row["cores"]) == 5]
        latest_5 = validate_fm_rows(five_rows, lambda row: row["case"], "packaged 5-core")
        if len(latest_5) != 100:
            raise ValueError("Packaged five-core fallback is missing validated FM provenance")
    packaged_1 = [row for row in packaged if int(row["cores"]) == 1]
    baselines = {row["case"]: int(row["makespan_cycles"]) for row in packaged_1
                 if int(row["cores"]) == 1}
    packaged_14 = [row for row in packaged if int(row["cores"]) in (1, 2, 3, 4)]
    rows = []
    for row in packaged_14:
        item = {key: row.get(key, "") for key in (
            "case", "cores", "status", "makespan_cycles", "added_copy_bytes",
            "speedup", "method", "plan_file")}
        if int(item["cores"]) == 1:
            item["speedup"] = "1.000000000"
        else:
            item["speedup"] = f"{baselines[item['case']] / int(item['makespan_cycles']):.9f}"
        rows.append(item)

    cases = sorted({row["case"] for row in rows})
    by_case = {(row["case"], int(row["cores"])): row for row in rows}
    expected_fm = {(case, cores) for case in cases for cores in (2, 3, 4)}
    if len(cases) != 100 or len(rows) != 400 or set(fm_rows) != expected_fm:
        raise ValueError(f"Expected 100 cases, 400 packaged rows, and 300 FM rows; got {len(cases)} / {len(rows)} / {len(fm_rows)}")
    if set(latest_5) != set(cases):
        raise ValueError("Latest 5-core table does not cover the packaged 100 cases")

    # Validate every source and provenance record before copying plans or
    # replacing the canonical report.
    source_plans = {}
    for case in cases:
        for cores in (2, 3, 4):
            source_plans[(case, cores)] = resolve_plan(fm_rows[(case, cores)])
        source_plans[(case, 5)] = resolve_plan(latest_5[case])
    PLANS.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for cores in (2, 3, 4):
            latest = fm_rows[(case, cores)]
            target_plan = PLANS / f"{case}_{cores}cores.json"
            shutil.copy2(source_plans[(case, cores)], target_plan)
            makespan = int(latest["makespan_cycles"])
            by_case[(case, cores)] = {
                "case": case, "cores": str(cores), "status": latest["status"],
                "makespan_cycles": str(makespan), "added_copy_bytes": latest["added_copy_bytes"],
                "speedup": f"{baselines[case] / makespan:.9f}",
                "method": "fm_selected_refined", "plan_file": f"plans/{case}_{cores}cores.json",
            }
        latest = latest_5[case]
        target_plan = PLANS / f"{case}_5cores.json"
        shutil.copy2(source_plans[(case, 5)], target_plan)
        baseline = int(by_case[(case, 1)]["makespan_cycles"])
        makespan = int(latest["makespan_cycles"])
        by_case[(case, 5)] = {
            "case": case,
            "cores": "5",
            "status": latest["status"],
            "makespan_cycles": str(makespan),
            "added_copy_bytes": latest["added_copy_bytes"],
            "speedup": f"{baseline / makespan:.9f}",
            "method": "fm_5core_latest",
            "plan_file": f"plans/{case}_5cores.json",
        }

    fields = ["case", "cores", "status", "makespan_cycles", "added_copy_bytes",
              "speedup", "method", "plan_file"]
    ordered = [by_case[(case, core)] for case in cases for core in range(1, 6)]
    with SOURCE_FINAL.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    averages = []
    for core in range(1, 6):
        values = [float(row["speedup"]) for row in ordered if int(row["cores"]) == core]
        averages.append({"cores": core, "scored_cases": len(values),
                         "average_speedup": "1.000000000" if core == 1
                         else f"{sum(values) / len(values):.9f}"})
    with AVERAGES.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cores", "scored_cases", "average_speedup"])
        writer.writeheader()
        writer.writerows(averages)

    x = [row["cores"] for row in averages]
    y = [float(row["average_speedup"]) for row in averages]
    width, height = 760, 500
    left, right, top, bottom = 85, 30, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = max(5.5, max(y) * 1.12)
    sx = lambda value: left + (value - 1) * plot_w / 4
    sy = lambda value: height - bottom - value * plot_h / ymax
    mean_points = " ".join(f"{sx(core):.1f},{sy(value):.1f}" for core, value in zip(x, y))
    ideal_points = f"{sx(1):.1f},{sy(1):.1f} {sx(5):.1f},{sy(5):.1f}"
    grid = "".join(
        f'<line x1="{left}" y1="{sy(tick):.1f}" x2="{width-right}" y2="{sy(tick):.1f}" stroke="#dddddd"/> '
        f'<text x="{left-12}" y="{sy(tick)+4:.1f}" text-anchor="end" font-size="12">{tick}</text>'
        for tick in range(0, 6))
    ticks = "".join(
        f'<text x="{sx(core):.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12">{core}</text>'
        for core in x)
    labels = "".join(
        f'<text x="{sx(core):.1f}" y="{sy(value)-12:.1f}" text-anchor="middle" font-size="12">{value:.3f}x</text>'
        for core, value in zip(x, y))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="25" text-anchor="middle" font-size="17" font-family="sans-serif">Problem 1: mean speedup across 100 cases</text>
{grid}<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/><line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>{ticks}
<polyline points="{ideal_points}" fill="none" stroke="#777" stroke-width="2" stroke-dasharray="7,5"/><polyline points="{mean_points}" fill="none" stroke="#176baf" stroke-width="3"/>
{''.join(f'<circle cx="{sx(core):.1f}" cy="{sy(value):.1f}" r="4" fill="#176baf"/>' for core, value in zip(x, y))}{labels}
<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="13" font-family="sans-serif">Number of cores</text><text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-size="13" font-family="sans-serif">Mean speedup vs. 1 core</text>
<line x1="{width-210}" y1="35" x2="{width-180}" y2="35" stroke="#176baf" stroke-width="3"/><text x="{width-172}" y="39" font-size="12">Mean speedup</text><line x1="{width-210}" y1="55" x2="{width-180}" y2="55" stroke="#777" stroke-width="2" stroke-dasharray="7,5"/><text x="{width-172}" y="59" font-size="12">Ideal linear y=k</text>
</svg>'''
    CHART.parent.mkdir(parents=True, exist_ok=True)
    CHART.write_text(svg, encoding="utf-8")
    print(f"Wrote {SOURCE_FINAL}, {AVERAGES}, and {CHART}")


if __name__ == "__main__":
    main()

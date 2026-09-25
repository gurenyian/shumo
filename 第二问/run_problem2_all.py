"""Run the official Scene-B solver for all requested cases and core counts."""

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiment import ROOT, _read_json


def write_tables(output_root, cases, cores, one_core):
    rows = []
    for case in cases:
        baseline = one_core.get(case)
        if baseline is not None:
            rows.append({'case': case, 'cores': 1, 'status': 'singlecore_reference',
                         'makespan_cycles': baseline, 'added_copy_bytes': 0,
                         'speedup': '1.000000000', 'spill_added_copy_bytes': '',
                         'official_evals': 0, 'solver_seconds': 0,
                         'plan_file': '', 'result_file': ''})
        for core_count in cores:
            folder = output_root / f'{case}_{core_count}cores'
            file = folder / 'summary.json'
            if not file.exists():
                continue
            try:
                summary = _read_json(file)
                best = summary['best']
            except (OSError, ValueError, KeyError):
                # Another worker may still be writing this summary.
                continue
            makespan = int(best['makespan'])
            rows.append({'case': case, 'cores': core_count, 'status': 'scored',
                         'makespan_cycles': makespan,
                         'added_copy_bytes': best['added_copy_bytes'],
                         'speedup': f'{baseline / makespan:.9f}' if baseline else '',
                         'spill_added_copy_bytes': best['spill_added_copy_bytes'],
                         'official_evals': summary['official_evaluator_calls'],
                         'solver_seconds': f'{summary["solver_runtime_seconds"]:.3f}',
                         'plan_file': str(folder / 'best_plan.json'),
                         'result_file': str(folder / 'best_result.json')})
    if rows:
        with (output_root / 'batch_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    averages = []
    for core_count in (1, 2, 3, 4, 5):
        values = [float(row['speedup']) for row in rows
                  if row['cores'] == core_count and row['speedup']]
        averages.append({'cores': core_count, 'scored_cases': len(values),
                         'average_speedup': f'{sum(values) / len(values):.9f}' if values else ''})
    with (output_root / 'average_speedup.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(averages[0]))
        writer.writeheader()
        writer.writerows(averages)
    return rows, averages


def write_chart(output_root, averages):
    if any(int(row['scored_cases']) == 0 for row in averages):
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [int(row['cores']) for row in averages]
    y = [float(row['average_speedup']) for row in averages]
    ax.plot(x, y, marker='o', linewidth=2)
    ax.set(xlabel='Number of cores', ylabel='Mean speedup',
           title='Scene B mean speedup')
    ax.set_xticks(x)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output_root / 'average_speedup.png', dpi=170)
    plt.close(fig)


def run_target(index, case, cores, args, data, config, output_root):
    """One worker owns one output folder; the main thread alone writes tables."""
    folder = output_root / f'{case}_{cores}cores'
    if (not args.force and (folder / 'summary.json').exists()
            and (folder / 'best_plan.json').exists()
            and (folder / 'best_result.json').exists()):
        return index, case, cores, 'saved result', None
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'solver_problem2.py'),
               str(data / f'{case}.json'), '-n', str(cores), '--config', str(config),
               '--max-evals', str(args.max_evals), '--seconds', str(args.seconds),
               '--eval-timeout', str(args.eval_timeout), '--output-dir', str(folder),
               '--base-scales', *map(str, args.base_scales)]
    log_path = folder / 'solver.log'
    started = time.perf_counter()
    try:
        with log_path.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.seconds + 90, check=False)
        status = 'completed' if completed.returncode == 0 else f'error exit={completed.returncode}'
    except subprocess.TimeoutExpired:
        status = 'outer timeout'
    except OSError as error:
        status = f'launch error: {error}'
    return index, case, cores, f'{status} ({time.perf_counter() - started:.1f}s)', log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores', nargs='+', type=int, choices=[2, 3, 4, 5],
                        default=[2, 3, 4, 5])
    parser.add_argument('--cases', nargs='*', help='Case names, e.g. case_001 case_002')
    parser.add_argument('--max-evals', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--eval-timeout', type=float, default=120)
    parser.add_argument('--base-scales', nargs='+', type=float, default=[0.75, 1.0])
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results_problem2')
    parser.add_argument('--workers', type=int, default=5,
                        help='Concurrent solver processes; use 5 for five parallel jobs')
    parser.add_argument('--force', action='store_true', help='Re-evaluate completed configurations')
    args = parser.parse_args()
    if args.max_evals < 1 or args.seconds <= 0 or args.eval_timeout <= 0 or args.workers < 1:
        parser.error('budgets must be positive')
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'official' / 'data'
    all_cases = sorted(path.stem for path in data.glob('case_*.json'))
    cases = sorted(set(args.cases)) if args.cases else all_cases
    unknown = sorted(set(cases) - set(all_cases))
    if unknown:
        parser.error(f'unknown cases: {unknown}')
    reference_table = ROOT / 'final_results' / 'final_results_100cases.csv'
    with reference_table.open(encoding='utf-8-sig', newline='') as stream:
        one_core = {row['case']: int(row['makespan_cycles']) for row in csv.DictReader(stream)
                    if row['cores'] == '1' and row['makespan_cycles']}
    targets = [(case, cores) for case in cases for cores in sorted(set(args.cores))]
    config = data / 'config.txt'
    core_counts = sorted(set(args.cores))
    print(f'Running {len(targets)} configurations with {args.workers} workers.', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_target, index, case, cores,
                               args, data, config, output_root)
                   for index, (case, cores) in enumerate(targets, 1)]
        for future in as_completed(futures):
            index, case, cores, status, log_path = future.result()
            suffix = f'; log: {log_path}' if log_path and not status.startswith('completed') else ''
            print(f'[{index}/{len(targets)}] {case} {cores} cores: {status}{suffix}',
                  flush=True)
            rows, averages = write_tables(output_root, cases, core_counts, one_core)
            print(f'  scored configurations: {sum(row["status"] == "scored" for row in rows)}',
                  flush=True)
    _, averages = write_tables(output_root, cases, sorted(set(args.cores)), one_core)
    write_chart(output_root, averages)
    print(f'Final tables: {output_root / "batch_results.csv"} and '
          f'{output_root / "average_speedup.csv"}', flush=True)


if __name__ == '__main__':
    main()

"""Historical V1 batch runner; final V2 runner is optimize_problem3_v2.py."""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def display_path(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output_root, cases, cores):
    rows, ablation = [], []
    for case in cases:
        for count in cores:
            folder = output_root / f'{case}_{count}cores'
            summary_path = folder / 'summary.json'
            if not summary_path.is_file():
                rows.append({'case': case, 'cores': count, 'status': 'missing'})
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                best = summary['best']
                no_l2 = int(summary['problem2_baseline_makespan'])
            except (OSError, ValueError, KeyError):
                rows.append({'case': case, 'cores': count, 'status': 'invalid_summary'})
                continue
            rows.append({
                'case': case, 'cores': count, 'status': 'scored',
                'no_l2_makespan': no_l2,
                'cache_makespan': best['makespan'],
                'l2_relative_speedup': f'{no_l2 / best["makespan"]:.9f}',
                'no_l2_added_copy_bytes': summary['problem2_baseline_added_copy_bytes'],
                'cache_added_copy_bytes': best['added_copy_bytes'],
                'cache_partition_added_copy_bytes': best['partition_added_copy_bytes'],
                'cache_spill_added_copy_bytes': best['spill_added_copy_bytes'],
                'no_l2_cache_hit_rate': '',
                'cache_hit_rate': f'{best["cache_hit_rate"]:.9f}',
                'cache_hit_bytes': best['cache_hit_bytes'],
                'cache_miss_bytes': best['cache_miss_bytes'],
                'cache_hits': best['cache_hits'],
                'cache_misses': best['cache_misses'],
                'official_evals': summary['official_evaluator_calls'],
                'solver_seconds': f'{summary["solver_runtime_seconds"]:.3f}',
                'plan_file': display_path(folder / 'best_plan.json'),
                'result_file': display_path(
                    folder / ('best_result.json' if (folder / 'best_result.json').is_file()
                              else 'best_result.json.gz')),
            })
            for entry in summary.get('ablation', []):
                if entry.get('official_makespan') is not None:
                    ablation.append({
                        'case': case, 'cores': count,
                        'candidate': entry['candidate'],
                        'operation': entry['operation'],
                        'makespan': entry['official_makespan'],
                        'added_copy_bytes': entry['added_copy_bytes'],
                        'cache_hit_rate': entry['official_cache_hit_rate'],
                        'accepted': entry['accepted'],
                    })
    columns = ['case', 'cores', 'status', 'no_l2_makespan', 'cache_makespan',
               'l2_relative_speedup', 'no_l2_added_copy_bytes',
               'cache_added_copy_bytes', 'cache_partition_added_copy_bytes',
               'cache_spill_added_copy_bytes', 'no_l2_cache_hit_rate',
               'cache_hit_rate', 'cache_hit_bytes', 'cache_miss_bytes',
               'cache_hits', 'cache_misses', 'official_evals', 'solver_seconds',
               'plan_file', 'result_file']
    normalized = [{name: row.get(name, '') for name in columns} for row in rows]
    write_csv(output_root / 'comparison_100cases.csv', normalized, columns)
    write_csv(output_root / 'appendix_100cases.csv', normalized, columns)
    write_csv(output_root / 'ablation_records.csv', ablation,
              ['case', 'cores', 'candidate', 'operation', 'makespan',
               'added_copy_bytes', 'cache_hit_rate', 'accepted'])
    by_key = {(r['case'], int(r['cores'])): r for r in rows if r['status'] == 'scored'}
    relative, makespan_curves, speedup_curves, movement_curves = [], [], [], []
    for count in (1, 2, 3, 4, 5):
        current = [by_key[(case, count)] for case in cases
                   if (case, count) in by_key]
        ratios = [float(r['l2_relative_speedup']) for r in current]
        relative.append({'cores': count, 'scored_cases': len(current),
                         'average_l2_relative_speedup': f'{sum(ratios) / len(ratios):.9f}' if ratios else ''})
        no = [int(r['no_l2_makespan']) for r in current]
        yes = [int(r['cache_makespan']) for r in current]
        makespan_curves.append({'cores': count, 'scored_cases': len(current),
                                'mean_no_l2_makespan': f'{sum(no) / len(no):.3f}' if no else '',
                                'mean_cache_makespan': f'{sum(yes) / len(yes):.3f}' if yes else ''})
        movement_curves.append({
            'cores': count, 'scored_cases': len(current),
            'mean_no_l2_added_copy_bytes':
                f'{sum(int(r["no_l2_added_copy_bytes"]) for r in current) / len(current):.3f}' if current else '',
            'mean_cache_added_copy_bytes':
                f'{sum(int(r["cache_added_copy_bytes"]) for r in current) / len(current):.3f}' if current else '',
            'mean_cache_hit_rate':
                f'{sum(float(r["cache_hit_rate"]) for r in current) / len(current):.9f}' if current else '',
        })
        paired = [(r, by_key[(r['case'], 1)]) for r in current
                  if (r['case'], 1) in by_key]
        if paired:
            no_speed = [int(one['no_l2_makespan']) / int(r['no_l2_makespan'])
                        for r, one in paired]
            cache_speed = [int(one['no_l2_makespan']) / int(r['cache_makespan'])
                           for r, one in paired]
        else:
            no_speed = cache_speed = []
        speedup_curves.append({'cores': count, 'paired_cases': len(paired),
                               'mean_no_l2_speedup': f'{sum(no_speed) / len(no_speed):.9f}' if paired else '',
                               'mean_cache_speedup': f'{sum(cache_speed) / len(cache_speed):.9f}' if paired else ''})
    write_csv(output_root / 'average_l2_relative_speedup.csv', relative,
              ['cores', 'scored_cases', 'average_l2_relative_speedup'])
    write_csv(output_root / 'average_makespan_curves.csv', makespan_curves,
              ['cores', 'scored_cases', 'mean_no_l2_makespan', 'mean_cache_makespan'])
    write_csv(output_root / 'average_speedup_curves.csv', speedup_curves,
              ['cores', 'paired_cases', 'mean_no_l2_speedup', 'mean_cache_speedup'])
    write_csv(output_root / 'average_transfer_and_hit_curves.csv', movement_curves,
              ['cores', 'scored_cases', 'mean_no_l2_added_copy_bytes',
               'mean_cache_added_copy_bytes', 'mean_cache_hit_rate'])
    return rows, relative, makespan_curves, speedup_curves


def write_charts(output_root, makespan, speedup, relative):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for name, data, left, right, ylabel in (
        ('average_makespan_curves.png', makespan, 'mean_no_l2_makespan',
         'mean_cache_makespan', 'Mean Makespan (cycles)'),
        ('average_speedup_curves.png', speedup, 'mean_no_l2_speedup',
         'mean_cache_speedup', 'Mean speedup vs. one-core no-L2'),
    ):
        valid = [r for r in data if r[left] and r[right]]
        if not valid:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.6))
        x = [int(r['cores']) for r in valid]
        ax.plot(x, [float(r[left]) for r in valid], marker='o', label='No L2')
        ax.plot(x, [float(r[right]) for r in valid], marker='s', label='Read-only L2')
        ax.set(xlabel='Number of cores', ylabel=ylabel, xticks=range(1, 6))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_root / name, dpi=170)
        plt.close(fig)
    valid = [r for r in relative if r['average_l2_relative_speedup']]
    if valid:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot([int(r['cores']) for r in valid],
                [float(r['average_l2_relative_speedup']) for r in valid],
                marker='o', color='#126E82')
        ax.axhline(1.0, color='gray', linewidth=1, linestyle='--')
        ax.set(xlabel='Number of cores',
               ylabel='Mean same-core speedup: no L2 / read-only L2',
               xticks=range(1, 6))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_root / 'average_l2_relative_speedup.png', dpi=170)
        plt.close(fig)


def run_target(index, case, cores, args, output_root):
    folder = output_root / f'{case}_{cores}cores'
    if (not args.force and (folder / 'summary.json').is_file()
            and (folder / 'best_plan.json').is_file()
            and ((folder / 'best_result.json').is_file()
                 or (folder / 'best_result.json.gz').is_file())):
        return index, case, cores, 'saved result'
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'solver_problem3.py'),
           str(ROOT / 'official/data' / f'{case}.json'), '-n', str(cores),
           '--config', str(ROOT / 'official/data/config.txt'),
           '--max-evals', str(args.max_evals), '--seconds', str(args.seconds),
           '--eval-timeout', str(args.eval_timeout), '--output-dir', str(folder)]
    with (folder / 'solver.log').open('w', encoding='utf-8') as log:
        try:
            run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 check=False, timeout=args.seconds + 90)
            status = 'completed' if run.returncode == 0 else f'error exit={run.returncode}'
        except subprocess.TimeoutExpired:
            status = 'outer timeout'
    return index, case, cores, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores', nargs='+', type=int, choices=range(1, 6),
                        default=[1, 2, 3, 4, 5])
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--max-evals', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--eval-timeout', type=float, default=120)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results_problem3_v1_rerun')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='rebuild tables and figures from saved results without solving')
    args = parser.parse_args()
    if args.workers < 1 or args.max_evals < 1 or args.seconds <= 0 or args.eval_timeout <= 0:
        parser.error('budgets must be positive')
    cases = sorted(p.stem for p in (ROOT / 'official/data').glob('case_*.json')
                   if re.fullmatch(r'case_\d{3}', p.stem))
    if args.cases:
        unknown = set(args.cases) - set(cases)
        if unknown:
            parser.error(f'unknown cases: {sorted(unknown)}')
        cases = sorted(set(args.cases))
    cores = sorted(set(args.cores))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    targets = [(case, count) for case in cases for count in cores]
    if args.aggregate_only:
        _, relative, makespan, speedup = write_tables(output_root, cases, cores)
        write_charts(output_root, makespan, speedup, relative)
        print(f'Rebuilt tables and figures from {output_root}', flush=True)
        return
    print(f'Running {len(targets)} case/core configurations with {args.workers} workers.', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_target, i, case, count, args, output_root)
                   for i, (case, count) in enumerate(targets, 1)]
        for future in as_completed(futures):
            index, case, count, status = future.result()
            print(f'[{index}/{len(targets)}] {case} {count} cores: {status}', flush=True)
            rows, _, _, _ = write_tables(output_root, cases, cores)
            print(f'  scored: {sum(r["status"] == "scored" for r in rows)}', flush=True)
    _, relative, makespan, speedup = write_tables(output_root, cases, cores)
    write_charts(output_root, makespan, speedup, relative)
    print(f'Results: {output_root / "comparison_100cases.csv"}', flush=True)
    print(f'L2 relative speedup: {relative}', flush=True)


if __name__ == '__main__':
    main()

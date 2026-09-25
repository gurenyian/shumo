"""Resumable standalone multilevel five-core optimizer."""
import argparse
import concurrent.futures
import csv
from pathlib import Path

from experiment import ROOT, _read_json, save, read_evaluation_config, read_scene_a_config
from hybrid_solver import official_score, signature
from multilevel_partition import candidates

OUT = ROOT / 'results' / 'experiments' / 'results_multilevel_5core'
SOURCE = ROOT / 'results' / 'final' / 'final_results_100cases.csv'


def key(result):
    return int(result['makespan']), int(result['data_movement_bytes']['added_copy_bytes'])


def five_core_rows():
    """Read only five-core baseline fields from the mixed historical table."""
    fields = ('case', 'cores', 'status', 'makespan_cycles', 'added_copy_bytes', 'plan_file')
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = {name: header.index(name) for name in fields}
        rows = []
        for values in reader:
            if values[index['cores']] != '5':
                continue
            row = {name: values[index[name]] for name in fields}
            if row['status'] == 'scored' and row['plan_file']:
                rows.append(row)
    return rows


def _completed_signatures(history):
    """Only reuse a completed score for the exact plan that was evaluated."""
    return {signature(item['plan']) for item in history
            if item.get('status') == 'valid' and isinstance(item.get('plan'), dict)}


def run_one(case, row, cfg, waits, timeout):
    folder = OUT / case
    folder.mkdir(parents=True, exist_ok=True)
    history_path = folder / 'history.json'
    history = _read_json(history_path) if history_path.exists() else []
    done = _completed_signatures(history)
    incumbent = _read_json(ROOT / 'results' / 'final' / row['plan_file'])
    best = (int(row['makespan_cycles']), int(row['added_copy_bytes']))
    best_plan = incumbent
    if (folder / 'best_result.json').exists() and (folder / 'best_plan.json').exists():
        saved = _read_json(folder / 'best_result.json')
        if key(saved) < best:
            best = key(saved)
            best_plan = _read_json(folder / 'best_plan.json')
    graph = _read_json(ROOT / 'official' / 'data' / f'{case}.json')
    for index, plan in enumerate(candidates(graph, incumbent, cfg, waits, 5, 3), 1):
        label = f'candidate_{index}'
        plan_signature = signature(plan)
        if plan_signature in done:
            continue
        result, info = official_score(case, plan, folder, label, timeout, fast_evaluator=True)
        record = {'candidate': label, 'plan': plan, **info, 'accepted': False}
        if result is not None and key(result) < best:
            best = key(result)
            best_plan = plan
            record['accepted'] = True
            save(folder / 'best_plan.json', plan)
            save(folder / 'best_result.json', result)
        history.append(record)
        save(history_path, history)
    if not (folder / 'best_plan.json').exists():
        save(folder / 'best_plan.json', best_plan)
    return case, best


def run_job(job):
    return run_one(*job)


def write_summary(rows, results):
    """Write the current five-core outcome without modifying final_results."""
    result_by_case = dict(results)
    output_rows = []
    for row in rows:
        makespan, added_bytes = result_by_case.get(
            row['case'], (int(row['makespan_cycles']), int(row['added_copy_bytes'])))
        output_rows.append({
            'case': row['case'],
            'status': 'scored',
            'baseline_makespan_cycles': row['makespan_cycles'],
            'baseline_added_copy_bytes': row['added_copy_bytes'],
            'makespan_cycles': makespan,
            'added_copy_bytes': added_bytes,
            'improved': str((makespan, added_bytes) <
                            (int(row['makespan_cycles']), int(row['added_copy_bytes']))).lower(),
            'plan_file': str(Path('results') / 'experiments' / 'results_multilevel_5core' / row['case'] / 'best_plan.json'),
        })
    target = OUT / 'results.csv'
    temporary = target.with_name(target.name + '.writing')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=50,
                        help='parallel case jobs (default: 50)')
    args = parser.parse_args()
    if args.timeout <= 0 or args.workers < 1:
        parser.error('timeout and workers must be positive')
    rows = five_core_rows()
    selected = set(args.cases) if args.cases else None
    rows = [row for row in rows if selected is None or row['case'] in selected]
    cfg = read_evaluation_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    waits = read_scene_a_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    OUT.mkdir(exist_ok=True)
    jobs = [(row['case'], row, cfg, waits, args.timeout) for row in rows]
    results = []
    if args.workers == 1:
        for job in jobs:
            result = run_one(*job)
            results.append(result)
            print(result, flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(run_job, jobs):
                results.append(result)
                print(result, flush=True)
    if rows:
        print('Wrote:', write_summary(rows, results), flush=True)


if __name__ == '__main__':
    main()

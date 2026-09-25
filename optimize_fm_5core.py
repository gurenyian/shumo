"""Resumable five core FM refinement runner.

Plans are generated from the iterative five core incumbent and are accepted
only after the official evaluator returns a valid result.  The script is
intentionally idle until invoked; in particular importing it never starts a
batch run.
"""
import argparse
import concurrent.futures
import csv
from pathlib import Path

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from hybrid_solver import official_score, signature
from fm_candidates import candidate_variants

OUT = ROOT / 'results' / 'experiments' / 'results_fm_5core'
SOURCE = ROOT / 'results' / 'experiments' / 'results_iterative_5core' / 'results.csv'
SUMMARY_FIELDS = ('case', 'status', 'baseline_makespan_cycles',
                  'baseline_added_copy_bytes', 'makespan_cycles',
                  'added_copy_bytes', 'improved', 'method', 'plan_file')


def five_core_rows():
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get('status') == 'scored' and row.get('case')]


def _done(history):
    return {signature(r['plan']) for r in history
            if r.get('status') == 'valid' and isinstance(r.get('plan'), dict)}


def _key(result):
    return int(result['makespan']), int(result['data_movement_bytes']['added_copy_bytes'])


def run_one(row, cfg, waits, timeout):
    case = row['case']
    folder = OUT / case
    folder.mkdir(parents=True, exist_ok=True)
    history_path = folder / 'history.json'
    history = _read_json(history_path) if history_path.exists() else []
    done = _done(history)
    plan_name = row.get('plan_file') or str(Path('results') / 'experiments' / 'results_iterative_5core' / case / 'best_plan.json')
    incumbent_path = ROOT / plan_name
    if not incumbent_path.exists():
        incumbent_path = ROOT / 'results' / 'experiments' / plan_name
    incumbent = _read_json(incumbent_path)
    best = (int(row.get('makespan_cycles', 10**30)), int(row.get('added_copy_bytes', 10**30)))
    best_plan = incumbent
    # A resumed run skips already valid candidates.  Start from its persisted
    # winner so those skipped candidates remain reflected in results.csv.
    saved_plan_path = folder / 'best_plan.json'
    saved_result_path = folder / 'best_result.json'
    if saved_plan_path.exists() and saved_result_path.exists():
        saved_result = _read_json(saved_result_path)
        saved_best = _key(saved_result)
        if saved_best < best:
            best, best_plan = saved_best, _read_json(saved_plan_path)
    graph = _read_json(ROOT / 'official' / 'data' / f'{case}.json')
    for index, plan in enumerate(candidate_variants(graph, incumbent, 5, cfg, waits), 1):
        if signature(plan) in done:
            continue
        result, info = official_score(case, plan, folder, f'candidate_{index}', timeout,
                                      fast_evaluator=False)
        record = {'candidate': f'candidate_{index}', 'plan': plan, **info, 'accepted': False}
        if result is not None and _key(result) < best:
            best, best_plan = _key(result), plan
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


def write_summary(rows, results, all_rows):
    target = OUT / 'results.csv'
    previous = {}
    if target.exists():
        with target.open(encoding='utf-8-sig', newline='') as stream:
            previous = {row['case']: row for row in csv.DictReader(stream)}
    updated = dict(results)
    output = []
    for row in all_rows:
        case = row['case']
        if case not in updated:
            if case in previous:
                output.append({field: previous[case].get(field, '')
                               for field in SUMMARY_FIELDS})
            continue
        makespan, bytes_ = updated[case]
        output.append({'case': case, 'status': 'scored',
                       'baseline_makespan_cycles': row['makespan_cycles'],
                       'baseline_added_copy_bytes': row['added_copy_bytes'],
                       'makespan_cycles': makespan, 'added_copy_bytes': bytes_,
                       'improved': str((makespan, bytes_) <
                                       (int(row['makespan_cycles']), int(row['added_copy_bytes']))).lower(),
                       'method': 'fm_5core_refined',
                       # Results live below results/experiments after the FM
                       # runner relocation; keep the exported path directly
                       # usable by the final report generator.
                       'plan_file': str(Path('results') / 'experiments' / 'results_fm_5core' / case / 'best_plan.json')})
    if not output:
        return target
    temporary = target.with_name(target.name + '.writing')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=50)
    args = parser.parse_args()
    if args.timeout <= 0 or args.workers < 1:
        parser.error('timeout and workers must be positive')
    all_rows = five_core_rows()
    rows = all_rows
    if args.cases:
        selected = set(args.cases)
        rows = [row for row in all_rows if row['case'] in selected]
    cfg = read_evaluation_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    waits = read_scene_a_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    OUT.mkdir(exist_ok=True)
    jobs = [(row, cfg, waits, args.timeout) for row in rows]
    if args.workers == 1:
        results = [run_one(*job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_job, jobs))
    if rows:
        print('Wrote:', write_summary(rows, results, all_rows), flush=True)


if __name__ == '__main__':
    main()

"""Resume-safe FM refinement for the packaged 2--4 core incumbents.

The fixed one-core baseline is never changed.  For each selected core count,
the packaged plan is the incumbent and acyclic FM generates a bounded set of
rescheduled candidates.  Existing five-core FM output is consumed by the
report generator and is never rescored here.
"""

import argparse
import concurrent.futures
import csv
from pathlib import Path

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from hybrid_solver import official_score, signature
from fm_candidates import candidate_variants


OUT = ROOT / 'results' / 'experiments' / 'results_fm_allcores'
SOURCE = ROOT / 'results' / 'final' / 'final_results_100cases.csv'
MAX_FINALS = 8
SUMMARY_FIELDS = ('case', 'cores', 'status', 'baseline_makespan_cycles',
                  'baseline_added_copy_bytes', 'makespan_cycles',
                  'added_copy_bytes', 'improved', 'method', 'plan_file')


def _key(result):
    return int(result['makespan']), int(result['data_movement_bytes']['added_copy_bytes'])


def _source_rows():
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {(row['case'], int(row['cores'])): row for row in rows
            if row.get('status') == 'scored' and row.get('case')}


def run_one(row, cores, cfg, waits, timeout):
    case = row['case']
    folder = OUT / f'{case}_{cores}cores'
    folder.mkdir(parents=True, exist_ok=True)
    history_path = folder / 'history.json'
    history = _read_json(history_path) if history_path.exists() else []
    done = {signature(r['plan']) for r in history
            if r.get('status') == 'valid' and isinstance(r.get('plan'), dict)}
    incumbent_path = ROOT / row['plan_file']
    if not incumbent_path.exists():
        incumbent_path = ROOT / 'results' / 'final' / row['plan_file']
    incumbent = _read_json(incumbent_path)
    baseline_key = (int(row['makespan_cycles']), int(row['added_copy_bytes']))
    best, best_plan = baseline_key, incumbent
    saved_result = folder / 'best_result.json'
    saved_plan = folder / 'best_plan.json'
    if saved_result.exists() and saved_plan.exists():
        saved = _read_json(saved_result)
        if _key(saved) < best:
            best, best_plan = _key(saved), _read_json(saved_plan)
    if not any(r.get('status') == 'incumbent' for r in history):
        history.append({'candidate': 'incumbent', 'status': 'incumbent',
                        'plan': incumbent, 'makespan_cycles': baseline_key[0],
                        'added_copy_bytes': baseline_key[1], 'accepted': True})
        save(saved_plan, incumbent)
        save(folder / 'incumbent_score.json', {
            'makespan': baseline_key[0],
            'data_movement_bytes': {'added_copy_bytes': baseline_key[1]},
            'source': 'results/final/final_results_100cases.csv'})
        save(history_path, history)
    graph = _read_json(ROOT / 'official' / 'data' / f'{case}.json')
    for index, plan in enumerate(candidate_variants(graph, incumbent, cores, cfg, waits), 1):
        if signature(plan) in done or index == 1:
            continue
        result, info = official_score(case, plan, folder, f'candidate_{index}', timeout,
                                      fast_evaluator=False)
        record = {'candidate': f'candidate_{index}', 'plan': plan, **info, 'accepted': False}
        if result is not None and _key(result) < best:
            best, best_plan = _key(result), plan
            record['accepted'] = True
            save(saved_plan, plan)
            save(saved_result, result)
        history.append(record)
        save(history_path, history)
    if not saved_plan.exists():
        save(saved_plan, best_plan)
    return case, cores, best


def run_job(job):
    return run_one(*job)


def write_summary(rows, results, all_rows):
    # Selective runs retain only rows that were previously scored.  Filling
    # every absent source row here would falsely claim an FM result and invent
    # a best-plan path for a case/core pair that this invocation did not run.
    previous = {}
    target = OUT / 'results.csv'
    if target.exists():
        with target.open(encoding='utf-8-sig', newline='') as stream:
            previous = {(r['case'], int(r['cores'])): r
                        for r in csv.DictReader(stream)}
    updated = {(case, cores): key for case, cores, key in results}
    output = []
    for row in all_rows:
        case, cores = row['case'], int(row['cores'])
        prior = previous.get((case, cores))
        if (case, cores) in updated:
            makespan, bytes_ = updated[(case, cores)]
            output.append({'case': case, 'cores': cores, 'status': 'scored',
                           'baseline_makespan_cycles': row['makespan_cycles'],
                           'baseline_added_copy_bytes': row['added_copy_bytes'],
                           'makespan_cycles': makespan, 'added_copy_bytes': bytes_,
                           'improved': str((makespan, bytes_) <
                                           (int(row['makespan_cycles']), int(row['added_copy_bytes']))).lower(),
                           'method': 'fm_selected_refined',
                           'plan_file': str(Path('results') / 'experiments' /
                                            'results_fm_allcores' / f'{case}_{cores}cores' / 'best_plan.json')})
        elif prior:
            output.append({field: prior.get(field, '') for field in SUMMARY_FIELDS})
    if not output:
        return target
    with target.with_name('results.csv.writing').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    target.with_name('results.csv.writing').replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores', nargs='+', type=int, choices=[2, 3, 4], default=[2, 3, 4])
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--workers', type=int, default=50)
    args = parser.parse_args()
    if args.timeout <= 0 or args.workers < 1:
        parser.error('timeout and workers must be positive')
    source = _source_rows()
    selected = set(args.cases) if args.cases else None
    all_rows = [source[(case, cores)] for case in sorted({c for c, _ in source})
                for cores in (2, 3, 4)]
    rows = [row for row in all_rows
            if int(row['cores']) in args.cores and
            (selected is None or row['case'] in selected)]
    cfg = read_evaluation_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    waits = read_scene_a_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(row, int(row['cores']), cfg, waits, args.timeout) for row in rows]
    if args.workers == 1:
        results = [run_one(*job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_job, jobs))
    if rows:
        print('Wrote:', write_summary(rows, results, all_rows), flush=True)


if __name__ == '__main__':
    main()

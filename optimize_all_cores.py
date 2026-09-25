"""Apply communication-aware adaptive partitioning to every multicore case.

The existing one-core rows remain the official single-core reference. This
script independently optimizes 2, 3, 4, and 5-core plans and writes a separate
result table, leaving the packaged final results untouched.
"""

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

from adaptive_clustering import adaptive_partition
from experiment import ROOT, _read_json, save, read_evaluation_config, read_scene_a_config
from fast_solver import GraphModel
from hybrid_solver import official_score, signature


SOURCE = ROOT / 'results' / 'final' / 'final_results_100cases.csv'
OUT = ROOT / 'results' / 'experiments' / 'results_allcores_adaptive'


def result_key(makespan, added_bytes):
    return int(makespan), int(added_bytes)


def run_one(case, cores, base_row, cfg, waits, penalties, timeout):
    folder = OUT / f'{case}_{cores}cores'
    folder.mkdir(parents=True, exist_ok=True)
    history_file = folder / 'history.json'
    base_plan_path = ROOT / 'results' / 'final' / base_row['plan_file']
    baseline_plan = _read_json(base_plan_path)
    baseline_key = result_key(base_row['makespan_cycles'], base_row['added_copy_bytes'])

    if (folder / 'best_plan.json').exists() and (folder / 'best_score.json').exists():
        best_plan = _read_json(folder / 'best_plan.json')
        best_score = _read_json(folder / 'best_score.json')
        best_key = result_key(best_score['makespan_cycles'], best_score['added_copy_bytes'])
    else:
        best_plan = baseline_plan
        best_key = baseline_key
        best_score = {'makespan_cycles': baseline_key[0], 'added_copy_bytes': baseline_key[1],
                      'method': base_row['method'], 'plan_file': str(base_plan_path)}
        save(folder / 'best_plan.json', best_plan)
        save(folder / 'best_score.json', best_score)

    history = _read_json(history_file) if history_file.exists() else []
    completed = {row['candidate'] for row in history
                 if row.get('status') not in {'timeout', 'error'}}
    graph = _read_json(ROOT / 'official' / 'data' / f'{case}.json')
    gm = GraphModel(graph)

    for penalty in penalties:
        label = 'adaptive_p' + str(penalty).replace('.', 'p')
        if label in completed:
            continue
        print(case, f'{cores}-core', label, '开始构造候选', flush=True)
        try:
            plan, detail = adaptive_partition(gm, cores, cfg, waits,
                                              parallel_penalty=penalty)
            if signature(plan) == signature(baseline_plan):
                record = {'candidate': label, 'status': 'duplicate_baseline',
                          'accepted': False, 'detail': detail}
            else:
                cached_plan = folder / f'{label}_plan.json'
                cached_result = folder / f'{label}_result.json'
                if (cached_plan.exists() and cached_result.exists() and
                        signature(_read_json(cached_plan)) == signature(plan)):
                    result = _read_json(cached_result)
                    info = {'status': 'valid_cached', 'makespan': result['makespan'],
                            'added_copy_bytes': result['data_movement_bytes']['added_copy_bytes']}
                else:
                    print(case, f'{cores}-core', label,
                          f'开始正式评分，单候选上限 {timeout:g} 秒', flush=True)
                    result, info = official_score(case, plan, folder, label, timeout,
                                                  fast_evaluator=True)
                candidate_key = (result_key(result['makespan'],
                                            result['data_movement_bytes']['added_copy_bytes'])
                                 if result is not None else None)
                accepted = candidate_key is not None and candidate_key < best_key
                record = {'candidate': label, **info, 'accepted': accepted, 'detail': detail}
                if accepted:
                    best_key = candidate_key
                    best_plan = plan
                    best_score = {'makespan_cycles': best_key[0],
                                  'added_copy_bytes': best_key[1],
                                  'method': f'adaptive_partition_penalty_{penalty}',
                                  'plan_file': str((folder / 'best_plan.json').relative_to(ROOT)),
                                  'result_file': str((folder / f'{label}_result.json').relative_to(ROOT))}
                    save(folder / 'best_plan.json', best_plan)
                    save(folder / 'best_score.json', best_score)
            history.append(record)
        except (ValueError, RuntimeError, KeyError) as error:
            history.append({'candidate': label, 'status': 'invalid',
                            'accepted': False, 'message': str(error)[:500]})
        save(history_file, history)
        print(case, f'{cores}-core', label, history[-1]['status'],
              'best_makespan', best_key[0], flush=True)

    return best_score, best_key


def run_job(job):
    case, cores, base_row, cfg, waits, penalties, timeout = job
    try:
        score, key = run_one(case, cores, base_row, cfg, waits, penalties, timeout)
        return case, cores, score, key, None
    except Exception as error:
        return case, cores, None, None, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores', nargs='+', type=int, default=[2, 3, 4, 5],
                        choices=[2, 3, 4, 5])
    parser.add_argument('--cases', nargs='*', help='Optional case names, e.g. case_001 case_002')
    parser.add_argument('--penalties', nargs='+', type=float, default=[1.0, 2.0])
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=25,
                        help='parallel case/core jobs (default: 25)')
    args = parser.parse_args()
    if args.timeout <= 0 or args.workers < 1 or any(p < 0 for p in args.penalties):
        parser.error('timeout and workers must be positive and penalties must be nonnegative')

    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        original_rows = list(csv.DictReader(stream))
    selected_cases = set(args.cases) if args.cases else None
    rows_by_config = {(row['case'], int(row['cores'])): row for row in original_rows}
    cfg = read_evaluation_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    waits = read_scene_a_config(str(ROOT / 'official' / 'data' / 'config.txt'))
    OUT.mkdir(exist_ok=True)

    jobs = []
    for case in sorted({row['case'] for row in original_rows}):
        if selected_cases is not None and case not in selected_cases:
            continue
        for cores in sorted(set(args.cores)):
            row = rows_by_config[(case, cores)]
            if row['status'] != 'scored' or not row['plan_file']:
                print('SKIP: missing scored baseline', case, cores, flush=True)
                continue
            jobs.append((case, cores, row, cfg, waits, args.penalties, args.timeout))

    completed = 0
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            case, cores, score, key, error = future.result()
            completed += 1
            if error:
                errors.append({'case': case, 'cores': cores, 'error': error})
                print(f'[{completed}/{len(jobs)}] ERROR {case} {cores}-core: {error}', flush=True)
            else:
                print(f'[{completed}/{len(jobs)}] DONE {case} {cores}-core makespan={key[0]}', flush=True)

    # Build an independent table, retaining the original one-core reference.
    output_rows = []
    one_core = {row['case']: int(row['makespan_cycles']) for row in original_rows
                if int(row['cores']) == 1 and row['makespan_cycles']}
    for row in original_rows:
        item = {key: row.get(key, '') for key in
                ('case', 'cores', 'status', 'makespan_cycles', 'added_copy_bytes', 'speedup')}
        item['method'], item['plan_file'], item['result_file'] = (
            row.get('method', ''), row.get('plan_file', ''), row.get('result_file', ''))
        cores = int(row['cores'])
        folder = OUT / f"{row['case']}_{cores}cores"
        score_file = folder / 'best_score.json'
        if cores > 1 and score_file.exists():
            score = _read_json(score_file)
            item['makespan_cycles'] = str(score['makespan_cycles'])
            item['added_copy_bytes'] = str(score['added_copy_bytes'])
            item['method'] = score['method']
            item['plan_file'] = str(Path('results') / 'experiments' / 'results_allcores_adaptive' /
                                    f"{row['case']}_{cores}cores" / 'best_plan.json')
            item['result_file'] = score.get('result_file', '')
        baseline = one_core.get(row['case'])
        if baseline and item['makespan_cycles']:
            item['speedup'] = f"{baseline / int(item['makespan_cycles']):.9f}"
        output_rows.append(item)

    result_file = OUT / 'final_results_all_cores.csv'
    with result_file.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    averages_file = OUT / 'average_speedup_all_cores.csv'
    with averages_file.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cores', 'scored_cases', 'average_speedup'])
        writer.writeheader()
        for cores in range(1, 6):
            values = [float(row['speedup']) for row in output_rows
                      if int(row['cores']) == cores and row['speedup']]
            writer.writerow({'cores': cores, 'scored_cases': len(values),
                             'average_speedup': f'{sum(values) / len(values):.9f}'
                             if values else ''})
    print('Finished:', result_file, averages_file, flush=True)
    if errors:
        error_file = OUT / 'errors.json'
        save(error_file, errors)
        print(f'Failed jobs: {len(errors)}; details: {error_file}', flush=True)


if __name__ == '__main__':
    main()

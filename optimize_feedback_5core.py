"""Resumable, standalone observed feedback refinement for five cores."""
import argparse
import concurrent.futures
import csv
from pathlib import Path

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from hybrid_solver import official_score, signature
from feedback_refinement import candidates

OUT = ROOT / 'results' / 'experiments' / 'results_feedback_5core'
SOURCE = ROOT / 'results' / 'experiments' / 'results_multilevel_5core' / 'results.csv'


def key(result):
    return int(result['makespan']), int(result['data_movement_bytes']['added_copy_bytes'])


def rows():
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return [r for r in reader if r.get('status') == 'scored' and r.get('plan_file')]


def run_one(row, cfg, waits, timeout):
    case = row['case']; folder = OUT / case; folder.mkdir(parents=True, exist_ok=True)
    history_path = folder / 'history.json'
    history = _read_json(history_path) if history_path.exists() else []
    plan_path = ROOT / row['plan_file']
    incumbent = _read_json(plan_path)
    graph = _read_json(ROOT / 'official' / 'data' / f'{case}.json')
    source_key = (int(row['makespan_cycles']), int(row['added_copy_bytes']))
    calibration = next((r for r in history if r.get('family') == 'incumbent_calibration'), None)
    if calibration is None:
        result, info = official_score(case, incumbent, folder, 'incumbent_calibration', timeout, fast_evaluator=True)
        if result is None:
            raise RuntimeError(f'{case}: incumbent calibration failed: {info}')
        if key(result) != source_key:
            raise RuntimeError(f'{case}: source key {source_key} != official incumbent {key(result)}')
        calibration = {'family': 'incumbent_calibration', 'status': 'valid', 'plan': incumbent,
                       'plan_signature': repr(signature(incumbent)), 'makespan': key(result)[0],
                       'added_copy_bytes': key(result)[1], **info}
        history.append(calibration); save(history_path, history)
    elif (calibration.get('makespan'), calibration.get('added_copy_bytes')) != source_key:
        raise RuntimeError(f'{case}: cached calibration does not match source')
    best = source_key; best_plan = incumbent
    if (folder / 'best_result.json').exists() and (folder / 'best_plan.json').exists():
        saved = _read_json(folder / 'best_result.json')
        if key(saved) < best:
            best, best_plan = key(saved), _read_json(folder / 'best_plan.json')
    result = _read_json(folder / 'incumbent_calibration_result.json')
    proposals, analysis = candidates(graph, incumbent, result, cfg, waits, 5, 4)
    # A timeout or evaluator process error is not a verdict on this plan.  Do
    # retain successful exact signatures, while allowing a resumed run to retry
    # interrupted scoring work.
    done = {r.get('plan_signature') for r in history if r.get('status') == 'valid'}
    for index, proposal in enumerate(proposals, 1):
        plan = proposal['plan']; token = repr(signature(plan))
        if token in done: continue
        label = f"candidate_{index}_{proposal['family']}"
        scored, info = official_score(case, plan, folder, label, timeout, fast_evaluator=True)
        record = {'family': proposal['family'], 'candidate': label, 'plan': plan,
                  'plan_signature': token, 'construction': proposal['detail'],
                  'proxy_makespan': proposal['proxy_makespan'], **info, 'accepted': False}
        if scored is not None and key(scored) < best:
            best, best_plan = key(scored), plan; record['accepted'] = True
            save(folder / 'best_plan.json', plan); save(folder / 'best_result.json', scored)
        history.append(record); done.add(token); save(history_path, history)
    if not (folder / 'best_plan.json').exists():
        save(folder / 'best_plan.json', best_plan)
    return case, best


def _job(args): return run_one(*args)


def write_summary(items, selected_rows):
    by_case = dict(items); out = []
    for row in selected_rows:
        value = by_case.get(row['case'], (int(row['makespan_cycles']), int(row['added_copy_bytes'])))
        out.append({'case': row['case'], 'status': 'scored', 'baseline_makespan_cycles': row['makespan_cycles'],
                    'baseline_added_copy_bytes': row['added_copy_bytes'], 'makespan_cycles': value[0],
                    'added_copy_bytes': value[1], 'improved': str(value < (int(row['makespan_cycles']), int(row['added_copy_bytes']))).lower(),
                    'plan_file': str(Path('results') / 'experiments' / 'results_feedback_5core' / row['case'] / 'best_plan.json')})
    target = OUT / 'results.csv'; temp = target.with_name(target.name + '.writing')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0])); writer.writeheader(); writer.writerows(out)
    temp.replace(target); return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='*'); parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=50)
    args = parser.parse_args()
    if args.timeout <= 0 or args.workers < 1: parser.error('timeout and workers must be positive')
    cfg = read_evaluation_config(str(ROOT / 'official/data/config.txt'))
    waits = read_scene_a_config(str(ROOT / 'official/data/config.txt'))
    selected = set(args.cases) if args.cases else None
    selected_rows = [r for r in rows() if selected is None or r['case'] in selected]
    OUT.mkdir(exist_ok=True)
    jobs = [(r, cfg, waits, args.timeout) for r in selected_rows]
    results = []
    if args.workers == 1:
        for job in jobs: results.append(run_one(*job)); print(results[-1], flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for value in pool.map(_job, jobs): results.append(value); print(value, flush=True)
    if selected_rows: print('Wrote:', write_summary(results, selected_rows), flush=True)


if __name__ == '__main__': main()

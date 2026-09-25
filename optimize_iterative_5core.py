"""Resumable four-round iterative refinement for standalone five-core plans."""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from hybrid_solver import official_score, signature
from iterative_refinement import candidates

OUT = ROOT / 'results' / 'experiments' / 'results_iterative_5core'
SOURCE = ROOT / 'results' / 'experiments' / 'results_feedback_5core' / 'results.csv'
ROUNDS = 4


def key(result):
    return int(result['makespan']), int(result['data_movement_bytes']['added_copy_bytes'])


def rows():
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        return [r for r in csv.DictReader(stream)
                if r.get('status') == 'scored' and r.get('plan_file')]


def _sig(plan):
    return repr(signature(plan))


def _clean(folder, label, keep=False):
    if keep: return
    for suffix in ('_plan.json', '_result.json', '_trace.json', '_log.txt'):
        (folder / f'{label}{suffix}').unlink(missing_ok=True)


def run_one(row, cfg, waits, timeout):
    case = row['case']; folder = OUT / case; folder.mkdir(parents=True, exist_ok=True)
    history_path = folder / 'history.json'
    history = _read_json(history_path) if history_path.exists() else []
    graph = _read_json(ROOT / 'official' / 'data' / f'{case}.json')
    plan_path = ROOT / row['plan_file']; incumbent = _read_json(plan_path)
    if len(incumbent.get('core_schedules', ())) != 5:
        raise ValueError(f'{case}: iterative refinement accepts standalone five-core plans only')
    source_key = (int(row['makespan_cycles']), int(row['added_copy_bytes']))
    calibration_path = folder / 'incumbent_calibration_result.json'
    source_signature = _sig(incumbent)
    calibration = next((r for r in history if r.get('family') == 'incumbent_calibration'
                        and r.get('status') == 'valid'
                        and r.get('plan_signature') == source_signature
                        and tuple(r.get('source_key', ())) == source_key
                        and calibration_path.exists()), None)
    if calibration is None:
        result, details = official_score(case, incumbent, folder, 'incumbent_calibration', timeout, fast_evaluator=True)
        if result is None: raise RuntimeError(f'{case}: incumbent calibration failed: {details}')
        if key(result) != source_key: raise RuntimeError(f'{case}: source key does not match official incumbent')
        calibration = {'family': 'incumbent_calibration', 'status': 'valid', 'round': 0,
                       'plan': incumbent, 'plan_signature': source_signature,
                       'source_key': source_key, **details}
        history.append(calibration); save(history_path, history)
    result = _read_json(calibration_path)
    best_plan = incumbent; best_key = source_key
    best_result_path = folder / 'best_result.json'
    state_path = folder / 'best_state.json'
    state = _read_json(state_path) if state_path.exists() else {}
    start_round = 1
    state_matches = (state.get('source_signature') == source_signature and
                     tuple(state.get('source_key', ())) == source_key)
    if state_matches:
        start_round = int(state.get('completed_round', 0)) + 1
        if (folder / 'best_plan.json').exists() and best_result_path.exists():
            best_plan = _read_json(folder / 'best_plan.json'); best_key = key(_read_json(best_result_path)); result = _read_json(best_result_path)
        if state.get('finished'):
            return case, best_key
    else:
        # Bind an existing output folder to this exact source before doing any
        # new work; stale plans cannot advance a resumed run.
        save(folder / 'best_plan.json', best_plan)
        save(state_path, {'source_signature': source_signature, 'source_key': source_key,
                          'plan_signature': _sig(best_plan), 'completed_round': 0})
    for round_no in range(start_round, ROUNDS + 1):
        proposals, _ = candidates(graph, best_plan, result, cfg, waits, 5, 8)
        parent = _sig(best_plan)
        done = {_sig(r['plan']) for r in history if r.get('round') == round_no and r.get('parent_signature') == parent
                and r.get('status') == 'valid' and r.get('plan')}
        # Scores are deterministic for a plan.  A plan that occurred under a
        # different parent is not evaluated again; parent-local records remain
        # the only inputs to this round's winner recovery.
        done.update(_sig(r['plan']) for r in history if r.get('status') == 'valid' and r.get('plan'))
        # Evaluate one bounded neighbourhood against a fixed official parent.
        # Select its official winner only after the batch, then regenerate from
        # that winner in the next round.
        prior = [r for r in history if r.get('round') == round_no and r.get('parent_signature') == parent
                 and r.get('status') == 'valid' and r.get('plan')]
        prior = [r for r in prior if (int(r['makespan']), int(r['added_copy_bytes'])) < best_key]
        if prior:
            recovered = min(prior, key=lambda r: (int(r['makespan']), int(r['added_copy_bytes'])))
            round_best = ((int(recovered['makespan']), int(recovered['added_copy_bytes'])),
                          recovered['plan'], None, recovered)
        else:
            round_best = None
        successful = True
        for index, proposal in enumerate(proposals, 1):
            plan = proposal['plan']; token = _sig(plan)
            if token in done: continue
            label = f"round_{round_no}_candidate_{index}_{proposal['family']}"
            scored, details = official_score(case, plan, folder, label, timeout, fast_evaluator=True)
            successful = successful and scored is not None
            record = {'family': proposal['family'], 'round': round_no, 'candidate': label,
                      'parent_signature': parent, 'plan': plan, 'plan_signature': token,
                      'construction': proposal['detail'], 'proxy_makespan': proposal['proxy_makespan'], **details,
                      'accepted': False}
            if scored is not None and key(scored) < best_key:
                candidate_key = key(scored)
                if round_best is None or candidate_key < round_best[0]:
                    round_best = candidate_key, plan, scored, record
            history.append(record); save(history_path, history)
        if round_best is None:
            if successful:
                save(state_path, {'source_signature': source_signature, 'source_key': source_key,
                                  'plan_signature': _sig(best_plan), 'completed_round': round_no,
                                  'finished': True})
                for index, proposal in enumerate(proposals, 1):
                    _clean(folder, f"round_{round_no}_candidate_{index}_{proposal['family']}")
            break
        best_key, best_plan, result, accepted_record = round_best
        if result is None:
            # A previous interruption may have retained the compact history
            # record while removing its transient evaluator result. Restore
            # only the selected winner, then continue from its real timeline.
            prior_result = folder / f"{accepted_record['candidate']}_result.json"
            if prior_result.exists():
                result = _read_json(prior_result)
            else:
                result, details = official_score(case, best_plan, folder,
                                                 f'round_{round_no}_resume_winner', timeout,
                                                 fast_evaluator=True)
                if result is None:
                    raise RuntimeError(f'{case}: could not restore round {round_no} winner: {details}')
                _clean(folder, f'round_{round_no}_resume_winner')
        accepted_record['accepted'] = True
        save(folder / 'best_plan.json', best_plan); save(folder / 'best_result.json', result)
        save(state_path, {'source_signature': source_signature, 'source_key': source_key,
                          'plan_signature': _sig(best_plan), 'completed_round': round_no})
        save(history_path, history)
        for index, proposal in enumerate(proposals, 1):
            _clean(folder, f"round_{round_no}_candidate_{index}_{proposal['family']}")
    if not (folder / 'best_plan.json').exists():
        save(folder / 'best_plan.json', best_plan)
        save(state_path, {'source_signature': source_signature, 'source_key': source_key,
                          'plan_signature': _sig(best_plan), 'completed_round': 0})
    return case, best_key


def _job(args): return run_one(*args)


def write_summary(items, selected):
    values = dict(items); out = []
    for row in selected:
        value = values.get(row['case'], (int(row['makespan_cycles']), int(row['added_copy_bytes'])))
        out.append({'case': row['case'], 'status': 'scored',
                    'baseline_makespan_cycles': row['makespan_cycles'],
                    'baseline_added_copy_bytes': row['added_copy_bytes'],
                    'makespan_cycles': value[0], 'added_copy_bytes': value[1],
                    'improved': str(value < (int(row['makespan_cycles']), int(row['added_copy_bytes']))).lower(),
                    'plan_file': str(Path('results') / 'experiments' / 'results_iterative_5core' / row['case'] / 'best_plan.json')})
    target = OUT / 'results.csv'; temp = target.with_name('results.csv.writing')
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
    wanted = set(args.cases) if args.cases else None
    selected = [r for r in rows() if wanted is None or r['case'] in wanted]; OUT.mkdir(exist_ok=True)
    jobs = [(r, cfg, waits, args.timeout) for r in selected]; results = []
    if args.workers == 1:
        for job in jobs: results.append(run_one(*job)); print(results[-1], flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for value in pool.map(_job, jobs): results.append(value); print(value, flush=True)
    if selected: print('Wrote:', write_summary(results, selected), flush=True)


if __name__ == '__main__': main()

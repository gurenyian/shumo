"""Ablate fixed-partition insertion scheduling and adaptive DAG clustering.

Checkpoints and candidate plans live in results/experiments/results_refine_v3.
Accepted results are published into results/experiments/results_opt5, while
every before-plan and score is preserved.
"""

import argparse
import csv
import math
import time

from adaptive_clustering import adaptive_partition, schedule_groups
from experiment import ROOT, _read_json, save, read_evaluation_config, read_scene_a_config
from fast_solver import GraphModel, topo
from hybrid_solver import groups_from_plan, official_score, signature
from rescue_fivecore import OUTPUT, key, rows_by_case, compare_table, write_final_results, write_chart


RUNS = ROOT / 'results' / 'experiments' / 'results_refine_v3'


def diagnose(gm, plan, result):
    groups, _ = groups_from_plan(plan)
    original_ids = sorted(set(plan['node_to_subgraph'].values()))
    measured = {task['subgraph_id']: task['duration']
                for core in result['per_core_timeline'] for task in core['tasks']}
    durations = [measured[v] for v in original_ids]
    _, pred, succ, _, _, proxy = gm.describe(groups, result['bandwidth_bytes_per_cycle'])
    finish = {}
    for v in topo(pred, succ):
        finish[v] = durations[v] + max((finish[p] for p in pred[v]), default=0)
    busy = [sum(task['duration'] for task in core['tasks'])
            for core in result['per_core_timeline']]
    return {'before_makespan': result['makespan'],
            'before_added_copy_bytes': result['data_movement_bytes']['added_copy_bytes'],
            'observed_durations': durations, 'task_count': len(groups),
            'core_busy_cycles': busy,
            'task_cp_at_observed_durations': max(finish.values(), default=0),
            'op_compute_critical_path': max(gm.critical.values(), default=0),
            'compute_lower_bound': max(max(gm.critical.values(), default=0),
                                       max(gm.pipes.values(), default=0) / len(busy)),
            'proxy_duration_sum': sum(proxy), 'observed_duration_sum': sum(durations)}


def run_case(case, cfg, waits, timeout, penalties, traffic_feedback=False):
    started = time.perf_counter()
    folder = RUNS / case
    incumbent_folder = OUTPUT / case
    incumbent = _read_json(incumbent_folder / 'best_result.json')
    incumbent_plan = _read_json(incumbent_folder / 'best_plan.json')
    summary = _read_json(incumbent_folder / 'summary.json')
    graph = _read_json(ROOT / 'official/data' / f'{case}.json')
    gm = GraphModel(graph)
    if not (folder / 'before_plan.json').exists():
        save(folder / 'before_plan.json', incumbent_plan)
        save(folder / 'diagnostics.json', diagnose(gm, incumbent_plan, incumbent))
    before_plan = _read_json(folder / 'before_plan.json')
    diagnostics = _read_json(folder / 'diagnostics.json')
    records = _read_json(folder / 'history.json') if (folder / 'history.json').exists() else []
    checked = {row['candidate'] for row in records}

    def score(label, construct):
        nonlocal incumbent, incumbent_plan
        if label in checked:
            return
        begin = time.perf_counter()
        try:
            candidate, detail = construct()
            construction_seconds = time.perf_counter() - begin
            if signature(candidate) == signature(incumbent_plan):
                result, info = None, {'status': 'duplicate', 'makespan': incumbent['makespan']}
            else:
                cached_plan = folder / f'{label}_plan.json'
                cached_result = folder / f'{label}_result.json'
                if (cached_plan.exists() and cached_result.exists() and
                        signature(_read_json(cached_plan)) == signature(candidate)):
                    result = _read_json(cached_result)
                    info = {'status': 'valid', 'makespan': result['makespan'], 'cached': True,
                            'added_copy_bytes': result['data_movement_bytes']['added_copy_bytes']}
                else:
                    result, info = official_score(case, candidate, folder, label, timeout,
                                                  fast_evaluator=True)
            info['construction'] = detail
            info['construction_seconds'] = construction_seconds
        except (ValueError, RuntimeError) as error:
            result, info = None, {'status': 'invalid', 'message': str(error)[:500]}
        accepted = result is not None and key(result) < key(incumbent)
        record = {'candidate': label, **info, 'accepted': accepted}
        records.append(record)
        summary['candidate_history'].append(record)
        if accepted:
            incumbent, incumbent_plan = result, candidate
            summary['selected'] = label
            summary['best_makespan'] = result['makespan']
            save(incumbent_folder / 'best_result.json', incumbent)
            save(incumbent_folder / 'best_plan.json', incumbent_plan)
        save(incumbent_folder / 'summary.json', summary)
        save(folder / 'history.json', records)
        print(case, label, info['status'], 'candidate', info.get('makespan'),
              'best', incumbent['makespan'], 'accepted', accepted, flush=True)

    groups, _ = groups_from_plan(before_plan)
    score('v3_observed_insertion', lambda: schedule_groups(
        gm, groups, 5, cfg, waits, diagnostics['observed_durations']))
    for penalty in penalties:
        label = 'v3_adaptive_' + str(penalty).replace('.', 'p')
        score(label, lambda penalty=penalty: adaptive_partition(
            gm, 5, cfg, waits, parallel_penalty=penalty))
    seed_result_file = folder / 'v3_adaptive_1p0_result.json'
    if traffic_feedback and seed_result_file.exists():
        seed = _read_json(seed_result_file)
        task_count = max(1, len(seed['step3_by_task']))
        ddr_floor = seed['data_movement_bytes']['scheduled_copy_bytes'] / cfg['bandwidth']
        # The base work target is sqrt(synchronization_cost * ideal_core_work).
        # Feed observed mean transfer service into that same overhead model.
        scale = math.sqrt(1 + ddr_floor / (task_count *
                          max(1, waits['task_cross_core_wait_cycles'])))
        if ddr_floor >= .5 * seed['makespan'] and scale >= 1.1:
            def communication_feedback():
                plan, info = adaptive_partition(gm, 5, cfg, waits,
                                                target_scale=scale)
                info.update({'feedback_scale': scale, 'observed_ddr_floor': ddr_floor,
                             'mean_transfer_service': ddr_floor / task_count,
                             'seed_makespan': seed['makespan']})
                return plan, info
            score('v3_traffic_feedback', communication_feedback)
    if incumbent_plan['node_to_subgraph'] != before_plan['node_to_subgraph']:
        def calibrated_reschedule():
            current_groups, _ = groups_from_plan(incumbent_plan)
            current_diagnostic = diagnose(gm, incumbent_plan, incumbent)
            return schedule_groups(gm, current_groups, 5, cfg, waits,
                                   current_diagnostic['observed_durations'])
        score('v3_final_insertion', calibrated_reschedule)
    save(folder / 'summary.json', {'case': case, 'before': diagnostics['before_makespan'],
                                  'after': incumbent['makespan'],
                                  'elapsed_seconds_this_run': time.perf_counter() - started})


def write_ablation():
    rows = []
    for file in sorted(RUNS.glob('case_*/diagnostics.json')):
        case = file.parent.name
        diagnostic = _read_json(file)
        records = _read_json(file.parent / 'history.json') if (file.parent / 'history.json').exists() else []
        row = {'case': case, 'before_makespan': diagnostic['before_makespan'],
               'after_makespan': _read_json(OUTPUT / case / 'best_result.json')['makespan'],
               'task_count_before': diagnostic['task_count'],
               'compute_lower_bound': diagnostic['compute_lower_bound'],
               'task_cp_observed': diagnostic['task_cp_at_observed_durations'],
               'max_core_busy': max(diagnostic['core_busy_cycles'], default=0),
               'min_core_busy': min(diagnostic['core_busy_cycles'], default=0)}
        for record in records:
            row[record['candidate']] = record.get('makespan', record['status'])
        rows.append(row)
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (RUNS / 'ablation.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', type=float, default=4.5)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--penalties', nargs='*', type=float, default=[1.0, 2.0])
    parser.add_argument('--traffic-feedback', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0 or args.threshold <= 0 or any(p < 0 for p in args.penalties):
        parser.error('timeout and threshold must be positive; penalties nonnegative')
    RUNS.mkdir(exist_ok=True)
    if not (RUNS / 'before_comparison.csv').exists():
        (RUNS / 'before_comparison.csv').write_bytes((OUTPUT / 'comparison.csv').read_bytes())
    source = rows_by_case()
    with (RUNS / 'before_comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
        selected = [row['case'] for row in csv.DictReader(stream)
                    if float(row['new_best_speedup']) < args.threshold and
                    (args.cases is None or row['case'] in args.cases)]
    cfg = read_evaluation_config(str(ROOT / 'official/data/config.txt'))
    waits = read_scene_a_config(str(ROOT / 'official/data/config.txt'))
    print('Selected cases:', len(selected), flush=True)
    for case in selected:
        run_case(case, cfg, waits, args.timeout, args.penalties, args.traffic_feedback)
        write_ablation()
    compare_table(source)
    write_final_results()
    write_chart(source)


if __name__ == '__main__':
    main()

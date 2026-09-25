"""V2: official-feedback local refinement of the immutable V1 Problem-3 results.

The official evaluator is the only acceptance oracle. Every (case, cores)
starts with its V1 incumbent; V2 cannot silently regress Makespan or added
copy bytes. The result directory and comparison table record provenance.
"""

import argparse
import csv
import gzip
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiment import _read_json, read_evaluation_config, save
from fast_solver import GraphModel, topo
from hybrid_solver import groups_from_plan, signature
from solver_problem2 import core_finish_times, result_key, validate_plan
from solver_problem3 import (Problem3Model, official_evaluate_problem3,
                             result_metrics)
from multicore_cut_evaluate_problem_3 import read_cache_config, read_scene_b_config
from run_problem3_all import write_charts, write_tables

ROOT = Path(__file__).resolve().parent
V1 = ROOT / 'results_problem3_v1_original'
V2 = ROOT / 'results_problem3_v2_final'


def read_result(folder):
    raw = folder / 'best_result.json'
    if raw.is_file():
        return _read_json(raw)
    with gzip.open(folder / 'best_result.json.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_result(folder, result):
    with gzip.open(folder / 'best_result.json.gz', 'wt', encoding='utf-8',
                   compresslevel=5) as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',', ':'))


def legal_order(model, schedules):
    if sum(map(len, schedules)) != len(model.groups):
        return False
    pred = [set(items) for items in model.pred]
    succ = [set(items) for items in model.succ]
    for order in schedules:
        for a, b in zip(order, order[1:]):
            pred[b].add(a)
            succ[a].add(b)
    try:
        topo(pred, succ)
        return True
    except ValueError:
        return False


def feedback_candidates(model, plan, result, limit=20):
    """Target groups responsible for official DDR misses on the latest critical core.

    Try a short in-core timing shift first; for cores with a long completion
    tail, also test moving the group to a less loaded core. Global DAG checks
    reject cyclic plans before invoking the expensive official evaluator.
    """
    _, schedules = groups_from_plan(plan)
    schedules = [list(order) for order in schedules]
    core_ends = core_finish_times(result, model.cores)
    critical = max(range(model.cores), key=lambda c: core_ends[c])
    group_miss = defaultdict(int)
    group_hit = defaultdict(int)
    for lane in result.get('per_core_timeline', []):
        for event in lane.get('ops', []):
            if event.get('op') != 'COPY_IN':
                continue
            group = event.get('subgraph_id')
            if group is None:
                continue
            tensor = event.get('cache_tensor_id')
            size = model.tensor_by_id.get(tensor, (0,))[0]
            if event.get('cache_hit'):
                group_hit[group] += size
            else:
                group_miss[group] += size
    busy = schedules[critical]
    ranked = sorted(busy, key=lambda g: (group_miss[g] / model.cfg['bandwidth']
                                         + 0.15 * group_hit[g] / model.cache_bandwidth
                                         + 0.05 * model.duration[g], -g), reverse=True)
    ranked = ranked[:min(10, len(ranked))]
    proposals, seen = [], {signature(plan)}

    def add(kind, modified, group):
        if not legal_order(model, modified):
            return
        candidate = model.plan({g: core for core, order in enumerate(modified)
                                for g in order}, modified)
        token = signature(candidate)
        if token in seen:
            return
        seen.add(token)
        proposals.append((kind, candidate, group))

    for group in ranked:
        pos = busy.index(group)
        # Shift a costly read ahead of nearby work to improve pipeline overlap,
        # or delay it until the L2 insertion from another core has completed.
        for offset in (-4, -2, -1, 1, 2, 4):
            target = pos + offset
            if not 0 <= target < len(busy):
                continue
            changed = [list(order) for order in schedules]
            changed[critical].pop(pos)
            changed[critical].insert(target, group)
            add('feedback_reorder', changed, group)
        # Only test a cross-core move when the official end times show slack.
        for other in sorted(range(model.cores), key=lambda c: core_ends[c]):
            if other == critical or core_ends[other] >= 0.98 * core_ends[critical]:
                continue
            for fraction in (0.5, 1.0):
                changed = [list(order) for order in schedules]
                changed[critical].remove(group)
                dest = changed[other]
                dest.insert(int(len(dest) * fraction), group)
                add('feedback_move', changed, group)
            break
        if len(proposals) >= limit:
            break
    # Ranking uses official miss counts; it is a search ordering, never a score.
    return proposals[:limit]


def solve_one(case, cores, original_root, output_root, max_evals, seconds,
              eval_timeout, force=False):
    source = original_root / f'{case}_{cores}cores'
    destination = output_root / source.name
    if not force and (destination / 'summary.json').is_file() and \
            (destination / 'best_plan.json').is_file() and \
            (destination / 'best_result.json.gz').is_file():
        return case, cores, 'saved', None
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=True)
    graph_path = ROOT / 'official/data' / f'{case}.json'
    config_path = ROOT / 'official/data/config.txt'
    original_plan = _read_json(source / 'best_plan.json')
    original_result = read_result(source)
    original_summary = _read_json(source / 'summary.json')
    best_plan, best_result = original_plan, original_result
    history = []
    evals = 0
    # The V1 search deliberately left >10k-op graphs as a verified incumbent.
    # Building group neighborhoods there can consume gigabytes and minutes.
    # Preserve those official V1 results and make the V2 scope explicit.
    large_graph = len(original_plan['node_to_subgraph']) > 10000
    if cores > 1 and max_evals > 0 and not large_graph:
        graph = _read_json(graph_path)
        validate_plan(graph, original_plan, cores)
        gm = GraphModel(graph)
        cfg = read_evaluation_config(str(config_path))
        cache = read_cache_config(str(config_path))
        delay = read_scene_b_config(str(config_path))['cross_core_copy_delay_cycles']
        groups, _ = groups_from_plan(original_plan)
        model = Problem3Model(gm, groups, cores, cfg, delay,
                              cache['cache_capacity_bytes'],
                              cache['cache_bandwidth_bytes_per_cycle'])
        # Rebuild the target list after a successful official improvement.
        seen = {signature(original_plan)}
        while evals < max_evals and time.perf_counter() - started < seconds:
            proposals = feedback_candidates(model, best_plan, best_result)
            proposals = [(kind, plan, group) for kind, plan, group in proposals
                         if signature(plan) not in seen]
            if not proposals:
                break
            improved = False
            for kind, plan, group in proposals:
                if evals >= max_evals or time.perf_counter() - started >= seconds:
                    break
                seen.add(signature(plan))
                evals += 1
                label = f'v2_candidate_{evals:02d}'
                try:
                    result, info = official_evaluate_problem3(
                        graph_path, config_path, graph, plan, destination,
                        label, max(1, min(eval_timeout,
                                          seconds - (time.perf_counter() - started))))
                except (ValueError, RuntimeError, KeyError) as exc:
                    result, info = None, {'status': 'invalid', 'message': str(exc)[:400]}
                accepted = result is not None and result_key(result) < result_key(best_result)
                if accepted:
                    best_plan, best_result = plan, result
                    improved = True
                history.append({'candidate': label, 'operation': kind,
                                'target_group': group, 'status': info['status'],
                                'makespan': result['makespan'] if result else None,
                                'added_copy_bytes': result['data_movement_bytes']['added_copy_bytes']
                                if result else None, 'accepted': accepted,
                                'seconds': info.get('seconds')})
                (destination / f'{label}_plan.json').unlink(missing_ok=True)
                (destination / f'{label}_log.txt').unlink(missing_ok=True)
                if improved:
                    break
            if not improved:
                break
    save(destination / 'best_plan.json', best_plan)
    write_result(destination, best_result)
    save(destination / 'search_history.json', history)
    old = result_metrics(original_result)
    new = result_metrics(best_result)
    summary = {'version': 'V2_FINAL', 'case': case, 'cores': cores,
               'original_version': 'V1_ORIGINAL',
               'original_result_folder': str(source.resolve()),
               'selected_source': 'V2_FEEDBACK_SEARCH' if result_key(best_result) < result_key(original_result)
               else 'V1_ORIGINAL_UNCHANGED',
               'problem2_baseline_makespan': original_summary['problem2_baseline_makespan'],
               'problem2_baseline_added_copy_bytes': original_summary['problem2_baseline_added_copy_bytes'],
               'problem3_initial': old, 'original_v1': old, 'best': new,
               'l2_relative_speedup': original_summary['problem2_baseline_makespan'] / new['makespan'],
               'v2_additional_official_evaluations': evals,
               'search_profile': ('baseline_only_large_graph' if large_graph else
                                  'baseline_only_single_core' if cores == 1 else
                                  'feedback_local_search'),
               'official_evaluator_calls': original_summary.get('official_evaluator_calls', 0) + evals,
               'solver_runtime_seconds': time.perf_counter() - started,
               'best_plan_path': str((destination / 'best_plan.json').resolve()),
               'best_result_path': str((destination / 'best_result.json.gz').resolve())}
    save(destination / 'summary.json', summary)
    return case, cores, summary['selected_source'], old['makespan'] - new['makespan']


def compare(original_root, output_root, cases, cores):
    rows = []
    for case in cases:
        for core in cores:
            old_path = original_root / f'{case}_{core}cores' / 'summary.json'
            new_path = output_root / f'{case}_{core}cores' / 'summary.json'
            if not old_path.is_file() or not new_path.is_file():
                rows.append({'case': case, 'cores': core, 'status': 'missing'})
                continue
            old, new = _read_json(old_path), _read_json(new_path)
            a, b = old['best'], new['best']
            rows.append({'case': case, 'cores': core, 'status': 'scored',
                         'selected_source': new['selected_source'],
                         'v1_makespan': a['makespan'], 'v2_final_makespan': b['makespan'],
                         'reduction_cycles': a['makespan'] - b['makespan'],
                         'reduction_pct': f'{100 * (a["makespan"] - b["makespan"]) / a["makespan"]:.6f}',
                         'v1_added_copy_bytes': a['added_copy_bytes'],
                         'v2_final_added_copy_bytes': b['added_copy_bytes'],
                         'v1_cache_hit_rate': f'{a["cache_hit_rate"]:.9f}',
                         'v2_final_cache_hit_rate': f'{b["cache_hit_rate"]:.9f}',
                         'additional_official_evaluations': new['v2_additional_official_evaluations']})
    fields = ['case', 'cores', 'status', 'selected_source', 'v1_makespan',
              'v2_final_makespan', 'reduction_cycles', 'reduction_pct',
              'v1_added_copy_bytes', 'v2_final_added_copy_bytes',
              'v1_cache_hit_rate', 'v2_final_cache_hit_rate',
              'additional_official_evaluations']
    with (output_root / 'V1_vs_V2_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(({field: row.get(field, '') for field in fields} for row in rows))
    averages = []
    for core in cores:
        sample = [row for row in rows if row['cores'] == core and row['status'] == 'scored']
        if not sample:
            continue
        averages.append({
            'cores': core, 'scored_cases': len(sample),
            'improved_cases': sum(row['selected_source'] == 'V2_FEEDBACK_SEARCH' for row in sample),
            'mean_v1_makespan': sum(row['v1_makespan'] for row in sample) / len(sample),
            'mean_v2_final_makespan': sum(row['v2_final_makespan'] for row in sample) / len(sample),
            'mean_v1_to_v2_speedup': sum(row['v1_makespan'] / row['v2_final_makespan']
                                         for row in sample) / len(sample),
            'mean_reduction_pct': sum(float(row['reduction_pct']) for row in sample) / len(sample),
            'total_additional_official_evaluations': sum(
                row['additional_official_evaluations'] for row in sample),
        })
    with (output_root / 'V1_vs_V2_average_by_core.csv').open(
            'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(averages[0]) if averages else
                                ['cores', 'scored_cases'])
        writer.writeheader()
        writer.writerows(averages)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, default=V1)
    parser.add_argument('--output-root', type=Path, default=V2)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--cores', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-evals', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--eval-timeout', type=float, default=45)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.max_evals < 0 or args.seconds <= 0 or args.eval_timeout <= 0:
        parser.error('invalid search budgets')
    cases = sorted(p.stem for p in (ROOT / 'official/data').glob('case_*.json')
                   if re.fullmatch(r'case_\d{3}', p.stem))
    if args.cases:
        unknown = set(args.cases) - set(cases)
        if unknown:
            parser.error(f'unknown cases: {sorted(unknown)}')
        cases = sorted(set(args.cases))
    cores = sorted(set(args.cores))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        targets = [(case, core) for case in cases for core in cores]
        print(f'V2 refining {len(targets)} configurations', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(solve_one, case, core, args.original_root,
                                args.output_root, args.max_evals, args.seconds,
                                args.eval_timeout, args.force) for case, core in targets]
            for index, job in enumerate(as_completed(jobs), 1):
                case, core, status, cycles = job.result()
                print(f'[{index}/{len(jobs)}] {case} {core}core: {status}; '
                      f'cycles_saved={cycles}', flush=True)
    rows = compare(args.original_root, args.output_root, cases, cores)
    _, relative, makespan, speedup = write_tables(args.output_root, cases, cores)
    write_charts(args.output_root, makespan, speedup, relative)
    improved = [row for row in rows if row.get('selected_source') == 'V2_FEEDBACK_SEARCH']
    print(f'V2: {len(rows)} rows, {len(improved)} improved, '
          f'{sum(int(row["reduction_cycles"]) for row in improved)} cycles saved', flush=True)


if __name__ == '__main__':
    main()

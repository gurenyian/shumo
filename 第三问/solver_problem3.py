"""Scene B with shared read-only L2: bounded, evaluator-guided optimization.

All L2/FIFO estimates here rank candidates only. The unmodified official
problem-3 evaluator alone chooses the final (makespan, added_copy_bytes).
"""

import argparse
import csv
import heapq
import json
import math
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from adaptive_clustering import adaptive_partition
from experiment import _read_json, read_evaluation_config, save
from fast_solver import GraphModel, make_plan, topo
from hybrid_solver import groups_from_plan, signature
from solver_problem2 import (Problem2Model, affinity_core_assignment,
                             cache_aware_reorder, candidate_neighbors,
                             core_finish_times, proxy_score, result_key,
                             validate_plan)
from multicore_cut_evaluate_problem_3 import read_cache_config, read_scene_b_config

ROOT = Path(__file__).resolve().parent


class PredictiveL2FIFO:
    """Sequential approximation of the official issue/completion-time FIFO.

    The evaluator may issue concurrent misses before any insertion. Thus this
    proxy is intentionally not treated as a source of actual hit statistics.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.entries = OrderedDict()
        self.used_bytes = 0

    def clone(self):
        other = PredictiveL2FIFO(self.capacity)
        other.entries = self.entries.copy()
        other.used_bytes = self.used_bytes
        return other

    def access(self, tensor_id, size):
        size = int(size)
        if tensor_id in self.entries:
            return True, []  # Hits do not refresh FIFO position.
        if size <= 0 or size > self.capacity:
            return False, []
        evicted = []
        while self.entries and self.used_bytes + size > self.capacity:
            key, old_size = self.entries.popitem(last=False)
            self.used_bytes -= old_size
            evicted.append((key, old_size))
        self.entries[tensor_id] = size
        self.used_bytes += size
        return False, evicted


class Problem3Model(Problem2Model):
    def __init__(self, gm, groups, cores, cfg, delay, cache_capacity,
                 cache_bandwidth):
        super().__init__(gm, groups, cores, cfg, delay)
        self.cache_capacity = int(cache_capacity)
        self.cache_bandwidth = float(cache_bandwidth)
        self.tensor_by_id = {}
        self.reads_by_group = [set() for _ in groups]
        self.hot = []
        rank_scale = max(self.rank.values(), default=1)
        for tid, size, pos, producers, consumers in self.tensors:
            self.tensor_by_id[tid] = (size, pos, producers, consumers)
            for g in consumers:
                self.reads_by_group[g].add(tid)
            if len(consumers) < 2:
                continue
            criticality = 1 + max((self.rank[g] for g in consumers), default=0) / max(1, rank_scale)
            gain = max(0, len(consumers) - 1) * size * (
                1 / cfg['bandwidth'] - 1 / self.cache_bandwidth)
            self.hot.append((gain * criticality, tid))
        self.hot.sort(reverse=True)

    def _needs_read(self, tid, core, assignment):
        _, _, producers, _ = self.tensor_by_id[tid]
        return not producers or any(assignment[g] != core for g in producers)

    def access_keys(self, group, assignment, seen):
        core = assignment[group]
        result = []
        for tid in sorted(self.reads_by_group[group]):
            if (tid, core) in seen or not self._needs_read(tid, core, assignment):
                continue
            result.append(tid)
        return result

    def simulate_fifo(self, plan):
        _, schedules = groups_from_plan(plan)
        assignment = {g: core for core, order in enumerate(schedules) for g in order}
        pred = [set(p) for p in self.pred]
        succ = [set(s) for s in self.succ]
        for order in schedules:
            for a, b in zip(order, order[1:]):
                pred[b].add(a)
                succ[a].add(b)
        order = topo(pred, succ)
        remaining = Counter()
        for tid, (_, _, _, consumers) in self.tensor_by_id.items():
            for core in {assignment[g] for g in consumers}:
                if self._needs_read(tid, core, assignment):
                    remaining[tid] += 1
        fifo = PredictiveL2FIFO(self.cache_capacity)
        seen = set()
        hit_bytes = miss_bytes = eviction_damage = insert_bytes = 0
        reuse_distance = {}
        last_insert_position = {}
        evictions = 0
        for group in order:
            core = assignment[group]
            for tid in self.access_keys(group, assignment, seen):
                size = self.tensor_by_id[tid][0]
                seen.add((tid, core))
                remaining[tid] -= 1
                if tid in last_insert_position:
                    reuse_distance[tid] = insert_bytes - last_insert_position[tid]
                hit, displaced = fifo.access(tid, size)
                if hit:
                    hit_bytes += size
                else:
                    miss_bytes += size
                    if 0 < size <= self.cache_capacity:
                        insert_bytes += size
                        last_insert_position[tid] = insert_bytes
                for old_tid, old_size in displaced:
                    evictions += 1
                    eviction_damage += max(0, remaining[old_tid]) * old_size * (
                        1 / self.cfg['bandwidth'] - 1 / self.cache_bandwidth)
        return {'predicted_ddr_bytes': miss_bytes,
                'predicted_l2_hit_bytes': hit_bytes,
                'predicted_evictions': evictions,
                'eviction_damage_cycles': eviction_damage,
                'reuse_distance_bytes': reuse_distance}


def proxy_score_problem3(model, plan):
    base, private = proxy_score(model, plan)
    l2 = model.simulate_fifo(plan)
    score = (base + 0.15 * l2['predicted_ddr_bytes'] / model.cfg['bandwidth']
             + 0.04 * l2['predicted_l2_hit_bytes'] / model.cache_bandwidth
             + 0.18 * l2['eviction_damage_cycles'])
    return score, {**private, **l2}


def l2_aware_reorder(model, assignment, variant=0):
    """Global topological ready list; prefer useful FIFO hits near equal rank."""
    degree = [len(p) for p in model.pred]
    ready = [(-model.rank[g], g) for g, d in enumerate(degree) if not d]
    heapq.heapify(ready)
    schedules = [[] for _ in range(model.cores)]
    fifo = PredictiveL2FIFO(model.cache_capacity)
    seen = set()
    remaining = Counter()
    for tid, (_, _, _, consumers) in model.tensor_by_id.items():
        for core in {assignment[g] for g in consumers}:
            if model._needs_read(tid, core, assignment):
                remaining[tid] += 1
    while ready:
        shortlist = [heapq.heappop(ready)[1] for _ in range(min(12, len(ready)))]

        def priority(group):
            preview = fifo.clone()
            hit_gain = damage = 0.0
            future = remaining.copy()
            for tid in model.access_keys(group, assignment, seen):
                size = model.tensor_by_id[tid][0]
                future[tid] -= 1
                hit, evicted = preview.access(tid, size)
                if hit:
                    hit_gain += size * (1 / model.cfg['bandwidth'] - 1 / model.cache_bandwidth)
                for old_tid, old_size in evicted:
                    damage += max(0, future[old_tid]) * old_size * (
                        1 / model.cfg['bandwidth'] - 1 / model.cache_bandwidth)
            pressure = sum(max(0, model.cache_footprint[group][name] / max(1, cap) - 0.7)
                           for name, cap in model.cfg['capacity'].items())
            return model.rank[group] + (0.3 + 0.25 * variant) * hit_gain - 0.2 * damage - 0.03 * model.ideal * pressure

        group = max(shortlist, key=lambda g: (priority(g), -g))
        for other in shortlist:
            if other != group:
                heapq.heappush(ready, (-model.rank[other], other))
        core = assignment[group]
        schedules[core].append(group)
        for tid in model.access_keys(group, assignment, seen):
            seen.add((tid, core))
            remaining[tid] -= 1
            fifo.access(tid, model.tensor_by_id[tid][0])
        for successor in model.succ[group]:
            degree[successor] -= 1
            if degree[successor] == 0:
                heapq.heappush(ready, (-model.rank[successor], successor))
    return schedules


def l2_aware_core_assignment(model, seed_plan, mode='scatter', aggressive=False):
    assignment = model.assignment(seed_plan)
    limits = 6 if aggressive else 2
    for _, tid in model.hot[:limits]:
        size, _, _, consumers = model.tensor_by_id[tid]
        if len(consumers) < 2:
            continue
        groups = sorted(consumers, key=lambda g: (-model.rank[g], g))
        loads = [sum(max(model.work[g].values(), default=0)
                     for g, core in assignment.items() if core == c)
                 for c in range(model.cores)]
        if mode == 'scatter' and size <= model.cache_capacity:
            source = max(groups, key=lambda g: loads[assignment[g]])
            target = min((c for c in range(model.cores) if c != assignment[source]),
                         key=lambda c: (loads[c], c), default=None)
            if target is not None:
                assignment[source] = target
        elif mode == 'cluster' and len({assignment[g] for g in groups}) > 1:
            target = min({assignment[g] for g in groups}, key=lambda c: (loads[c], c))
            source = next((g for g in groups if assignment[g] != target), None)
            if source is not None:
                assignment[source] = target
    schedules = l2_aware_reorder(model, assignment, int(aggressive))
    return model.plan(assignment, schedules)


def official_evaluate_problem3(graph_path, config_path, graph, plan, folder,
                               label, timeout):
    validate_plan(graph, plan, len(plan['core_schedules']))
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / f'{label}_plan.json'
    result_path = folder / f'{label}_result.json'
    log_path = folder / f'{label}_log.txt'
    trace_path = folder / f'{label}_trace.json'
    save(plan_path, plan)
    cmd = [sys.executable, str(ROOT / 'official/code/multicore_cut_evaluate_problem_3.py'),
           str(graph_path), str(plan_path), '--config', str(config_path),
           '-o', str(result_path), '--log-output', str(log_path),
           '--trace-output', str(trace_path)]
    started = time.perf_counter()
    try:
        run = subprocess.run(cmd, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        result_path.unlink(missing_ok=True)
        trace_path.unlink(missing_ok=True)
        return None, {'status': 'timeout', 'seconds': time.perf_counter() - started}
    if run.returncode:
        result_path.unlink(missing_ok=True)
        trace_path.unlink(missing_ok=True)
        return None, {'status': 'error', 'seconds': time.perf_counter() - started,
                      'message': run.stderr.strip()[-1000:]}
    result = _read_json(result_path)
    result_path.unlink(missing_ok=True)
    trace_path.unlink(missing_ok=True)
    return result, {'status': 'valid', 'seconds': time.perf_counter() - started}


def no_l2_onecore(graph_path, config_path, graph, plan, folder, timeout):
    out = folder / 'no_l2_result.json'
    plan_file = folder / 'no_l2_plan.json'
    trace_file = folder / 'no_l2_trace.json'
    log_file = folder / 'no_l2_log.txt'
    save(plan_file, plan)
    cmd = [sys.executable, str(ROOT / 'official/code/multicore_cut_evaluate_problem_2.py'),
           str(graph_path), str(plan_file), '--config', str(config_path), '-o', str(out),
           '--trace-output', str(trace_file), '--log-output', str(log_file)]
    run = subprocess.run(cmd, capture_output=True, text=True,
                         encoding='utf-8', errors='replace', timeout=timeout)
    if run.returncode:
        raise RuntimeError(run.stderr.strip()[-1000:])
    result = _read_json(out)
    out.unlink(missing_ok=True)
    trace_file.unlink(missing_ok=True)
    return {'makespan': int(result['makespan']),
            'added_copy_bytes': int(result['data_movement_bytes']['added_copy_bytes'])}


def load_no_l2_table():
    path = ROOT / 'results_problem2' / 'batch_results.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {(row['case'], int(row['cores'])): row for row in csv.DictReader(stream)
                if row['makespan_cycles']}


def baseline_plan(graph_path, gm, cores):
    if cores == 1:
        return make_plan([list(gm.order)], [[0]]), None
    path = ROOT / 'results_problem2' / f'{graph_path.stem}_{cores}cores' / 'best_plan.json'
    if not path.is_file():
        raise FileNotFoundError(f'Problem-2 baseline plan missing: {path}')
    return _read_json(path), path


def result_metrics(result):
    movement = result['data_movement_bytes']
    cache = result['cache_stats']
    return {'makespan': int(result['makespan']),
            'added_copy_bytes': int(movement['added_copy_bytes']),
            'partition_added_copy_bytes': int(movement['partition_added_copy_bytes']),
            'spill_added_copy_bytes': int(movement['spill_added_copy_bytes']),
            'cache_hit_rate': float(cache['hit_rate']),
            'cache_hit_bytes': int(cache['hit_bytes']),
            'cache_miss_bytes': int(cache['miss_bytes']),
            'cache_hits': int(cache['copy_in_hits']),
            'cache_misses': int(cache['copy_in_misses']),
            'task_dependencies': len(result.get('task_dependencies', [])),
            'memory_peak_by_core': result.get('memory_peak_by_core', {})}


def generate_neighbors(model, plan, result, max_neighbors=100):
    assignment = model.assignment(plan)
    generated = []
    for item in candidate_neighbors(model, plan, result, max_neighbors=70):
        other = item['model']
        new_model = (model if other.groups is model.groups else
                     Problem3Model(model.gm, other.groups, model.cores, model.cfg,
                                   model.delay, model.cache_capacity,
                                   model.cache_bandwidth))
        try:
            proxy, detail = proxy_score_problem3(new_model, item['plan'])
            generated.append({'kind': item['kind'], 'plan': item['plan'],
                              'model': new_model, 'proxy': proxy, 'detail': detail,
                              'target_tensor': None})
        except (KeyError, ValueError):
            pass
    for variant in (0, 1):
        try:
            candidate = model.plan(assignment, l2_aware_reorder(model, assignment, variant))
            proxy, detail = proxy_score_problem3(model, candidate)
            generated.append({'kind': 'reuse_window_reorder', 'plan': candidate,
                              'model': model, 'proxy': proxy, 'detail': detail,
                              'target_tensor': None})
        except (KeyError, ValueError):
            pass
    for mode in ('scatter', 'cluster'):
        for aggressive in (False, True):
            try:
                candidate = l2_aware_core_assignment(model, plan, mode, aggressive)
                proxy, detail = proxy_score_problem3(model, candidate)
                generated.append({'kind': f'{mode}_shared_tensor', 'plan': candidate,
                                  'model': model, 'proxy': proxy, 'detail': detail,
                                  'target_tensor': model.hot[0][1] if model.hot else None})
            except (KeyError, ValueError):
                pass
    generated.sort(key=lambda x: x['proxy'])
    unique, seen = [], set()
    for item in generated:
        token = signature(item['plan'])
        if token not in seen:
            unique.append(item)
            seen.add(token)
        if len(unique) >= max_neighbors:
            break
    return unique


def select_neighbors(neighbors, count=3):
    selected = neighbors[:count]
    alternatives = [x for x in neighbors[count:] if x['kind'] in
                    {'scatter_shared_tensor', 'cluster_shared_tensor',
                     'reuse_window_reorder', 'recluster'}]
    if alternatives:
        selected.append(min(alternatives, key=lambda x: x['proxy']))
    return selected


def solve(graph_path, cores, config_path, output_dir, baseline_path=None,
          max_evals=16, seconds=300, eval_timeout=120, patience=2):
    started = time.perf_counter()
    deadline = started + seconds
    graph = _read_json(graph_path)
    gm = GraphModel(graph)
    # On the largest official graphs, repeated Python-side neighbor construction
    # can exceed the per-case wall-clock budget even after the official baseline
    # has scored. Keep that validated incumbent and finish the configuration.
    large_graph = len(gm.ops) > 10000
    cfg = read_evaluation_config(str(config_path))
    delay = read_scene_b_config(str(config_path))['cross_core_copy_delay_cycles']
    cache_cfg = read_cache_config(str(config_path))
    cache_capacity = cache_cfg['cache_capacity_bytes']
    cache_bandwidth = cache_cfg['cache_bandwidth_bytes_per_cycle']
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline_path:
        baseline, source = _read_json(baseline_path), baseline_path
    else:
        baseline, source = baseline_plan(graph_path, gm, cores)
    validate_plan(graph, baseline, cores)
    no_l2_rows = load_no_l2_table()
    if cores == 1:
        no_l2 = no_l2_onecore(graph_path, config_path, graph, baseline,
                              output_dir, eval_timeout)
    else:
        row = no_l2_rows[(graph_path.stem, cores)]
        no_l2 = {'makespan': int(row['makespan_cycles']),
                 'added_copy_bytes': int(row['added_copy_bytes'])}
    lower_bound = max(max(gm.critical.values(), default=0),
                      max(gm.pipes.values(), default=0) / cores)
    groups, _ = groups_from_plan(baseline)
    model = Problem3Model(gm, groups, cores, cfg, delay,
                          cache_capacity, cache_bandwidth)
    history, ablation = [], []
    seen = set()
    accepted = Counter()
    calls = 0
    best_plan = best_result = best_model = None
    initial = None

    def score(label, kind, candidate, candidate_model, iteration=0,
              proxy=None, detail=None, target_tensor=None):
        nonlocal calls, best_plan, best_result, best_model, initial
        token = signature(candidate)
        if token in seen or calls >= max_evals or time.perf_counter() >= deadline:
            return False
        seen.add(token)
        calls += 1
        try:
            result, info = official_evaluate_problem3(
                graph_path, config_path, graph, candidate, output_dir, label,
                min(eval_timeout, max(1, deadline - time.perf_counter())))
        except (ValueError, RuntimeError, KeyError) as exc:
            result, info = None, {'status': 'invalid', 'message': str(exc)[:500]}
        metrics = result_metrics(result) if result is not None else {}
        record = {'iteration': iteration, 'candidate': label, 'operation': kind,
                  'target_tensor': target_tensor, 'proxy': proxy,
                  'predicted_ddr_bytes': (detail or {}).get('predicted_ddr_bytes'),
                  'predicted_l2_hit_bytes': (detail or {}).get('predicted_l2_hit_bytes'),
                  'predicted_evictions': (detail or {}).get('predicted_evictions'),
                  'official_makespan': metrics.get('makespan'),
                  'official_cache_hit_rate': metrics.get('cache_hit_rate'),
                  'added_copy_bytes': metrics.get('added_copy_bytes'),
                  'status': info['status'], 'seconds': info.get('seconds'),
                  'message': info.get('message'), 'accepted': False}
        if result is not None:
            if initial is None:
                initial = metrics
            if best_result is None or result_key(result) < result_key(best_result):
                best_plan, best_result, best_model = candidate, result, candidate_model
                record['accepted'] = True
                accepted[kind] += 1
                save(output_dir / 'best_plan.json', best_plan)
                save(output_dir / 'best_result.json', best_result)
        history.append(record)
        if iteration == 0:
            ablation.append(record)
        save(output_dir / 'search_history.json', history)
        print(f'[{calls}/{max_evals}] {label}: {info["status"]}, '
              f'makespan={metrics.get("makespan")}, '
              f'hit_rate={metrics.get("cache_hit_rate")}, '
              f'best={best_result["makespan"] if best_result else None}', flush=True)
        return record['accepted']

    baseline_proxy, baseline_detail = proxy_score_problem3(model, baseline)
    score('baseline_a_problem2_plan', 'problem2_plan', baseline, model,
          proxy=baseline_proxy, detail=baseline_detail)
    if best_result is None:
        raise RuntimeError('Problem-3 baseline did not score; inspect search_history.json')
    if (cores > 1 and not large_graph and calls < max_evals
            and time.perf_counter() < deadline):
        direct_assignment, direct_schedule = affinity_core_assignment(
            model, shared=False, cache=False)
        candidate = model.plan(direct_assignment, direct_schedule)
        proxy, detail = proxy_score_problem3(model, candidate)
        score('direct_affinity', 'direct_affinity', candidate, model,
              proxy=proxy, detail=detail)
        assignment, _ = affinity_core_assignment(model, shared=True, cache=True)
        candidate = model.plan(assignment, cache_aware_reorder(model, assignment))
        proxy, detail = proxy_score_problem3(model, candidate)
        score('private_cache_affinity', 'private_cache_affinity', candidate, model,
              proxy=proxy, detail=detail)
        for aggressive in (False, True):
            candidate = l2_aware_core_assignment(model, baseline, 'scatter', aggressive)
            proxy, detail = proxy_score_problem3(model, candidate)
            score(f'l2_scatter_{int(aggressive)}', 'l2_scatter', candidate, model,
                  proxy=proxy, detail=detail,
                  target_tensor=model.hot[0][1] if model.hot else None)
        candidate = model.plan(model.assignment(baseline),
                               l2_aware_reorder(model, model.assignment(baseline)))
        proxy, detail = proxy_score_problem3(model, candidate)
        score('fifo_reuse_reorder', 'reuse_window_reorder', candidate, model,
              proxy=proxy, detail=detail)
        # Alternative partition reuses the proven safe contraction machinery.
        if calls < max_evals and time.perf_counter() < deadline:
            try:
                partition, _ = adaptive_partition(
                    gm, cores, cfg,
                    {'task_cross_core_wait_cycles': delay,
                     'task_same_core_wait_cycles': 0}, target_scale=0.75)
                new_groups, _ = groups_from_plan(partition)
                new_model = Problem3Model(gm, new_groups, cores, cfg, delay,
                                          cache_capacity, cache_bandwidth)
                assign, _ = affinity_core_assignment(new_model, shared=False, cache=True)
                candidate = new_model.plan(assign, l2_aware_reorder(new_model, assign))
                proxy, detail = proxy_score_problem3(new_model, candidate)
                score('adaptive_partition_l2', 'adaptive_partition', candidate,
                      new_model, proxy=proxy, detail=detail)
            except (ValueError, KeyError):
                pass
    no_improvement = 0
    iteration = 0
    while (cores > 1 and not large_graph and best_result is not None and no_improvement < patience
           and calls < max_evals and time.perf_counter() < deadline):
        iteration += 1
        neighbors = generate_neighbors(best_model, best_plan, best_result)
        choices = select_neighbors([x for x in neighbors if signature(x['plan']) not in seen])
        if not choices:
            break
        improved = False
        for index, item in enumerate(choices):
            improved |= score(f'search_{iteration}_{index}_{item["kind"]}',
                              item['kind'], item['plan'], item['model'],
                              iteration=iteration, proxy=item['proxy'],
                              detail=item['detail'],
                              target_tensor=item['target_tensor'])
            if calls >= max_evals or time.perf_counter() >= deadline:
                break
        no_improvement = 0 if improved else no_improvement + 1
    best = result_metrics(best_result)
    summary = {'case': graph_path.stem, 'cores': cores,
               'problem2_baseline_plan': str(source) if source else None,
               'problem2_baseline_makespan': no_l2['makespan'],
               'problem2_baseline_added_copy_bytes': no_l2['added_copy_bytes'],
               'problem3_initial': initial, 'best': best,
               'l2_relative_speedup': no_l2['makespan'] / best['makespan'],
               'official_evaluator_calls': calls,
               'search_profile': ('baseline_only_large_graph' if large_graph else
                                  'baseline_only_budget' if max_evals == 1 else
                                  'bounded_local_search'),
               'solver_runtime_seconds': time.perf_counter() - started,
               'accepted_operations': dict(accepted),
               'optimistic_lower_bound': lower_bound,
               'gap_to_optimistic_lower_bound': (best['makespan'] - lower_bound) / max(1, lower_bound),
               'ablation': ablation,
               'best_plan_path': str(output_dir / 'best_plan.json'),
               'best_result_path': str(output_dir / 'best_result.json')}
    save(output_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('-n', '--cores', type=int, required=True, choices=range(1, 6))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--problem2-baseline', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-evals', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--eval-timeout', type=float, default=120)
    parser.add_argument('--patience', type=int, default=2)
    args = parser.parse_args(argv)
    if args.max_evals < 1 or args.seconds <= 0 or args.eval_timeout <= 0 or args.patience < 1:
        parser.error('budgets must be positive')
    out = args.output_dir or ROOT / 'results_problem3' / f'{args.graph.stem}_{args.cores}cores'
    summary = solve(args.graph.resolve(), args.cores, args.config.resolve(),
                    out.resolve(), args.problem2_baseline,
                    args.max_evals, args.seconds, args.eval_timeout, args.patience)
    print(json.dumps({'makespan': summary['best']['makespan'],
                      'cache_hit_rate': summary['best']['cache_hit_rate'],
                      'l2_relative_speedup': summary['l2_relative_speedup'],
                      'best_plan_path': summary['best_plan_path']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

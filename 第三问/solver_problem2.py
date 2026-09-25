"""Scene B: cache-aware core affinity and bounded official-evaluator search.

The proxy constructs and ranks candidates. Only the unmodified official
problem-2 evaluator decides whether a plan replaces the incumbent.
"""

import argparse
import csv
import heapq
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_clustering import adaptive_partition
from experiment import _read_json, read_evaluation_config, save
from fast_solver import GraphModel, make_plan, topo
from hybrid_solver import groups_from_plan, signature
from multicore_cut_evaluate_problem_2 import read_scene_b_config
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order

ROOT = Path(__file__).resolve().parent
PIPES = ('PIPE_MTE2', 'PIPE_MTE3', 'PIPE_M', 'PIPE_V')


def result_key(result):
    traffic = result['data_movement_bytes']
    return int(result['makespan']), int(traffic['added_copy_bytes'])


def validate_plan(graph, plan, cores):
    if len(plan['core_schedules']) != cores:
        raise ValueError('wrong number of core schedules')
    view = derive_multicore_plan(graph, plan)
    validate_task_order(view)
    return view


class Problem2Model:
    """Features for one fixed acyclic partition of the original DAG."""

    def __init__(self, gm, groups, cores, cfg, delay, shared_weight=0.5):
        self.gm, self.groups, self.cores = gm, groups, cores
        self.cfg, self.delay = cfg, delay
        self.mapping, self.pred, self.succ, self.work, self.traffic, self.duration = (
            gm.describe(groups, cfg['bandwidth']))
        self.order = topo(self.pred, self.succ)
        self.total_work = Counter()
        for work in self.work:
            self.total_work.update(work)
        self.ideal = max(self.total_work.values(), default=1) / cores
        self.flow = Counter()
        self.shared = Counter()
        self.tensors = []
        self.group_tensors = [set() for _ in groups]
        self.affinity = [defaultdict(float) for _ in groups]
        self.direct_affinity = [defaultdict(int) for _ in groups]
        self.output_bytes = [0] * len(groups)
        self.input_bytes = [0] * len(groups)
        for tensor in gm.graph['tensors']:
            ident, size = tensor['id'], int(tensor['size'])
            producers = {self.mapping[v] for v in gm.producers[ident] if v in self.mapping}
            consumers = {self.mapping[v] for v in gm.consumers[ident] if v in self.mapping}
            if not producers and not consumers:
                continue
            pos = tensor.get('pos', 'UB')
            if pos == 'DDR':
                pos = 'UB'
            self.tensors.append((ident, size, pos, producers, consumers))
            for group in producers | consumers:
                self.group_tensors[group].add(len(self.tensors) - 1)
            for a in producers:
                self.output_bytes[a] += size
                for b in consumers - {a}:
                    self.flow[a, b] += size
            if not producers:
                for b in consumers:
                    self.input_bytes[b] += size
                if 1 < len(consumers) <= 16:
                    ordered = sorted(consumers)
                    for index, a in enumerate(ordered):
                        for b in ordered[index + 1:]:
                            self.shared[a, b] += size
        for (a, b), size in self.flow.items():
            self.affinity[a][b] += size
            self.affinity[b][a] += size
            self.direct_affinity[a][b] += size
            self.direct_affinity[b][a] += size
        for (a, b), size in self.shared.items():
            self.affinity[a][b] += shared_weight * size
            self.affinity[b][a] += shared_weight * size
        # An internal subgraph boundary has no mandatory DDR trip in Scene B.
        self.duration = [max(work.values(), default=0) + self.input_bytes[i] /
                         cfg['bandwidth'] for i, work in enumerate(self.work)]
        self.rank = {}
        for group in reversed(self.order):
            self.rank[group] = self.duration[group] + max(
                (self.delay + self.flow.get((group, other), 0) / cfg['bandwidth']
                 + self.rank[other] for other in self.succ[group]), default=0)
        self.cache_footprint = []
        for group in range(len(groups)):
            footprint = Counter()
            for index in self.group_tensors[group]:
                _, size, pos, _, _ = self.tensors[index]
                if pos in cfg['capacity']:
                    footprint[pos] += size
            self.cache_footprint.append(footprint)

    def assignment(self, plan):
        _, schedules = groups_from_plan(plan)
        return {
            group: core for core, order in enumerate(schedules) for group in order}

    def plan(self, assignment, schedules=None):
        if schedules is None:
            schedules = [[] for _ in range(self.cores)]
            for group in self.order:
                schedules[assignment[group]].append(group)
        return make_plan(self.groups, schedules)


def build_base_groups(gm, cores, cfg, delay, scale):
    # Scene-A routine supplies only safe DAG contractions; its task wait proxy
    # is replaced by Scene-B's 500-cycle cross-core release and zero same-core wait.
    waits = {'task_cross_core_wait_cycles': delay,
             'task_same_core_wait_cycles': 0}
    plan, detail = adaptive_partition(gm, cores, cfg, waits,
                                      target_scale=scale)
    groups, _ = groups_from_plan(plan)
    return groups, plan, detail


def estimate_core_cache_pressure(model, schedules):
    """Soft lifetime proxy: birth to last local consumer in schedule positions."""
    capacities = model.cfg['capacity']
    peaks, residence = [], 0
    for order in schedules:
        position = {group: index for index, group in enumerate(order)}
        events = {name: [0] * (len(order) + 1) for name in capacities}
        for _, size, pos, producers, consumers in model.tensors:
            if pos not in capacities:
                continue
            local_producers = producers & position.keys()
            local_consumers = consumers & position.keys()
            touched = local_producers | local_consumers
            if not touched:
                continue
            birth = min(position[group] for group in local_producers) if local_producers else min(
                position[group] for group in local_consumers)
            last_use = max((position[group] for group in local_consumers), default=birth)
            last_use = max(birth, last_use)
            events[pos][birth] += size
            events[pos][last_use + 1] -= size
            residence += size * (last_use - birth)
        core_peak = {}
        for pos, differences in events.items():
            live, peak = 0, 0
            for change in differences:
                live += change
                peak = max(peak, live)
            core_peak[pos] = peak
        peaks.append(core_peak)
    return peaks, residence


def cache_cost(model, peaks):
    cost = 0.0
    for core_peak in peaks:
        for name, capacity in model.cfg['capacity'].items():
            rho = core_peak.get(name, 0) / max(1, capacity)
            if rho > 1:
                cost += 10 + 40 * (rho - 1) ** 2
            elif rho > 0.7:
                cost += (rho - 0.7) ** 2
    return cost


def affinity_core_assignment(model, shared=True, cache=True):
    """Critical-rank ready list with communication, load and cache terms."""
    degree = [len(pred) for pred in model.pred]
    ready = [(-model.rank[i], i) for i, count in enumerate(degree) if not count]
    heapq.heapify(ready)
    assigned, finish = {}, {}
    schedules = [[] for _ in range(model.cores)]
    loads = [Counter() for _ in range(model.cores)]
    free = [0.0] * model.cores
    recent = [Counter() for _ in range(model.cores)]
    while ready:
        _, group = heapq.heappop(ready)
        choices = []
        for core in range(model.cores):
            affinity_bytes = sum(value for other, value in model.affinity[group].items()
                                 if assigned.get(other) == core)
            if not shared:
                affinity_bytes = sum(size for other, size in
                                     model.direct_affinity[group].items()
                                     if assigned.get(other) == core)
            release = max((finish[p] + (model.delay if assigned[p] != core else 0)
                           + (model.flow.get((p, group), 0) / model.cfg['bandwidth']
                              if assigned[p] != core else 0)
                           for p in model.pred[group]), default=0)
            start = max(free[core], release)
            projected = loads[core] + model.work[group]
            overflow = max(0, max(projected.values(), default=0) - 1.15 * model.ideal)
            pressure = 0.0
            if cache:
                for name, capacity in model.cfg['capacity'].items():
                    rho = (recent[core][name] + model.cache_footprint[group][name]) / max(1, capacity)
                    pressure += max(0, rho - 0.7) ** 2 * model.ideal * 0.05
            score = (start + model.duration[group] + 0.7 * overflow + pressure
                     - 0.65 * affinity_bytes / model.cfg['bandwidth'])
            choices.append((score, start + model.duration[group], core))
        _, end, core = min(choices)
        assigned[group], finish[group] = core, end
        free[core] = end
        loads[core].update(model.work[group])
        schedules[core].append(group)
        for name in model.cfg['capacity']:
            recent[core][name] = min(model.cfg['capacity'][name],
                                     0.5 * recent[core][name] +
                                     model.cache_footprint[group][name])
        for next_group in model.succ[group]:
            degree[next_group] -= 1
            if degree[next_group] == 0:
                heapq.heappush(ready, (-model.rank[next_group], next_group))
    return assigned, schedules


def cache_aware_reorder(model, assignment, variant=0):
    """One global topological order, projected to cores, cannot make a wait cycle."""
    degree = [len(pred) for pred in model.pred]
    ready = [(-model.rank[i], i) for i, count in enumerate(degree) if not count]
    heapq.heapify(ready)
    schedules = [[] for _ in range(model.cores)]
    live = [set() for _ in range(model.cores)]
    live_bytes = [Counter() for _ in range(model.cores)]
    remaining = Counter()
    for index, (_, _, _, _, consumers) in enumerate(model.tensors):
        for core in range(model.cores):
            remaining[index, core] = sum(assignment[group] == core for group in consumers)
    while ready:
        # Limit dynamic scoring to the highest-rank ready vertices on big DAGs.
        shortlist = [heapq.heappop(ready)[1] for _ in range(min(12, len(ready)))]
        def priority(group):
            core = assignment[group]
            released = 0
            for index in model.group_tensors[group] & live[core]:
                if group in model.tensors[index][4] and remaining[index, core] == 1:
                    released += model.tensors[index][1]
            pressure = sum(max(0, live_bytes[core][name] +
                               model.cache_footprint[group][name] - cap) / max(1, cap)
                           for name, cap in model.cfg['capacity'].items())
            return (model.rank[group] + (0.3 + 0.2 * variant) * released /
                    model.cfg['bandwidth'] - pressure * model.ideal * 0.05)
        group = max(shortlist, key=lambda g: (priority(g), -g))
        for other in shortlist:
            if other != group:
                heapq.heappush(ready, (-model.rank[other], other))
        core = assignment[group]
        schedules[core].append(group)
        for index in model.group_tensors[group]:
            _, size, pos, producers, consumers = model.tensors[index]
            if pos not in model.cfg['capacity']:
                continue
            if (group in producers or (not producers and group in consumers)) and remaining[index, core] > 0:
                if index not in live[core]:
                    live[core].add(index)
                    live_bytes[core][pos] += size
            if group in consumers:
                remaining[index, core] -= 1
                if remaining[index, core] == 0 and index in live[core]:
                    live[core].remove(index)
                    live_bytes[core][pos] -= size
        for successor in model.succ[group]:
            degree[successor] -= 1
            if degree[successor] == 0:
                heapq.heappush(ready, (-model.rank[successor], successor))
    return schedules


def proxy_score(model, plan):
    """Approximate makespan; never used to accept a final answer."""
    _, schedules = groups_from_plan(plan)
    assigned = {group: core for core, order in enumerate(schedules) for group in order}
    pred = [set(items) for items in model.pred]
    succ = [set(items) for items in model.succ]
    for order in schedules:
        for a, b in zip(order, order[1:]):
            pred[b].add(a)
            succ[a].add(b)
    order = topo(pred, succ)
    finish = {}
    for group in order:
        core = assigned[group]
        finish[group] = model.duration[group] + max(
            (finish[p] + (model.delay + model.flow.get((p, group), 0) /
                          model.cfg['bandwidth'] if assigned[p] != core else 0)
             for p in pred[group]), default=0)
    cross_bytes = sum(size for (a, b), size in model.flow.items()
                      if assigned[a] != assigned[b])
    peaks, residence = estimate_core_cache_pressure(model, schedules)
    return (max(finish.values(), default=0) + 0.15 * cross_bytes /
            model.cfg['bandwidth'] + 0.02 * model.ideal * cache_cost(model, peaks),
            {'cross_bytes_proxy': cross_bytes, 'peak_proxy': peaks,
             'residence_byte_steps': residence})


def official_evaluate_problem2(graph_path, config_path, graph, plan, folder,
                               label, timeout):
    """Validate, invoke the official CLI, and retain result/log for review."""
    validate_plan(graph, plan, len(plan['core_schedules']))
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / f'{label}_plan.json'
    result_path = folder / f'{label}_result.json'
    trace_path = folder / f'{label}_trace.json'
    log_path = folder / f'{label}_log.txt'
    save(plan_path, plan)
    command = [sys.executable, str(ROOT / 'official/code/multicore_cut_evaluate_problem_2.py'),
               str(graph_path), str(plan_path), '--config', str(config_path),
               '-o', str(result_path), '--trace-output', str(trace_path),
               '--log-output', str(log_path)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        trace_path.unlink(missing_ok=True)
        return None, {'status': 'timeout', 'seconds': time.perf_counter() - started}
    trace_path.unlink(missing_ok=True)
    if completed.returncode:
        return None, {'status': 'error', 'seconds': time.perf_counter() - started,
                      'message': completed.stderr.strip()[-1000:]}
    result = _read_json(result_path)
    return result, {'status': 'valid', 'seconds': time.perf_counter() - started,
                    'makespan': result['makespan'],
                    'added_copy_bytes': result['data_movement_bytes']['added_copy_bytes'],
                    'spill_added_copy_bytes': result['data_movement_bytes']['spill_added_copy_bytes']}


def core_finish_times(result, cores):
    ends = [0] * cores
    for entry in result.get('per_core_timeline', []):
        ends[int(entry['core_id'])] = max((op['end'] for op in entry['ops']), default=0)
    return ends


def result_metrics(result):
    traffic = result['data_movement_bytes']
    peaks = result.get('memory_peak_by_core', {})
    return {'makespan': result['makespan'],
            'added_copy_bytes': traffic['added_copy_bytes'],
            'partition_added_copy_bytes': traffic['partition_added_copy_bytes'],
            'spill_added_copy_bytes': traffic['spill_added_copy_bytes'],
            'memory_peak_by_core': peaks,
            'task_dependencies': len(result.get('task_dependencies', []))}


def candidate_neighbors(model, incumbent_plan, official_result, max_neighbors=100):
    """Target heavy cores, hot cut edges, cache pressure and different granularity."""
    _, schedules = groups_from_plan(incumbent_plan)
    assignment = {group: core for core, order in enumerate(schedules) for group in order}
    if len(assignment) != len(model.groups):
        return []
    core_ends = core_finish_times(official_result, model.cores)
    busiest = max(range(model.cores), key=lambda c: core_ends[c])
    light = sorted(range(model.cores), key=lambda c: core_ends[c])
    hot = sorted(schedules[busiest], key=lambda g: (
        -max(model.work[g].values(), default=0)
        -sum(value for other, value in model.affinity[g].items()
             if assignment.get(other) != busiest) / model.cfg['bandwidth'], g))[:10]
    generated = []

    def propose(kind, changed, new_model=model, order_variant=None, magnitude=1):
        try:
            if order_variant is None:
                plan = new_model.plan(changed)
            else:
                plan = new_model.plan(changed,
                                      cache_aware_reorder(new_model, changed, order_variant))
            proxy, detail = proxy_score(new_model, plan)
            generated.append({'kind': kind, 'plan': plan, 'model': new_model,
                              'proxy': proxy, 'detail': detail, 'magnitude': magnitude})
        except (ValueError, KeyError):
            pass

    # Move: heavy-core groups and groups strongly attracted to another core.
    move_seeds = list(dict.fromkeys(hot + sorted(range(len(model.groups)),
        key=lambda g: -sum(value for other, value in model.affinity[g].items()
                           if assignment.get(other) != assignment[g]))[:6]))
    for group in move_seeds:
        targets = sorted((c for c in light if c != assignment[group]),
                         key=lambda c: (-sum(value for other, value in
                                             model.affinity[group].items()
                                             if assignment.get(other) == c), core_ends[c]))[:2]
        for core in targets:
            changed = dict(assignment)
            changed[group] = core
            propose('move', changed)

    # Swap: exchange a heavy-core group with a small group on a light core.
    for source in hot[:4]:
        for target_core in light[:2]:
            if target_core == busiest:
                continue
            other_groups = sorted(schedules[target_core],
                                  key=lambda g: abs(max(model.work[g].values(), default=0)
                                                    - max(model.work[source].values(), default=0)))[:2]
            for target in other_groups:
                changed = dict(assignment)
                changed[source], changed[target] = changed[target], changed[source]
                propose('swap', changed, magnitude=2)

    # ChainMove: keep two adjacent strongly communicating groups together.
    hot_pairs = sorted(((value, a, b) for (a, b), value in model.flow.items()
                        if assignment[a] != assignment[b] or assignment[a] == busiest),
                       reverse=True)[:8]
    for _, a, b in hot_pairs:
        for core in (assignment[a], assignment[b]):
            if assignment[a] == assignment[b] == core:
                core = light[0]
            changed = dict(assignment)
            changed[a] = changed[b] = core
            if changed != assignment:
                propose('chain_move', changed, magnitude=2)

    # Reorder: same assignment, new globally topological priority.
    for variant in (0, 1):
        propose('reorder', assignment, order_variant=variant)

    # Recluster: split one large group by its original op topological order,
    # then reassign one half. Invalid contractions are discarded by topo/validator.
    big = sorted((g for g in schedules[busiest] if len(model.groups[g]) >= 4),
                 key=lambda g: -max(model.work[g].values(), default=0))[:2]
    op_position = {op: index for index, op in enumerate(model.gm.order)}
    for group in big:
        members = sorted(model.groups[group], key=lambda op: op_position[op])
        halfway = len(members) // 2
        if not halfway:
            continue
        groups = [list(members_) for members_ in model.groups]
        groups[group] = members[:halfway]
        groups.append(members[halfway:])
        try:
            refined = Problem2Model(model.gm, groups, model.cores,
                                    model.cfg, model.delay)
            for core in light:
                if core != busiest:
                    changed = dict(assignment)
                    changed[len(groups) - 1] = core
                    propose('recluster', changed, refined, magnitude=len(members) // 2)
                    break
        except (ValueError, KeyError):
            pass
    generated.sort(key=lambda item: item['proxy'])
    unique, seen = [], set()
    for item in generated:
        token = signature(item['plan'])
        if token not in seen:
            seen.add(token)
            unique.append(item)
        if len(unique) >= max_neighbors:
            break
    return unique


def choose_neighbors(neighbors, top_k=3):
    selected = neighbors[:top_k]
    # Keep one structurally different candidate when the proxy is biased.
    alternatives = [item for item in neighbors[top_k:]
                    if item['kind'] in {'chain_move', 'recluster', 'reorder'}]
    if alternatives:
        diversity = max(alternatives, key=lambda item: (item['magnitude'],
                                                        -item['proxy']))
        selected.append(diversity)
    return selected


def default_baseline_path(graph_path, cores):
    case = graph_path.stem
    newest = ROOT / 'results_allcores_adaptive' / f'{case}_{cores}cores' / 'best_plan.json'
    if newest.exists():
        return newest
    original = ROOT / 'final_results' / 'plans' / f'{case}_{cores}cores.json'
    return original if original.exists() else None


def fallback_baseline(gm, cores):
    # Valid and deterministic when no problem-1 plan is available.
    return make_plan([list(gm.order)], [[0]] + [[] for _ in range(cores - 1)])


def solve(graph_path, cores, config_path, output_dir, baseline_path=None,
          max_evals=12, seconds=300, eval_timeout=120,
          base_scales=(0.75, 1.0), patience=2):
    started = time.perf_counter()
    deadline = started + seconds
    graph = _read_json(graph_path)
    gm = GraphModel(graph)
    cfg = read_evaluation_config(str(config_path))
    scene = read_scene_b_config(str(config_path))
    delay = scene['cross_core_copy_delay_cycles']
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline_path is None:
        baseline_path = default_baseline_path(graph_path, cores)
    baseline = _read_json(baseline_path) if baseline_path else fallback_baseline(gm, cores)
    validate_plan(graph, baseline, cores)
    lower_bound = max(max(gm.critical.values(), default=0),
                      max(gm.pipes.values(), default=0) / cores)
    ablation, history = [], []
    official_calls = 0
    best_plan = best_result = best_model = None
    initial_result = None
    accepted = Counter()
    seen = {}

    def score(label, kind, plan, model, proxy=None, detail=None, iteration=0):
        nonlocal official_calls, best_plan, best_result, best_model, initial_result
        token = signature(plan)
        if token in seen:
            previous = seen[token]
            record = {'iteration': iteration, 'candidate': label, 'operation': kind,
                      'proxy': proxy, 'status': 'duplicate',
                      'same_as': previous['candidate'],
                      'official_makespan': previous.get('official_makespan'),
                      'added_copy_bytes': previous.get('added_copy_bytes'),
                      'accepted': False, 'detail': detail}
            history.append(record)
            if iteration == 0:
                ablation.append(record)
            save(output_dir / 'search_history.json', history)
            return False
        if official_calls >= max_evals or time.perf_counter() >= deadline:
            return False
        seen[token] = {'candidate': label}
        remaining = deadline - time.perf_counter()
        if remaining < 1:
            return False
        official_calls += 1
        print(f'[{official_calls}/{max_evals}] {label}: official scene-B evaluation',
              flush=True)
        try:
            result, info = official_evaluate_problem2(
                graph_path, config_path, graph, plan, output_dir, label,
                min(eval_timeout, remaining))
        except (ValueError, RuntimeError, KeyError) as error:
            result, info = None, {'status': 'invalid', 'message': str(error)[:500]}
        record = {'iteration': iteration, 'candidate': label, 'operation': kind,
                  'proxy': proxy, 'status': info['status'],
                  'official_makespan': info.get('makespan'),
                  'added_copy_bytes': info.get('added_copy_bytes'),
                  'accepted': False, 'seconds': info.get('seconds'),
                  'detail': detail}
        if result is not None:
            metrics = result_metrics(result)
            record.update(metrics)
            if initial_result is None:
                initial_result = metrics
            if best_result is None or result_key(result) < result_key(best_result):
                best_plan, best_result, best_model = plan, result, model
                record['accepted'] = True
                accepted[kind] += 1
                save(output_dir / 'best_plan.json', best_plan)
                save(output_dir / 'best_result.json', best_result)
        history.append(record)
        seen[token] = record
        if iteration == 0:
            ablation.append(record)
        save(output_dir / 'search_history.json', history)
        print(f"  {info['status']} makespan={info.get('makespan')} "
              f"best={best_result['makespan'] if best_result else None}", flush=True)
        return record['accepted']

    # A: final problem-1 plan evaluated under Scene B rules.
    baseline_groups, _ = groups_from_plan(baseline)
    baseline_model = Problem2Model(gm, baseline_groups, cores, cfg, delay)
    score('baseline_a', 'problem1_plan', baseline, baseline_model)

    # B/C/D: same fine partition, then progressively change core assignment.
    for scale in base_scales:
        if official_calls >= max_evals or time.perf_counter() >= deadline:
            break
        groups, eft_plan, partition_detail = build_base_groups(
            gm, cores, cfg, delay, scale)
        model = Problem2Model(gm, groups, cores, cfg, delay)
        marker = str(scale).replace('.', 'p')
        score(f'baseline_b_{marker}', 'eft', eft_plan, model,
              detail={'scale': scale, **partition_detail})
        if official_calls >= max_evals or time.perf_counter() >= deadline:
            break
        direct_assignment, direct_schedules = affinity_core_assignment(
            model, shared=False, cache=False)
        score(f'method_c_{marker}', 'direct_affinity',
              model.plan(direct_assignment, direct_schedules), model,
              detail={'scale': scale})
        if official_calls >= max_evals or time.perf_counter() >= deadline:
            break
        assignment, _ = affinity_core_assignment(model, shared=True, cache=True)
        reordered = cache_aware_reorder(model, assignment)
        plan = model.plan(assignment, reordered)
        proxy, proxy_detail = proxy_score(model, plan)
        score(f'method_d_{marker}', 'cache_aware', plan, model,
              proxy=proxy, detail={'scale': scale, **proxy_detail})

    # E: bounded, official-evaluator-guided local search.
    no_improvement = 0
    iteration = 0
    while (best_result is not None and no_improvement < patience and
           official_calls < max_evals and time.perf_counter() < deadline):
        iteration += 1
        neighbors = candidate_neighbors(best_model, best_plan, best_result)
        choices = choose_neighbors(neighbors)
        if not choices:
            break
        improved = False
        for index, item in enumerate(choices):
            label = f'search_{iteration}_{index}_{item["kind"]}'
            improved |= score(label, item['kind'], item['plan'], item['model'],
                              proxy=item['proxy'], detail=item['detail'],
                              iteration=iteration)
            if official_calls >= max_evals or time.perf_counter() >= deadline:
                break
        no_improvement = 0 if improved else no_improvement + 1

    if best_result is None:
        raise RuntimeError('No valid Scene-B evaluation; inspect search_history.json')
    summary = {'case': graph_path.stem, 'cores': cores,
               'baseline_problem1_plan': str(baseline_path) if baseline_path else None,
               'base_target_scales': list(base_scales),
               'initial_tasks': len(baseline_groups),
               'initial_proxy_score': proxy_score(baseline_model, baseline)[0],
               'initial_official': initial_result,
               'best': result_metrics(best_result),
               'official_evaluator_calls': official_calls,
               'accepted_operations': dict(accepted),
               'solver_runtime_seconds': time.perf_counter() - started,
               'optimistic_lower_bound': lower_bound,
               'gap_to_optimistic_lower_bound':
                   (best_result['makespan'] - lower_bound) / max(1, lower_bound),
               'ablation': ablation + [{'iteration': iteration,
                                        'candidate': 'method_e_final',
                                        'operation': 'local_search',
                                        **result_metrics(best_result)}],
               'best_plan_path': str(output_dir / 'best_plan.json'),
               'best_result_path': str(output_dir / 'best_result.json')}
    save(output_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path, help='case JSON')
    parser.add_argument('-n', '--cores', type=int, required=True, choices=range(2, 6))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline-plan', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-evals', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--eval-timeout', type=float, default=120)
    parser.add_argument('--base-scales', nargs='+', type=float, default=[0.75, 1.0])
    parser.add_argument('--patience', type=int, default=2)
    args = parser.parse_args(argv)
    if (args.max_evals < 1 or args.seconds <= 0 or args.eval_timeout <= 0 or
            args.patience < 1 or any(scale <= 0 for scale in args.base_scales)):
        parser.error('budgets and scales must be positive')
    if not args.graph.is_file() or not args.config.is_file():
        parser.error('graph and config files must exist')
    out = args.output_dir or ROOT / 'results_problem2' / f'{args.graph.stem}_{args.cores}cores'
    summary = solve(args.graph.resolve(), args.cores, args.config.resolve(),
                    out.resolve(), args.baseline_plan, args.max_evals,
                    args.seconds, args.eval_timeout,
                    tuple(args.base_scales), args.patience)
    print(json.dumps({'makespan': summary['best']['makespan'],
                      'added_copy_bytes': summary['best']['added_copy_bytes'],
                      'best_plan_path': summary['best_plan_path'],
                      'official_evaluator_calls': summary['official_evaluator_calls']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""第一问混合求解：结构聚合 + HEFT式分核 + 有预算的瓶颈局部搜索。

运行 example: python hybrid_solver.py --case case_002 --cores 4 --max-evals 8 --seconds 120
max-evals 包括初始方案的评分；不会按固定网格枚举粒度。
"""
import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from experiment import ROOT, _read_json, read_evaluation_config, read_scene_a_config, save
from fast_solver import GraphModel, construct, estimate, independent_component_plan, topo
from region_solver import region_partition
from adaptive_clustering import adaptive_partition, schedule_groups
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order


def groups_from_plan(plan):
    mapping = plan['node_to_subgraph']
    ids = sorted(set(mapping.values()))
    by_id = {old: new for new, old in enumerate(ids)}
    groups = [[] for _ in ids]
    for v, old in mapping.items():
        groups[by_id[old]].append(int(v))
    schedules = [[by_id[v] for v in order] for order in plan['core_schedules']]
    return groups, schedules


def plan_from_groups(groups, schedules):
    return {'node_to_subgraph': {str(v): i for i, group in enumerate(groups) for v in group},
            'core_schedules': schedules}


def signature(plan):
    return (tuple(sorted((int(v), i) for v, i in plan['node_to_subgraph'].items())),
            tuple(tuple(order) for order in plan['core_schedules']))


def compact(groups, schedules):
    kept = [i for i, group in enumerate(groups) if group]
    new_id = {old: new for new, old in enumerate(kept)}
    return [groups[i] for i in kept], [[new_id[i] for i in order if i in new_id] for order in schedules]


def tensor_affinities(gm, mapping):
    """张量按子图去重，分别记录生产消费边和共享外部输入。"""
    flow, shared = Counter(), Counter()
    for t in gm.graph['tensors']:
        ident, size = t['id'], t['size']
        prod = {mapping[v] for v in gm.producers[ident] if v in mapping}
        cons = {mapping[v] for v in gm.consumers[ident] if v in mapping}
        for a in prod:
            for b in cons - {a}:
                flow[a, b] += size
        # 输入不在候选子图中生产。限制高扇出 pair 展开，防止大图二次方。
        if not prod and 1 < len(cons) <= 16:
            c = sorted(cons)
            for i, a in enumerate(c):
                for b in c[i + 1:]:
                    shared[a, b] += size
    return flow, shared


def observed_core_ends(result, cores):
    ends = [0] * cores
    for core in result['per_core_timeline']:
        ends[core['core_id']] = max((op['end'] for op in core['ops']), default=0)
    return ends


def neighbors(gm, groups, schedules, result, bandwidth, waits=None):
    """只生成重核、热边、大块附近的候选；每类邻域有固定的小上限。"""
    count, cores = len(groups), len(schedules)
    if count == 0:
        return []
    mapping, pred, succ, work, traffic, duration = gm.describe(groups, bandwidth)
    order = topo(pred, succ)
    position = {v: i for i, v in enumerate(order)}
    core_ends = observed_core_ends(result, cores)
    busy = {core['core_id']: sum(task['duration'] for task in core.get('tasks', []))
            for core in result['per_core_timeline']}
    loads = [busy.get(c) or core_ends[c] for c in range(cores)]
    heavy = sorted(range(cores), key=lambda c: (-loads[c], -core_ends[c], c))[:min(2, cores)]
    light = sorted(range(cores), key=lambda c: (loads[c], core_ends[c], c))[:min(2, cores)]
    result_items = []

    def add(kind, detail, new_groups, new_schedules, score_hint=0):
        result_items.append((kind, detail, new_groups, new_schedules, score_hint))

    # Move：从最晚结束核心挑瓶颈块，向较早结束核心移动。
    for source in heavy:
        ranked = sorted(schedules[source], key=lambda v: (-duration[v], -traffic[v], v))[:3]
        for v in ranked:
            for target in light:
                if source == target:
                    continue
                orders = [x[:] for x in schedules]
                orders[source].remove(v)
                insert_at = next((i for i, w in enumerate(orders[target]) if position[w] > position[v]), len(orders[target]))
                orders[target].insert(insert_at, v)
                add('move', {'subgraph': v, 'from_core': source, 'to_core': target}, groups, orders, duration[v])

    # Swap：互换重核上的大块与轻核上的块，然后检查全局依赖。
    for source in heavy[:1]:
        if not schedules[source]:
            continue
        a = max(schedules[source], key=lambda v: (duration[v], -v))
        for target in light:
            if target == source:
                continue
            for b in sorted(schedules[target], key=lambda v: (-duration[v], v))[:2]:
                orders = [x[:] for x in schedules]
                orders[source][orders[source].index(a)] = b
                orders[target][orders[target].index(b)] = a
                add('swap', {'subgraphs': [a, b], 'cores': [source, target]}, groups, orders,
                    abs(duration[a] - duration[b]))

    # Merge：先试跨任务 Tensor 大的边，也考虑共享外部输入。
    flow, shared = tensor_affinities(gm, mapping)
    pair_scores = Counter()
    for (a, b), size in flow.items():
        pair_scores[tuple(sorted((a, b)))] += 2 * size
    for pair, size in shared.items():
        pair_scores[pair] += size
    for pair_index, ((a, b), score) in enumerate(
            sorted(pair_scores.items(), key=lambda x: (-x[1], x[0]))[:8]):
        # A/B 合入 A 的位置；包含跨核尝试，最终仍须验证 Task 图无环。
        for keep, drop in ((a, b), (b, a)) if a != b else ((a, b),):
            members = [g[:] for g in groups]
            members[keep].extend(members[drop])
            members[drop] = []
            orders = [[v for v in s if v != drop] for s in schedules]
            members, orders = compact(members, orders)
            add('merge', {'keep': keep, 'drop': drop, 'weighted_tensor_bytes': score}, members, orders, score)
            if waits is not None and pair_index < 2:
                try:
                    replanned, _ = schedule_groups(gm, members, cores, {'bandwidth': bandwidth}, waits)
                    add('merge_reschedule', {'keep': keep, 'drop': drop}, members,
                        replanned['core_schedules'], score)
                except (RuntimeError, ValueError):
                    pass

    # Split：只考虑工作量最大、至少四个操作的大组；按原 DAG 拓扑序二分。
    op_pos = {v: i for i, v in enumerate(gm.order)}
    for v in sorted(range(count), key=lambda i: (-max(work[i].values(), default=0), i))[:2]:
        if len(groups[v]) < 4:
            continue
        ordered = sorted(groups[v], key=lambda op: op_pos[op])
        half = sum(gm.ops[op]['cycles'] for op in ordered) / 2
        consumed, mid = 0, 0
        while mid < len(ordered) - 1 and (consumed < half or mid == 0):
            consumed += gm.ops[ordered[mid]]['cycles']
            mid += 1
        members = [g[:] for g in groups]
        members[v] = ordered[:mid]
        new_id = len(members)
        members.append(ordered[mid:])
        orders = [s[:] for s in schedules]
        for s in orders:
            if v in s:
                s.insert(s.index(v) + 1, new_id)
                break
        add('split', {'subgraph': v, 'left_ops': mid, 'right_ops': len(ordered) - mid}, members, orders,
            max(work[v].values(), default=0))
        if waits is not None:
            replanned, _ = schedule_groups(gm, members, cores, {'bandwidth': bandwidth}, waits)
            add('split_reschedule', {'subgraph': v, 'left_ops': mid,
                                    'right_ops': len(ordered) - mid}, members,
                replanned['core_schedules'], max(work[v].values(), default=0))
    if waits is not None:
        measured = {task['subgraph_id']: task['duration']
                    for core in result['per_core_timeline'] for task in core.get('tasks', [])}
        if all(v in measured for v in range(count)):
            replanned, _ = schedule_groups(gm, groups, cores, {'bandwidth': bandwidth}, waits,
                                            [measured[v] for v in range(count)])
            add('observed_reschedule', {}, groups, replanned['core_schedules'])
    return result_items


def official_score(case, plan, folder, label, timeout, fast_evaluator=False):
    plan_file = folder / f'{label}_plan.json'
    result_file = folder / f'{label}_result.json'
    trace_file = folder / f'{label}_trace.json'
    log_file = folder / f'{label}_log.txt'
    save(plan_file, plan)
    for path in (result_file, trace_file, log_file):
        path.unlink(missing_ok=True)
    evaluator = ('optimized_official/code/multicore_cut_evaluate_problem_1.py' if fast_evaluator
                 else 'official/code/multicore_cut_evaluate_problem_1.py')
    command = [sys.executable, str(ROOT / evaluator),
               str(ROOT / 'official/data' / f'{case}.json'), str(plan_file),
               '--config', str(ROOT / 'official/data/config.txt'), '-o', str(result_file),
               '--trace-output', str(trace_file), '--log-output', str(log_file)]
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                   timeout=max(.01, timeout))
    except subprocess.TimeoutExpired:
        for path in (result_file, trace_file, log_file):
            path.unlink(missing_ok=True)
        return None, {'status': 'timeout', 'seconds': time.perf_counter() - start}
    if completed.returncode:
        return None, {'status': 'error', 'seconds': time.perf_counter() - start,
                      'message': completed.stderr.strip()[:1000]}
    result = _read_json(result_file)
    return result, {'status': 'valid', 'seconds': time.perf_counter() - start,
                    'makespan': result['makespan'],
                    'added_copy_bytes': result['data_movement_bytes']['added_copy_bytes']}


def run(case='case_002', cores=4, max_evals=8, seconds=120, starts=3,
        output_root=None, initial_score_timeout=120, fast_evaluator=False):
    if max_evals < 1 or seconds <= 0 or initial_score_timeout <= 0 or starts not in (1, 3, 5):
        raise ValueError('max-evals, seconds and initial_score_timeout must be positive; starts must be 1, 3 or 5')
    started = time.perf_counter()
    graph = _read_json(ROOT / 'official/data' / f'{case}.json')
    config = str(ROOT / 'official/data/config.txt')
    cfg, waits = read_evaluation_config(config), read_scene_a_config(config)
    gm = GraphModel(graph)
    out = Path(output_root) / f'{case}_{cores}cores' if output_root else ROOT / 'results_hybrid' / f'{case}_{cores}cores'
    scales = {1: (1.0,), 3: (1.0, .75, 1.25), 5: (1.0, .75, 1.25, .6, 1.5)}[starts]
    if max_evals == 1:
        scales = (1.0,)
    pool, seen_seeds = [], set()
    for scale in scales:
        if time.perf_counter() - started >= seconds:
            break
        plan, explanation = construct(graph, cores, cfg, waits, target_scale=scale)
        token = signature(plan)
        if token in seen_seeds:
            continue
        seen_seeds.add(token)
        groups, schedules = groups_from_plan(plan)
        proxy = estimate(gm, groups, schedules, cfg['bandwidth'], waits)
        pool.append((scale, proxy, plan, explanation))
    if len(gm.components) > 1 and time.perf_counter() - started < seconds:
        plan = independent_component_plan(gm, cores)
        token = signature(plan)
        if token not in seen_seeds:
            validate_task_order(derive_multicore_plan(graph, plan))
            seen_seeds.add(token)
            groups, schedules = groups_from_plan(plan)
            proxy = estimate(gm, groups, schedules, cfg['bandwidth'], waits)
            pool.append(('independent_components', proxy, plan,
                         {'final_task_count': len(groups), 'selected': 'independent_components'}))
    if max_evals > 1 and len(gm.ops) >= cores and time.perf_counter() - started < seconds:
        plan, explanation = adaptive_partition(gm, cores, cfg, waits)
        token = signature(plan)
        if token not in seen_seeds:
            seen_seeds.add(token)
            groups, schedules = groups_from_plan(plan)
            proxy = estimate(gm, groups, schedules, cfg['bandwidth'], waits)
            pool.append(('adaptive_clustering', proxy, plan,
                         {'final_task_count': len(groups), **explanation}))
    initial_used_cores = sum(bool(order) for order in pool[0][2]['core_schedules']) if pool else 0
    if initial_used_cores < cores and len(gm.ops) >= 100 and time.perf_counter() - started < seconds:
        try:
            plan, explanation = region_partition(gm, cores, cfg, waits)
            token = signature(plan)
            if token not in seen_seeds:
                seen_seeds.add(token)
                groups, schedules = groups_from_plan(plan)
                proxy = estimate(gm, groups, schedules, cfg['bandwidth'], waits)
                pool.append(('dependency_regions', proxy, plan,
                             {'final_task_count': len(groups), **explanation}))
        except (RuntimeError, ValueError):
            pass
    if output_root and cores > 2:
        for lower_cores in range(2, cores):
            previous = out.parent / f'{case}_{lower_cores}cores'
            previous_plan = previous / 'best_plan.json'
            previous_result = previous / 'best_result.json'
            if not previous_plan.exists() or not previous_result.exists():
                continue
            plan = _read_json(previous_plan)
            plan['core_schedules'] += [[] for _ in range(cores - len(plan['core_schedules']))]
            token = signature(plan)
            if token in seen_seeds:
                continue
            validate_task_order(derive_multicore_plan(graph, plan))
            seen_seeds.add(token)
            proxy = _read_json(previous_result)['makespan']
            pool.append((f'carry_{lower_cores}core', proxy, plan,
                         {'final_task_count': len(set(plan['node_to_subgraph'].values())),
                          'selected': 'carry_from_lower_core'}))
    if not pool:
        raise RuntimeError('time budget expired before constructing a plan')
    # 代理可以建议比较单 Task，但绝不能仅凭代理把多核方案丢弃。
    if pool[0][3].get('proxy_prefers_serial') and max_evals > 1:
        first_plan = pool[0][2]
        serial_plan = {'node_to_subgraph': {op_id: 0 for op_id in first_plan['node_to_subgraph']},
                       'core_schedules': [[0]] + [[] for _ in range(cores - 1)]}
        token = signature(serial_plan)
        if token not in seen_seeds:
            seen_seeds.add(token)
            pool.append(('single_task', pool[0][3]['estimated_serial_cycles'], serial_plan,
                         {'final_task_count': 1, 'selected': 'single_task_for_official_comparison'}))
    save(out / 'start_pool.json', [{'scale': scale, 'proxy_cycles': proxy,
                                    'tasks': explanation['final_task_count']}
                                   for scale, proxy, _, explanation in pool])
    # 默认规模必须进入官方评分；代理只用来挑选另一份不同的方案。
    initial_scale, _, initial_plan, initial_explanation = pool[0]
    save(out / 'construction_explanation.json', initial_explanation)
    history = []
    remaining = seconds - (time.perf_counter() - started)
    if remaining <= 0:
        save(out / 'best_plan.json', initial_plan)
        save(out / 'summary.json', {'status': 'construction_only_budget_expired', 'official_score_attempts': 0})
        return None
    initial_timeout = min(initial_score_timeout, remaining)
    result, info = official_score(case, initial_plan, out, 'initial', initial_timeout,
                                  fast_evaluator=fast_evaluator)
    calls = 1
    history.append({'step': 0, 'kind': 'initial', 'scale': initial_scale, **info})
    seen = {signature(initial_plan)}
    best_plan, best_result = initial_plan, result
    if best_result is None:
        # A slow initial partition must not make the entire configuration fail.
        fallback = sorted(pool[1:], key=lambda x: (x[0] == 'single_task', x[1], str(x[0])))
        for scale, proxy, candidate, explanation in fallback:
            if calls >= max_evals or time.perf_counter() - started >= seconds:
                break
            remaining = seconds - (time.perf_counter() - started)
            alternate, alt_info = official_score(case, candidate, out, f'fallback_{calls}',
                                                 min(initial_score_timeout, remaining),
                                                 fast_evaluator=fast_evaluator)
            history.append({'step': calls, 'kind': 'fallback', 'scale': scale,
                            'proxy_cycles': proxy, **alt_info})
            calls += 1
            seen.add(signature(candidate))
            if alternate is not None:
                best_plan, best_result = candidate, alternate
                history[-1]['accepted'] = True
                break
    if best_result is None:
        save(out / 'search_history.json', history)
        failure = {'status': 'initial_score_unavailable', 'initial_score_status': info['status'],
                   'official_score_attempts': calls,
                   'elapsed_seconds': time.perf_counter() - started,
                   'initial_score_timeout': initial_score_timeout,
                   'seconds_budget': seconds}
        if 'message' in info:
            failure['message'] = info['message']
        save(out / 'summary.json', failure)
        print(failure, flush=True)
        return None
    best_key = (best_result['makespan'], best_result['data_movement_bytes']['added_copy_bytes'])
    lower_bound = max(gm.pipes.values(), default=0) / cores
    certified_near_bound = bool(lower_bound and best_result['makespan'] <= 1.02 * lower_bound)
    if not certified_near_bound and len(pool) > 1:
        # A serial fallback must not crowd out all parallel seeds. Score the two
        # strongest distinct parallel constructions first, then compare serial.
        parallel = sorted((item for item in pool[1:] if item[0] != 'single_task'),
                          key=lambda x: (x[1], str(x[0])))[:2]
        region = next((item for item in pool[1:] if item[0] == 'dependency_regions'), None)
        if region is not None and region not in parallel:
            parallel.append(region)
        adaptive = next((item for item in pool[1:] if item[0] == 'adaptive_clustering'), None)
        if adaptive is not None:
            parallel = [adaptive] + [item for item in parallel if item is not adaptive]
        serial = [item for item in pool[1:] if item[0] == 'single_task']
        for scale, proxy, candidate, explanation in parallel + serial:
            if calls >= max_evals or time.perf_counter() - started >= seconds:
                break
            if signature(candidate) in seen:
                continue
            save(out / f'alternative_{calls}_construction_explanation.json', explanation)
            remaining = seconds - (time.perf_counter() - started)
            second, info = official_score(case, candidate, out, f'alternative_initial_{calls}',
                                          min(initial_score_timeout, remaining),
                                          fast_evaluator=fast_evaluator)
            history.append({'step': calls, 'kind': 'alternative_initial', 'scale': scale,
                            'proxy_cycles': proxy, **info})
            calls += 1
            seen.add(signature(candidate))
            if second is not None:
                key = (second['makespan'], second['data_movement_bytes']['added_copy_bytes'])
                if key < best_key:
                    best_plan, best_result, best_key = candidate, second, key
                    history[-1]['accepted'] = True
    while not certified_near_bound and calls < max_evals and time.perf_counter() - started < seconds:
        groups, schedules = groups_from_plan(best_plan)
        proposal = []
        queued = set()
        for kind, detail, new_groups, new_schedules, hint in neighbors(
                gm, groups, schedules, best_result, cfg['bandwidth'], waits):
            candidate = plan_from_groups(new_groups, new_schedules)
            token = signature(candidate)
            if token in seen or token in queued:
                continue
            queued.add(token)
            try:
                validate_task_order(derive_multicore_plan(graph, candidate))
                proxy = estimate(gm, new_groups, new_schedules, cfg['bandwidth'], waits)
            except (RuntimeError, ValueError):
                continue
            proposal.append((proxy, -hint, kind, detail, candidate))
        if not proposal:
            break
        proposal.sort(key=lambda item: (item[0], item[1], item[2]))
        # 每类邻域先选一份；未评分的候选留在后续轮次，不提前标记为已搜索。
        chosen, kinds = [], set()
        for item in proposal:
            if item[2] not in kinds:
                chosen.append(item)
                kinds.add(item[2])
            if len(chosen) == 4:
                break
        for item in proposal:
            if len(chosen) == 4:
                break
            if item not in chosen:
                chosen.append(item)
        timed_out = False
        for proxy, _, kind, detail, candidate in chosen:
            if calls >= max_evals or time.perf_counter() - started >= seconds:
                break
            remaining = seconds - (time.perf_counter() - started)
            result, info = official_score(case, candidate, out, f'candidate_{calls}', min(60, remaining),
                                          fast_evaluator=fast_evaluator)
            seen.add(signature(candidate))
            history.append({'step': calls, 'kind': kind, 'detail': detail, 'proxy_cycles': proxy, **info})
            calls += 1
            if result is not None:
                key = (result['makespan'], result['data_movement_bytes']['added_copy_bytes'])
                if key < best_key:
                    best_plan, best_result, best_key = candidate, result, key
                    history[-1]['accepted'] = True
                    break
            history[-1]['accepted'] = False
            if info['status'] == 'timeout':
                timed_out = True
                break
        if timed_out:
            break
    save(out / 'best_plan.json', best_plan)
    save(out / 'best_result.json', best_result)
    save(out / 'search_history.json', history)
    serial_history = next((item for item in history if item.get('scale') == 'single_task'), None)
    if serial_history is not None:
        serial_comparison_status = serial_history['status']
    elif initial_explanation.get('proxy_prefers_serial') and initial_explanation['final_task_count'] > 1:
        serial_comparison_status = 'not_attempted'
    else:
        serial_comparison_status = 'not_needed'
    summary = {'status': 'scored', 'case': case, 'cores': cores,
               'initial_makespan': history[0].get('makespan'),
               'best_makespan': best_result['makespan'],
               'best_added_copy_bytes': best_result['data_movement_bytes']['added_copy_bytes'],
               'official_score_attempts': calls, 'accepted_changes': sum(x.get('accepted', False) for x in history),
               'serial_comparison_status': serial_comparison_status,
               'elapsed_seconds': time.perf_counter() - started,
               'initial_score_timeout': initial_score_timeout,
               'max_evals': max_evals, 'seconds_budget': seconds, 'starts': starts,
               'starts_generated': len(pool),
               'pipe_lower_bound': lower_bound,
               'certified_gap_to_pipe_lower_bound_percent':
                   100 * (best_result['makespan'] / lower_bound - 1) if lower_bound else None,
               'stopped_near_bound': certified_near_bound}
    save(out / 'summary.json', summary)
    print(summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='case_002')
    parser.add_argument('--cores', type=int, choices=[2, 3, 4, 5], default=4)
    parser.add_argument('--max-evals', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--starts', type=int, choices=[1, 3, 5], default=3)
    parser.add_argument('--output-root', help='评估结果目录；默认 results_hybrid')
    parser.add_argument('--initial-score-timeout', type=float, default=120,
                        help='初始和第二候选的单次官方评分最长秒数，默认120')
    parser.add_argument('--fast-evaluator', action='store_true',
                        help='使用经过等价性对照的事件模拟加速副本')
    args = parser.parse_args()
    if run(args.case, args.cores, args.max_evals, args.seconds, args.starts,
           args.output_root, args.initial_score_timeout, args.fast_evaluator) is None:
        sys.exit(2)

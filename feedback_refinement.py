"""Observed-timeline refinement for a standalone five-core incumbent.

The evaluator remains the authority.  This module only constructs a bounded,
deterministic neighbourhood from one graph, config, and incumbent plan.
"""
from collections import Counter

from adaptive_clustering import insertion_schedule
from fast_solver import GraphModel, topo
from hybrid_solver import groups_from_plan, signature
from stub_multicore_cut_and_schedule import MulticoreCutError, derive_multicore_plan
from evaluation_validation import validate_task_order


def _plan(groups, schedules):
    return {'node_to_subgraph': {str(v): i for i, group in enumerate(groups) for v in group},
            'core_schedules': [list(s) for s in schedules]}


def _result_durations(result, count):
    values = {}
    for core in result.get('per_core_timeline', []):
        for task in core.get('tasks', []):
            values[int(task['subgraph_id'])] = int(task['duration'])
    return [values.get(i, 0) for i in range(count)]


def analyze(gm, plan, result, bandwidth, waits):
    """Return quotient plus observed critical tasks and augmented edges."""
    groups, schedules = groups_from_plan(plan)
    _, pred, succ, work, traffic, proxy = gm.describe(groups, bandwidth)
    observed = _result_durations(result, len(groups))
    durations = [observed[i] or proxy[i] for i in range(len(groups))]
    augmented_pred = [set(v) for v in pred]
    augmented_succ = [set(v) for v in succ]
    core_of = {}
    for core, order in enumerate(schedules):
        for task in order:
            core_of[task] = core
        for a, b in zip(order, order[1:]):
            augmented_pred[b].add(a)
            augmented_succ[a].add(b)
    order = topo(augmented_pred, augmented_succ)
    finish, parent, edge_delay = {}, {}, {}
    cross, same = waits['task_cross_core_wait_cycles'], waits['task_same_core_wait_cycles']
    for task in order:
        best = (0, -1, 0)
        for p in sorted(augmented_pred[task]):
            delay = same if core_of.get(p) == core_of.get(task) else cross
            value = finish[p] + delay
            if value > best[0]:
                best = (value, p, delay)
        finish[task] = best[0] + durations[task]
        parent[task] = best[1]
        edge_delay[task] = best[2]
    tail = max(order, key=lambda t: (finish[t], -t), default=None)
    critical = []
    while tail is not None and tail >= 0:
        critical.append(tail)
        tail = parent[tail]
    critical.reverse()
    edges = []
    for b in order:
        for a in sorted(augmented_pred[b]):
            edges.append({'source': a, 'target': b, 'delay': same if core_of.get(a) == core_of.get(b) else cross,
                          'critical': b in critical and (parent[b] == a)})
    return {'groups': groups, 'schedules': schedules, 'pred': pred, 'succ': succ,
            'augmented_pred': augmented_pred, 'augmented_succ': augmented_succ,
            'work': work, 'traffic': traffic, 'proxy_durations': proxy,
            'observed_durations': durations, 'critical_tasks': critical,
            'critical_edges': edges, 'finish': finish, 'core_of': core_of,
            'makespan_proxy': max(finish.values(), default=0)}


def _valid(graph, plan):
    mapping = {int(v): int(i) for v, i in plan['node_to_subgraph'].items()}
    # Plans intentionally omit COPY_IN/COPY_OUT nodes; the official plan
    # derivation reconstructs those around the scheduled compute subgraphs.
    if set(mapping) != {int(op['id']) for op in graph['ops']
                        if op.get('op') not in {'COPY_IN', 'COPY_OUT'}}:
        return False
    seen = [v for order in plan['core_schedules'] for v in order]
    ids = sorted(set(mapping.values()))
    if sorted(seen) != ids or any(seen.count(i) != 1 for i in ids):
        return False
    try:
        validate_task_order(derive_multicore_plan(graph, plan))
    except (ValueError, MulticoreCutError, KeyError):
        return False
    return True


def _proxy(gm, groups, schedules, bandwidth, waits, durations=None):
    _, pred, succ, _, _, modeled = gm.describe(groups, bandwidth)
    duration = modeled if durations is None else durations
    core_of = {v: c for c, s in enumerate(schedules) for v in s}
    for c, order in enumerate(schedules):
        for a, b in zip(order, order[1:]):
            pred[b].add(a); succ[a].add(b)
    finish = {}
    for task in topo(pred, succ):
        release = max((finish[p] + (waits['task_same_core_wait_cycles'] if core_of[p] == core_of[task]
                                    else waits['task_cross_core_wait_cycles']) for p in pred[task]), default=0)
        finish[task] = release + duration[task]
    return max(finish.values(), default=0)


def _split_options(gm, info, group_id):
    members = sorted(info['groups'][group_id], key={v: i for i, v in enumerate(gm.order)}.__getitem__)
    if len(members) < 3:
        return []
    positions = {v: i for i, v in enumerate(members)}
    boundary_delta = [0] * (len(members) + 1)
    # A tensor crosses a prefix cut exactly when a producer lies before it and
    # a consumer lies after it.  Accumulating its active cut interval keeps
    # this linear in the task and tensor counts.
    for tensor in gm.graph['tensors']:
        producers = [positions[v] for v in gm.producers[tensor['id']] if v in positions]
        consumers = [positions[v] for v in gm.consumers[tensor['id']] if v in positions]
        if producers and consumers:
            first_producer, last_consumer = min(producers), max(consumers)
            if first_producer < last_consumer:
                boundary_delta[first_producer + 1] += tensor['size']
                boundary_delta[last_consumer + 1] -= tensor['size']
    total_work = Counter()
    for v in members:
        total_work[gm.ops[v]['pipe']] += gm.ops[v]['cycles']
    left_work, boundary, options = Counter(), 0, []
    for cut in range(1, len(members)):
        op = gm.ops[members[cut - 1]]
        left_work[op['pipe']] += op['cycles']
        right_work = total_work - left_work
        boundary += boundary_delta[cut]
        split_max = max(max(left_work.values(), default=0), max(right_work.values(), default=0))
        options.append({'cut': cut, 'boundary': boundary, 'left_work': dict(left_work),
                        'right_work': dict(right_work), 'max_pipe_work': split_max,
                        'spill_relief': max(total_work.values(), default=0) - split_max})
    return members, options


def _split_plan(info, group_id, left, right, reschedule, gm, waits, bandwidth):
    groups = [g[:] for g in info['groups']]
    groups[group_id] = left
    groups.append(right)
    schedules = [s[:] for s in info['schedules']]
    new_id = len(groups) - 1
    for s in schedules:
        if group_id in s:
            at = s.index(group_id); s[at:at + 1] = [group_id, new_id]
            break
    if reschedule:
        _, pred, succ, _, _, durations = gm.describe(groups, bandwidth)
        schedules, _ = insertion_schedule(pred, succ, durations, len(schedules), waits)
    return _plan(groups, schedules)


def candidates(graph, incumbent, result, cfg, waits, cores=5, limit=4):
    """Return at most four distinct plans in A/B/C/D family order."""
    if cores != 5 or limit < 1:
        raise ValueError('feedback refinement is standalone five-core only')
    gm = GraphModel(graph)
    info = analyze(gm, incumbent, result, cfg['bandwidth'], waits)
    output, seen = [], {signature(incumbent)}
    def add(plan, family, detail):
        if len(output) >= limit or signature(plan) in seen or not _valid(graph, plan): return
        candidate_groups, candidate_schedules = groups_from_plan(plan)
        observed = info['observed_durations'] if candidate_groups == groups else None
        seen.add(signature(plan)); output.append({'family': family, 'plan': plan, 'detail': detail,
                                                   'proxy_makespan': _proxy(gm, candidate_groups, candidate_schedules,
                                                                            cfg['bandwidth'], waits, observed)})
    groups, _ = groups_from_plan(incumbent)
    _, pred, succ, _, _, _ = gm.describe(groups, cfg['bandwidth'])
    schedules, detail = insertion_schedule(pred, succ, info['observed_durations'], cores, waits)
    add(_plan(groups, schedules), 'A_observed_insertion', detail)
    critical_groups = [g for g in info['critical_tasks'] if len(groups[g]) >= 3]
    if critical_groups:
        # Inspect one observed bottleneck group, rather than repeatedly
        # scanning every tensor for every group on a long critical path.
        g = max(critical_groups, key=lambda task: (info['observed_durations'][task], len(groups[task]), -task))
        members, options = _split_options(gm, info, g)
        b = min(options, key=lambda x: (x['max_pipe_work'], x['boundary'], x['cut']))
        add(_split_plan(info, g, members[:b['cut']], members[b['cut']:], True, gm, waits, cfg['bandwidth']), 'B_critical_prefix_split',
            {'subgraph': g, 'cut': b['cut'], 'predicted_finish': b['max_pipe_work'],
             'left_pipe_work': b['left_work'], 'right_pipe_work': b['right_work'],
             'boundary_bytes_per_60': b['boundary'] / cfg['bandwidth'], 'spill_relief': b['spill_relief']})
        pareto = sorted(options, key=lambda x: (x['boundary'], -x['spill_relief'], x['cut']))
        for c in pareto:
            if c['cut'] != b['cut']:
                add(_split_plan(info, g, members[:c['cut']], members[c['cut']:], True, gm, waits, cfg['bandwidth']), 'C_pareto_locality_split',
                    {'subgraph': g, 'cut': c['cut'], 'boundary_bytes_per_60': c['boundary'] / cfg['bandwidth'],
                     'spill_relief': c['spill_relief']})
                break
    # D: choose the best valid target core for one critical bottleneck task.
    moves = []
    loads = [sum(info['observed_durations'][v] for v in s) for s in info['schedules']]
    movable = [task for task in info['critical_tasks'] if task in info['core_of']]
    if movable:
        # One observed bottleneck task and four target cores bound validation
        # and proxy work even when a case has thousands of critical tasks.
        task = max(movable, key=lambda v: (info['observed_durations'][v], info['finish'][v], -v))
        source = info['core_of'][task]
        for target in sorted((c for c in range(cores) if c != source), key=lambda c: (loads[c], c)):
            orders = [s[:] for s in info['schedules']]
            orders[source].remove(task); orders[target].append(task)
            orders[target].sort(key=lambda v: (info['finish'].get(v, 0), v))
            candidate = _plan(groups, orders)
            if _valid(graph, candidate):
                moves.append((_proxy(gm, groups, orders, cfg['bandwidth'], waits, info['observed_durations']),
                              task, source, target, candidate))
    if moves:
        _, task, source, target, candidate = min(moves, key=lambda item: (item[0], item[1], item[2], item[3]))
        add(candidate, 'D_critical_move_reinsert', {'task': task, 'from_core': source, 'to_core': target,
                                                    'selected_proxy': min(m[0] for m in moves)})
    return output, info

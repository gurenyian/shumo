"""Bounded, generic iterative neighbourhood refinement for five-core plans.

The official evaluator decides acceptance.  This module only builds and checks
small deterministic neighbourhoods from the current official winner.
"""
from collections import Counter
import heapq

from fast_solver import GraphModel, topo
from hybrid_solver import groups_from_plan, signature
from stub_multicore_cut_and_schedule import MulticoreCutError, derive_multicore_plan
from evaluation_validation import validate_task_order

CORES = 5
MAX_PROPOSALS = 8
MAX_MOVE_GAPS = 3


def _plan(groups, schedules):
    return {'node_to_subgraph': {str(v): i for i, g in enumerate(groups) for v in g},
            'core_schedules': [list(s) for s in schedules]}


def _durations(result, count):
    values = {}
    for core in result.get('per_core_timeline', []):
        for task in core.get('tasks', []):
            values[int(task['subgraph_id'])] = int(task['duration'])
    return [values.get(i, 0) for i in range(count)]


def _observed_schedule(pred, succ, durations, waits):
    """Linear list schedule using the incumbent's measured task durations.

    Gap insertion is useful on small quotients, but it scans an ever-growing
    core timeline.  This version is bounded by the quotient graph instead.
    """
    order = topo(pred, succ)
    rank = {}
    for task in reversed(order):
        rank[task] = durations[task] + max((rank[n] for n in succ[task]), default=0)
    degree = [len(item) for item in pred]
    ready = [(-rank[task], task) for task in order if not degree[task]]
    heapq.heapify(ready)
    schedules, free, finish, core_of = [[] for _ in range(CORES)], [0] * CORES, {}, {}
    same, cross = waits['task_same_core_wait_cycles'], waits['task_cross_core_wait_cycles']
    while ready:
        _, task = heapq.heappop(ready)
        ends = []
        for core in range(CORES):
            start = free[core] + (same if schedules[core] else 0)
            for parent in pred[task]:
                start = max(start, finish[parent] + (cross if core_of[parent] != core else 0))
            ends.append(start + durations[task])
        core = min(range(CORES), key=lambda item: (ends[item], item))
        schedules[core].append(task); free[core] = finish[task] = ends[core]; core_of[task] = core
        for child in sorted(succ[task]):
            degree[child] -= 1
            if not degree[child]:
                heapq.heappush(ready, (-rank[child], child))
    return schedules, {'proxy_makespan': max(finish.values(), default=0),
                       'scheduler': 'observed_linear_list'}


def _valid(graph, plan):
    eligible = {int(op['id']) for op in graph['ops']
                if op.get('op') not in {'COPY_IN', 'COPY_OUT'}}
    mapping = {int(v): int(g) for v, g in plan['node_to_subgraph'].items()}
    if set(mapping) != eligible or len(plan['core_schedules']) != CORES:
        return False
    flat = [g for order in plan['core_schedules'] for g in order]
    ids = sorted(set(mapping.values()))
    if sorted(flat) != ids or any(flat.count(g) != 1 for g in ids):
        return False
    try:
        validate_task_order(derive_multicore_plan(graph, plan))
    except (ValueError, KeyError, MulticoreCutError):
        return False
    return True


def _edge_info(gm, groups, schedules, durations, waits):
    """Build typed quotient edges and an observed critical path.

    Graph dependencies have zero delay.  Adjacent same-core schedule edges
    carry the configured same-core wait; cross-core dependencies carry the
    configured cross-core wait.  A same-core
    non-adjacent graph dependency remains a zero-delay dependency.
    """
    _, pred, succ, work, traffic, proxy = gm.describe(groups, 1)
    core_of = {g: c for c, order in enumerate(schedules) for g in order}
    adjacent = {(a, b) for order in schedules for a, b in zip(order, order[1:])}
    typed = []
    typed_pairs = set()
    for b in range(len(groups)):
        for a in sorted(pred[b]):
            if (a, b) in adjacent:
                delay, kind = waits['task_same_core_wait_cycles'], 'same_core_adjacent'
            elif core_of.get(a) != core_of.get(b):
                delay, kind = waits['task_cross_core_wait_cycles'], 'cross_core'
            else:
                delay, kind = 0, 'same_core_dependency'
            typed.append({'source': a, 'target': b, 'delay': delay, 'kind': kind})
            typed_pairs.add((a, b))
    # A core order is an augmented edge even when the original quotient graph
    # has no dependency between the adjacent tasks.
    for a, b in sorted(adjacent):
        if (a, b) not in typed_pairs:
            typed.append({'source': a, 'target': b,
                          'delay': waits['task_same_core_wait_cycles'],
                          'kind': 'same_core_adjacent'})
    ap = [set(v) for v in pred]; ass = [set(v) for v in succ]
    for edge in adjacent:
        a, b = edge
        ap[b].add(a); ass[a].add(b)
    order = topo(ap, ass)
    finish, parent = {}, {}
    for b in order:
        best = (0, -1)
        for a in sorted(ap[b]):
            delay = (waits['task_same_core_wait_cycles'] if (a, b) in adjacent
                     else (waits['task_cross_core_wait_cycles'] if core_of[a] != core_of[b] else 0))
            item = (finish[a] + delay, a)
            if item > best:
                best = item
        finish[b] = best[0] + durations[b]
        parent[b] = best[1]
    tail = max(order, key=lambda g: (finish[g], -g), default=None)
    critical = []
    while tail is not None and tail >= 0:
        critical.append(tail); tail = parent[tail]
    critical.reverse()
    return {'pred': pred, 'succ': succ, 'work': work, 'traffic': traffic,
            'proxy': proxy, 'core_of': core_of, 'adjacent': adjacent,
            'typed_edges': typed, 'finish': finish, 'critical': critical,
            'durations': durations, 'makespan_proxy': max(finish.values(), default=0)}


def analyze(graph, plan, result, bandwidth, waits):
    gm = GraphModel(graph); groups, schedules = groups_from_plan(plan)
    observed = _durations(result, len(groups))
    _, _, _, _, _, modeled = gm.describe(groups, bandwidth)
    durations = [observed[i] or modeled[i] for i in range(len(groups))]
    info = _edge_info(gm, groups, schedules, durations, waits)
    info.update(groups=groups, schedules=schedules, observed_durations=durations,
                bandwidth=bandwidth, waits=waits)
    return info


def _boundary_options(gm, members):
    """Rank topological prefix splits by their internal copy contribution.

    An internal tensor that crosses a new task boundary requires both the
    producer COPY_OUT and consumer COPY_IN.  The selected proposals below
    replace this local ranker with an exact whole-partition delta.
    """
    pos = {v: i for i, v in enumerate(members)}
    delta = [0] * (len(members) + 1)
    for tensor in gm.graph['tensors']:
        p = [pos[v] for v in gm.producers[tensor['id']] if v in pos]
        c = [pos[v] for v in gm.consumers[tensor['id']] if v in pos]
        if p and c and min(p) < max(c):
            delta[min(p) + 1] += tensor['size']; delta[max(c) + 1] -= tensor['size']
    work = Counter(); boundary = 0; options = []
    total = Counter()
    for v in members:
        total[gm.ops[v]['pipe']] += gm.ops[v]['cycles']
    for v in members[:-1]:
        work[gm.ops[v]['pipe']] += gm.ops[v]['cycles']
        boundary += 2 * delta[pos[v] + 1]
        right = total - work
        options.append({'cut': pos[v] + 1, 'boundary': boundary,
                        'added_copy_bytes': boundary,
                        'balance': max(max(work.values(), default=0), max(right.values(), default=0)),
                        'relief': max(total.values(), default=0) - max(max(work.values(), default=0), max(right.values(), default=0))})
    return options


def _partition_copy_bytes(gm, groups):
    """Match the evaluator's task-boundary COPY_IN/COPY_OUT byte count."""
    mapping = {op: group for group, members in enumerate(groups) for op in members}
    copy_bytes = 0
    for tensor in gm.graph['tensors']:
        producers = {mapping[op] for op in gm.producers[tensor['id']] if op in mapping}
        consumers = {mapping[op] for op in gm.consumers[tensor['id']] if op in mapping}
        has_copy_out = bool(gm.consumers[tensor['id']] & gm.copyouts)
        for group in producers | consumers:
            if group in consumers - producers:
                copy_bytes += tensor['size']
            if group in producers and (has_copy_out or not consumers or
                                       bool(consumers - {group})):
                copy_bytes += tensor['size']
    return copy_bytes


def _split(gm, info, gid, cut, reschedule=True):
    groups = [g[:] for g in info['groups']]
    members = sorted(groups[gid], key={v: i for i, v in enumerate(gm.order)}.__getitem__)
    groups[gid], right = members[:cut], members[cut:]
    groups.append(right); new = len(groups) - 1
    schedules = [s[:] for s in info['schedules']]
    for order in schedules:
        if gid in order:
            i = order.index(gid); order[i:i + 1] = [gid, new]; break
    if reschedule:
        _, pred, succ, _, _, duration = gm.describe(groups, info['bandwidth'])
        schedules, _ = _observed_schedule(pred, succ, duration, info['waits'])
    return _plan(groups, schedules)


def _move(gm, info, task, target):
    schedules = [s[:] for s in info['schedules']]
    source = info['core_of'][task]
    schedules[source].remove(task)
    # A full validator walks the graph.  Bound it to representative gaps near
    # the task's dependency window, plus each end, rather than validating every
    # gap of a large core schedule.
    output = []
    order = schedules[target]
    related = set(info['pred'][task]) | set(info['succ'][task])
    gaps = {0, len(order)}
    for index, other in enumerate(order):
        if other in related:
            gaps.update((index, index + 1))
    midpoint = len(order) // 2
    gaps.add(midpoint)
    ranked = sorted(gaps, key=lambda at: (min((abs(at - i) for i, other in enumerate(order)
                                               if other in related), default=0), at))
    for at in ranked[:MAX_MOVE_GAPS]:
        orders = [s[:] for s in schedules]; orders[target].insert(at, task)
        plan = _plan(info['groups'], orders)
        if _valid(gm.graph, plan): output.append((at, plan))
    return output


def _merge(gm, info, a, b):
    groups = [g[:] for g in info['groups']]
    groups[a].extend(groups[b]); groups.pop(b)
    schedules = [[(g - (1 if g > b else 0)) for g in s if g != b] for s in info['schedules']]
    return _plan(groups, schedules)


def candidates(graph, incumbent, result, cfg, waits, cores=CORES, limit=MAX_PROPOSALS):
    if cores != CORES or limit < 1: raise ValueError('iterative refinement is standalone five-core only')
    gm = GraphModel(graph); info = analyze(graph, incumbent, result, cfg['bandwidth'], waits)
    seen = {signature(incumbent)}; output = []
    def add(family, plan, detail, output):
        if len(output) >= min(limit, MAX_PROPOSALS) or signature(plan) in seen or not _valid(graph, plan): return
        groups, schedules = groups_from_plan(plan)
        if len(groups) == len(info['groups']):
            durations = info['observed_durations']
        else:
            durations = gm.describe(groups, cfg['bandwidth'])[5]
        proxy = _edge_info(gm, groups, schedules, durations, waits)['makespan_proxy']
        seen.add(signature(plan)); output.append({'family': family, 'plan': plan, 'detail': detail, 'proxy_makespan': proxy})
    by_family = [[], [], [], []]
    # Regenerate observed schedule from the current official winner.
    _, pred, succ, _, _, duration = gm.describe(info['groups'], cfg['bandwidth'])
    measured = info['observed_durations']
    scheduled, detail = _observed_schedule(pred, succ, measured, waits)
    add('observed_reschedule', _plan(info['groups'], scheduled), {'schedule': detail}, by_family[0])
    critical = info['critical']
    current_copy_bytes = _partition_copy_bytes(gm, info['groups'])
    # Critical split: rank all legal prefix cuts by pipe balance and boundary.
    for gid in critical:
        if len(info['groups'][gid]) < 3: continue
        members = sorted(info['groups'][gid], key={v: i for i, v in enumerate(gm.order)}.__getitem__)
        options = _boundary_options(gm, members)
        for option in sorted(options, key=lambda x: (x['balance'], x['boundary'], -x['relief'], x['cut']))[:3]:
            plan = _split(gm, info, gid, option['cut'])
            split_groups, _ = groups_from_plan(plan)
            exact = _partition_copy_bytes(gm, split_groups) - current_copy_bytes
            detail = {'subgraph': gid, **option, 'boundary': exact,
                      'added_copy_bytes': exact}
            add('critical_split', plan, detail, by_family[1])
        if len(by_family[1]) >= 3: break
    # Legal move/reinsert into every core gap for the hottest critical task.
    for task in sorted(critical, key=lambda x: (-info['durations'][x], -info['finish'][x], x))[:2]:
        for target in sorted(range(CORES), key=lambda c: (sum(info['durations'][x] for x in info['schedules'][c]), c)):
            if target == info['core_of'][task]: continue
            for at, plan in _move(gm, info, task, target):
                add('critical_move_reinsert', plan, {'task': task, 'to_core': target, 'gap': at}, by_family[2])
            if len(by_family[2]) >= 3: break
        if len(by_family[2]) >= 3: break
    # A small legal merge neighbourhood, ordered by boundary traffic.
    pairs = []
    for a in range(len(info['groups'])):
        for b in sorted(info['succ'][a]):
            pairs.append((info['traffic'][a] + info['traffic'][b], a, b))
    for _, a, b in sorted(pairs, reverse=True)[:4]:
        plan = _merge(gm, info, a, b)
        add('legal_partition_merge', plan, {'keep': a, 'drop': b}, by_family[3])
        if len(by_family[3]) >= 3: break
    # One candidate from each useful transformation precedes extra variants,
    # so a split-heavy critical path cannot consume the entire bounded budget.
    output = []
    while any(by_family) and len(output) < min(limit, MAX_PROPOSALS):
        for family in by_family:
            if family and len(output) < min(limit, MAX_PROPOSALS):
                output.append(family.pop(0))
    return output, info

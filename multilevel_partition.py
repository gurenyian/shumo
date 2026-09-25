"""Generic multilevel partitioning for arbitrary DAGs.

The official evaluator is the authority for scores.  This module only builds a
small deterministic set of structurally different plans from graph/config data.
"""
from collections import Counter

from adaptive_clustering import schedule_groups
from fast_solver import GraphModel, topo
from hybrid_solver import groups_from_plan, signature
from stub_multicore_cut_and_schedule import MulticoreCutError, derive_multicore_plan
from evaluation_validation import validate_task_order


def _quotient(gm, groups, bandwidth):
    _, pred, succ, work, traffic, duration = gm.describe(groups, bandwidth)
    order = topo(pred, succ)
    return pred, succ, work, traffic, duration, order


def _reachable_without(succ, source, target):
    """Exact path query excluding the direct edge source -> target."""
    stack = [v for v in succ[source] if v != target]
    seen = set(stack)
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for nxt in succ[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _flow_bytes(gm, mapping):
    flow = Counter()
    for tensor in gm.graph['tensors']:
        producers = {mapping[v] for v in gm.producers[tensor['id']] if v in mapping}
        consumers = {mapping[v] for v in gm.consumers[tensor['id']] if v in mapping}
        for a in producers:
            for b in consumers - {a}:
                flow[min(a, b), max(a, b)] += tensor['size']
    return flow


def _proxy(gm, groups, bandwidth, cores):
    pred, succ, work, traffic, duration, order = _quotient(gm, groups, bandwidth)
    finish = {}
    for task in order:
        finish[task] = duration[task] + max((finish[p] for p in pred[task]), default=0)
    ideal = sum(max(w.values(), default=0) for w in work) / max(cores, 1)
    overload = sum(max(0, max(w.values(), default=0) - ideal) for w in work)
    return (max(finish.values(), default=0) + overload,
            sum(traffic) / bandwidth, len(groups))


def _canonical(groups):
    return [sorted(group) for group in groups if group]


def _merge(groups, a, b):
    return _canonical([groups[i] + groups[b] if i == a else [] if i == b else group
                       for i, group in enumerate(groups)])


def _edge_delta(a, b, work, traffic, duration, flow, bandwidth, ideal, wait):
    """Cheap local form of the hierarchy objective for one legal edge."""
    merged_work = work[a] + work[b]
    boundary = flow[min(a, b), max(a, b)]
    merged_traffic = max(0, traffic[a] + traffic[b] - boundary)
    merged_duration = max(merged_work.values(), default=0) + merged_traffic / bandwidth
    path_increase = max(0, merged_duration - max(duration[a], duration[b]))
    old_over = max(0, max(work[a].values(), default=0) - ideal)
    old_over += max(0, max(work[b].values(), default=0) - ideal)
    new_over = max(0, max(merged_work.values(), default=0) - ideal)
    overload_increase = max(0, new_over - old_over)
    return boundary / bandwidth + wait - path_increase - overload_increase


def _legal_edges(gm, groups, bandwidth, cores, waits, checks=32):
    """Return a bounded set of beneficial, exactly safe contraction edges."""
    pred, succ, work, traffic, duration, _ = _quotient(gm, groups, bandwidth)
    flow = _flow_bytes(gm, {v: i for i, group in enumerate(groups) for v in group})
    ideal = sum(max(w.values(), default=0) for w in work) / cores
    offers = []
    for a in range(len(groups)):
        for b in sorted(succ[a]):
            score = _edge_delta(a, b, work, traffic, duration, flow, bandwidth, ideal,
                                waits['task_cross_core_wait_cycles'])
            if score > 0:
                offers.append((score, a, b))
    # Reachability is exact, but a hierarchy level examines only the strongest
    # constant-size set of locally beneficial edges.  This keeps construction
    # bounded on the largest supported DAGs without treating an untested edge
    # as safe.
    edges = [(a, b) for _, a, b in sorted(offers, key=lambda x: (-x[0], x[1], x[2]))[:checks]
             if not _reachable_without(succ, a, b)]
    return edges, pred, succ, work, traffic, duration


def _hierarchy(gm, cores, cfg, waits, seed_groups=None):
    groups = _canonical(seed_groups) if seed_groups else [[v] for v in gm.order]
    states = []
    seen = set()
    for _ in range(6):
        token = tuple(tuple(group) for group in groups)
        if token not in seen:
            states.append((groups, _proxy(gm, groups, cfg['bandwidth'], cores)))
            seen.add(token)
        edges, pred, succ, work, traffic, duration = _legal_edges(
            gm, groups, cfg['bandwidth'], cores, waits)
        flow = _flow_bytes(gm, {v: i for i, group in enumerate(groups) for v in group})
        ideal = sum(max(w.values(), default=0) for w in work) / max(cores, 1)
        scored = [(_edge_delta(a, b, work, traffic, duration, flow,
                              cfg['bandwidth'], ideal,
                              waits['task_cross_core_wait_cycles']), a, b)
                  for a, b in edges]
        if not scored:
            break
        used = set()
        chosen = []
        for score, a, b in sorted(scored, key=lambda x: (-x[0], x[1], x[2])):
            if a not in used and b not in used:
                chosen.append((a, b))
                used.update((a, b))
        if not chosen:
            break
        # Removing higher identifiers first keeps the remaining pair indices
        # valid while applying a matching in one hierarchy level.
        changed = False
        for a, b in sorted(chosen, key=lambda pair: (-max(pair), -min(pair))):
            a, b = sorted((a, b))
            if b < len(groups):
                trial = _merge(groups, a, b)
                try:
                    _quotient(gm, trial, cfg['bandwidth'])
                except ValueError:
                    continue
                groups = trial
                changed = True
        if not changed:
            break
    return states


def _local_variants(gm, groups, base, cores, cfg, waits):
    """Return a few deterministic, valid split and core-move neighbours.

    These are deliberately small local changes.  The official evaluator, not
    this proxy, decides whether any of them replaces the incumbent.
    """
    variants = []
    positions = {op: index for index, op in enumerate(gm.order)}

    # Split the largest contracted task along the original DAG topological
    # order, then reschedule all quotient tasks.
    splittable = [i for i, group in enumerate(groups) if len(group) > 1]
    if splittable:
        source = min(splittable, key=lambda i: (-len(groups[i]), i))
        ordered = sorted(groups[source], key=positions.__getitem__)
        mid = len(ordered) // 2
        trial = _canonical([group if i != source else ordered[:mid]
                            for i, group in enumerate(groups)] + [ordered[mid:]])
        try:
            variants.append(schedule_groups(gm, trial, cores, cfg, waits)[0])
        except (ValueError, MulticoreCutError):
            pass

    # Move one quotient task to each other core, preserving the quotient
    # topological order at the insertion point.  Validation rejects a move
    # whose new same-core ordering conflicts with a dependency.
    _, _, _, _, _, order = _quotient(gm, groups, cfg['bandwidth'])
    position = {task: index for index, task in enumerate(order)}
    schedules = base['core_schedules']
    movable = [(core, task) for core, schedule in enumerate(schedules)
               for task in schedule]
    for source, task in sorted(movable, key=lambda item: (-position[item[1]], item[0], item[1]))[:2]:
        for target in range(cores):
            if target == source:
                continue
            trial_schedules = [schedule[:] for schedule in schedules]
            trial_schedules[source].remove(task)
            insert_at = next((i for i, other in enumerate(trial_schedules[target])
                              if position[other] > position[task]), len(trial_schedules[target]))
            trial_schedules[target].insert(insert_at, task)
            plan = {'node_to_subgraph': dict(base['node_to_subgraph']),
                    'core_schedules': trial_schedules}
            try:
                validate_task_order(derive_multicore_plan(gm.graph, plan))
            except (ValueError, MulticoreCutError):
                continue
            variants.append(plan)
    return variants


def candidates(graph, incumbent, cfg, waits, cores=5, limit=3):
    """Build <= limit deterministic plans, retaining only valid distinct plans."""
    if limit < 1 or cores < 1:
        raise ValueError('limit and cores must be positive')
    gm = GraphModel(graph)
    incumbent_groups, _ = groups_from_plan(incumbent)
    # A five-core incumbent is already a compact valid quotient.  Starting
    # there avoids constructing and list-scheduling a one-task-per-operation
    # plan for very large DAGs; local splits provide the bounded refinement
    # direction while contraction supplies the coarsening direction.
    states = _hierarchy(gm, cores, cfg, waits, incumbent_groups)
    nondominated = []
    for state in states:
        value = state[1]
        if any(all(other[1][i] <= value[i] for i in range(3)) and
               any(other[1][i] < value[i] for i in range(3))
               for other in states if other is not state):
            continue
        nondominated.append(state)
    nondominated.sort(key=lambda item: (item[1][0], item[1][1], item[1][2]))
    plans = []
    seen = {signature(incumbent)}
    for groups, _ in nondominated:
        try:
            plan = schedule_groups(gm, groups, cores, cfg, waits)[0]
        except (ValueError, MulticoreCutError):
            continue
        # The first hierarchy state supplies bounded split/move neighbours;
        # later hierarchy levels remain useful distinct coarsenings.
        pool = _local_variants(gm, groups, plan, cores, cfg, waits) + [plan] if not plans else [plan]
        for candidate in pool:
            if signature(candidate) in seen:
                continue
            try:
                validate_task_order(derive_multicore_plan(graph, candidate))
            except (ValueError, MulticoreCutError):
                continue
            seen.add(signature(candidate))
            plans.append(candidate)
            if len(plans) == limit:
                return plans
    return plans

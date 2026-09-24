"""Communication-aware acyclic clustering and insertion list scheduling.

The cost model only chooses candidates. Final acceptance always uses the
contest evaluator, including memory capacities and shared DDR bandwidth.
"""

import heapq
import math
from collections import Counter

from fast_solver import Groups, make_plan, topo
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order


def insertion_schedule(pred, succ, duration, cores, waits):
    """Schedule ready tasks into the earliest feasible gap on each core.

    A gap must leave the same-core task-switch delay on both sides. Dependency
    release times include cross-core synchronization. The schedule is still a
    proxy: changing DDR contention can change observed task durations.
    """
    if cores < 1 or len(pred) != len(duration):
        raise ValueError("invalid core count or task durations")
    order = topo(pred, succ)
    rank = {}
    for v in reversed(order):
        rank[v] = duration[v] + max((rank[w] for w in succ[v]), default=0)
    degree = [len(p) for p in pred]
    ready = [(-rank[v], v) for v in order if not degree[v]]
    heapq.heapify(ready)
    slots = [[] for _ in range(cores)]
    assigned, finish = {}, {}
    same = waits['task_same_core_wait_cycles']
    cross = waits['task_cross_core_wait_cycles']
    inserted = 0
    while ready:
        _, v = heapq.heappop(ready)
        best = None
        for c in range(cores):
            release = max((finish[p] + (cross if assigned[p] != c else 0)
                           for p in pred[v]), default=0)
            after = 0
            for index in range(len(slots[c]) + 1):
                start = max(release, after)
                end = start + duration[v]
                if index == len(slots[c]) or end + same <= slots[c][index][0]:
                    candidate = (end, start, c, index)
                    if best is None or candidate < best:
                        best = candidate
                    break
                after = slots[c][index][1] + same
        end, start, core, index = best
        inserted += index < len(slots[core])
        slots[core].insert(index, (start, end, v))
        assigned[v], finish[v] = core, end
        for w in sorted(succ[v]):
            degree[w] -= 1
            if not degree[w]:
                heapq.heappush(ready, (-rank[w], w))
    return [[v for _, _, v in entries] for entries in slots], {
        'insertions_before_existing_tasks': inserted,
        'proxy_makespan': max(finish.values(), default=0)}


def schedule_groups(gm, groups, cores, cfg, waits, observed=None):
    _, pred, succ, _, _, proxy = gm.describe(groups, cfg['bandwidth'])
    durations = proxy if observed is None else observed
    schedules, detail = insertion_schedule(pred, succ, durations, cores, waits)
    plan = make_plan(groups, schedules)
    validate_task_order(derive_multicore_plan(gm.graph, plan))
    return plan, detail


def adaptive_partition(gm, cores, cfg, waits, parallel_penalty=1.0, target_scale=1.0):
    """Greedily contract safe edges while preserving useful branch parallelism.

    A sufficient cycle check permits u->v when u has one successor OR v has
    one predecessor. Unlike whole-join aggregation, this allows partial joins.
    The score rewards saved tensor traffic/task switches and penalizes packing
    originally overlapping work into one task. Pipe workload bounds cluster
    size. Dynamic scores are updated only around each contraction.
    """
    if parallel_penalty < 0 or target_scale <= 0:
        raise ValueError('invalid clustering parameters')
    overhead = waits['task_cross_core_wait_cycles']
    share = max(gm.pipes.values(), default=0) / cores
    target = max(min(share, target_scale * math.sqrt(max(1, overhead) * share)),
                 max((o['cycles'] for o in gm.ops.values()), default=1))
    groups = Groups({v: [v] for v in gm.order}, gm.pred, gm.succ,
                    {v: Counter({gm.ops[v]['pipe']: gm.ops[v]['cycles']}) for v in gm.order})
    first = {v: gm.critical[v] - gm.ops[v]['cycles'] for v in gm.order}
    last = dict(gm.critical)
    affinity = {v: {} for v in gm.order}
    for tensor in gm.graph['tensors']:
        producers = gm.producers[tensor['id']] & gm.ops.keys()
        consumers = gm.consumers[tensor['id']] & gm.ops.keys()
        for a in producers:
            for b in consumers:
                if a != b:
                    affinity[a][b] = affinity[a].get(b, 0) + tensor['size']
                    affinity[b][a] = affinity[b].get(a, 0) + tensor['size']
    versions = {v: 0 for v in gm.order}
    heap = []

    def loss(v):
        return max(0, max(groups.work[v].values(), default=0) - (last[v] - first[v]))

    def offer(a, b):
        if a == b or a not in groups.members or b not in groups.members or not groups.safe(a, b):
            return
        work = groups.work[a] + groups.work[b]
        maximum = max(work.values(), default=0)
        if maximum > target:
            return
        merged_loss = max(0, maximum - (max(last[a], last[b]) - min(first[a], first[b])))
        penalty = max(0, merged_loss - loss(a) - loss(b))
        saved = affinity[a].get(b, 0) / cfg['bandwidth'] + overhead
        benefit = saved - parallel_penalty * penalty
        if benefit > 0:
            priority = benefit / (1 + maximum / max(target, 1))
            heapq.heappush(heap, (-priority, a, b, versions[a], versions[b]))

    for a in gm.order:
        for b in sorted(groups.succ[a]):
            offer(a, b)
    merges = 0
    while heap:
        _, a, b, va, vb = heapq.heappop(heap)
        if (a not in groups.members or b not in groups.members or
                versions[a] != va or versions[b] != vb or not groups.safe(a, b)):
            continue
        neighbors = (groups.pred[a] | groups.pred[b] | groups.succ[a] | groups.succ[b]) - {a, b}
        weights = Counter(affinity[a]) + Counter(affinity[b])
        weights.pop(a, None)
        weights.pop(b, None)
        begin, end = min(first[a], first[b]), max(last[a], last[b])
        keep = groups.merge(a, b)
        drop = b if keep == a else a
        versions[keep] += 1
        first[keep], last[keep] = begin, end
        for other, weight in weights.items():
            affinity[other].pop(a, None)
            affinity[other].pop(b, None)
            affinity[other][keep] = weight
        affinity[keep] = dict(weights)
        affinity.pop(drop)
        merges += 1
        # New edges touch keep; changed neighbor degrees can also make a
        # previously blocked edge between that neighbor and a third task safe.
        for v in sorted(neighbors | {keep}):
            for p in sorted(groups.pred[v]):
                offer(p, v)
            for s in sorted(groups.succ[v]):
                offer(v, s)
    members = [groups.members[v] for v in sorted(groups.members)]
    plan, scheduling = schedule_groups(gm, members, cores, cfg, waits)
    return plan, {'algorithm': 'communication_parallelism_clustering',
                  'parallel_penalty': parallel_penalty, 'pipe_work_target': target,
                  'tasks': len(members), 'merged_edges': merges, **scheduling}

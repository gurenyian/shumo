"""Dependency-region partition for problem 1.

Each operation is labelled by its reachable source and sink operations. Along
every DAG edge, the source set can only grow and the sink set can only shrink.
Grouping equal labels therefore preserves an acyclic quotient. Large labels
are split in topological order by pipe work, keeping the same property.
"""

from collections import Counter, defaultdict

from fast_solver import coalesce_on_cores, list_schedule, make_plan
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order


def region_partition(gm, cores, cfg, waits, work_divisor=4, label_mode="both"):
    if work_divisor <= 0:
        raise ValueError("work_divisor must be positive")
    if label_mode not in ("both", "sources", "sinks"):
        raise ValueError("label_mode must be both, sources or sinks")
    roots = [v for v in gm.order if not gm.pred[v]]
    leaves = [v for v in gm.order if not gm.succ[v]]
    root_bit = {v: 1 << i for i, v in enumerate(roots)}
    leaf_bit = {v: 1 << i for i, v in enumerate(leaves)}
    ancestors = {}
    for v in gm.order:
        bits = root_bit.get(v, 0)
        for p in gm.pred[v]:
            bits |= ancestors[p]
        ancestors[v] = bits
    descendants = {}
    for v in reversed(gm.order):
        bits = leaf_bit.get(v, 0)
        for s in gm.succ[v]:
            bits |= descendants[s]
        descendants[v] = bits

    target = max(max(gm.pipes.values(), default=0) / max(1, cores * work_divisor),
                 max((op['cycles'] for op in gm.ops.values()), default=1))
    groups, group_work = [], []
    current = {}
    for v in gm.order:
        label = ((ancestors[v], descendants[v]) if label_mode == "both" else
                 ancestors[v] if label_mode == "sources" else descendants[v])
        pipe, cycles = gm.ops[v]['pipe'], gm.ops[v]['cycles']
        i = current.get(label)
        if i is None or group_work[i][pipe] + cycles > target:
            i = len(groups)
            current[label] = i
            groups.append([])
            group_work.append(Counter())
        groups[i].append(v)
        group_work[i][pipe] += cycles

    schedules, decisions = list_schedule(gm, groups, cores, cfg, waits)
    groups, schedules, merges = coalesce_on_cores(gm, groups, schedules, cfg['bandwidth'])
    plan = make_plan(groups, schedules)
    validate_task_order(derive_multicore_plan(gm.graph, plan))
    return plan, {"algorithm": "dependency_regions", "label_mode": label_mode,
                  "roots": len(roots),
                  "leaves": len(leaves), "work_target": target,
                  "tasks": len(groups), "core_tasks": [len(x) for x in schedules],
                  "same_core_merges": len(merges)}

"""第一问实验入口。仅依赖 Python 标准库，评分直接调用未修改的官方代码。"""
import argparse
import csv
import heapq
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'official' / 'code'))
from stub_multicore_cut_and_schedule import (
    _build_op_adjacency, _contract_excluded_copy_nodes, generate_multicore_plan)
from evaluation_validation import validate_graph, read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from contest_io import _read_json, format_scene_a_trace_json


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def model(graph):
    """COPY 节点收缩时保留依赖；不能简单删除所有 COPY 边。"""
    validate_graph(graph)
    ops = {o['id']: o for o in graph['ops'] if o['op'] not in {'COPY_IN', 'COPY_OUT'}}
    _, full = _build_op_adjacency(graph)
    pred, succ = _contract_excluded_copy_nodes(sorted(ops), full)
    degree = {v: len(pred[v]) for v in ops}
    ready = [v for v in ops if not degree[v]]
    heapq.heapify(ready)
    order, depth, critical = [], {}, {}
    while ready:
        v = heapq.heappop(ready)
        order.append(v)
        depth[v] = 1 + max((depth[p] for p in pred[v]), default=0)
        critical[v] = ops[v]['cycles'] + max((critical[p] for p in pred[v]), default=0)
        for w in sorted(succ[v]):
            degree[w] -= 1
            if degree[w] == 0:
                heapq.heappush(ready, w)
    # 无向连通分量：不同分量没有计算依赖，可作为保留局部性的自然块。
    unseen, components = set(ops), []
    for start in order:
        if start not in unseen:
            continue
        unseen.remove(start)
        stack, component = [start], []
        while stack:
            v = stack.pop()
            component.append(v)
            for w in sorted(pred[v] | succ[v]):
                if w in unseen:
                    unseen.remove(w)
                    stack.append(w)
        components.append(component)
    return ops, pred, succ, order, depth, critical, components


def inventory():
    rows = []
    for path in sorted((ROOT / 'official/data').glob('case_*.json')):
        g = _read_json(path)
        ops, pred, succ, order, depth, critical, components = model(g)
        pipes = Counter()
        for op in ops.values():
            pipes[op['pipe']] += op['cycles']
        rows.append(dict(case=path.stem, ops=len(g['ops']), compute_ops=len(ops),
                         tensors=len(g['tensors']), edges=len(g['edges']),
                         components=len(components), largest_component=max(map(len, components), default=0),
                         max_layer_width=max(Counter(depth.values()).values(), default=0),
                         critical_compute_cycles=max(critical.values(), default=0),
                         total_compute_cycles=sum(pipes.values()), **dict(pipes)))
    out = ROOT / 'results' / 'experiments' / 'inventory.csv'
    out.parent.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with out.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save(ROOT / 'results' / 'experiments' / 'inventory.json', rows)
    print(json.dumps({'cases': len(rows), 'compute_ops_range': [min(r['compute_ops'] for r in rows), max(r['compute_ops'] for r in rows)],
                      'components_range': [min(r['components'] for r in rows), max(r['components'] for r in rows)]}))


def schedule(graph, groups, cores, cfg, waits):
    """先调度依赖就绪且剩余路径长的子图，再选择预计完成最早的核。"""
    ops, pred, succ, order, *_ = model(graph)
    mapping = {v: i for i, group in enumerate(groups) for v in group}
    gp = [set() for _ in groups]
    gs = [set() for _ in groups]
    for v in ops:
        for w in succ[v]:
            a, b = mapping[v], mapping[w]
            if a != b:
                gp[b].add(a)
                gs[a].add(b)
    # 代理耗时：各 Pipe 工作量最大值 + 边界搬运量/DDR带宽。
    # 它不模拟流水重叠和拥塞，最终方案必须通过官方评分选取。
    work = [Counter() for _ in groups]
    for v, op in ops.items():
        work[mapping[v]][op['pipe']] += op['cycles']
    producers, consumers = defaultdict(set), defaultdict(set)
    allops = {o['id'] for o in graph['ops']}
    copyouts = {o['id'] for o in graph['ops'] if o['op'] == 'COPY_OUT'}
    for e in graph['edges']:
        a, b = e['source'], e['target']
        if a in allops and b not in allops:
            producers[b].add(a)
        elif a not in allops and b in allops:
            consumers[a].add(b)
    traffic = [0] * len(groups)
    for t in graph['tensors']:
        p = {mapping[v] for v in producers[t['id']] if v in mapping}
        c = {mapping[v] for v in consumers[t['id']] if v in mapping}
        for group in c - p:
            traffic[group] += t['size']
        for group in p:
            if c - {group} or not c or consumers[t['id']] & copyouts:
                traffic[group] += t['size']
    duration = [max(w.values(), default=0) + traffic[i] / cfg['bandwidth'] for i, w in enumerate(work)]
    degree = [len(p) for p in gp]
    ready = [i for i, d in enumerate(degree) if d == 0]
    topo = []
    while ready:
        v = ready.pop()
        topo.append(v)
        for w in sorted(gs[v]):
            degree[w] -= 1
            if not degree[w]:
                ready.append(w)
    if len(topo) != len(groups):
        raise ValueError('partition introduced a cycle')
    rank = {}
    for v in reversed(topo):
        rank[v] = duration[v] + max((rank[w] for w in gs[v]), default=0)
    degree = [len(p) for p in gp]
    ready = [(-rank[i], i) for i, d in enumerate(degree) if d == 0]
    heapq.heapify(ready)
    schedules, free, finish, assigned = [[] for _ in range(cores)], [0] * cores, {}, {}
    while ready:
        _, v = heapq.heappop(ready)
        options = []
        for core in range(cores):
            release = free[core] + (waits['task_same_core_wait_cycles'] if schedules[core] else 0)
            for p in gp[v]:
                release = max(release, finish[p] + (waits['task_cross_core_wait_cycles'] if assigned[p] != core else 0))
            options.append((release + duration[v], core))
        end, core = min(options)
        schedules[core].append(v)
        free[core] = finish[v] = end
        assigned[v] = core
        for w in sorted(gs[v]):
            degree[w] -= 1
            if not degree[w]:
                heapq.heappush(ready, (-rank[w], w))
    return {'node_to_subgraph': {str(v): mapping[v] for v in sorted(mapping)}, 'core_schedules': schedules}


def solve(case, cores):
    path = ROOT / 'official/data' / (case + '.json')
    graph = _read_json(path)
    cfg = read_evaluation_config(str(ROOT / 'official/data/config.txt'))
    waits = read_scene_a_config(str(ROOT / 'official/data/config.txt'))
    ops, pred, succ, order, depth, _, components = model(graph)
    out = ROOT / 'results' / 'experiments' / f'{case}_{cores}cores'
    candidates = [('one_task', {'node_to_subgraph': {str(v): 0 for v in ops}, 'core_schedules': [[0]] + [[] for _ in range(cores - 1)]}),
                  ('official_random_example', generate_multicore_plan(graph, num_cores=cores, seed=0))]
    candidates.append(('components', schedule(graph, components, cores, cfg, waits)))
    for count in (cores, cores * 4):
        bins = [[] for _ in range(min(count, len(components))) ]
        loads = [0] * len(bins)
        for component in sorted(components, key=lambda c: (-sum(ops[v]['cycles'] for v in c), min(c))):
            i = min(range(len(bins)), key=lambda j: (loads[j], j))
            bins[i].extend(component)
            loads[i] += sum(ops[v]['cycles'] for v in component)
        candidates.append((f'component_bins_{count}', schedule(graph, bins, cores, cfg, waits)))
    # 连续拓扑块保证收缩图无环。层内分块提供另一个并行性较强的对照。
    # 粒度扫描：连通图只试两种块长容易错过更合适的切点。
    for size in (16, 32, 48, 64, 96, 128, 192, 256):
        groups = [order[i:i + size] for i in range(0, len(order), size)]
        candidates.append((f'topo_{size}', schedule(graph, groups, cores, cfg, waits)))
    layers = defaultdict(list)
    for v in order:
        layers[depth[v]].append(v)
    groups = [layer[i:i + 32] for _, layer in sorted(layers.items()) for i in range(0, len(layer), 32)]
    candidates.append(('layer_32', schedule(graph, groups, cores, cfg, waits)))
    rows, best, seen = [], None, set()
    for name, plan in candidates:
        signature = json.dumps(plan, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        save(out / f'{name}_plan.json', plan)
        start = time.perf_counter()
        try:
            result = evaluate_scene_a(graph, plan, bandwidth=cfg['bandwidth'], capacity=cfg['capacity'],
                                      cross_core_wait=waits['task_cross_core_wait_cycles'], same_core_wait=waits['task_same_core_wait_cycles'])
            row = dict(candidate=name, valid=True, tasks=len(set(plan['node_to_subgraph'].values())),
                       makespan=result['makespan'], **result['data_movement_bytes'])
            key = (result['makespan'], result['data_movement_bytes']['added_copy_bytes'])
            if best is None or key < best[0]:
                best = (key, name, plan, result)
        except (ValueError, RuntimeError) as error:
            row = dict(candidate=name, valid=False, error=str(error))
        row['seconds'] = round(time.perf_counter() - start, 3)
        rows.append(row)
        save(out / 'comparison.json', rows)
        print(case, cores, row, flush=True)
    if best is None:
        raise RuntimeError('No feasible candidate; see comparison.json')
    save(out / 'best_plan.json', best[2])
    save(out / 'best_result.json', best[3])
    # 每核同一 Pipe 同时只执行一个操作，因此总计算量/核心数是严格下界。
    pipe_work = Counter()
    for op in ops.values():
        pipe_work[op['pipe']] += op['cycles']
    pipe_lower_bound = max((value / cores for value in pipe_work.values()), default=0)
    save(out / 'selection.json', {'candidate': best[1], 'makespan': best[0][0],
                                  'pipe_work_lower_bound': pipe_lower_bound,
                                  'gap_to_pipe_bound_percent': round(100 * (best[0][0] / pipe_lower_bound - 1), 3) if pipe_lower_bound else None})
    (out / 'best_trace.json').write_text(format_scene_a_trace_json(path.name, best[3]), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'solve'])
    parser.add_argument('--case', default='case_001')
    parser.add_argument('--cores', type=int, choices=[2, 3, 4, 5], default=4)
    args = parser.parse_args()
    inventory() if args.action == 'inventory' else solve(args.case, args.cores)

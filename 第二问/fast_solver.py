"""第一问：结构感知的有界聚合、关键路径列表调度、同核安全合并。

构造一个多核方案；廉价估算只提示是否值得与单 Task 做官方评分比较。
无随机搜索，无块长枚举；详见 快速算法说明.md。
"""
import argparse
import heapq
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

from experiment import (ROOT, model, save, _read_json, read_evaluation_config,
                        read_scene_a_config)
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order


class Groups:
    """维护收缩图。仅收缩无替代路径的边，避免把 DAG 合成环。"""
    def __init__(self, members, pred, succ, work):
        self.members = {i: list(v) for i, v in members.items()}
        self.pred = {i: set(v) for i, v in pred.items()}
        self.succ = {i: set(v) for i, v in succ.items()}
        self.work = {i: Counter(v) for i, v in work.items()}
        self.parent = {i: i for i in members}

    def root(self, v):
        r = v
        while self.parent[r] != r:
            r = self.parent[r]
        while self.parent[v] != v:
            next_v = self.parent[v]
            self.parent[v] = r
            v = next_v
        return r

    def safe(self, a, b):
        # 如果 a 只有 b 这一个后继，或者 b 只有 a 这一个前驱，
        # 不可能还有 a -> ... -> b 的替代路径，因而收缩不会引入环。
        return b in self.succ[a] and (len(self.succ[a]) == 1 or len(self.pred[b]) == 1)

    def merge(self, a, b):
        assert self.safe(a, b)
        # 保留成员多的一端作为并查集代表，减少成员列表反复复制。
        keep, drop = (a, b) if len(self.members[a]) >= len(self.members[b]) else (b, a)
        incoming = (self.pred[a] | self.pred[b]) - {a, b}
        outgoing = (self.succ[a] | self.succ[b]) - {a, b}
        for p in incoming:
            self.succ[p].difference_update({a, b})
            self.succ[p].add(keep)
        for s in outgoing:
            self.pred[s].difference_update({a, b})
            self.pred[s].add(keep)
        self.pred[keep], self.succ[keep] = incoming, outgoing
        self.members[keep].extend(self.members.pop(drop))
        self.work[keep].update(self.work.pop(drop))
        self.pred.pop(drop)
        self.succ.pop(drop)
        self.parent[drop] = keep
        return keep


class GraphModel:
    def __init__(self, graph):
        self.graph = graph
        (self.ops, self.pred, self.succ, self.order, self.depth,
         self.critical, self.components) = model(graph)
        self.pipes = Counter()
        for op in self.ops.values():
            self.pipes[op['pipe']] += op['cycles']
        self.producers, self.consumers = defaultdict(set), defaultdict(set)
        allops = {o['id'] for o in graph['ops']}
        self.copyouts = {o['id'] for o in graph['ops'] if o['op'] == 'COPY_OUT'}
        for edge in graph['edges']:
            a, b = edge['source'], edge['target']
            if a in allops and b not in allops:
                self.producers[b].add(a)
            elif a not in allops and b in allops:
                self.consumers[a].add(b)

    def describe(self, groups, bandwidth):
        """按张量的生产者/消费者子图集合计算边界字节，避免一组内重复计数。"""
        mapping = {v: i for i, group in enumerate(groups) for v in group}
        pred, succ = [set() for _ in groups], [set() for _ in groups]
        work, traffic = [Counter() for _ in groups], [0 for _ in groups]
        for v, op in self.ops.items():
            a = mapping[v]
            work[a][op['pipe']] += op['cycles']
            for w in self.succ[v]:
                b = mapping[w]
                if a != b:
                    pred[b].add(a)
                    succ[a].add(b)
        for t in self.graph['tensors']:
            p = {mapping[v] for v in self.producers[t['id']] if v in mapping}
            c = {mapping[v] for v in self.consumers[t['id']] if v in mapping}
            for group in c - p:
                traffic[group] += t['size']
            for group in p:
                if c - {group} or not c or self.consumers[t['id']] & self.copyouts:
                    traffic[group] += t['size']
        duration = [max(w.values(), default=0) + b / bandwidth for w, b in zip(work, traffic)]
        return mapping, pred, succ, work, traffic, duration


def topo(pred, succ):
    degree = [len(p) for p in pred]
    ready = [i for i, d in enumerate(degree) if not d]
    heapq.heapify(ready)
    order = []
    while ready:
        v = heapq.heappop(ready)
        order.append(v)
        for w in sorted(succ[v]):
            degree[w] -= 1
            if not degree[w]:
                heapq.heappush(ready, w)
    if len(order) != len(pred):
        raise ValueError('partition/task order contains a cycle')
    return order


def coarsen(gm, cores, waits, target_scale=1.0):
    """先保留小的完整分量，再沿拓扑序聚合串行链和完整汇合子树。"""
    # 代理额外代价 t + H * W/(N*t)：块太大损失均衡，块太小累积等待。
    # 连续最小点为 sqrt(H*W/N)，无需扫描块长。它不是实际 Makespan 公式。
    ideal_share = max(gm.pipes.values(), default=0) / cores
    overhead = waits['task_same_core_wait_cycles'] + waits['task_cross_core_wait_cycles']
    target = max(min(ideal_share, target_scale * math.sqrt(overhead * ideal_share)),
                 max((o['cycles'] for o in gm.ops.values()), default=0))
    initial, mapping = {}, {}
    for component in gm.components:
        work = Counter()
        for v in component:
            work[gm.ops[v]['pipe']] += gm.ops[v]['cycles']
        pieces = [component] if max(work.values(), default=0) <= target else [[v] for v in component]
        for piece in pieces:
            root = min(piece)
            initial[root] = sorted(piece)
            mapping.update({v: root for v in piece})
    pred, succ, work = {i: set() for i in initial}, {i: set() for i in initial}, {i: Counter() for i in initial}
    for v, op in gm.ops.items():
        a = mapping[v]
        work[a][op['pipe']] += op['cycles']
        for w in gm.succ[v]:
            b = mapping[w]
            if a != b:
                pred[b].add(a)
                succ[a].add(b)
    groups = Groups(initial, pred, succ, work)
    reasons = Counter()
    log = []
    for v in gm.order:
        b = groups.root(mapping[v])
        incoming = sorted(groups.pred[b])
        if not incoming:
            continue
        # 汇合节点要么与所有独占前驱一起合并，要么保持独立。
        # 只吞并某一个分支会让整个大 Task 等待其他分支，扩大同步范围。
        if any(groups.succ[a] != {b} for a in incoming):
            reasons['preserve_shared_branch'] += 1
            continue
        merged_work = groups.work[b].copy()
        for a in incoming:
            merged_work.update(groups.work[a])
        if max(merged_work.values(), default=0) > target:
            reasons['would_exceed_work_target'] += 1
            continue
        log.append({'target_op': v, 'predecessor_groups': incoming,
                    'merged_pipe_work_max': max(merged_work.values(), default=0),
                    'reason': 'whole_join' if len(incoming) > 1 else 'exclusive_chain'})
        for a in incoming:
            b = groups.merge(groups.root(a), groups.root(b))
            reasons['merged'] += 1
    result = [sorted(groups.members[i]) for i in sorted(groups.members)]
    return result, {'pipe_work_target': target, 'target_scale': target_scale,
                    'ideal_pipe_share': ideal_share,
                    'waiting_cost_proxy': overhead, 'initial_component_groups': len(initial),
                    'aggregation_decisions': dict(reasons), 'merges': log}


def list_schedule(gm, groups, cores, cfg, waits):
    mapping, pred, succ, work, traffic, duration = gm.describe(groups, cfg['bandwidth'])
    order = topo(pred, succ)
    rank = {}
    for v in reversed(order):
        rank[v] = duration[v] + max((rank[w] for w in succ[v]), default=0)
    degree = [len(p) for p in pred]
    ready = [(-rank[i], i) for i, d in enumerate(degree) if not d]
    heapq.heapify(ready)
    schedules, free, finish, assigned = [[] for _ in range(cores)], [0] * cores, {}, {}
    log = []
    while ready:
        _, v = heapq.heappop(ready)
        ends = []
        for c in range(cores):
            start = free[c] + (waits['task_same_core_wait_cycles'] if schedules[c] else 0)
            for p in pred[v]:
                start = max(start, finish[p] + (waits['task_cross_core_wait_cycles'] if assigned[p] != c else 0))
            ends.append(start + duration[v])
        c = min(range(cores), key=lambda c: (ends[c], c))
        schedules[c].append(v)
        free[c] = finish[v] = ends[c]
        assigned[v] = c
        log.append({'subgraph': v, 'chosen_core': c, 'estimated_finish_by_core': ends,
                    'rank': rank[v], 'boundary_bytes': traffic[v]})
        for w in sorted(succ[v]):
            degree[w] -= 1
            if not degree[w]:
                heapq.heappush(ready, (-rank[w], w))
    return schedules, log


def coalesce_on_cores(gm, members, schedules, bandwidth):
    """同核连续 Task 可安全合并时执行，减少重复输入和 Task 切换。

    安全检查必须同时纳入计算依赖与各核心 Task 顺序，不能只看原始图。
    """
    _, pred, succ, work, *_ = gm.describe(members, bandwidth)
    for order in schedules:
        for a, b in zip(order, order[1:]):
            pred[b].add(a)
            succ[a].add(b)
    groups = Groups(dict(enumerate(members)), dict(enumerate(pred)), dict(enumerate(succ)), dict(enumerate(work)))
    decisions = []
    for core, order in enumerate(schedules):
        for v, w in zip(order, order[1:]):
            a, b = groups.root(v), groups.root(w)
            # 无环仅是可行性要求：有外部前驱/后继时合并会把等待扩大到
            # 整个 Task，可能损失并行。只合并组合图上的独占串行边。
            if a != b and groups.succ[a] == {b} and groups.pred[b] == {a}:
                groups.merge(a, b)
                decisions.append({'core': core, 'previous_task': v, 'next_task': w})
    roots = sorted(groups.members)
    number = {root: i for i, root in enumerate(roots)}
    final_members = [sorted(groups.members[root]) for root in roots]
    final_schedules = []
    for order in schedules:
        compact = []
        for v in order:
            i = number[groups.root(v)]
            if not compact or compact[-1] != i:
                compact.append(i)
        final_schedules.append(compact)
    return final_members, final_schedules, decisions


def make_plan(groups, schedules):
    return {'node_to_subgraph': {str(v): i for i, group in enumerate(groups) for v in group},
            'core_schedules': schedules}


def independent_component_plan(gm, cores):
    """Keep disconnected components intact and balance their operation work.

    This gives a cheap, dependency-safe alternative when structural coarsening
    accidentally joins otherwise parallel branches into topological blocks.
    """
    groups = [sorted(component) for component in gm.components]
    schedules = [[] for _ in range(cores)]
    loads = [0] * cores
    order = sorted(range(len(groups)),
                   key=lambda i: (-sum(gm.ops[v]['cycles'] for v in groups[i]), i))
    for i in order:
        core = min(range(cores), key=lambda c: (loads[c], c))
        schedules[core].append(i)
        loads[core] += sum(gm.ops[v]['cycles'] for v in groups[i])
    return make_plan(groups, schedules)


def bound_fragmentation(gm, groups, target, cores):
    """复杂分叉使安全聚合留下大量碎片时，使用计算量驱动的拓扑块。

    这是一条结构触发规则，不对两种划分分别评分或搜索块长。
    """
    budget = max(cores, math.ceil(sum(gm.pipes.values()) / max(target, 1)) * cores)
    # 全部都是完整独立分量时，同核合并会自然消除碎片，无需拓扑切块。
    if len(groups) <= budget or len(groups) == len(gm.components):
        return groups, {'used': False, 'group_budget': budget}
    bounded, current, work = [], [], Counter()
    for v in gm.order:
        op = gm.ops[v]
        if current and work[op['pipe']] + op['cycles'] > target:
            bounded.append(current)
            current, work = [], Counter()
        current.append(v)
        work[op['pipe']] += op['cycles']
    if current:
        bounded.append(current)
    return bounded, {'used': True, 'group_budget': budget, 'before_groups': len(groups),
                     'after_groups': len(bounded), 'reason': 'too_many_small_structural_fragments'}


def estimate(gm, groups, schedules, bandwidth, waits):
    """固定方案的粗略时间；不含真实 DDR 拥塞、Spill 和 Pipe 重叠。"""
    _, pred, succ, _, traffic, duration = gm.describe(groups, bandwidth)
    core_of = {v: c for c, order in enumerate(schedules) for v in order}
    previous = {}
    for order in schedules:
        for a, b in zip(order, order[1:]):
            previous[b] = a
            pred[b].add(a)
            succ[a].add(b)
    end = {}
    for v in topo(pred, succ):
        start = 0
        for p in pred[v]:
            delay = waits['task_cross_core_wait_cycles'] if core_of[p] != core_of[v] else 0
            if previous.get(v) == p:
                delay = max(delay, waits['task_same_core_wait_cycles'])
            start = max(start, end[p] + delay)
        end[v] = start + duration[v]
    # 共享 DDR 不可能在少于总边界字节/总带宽的时间内完成传输。
    return max(max(end.values(), default=0), sum(traffic) / bandwidth)


def construct(graph, cores, cfg, waits, target_scale=1.0):
    start = time.perf_counter()
    gm = GraphModel(graph)
    if not gm.ops:
        return make_plan([], [[] for _ in range(cores)]), {'construction_seconds': time.perf_counter() - start, 'empty_graph': True}
    if target_scale <= 0:
        raise ValueError('target_scale must be positive')
    groups, explanation = coarsen(gm, cores, waits, target_scale)
    groups, fragment_info = bound_fragmentation(gm, groups, explanation['pipe_work_target'], cores)
    explanation['fragmentation_guard'] = fragment_info
    schedules, decisions = list_schedule(gm, groups, cores, cfg, waits)
    explanation['initial_group_count'] = len(groups)
    groups, schedules, merged = coalesce_on_cores(gm, groups, schedules, cfg['bandwidth'])
    parallel_estimate = estimate(gm, groups, schedules, cfg['bandwidth'], waits)
    serial_groups = [sorted(gm.ops)]
    serial_schedules = [[0]] + [[] for _ in range(cores - 1)]
    serial_estimate = estimate(gm, serial_groups, serial_schedules, cfg['bandwidth'], waits)
    proxy_prefers_serial = serial_estimate <= parallel_estimate
    plan = make_plan(groups, schedules)
    # 独立使用官方校验器检查覆盖、收缩图与所有同核顺序的组合依赖。
    validate_task_order(derive_multicore_plan(graph, plan))
    explanation.update({'algorithm': 'structural_bounded_coarsening_list_scheduling',
                        'scheduling_decisions': decisions, 'same_core_merges': merged,
                        'estimated_parallel_cycles': parallel_estimate, 'estimated_serial_cycles': serial_estimate,
                        'proxy_prefers_serial': proxy_prefers_serial,
                        'selected': 'constructed_parallel',
                        'final_task_count': len(groups), 'pipe_lower_bound': max(gm.pipes.values(), default=0) / cores,
                        'construction_seconds': time.perf_counter() - start})
    return plan, explanation


def run(case, cores, evaluate=True, evaluation_timeout=120):
    graph = _read_json(ROOT / 'official/data' / f'{case}.json')
    config = str(ROOT / 'official/data/config.txt')
    cfg, waits = read_evaluation_config(config), read_scene_a_config(config)
    plan, explanation = construct(graph, cores, cfg, waits)
    suffix = '' if evaluate else '_plan_only'
    out = ROOT / 'results_fast' / f'{case}_{cores}cores{suffix}'
    save(out / 'plan.json', plan)
    save(out / 'explanation.json', explanation)
    summary = {k: explanation.get(k) for k in ('selected', 'final_task_count', 'construction_seconds', 'pipe_lower_bound')}
    summary.update({'case': case, 'cores': cores, 'official_evaluations': 0})
    if evaluate:
        start = time.perf_counter()
        for name in ('result.json', 'trace.json', 'official_log.txt'):
            (out / name).unlink(missing_ok=True)

        def score(label, candidate):
            save(out / f'{label}_plan.json', candidate)
            for extension in ('result.json', 'trace.json', 'log.txt'):
                (out / f'{label}_{extension}').unlink(missing_ok=True)
            command = [sys.executable, str(ROOT / 'official/code/multicore_cut_evaluate_problem_1.py'),
                       str(ROOT / 'official/data' / f'{case}.json'), str(out / f'{label}_plan.json'),
                       '--config', config, '-o', str(out / f'{label}_result.json'),
                       '--trace-output', str(out / f'{label}_trace.json'),
                       '--log-output', str(out / f'{label}_log.txt')]
            summary['official_evaluations'] += 1
            try:
                completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                           errors='replace', timeout=evaluation_timeout)
            except subprocess.TimeoutExpired:
                for extension in ('result.json', 'trace.json', 'log.txt'):
                    (out / f'{label}_{extension}').unlink(missing_ok=True)
                return None, 'timeout'
            if completed.returncode:
                summary[f'{label}_error'] = completed.stderr.strip()[:1000]
                return None, 'error'
            return _read_json(out / f'{label}_result.json'), 'valid'

        parallel_result, parallel_status = score('parallel', plan)
        serial_plan = {'node_to_subgraph': {op_id: 0 for op_id in plan['node_to_subgraph']},
                       'core_schedules': [[0]] + [[] for _ in range(cores - 1)]}
        compare_serial = explanation['proxy_prefers_serial'] and serial_plan != plan
        serial_result, serial_status = (score('serial', serial_plan) if compare_serial else (None, 'not_requested'))
        scored = [(label, candidate, result) for label, candidate, result in
                  (('parallel', plan, parallel_result), ('serial', serial_plan, serial_result))
                  if result is not None]
        summary.update({'parallel_score_status': parallel_status,
                        'serial_score_status': serial_status,
                        'proxy_prefers_serial': explanation['proxy_prefers_serial'],
                        'evaluation_seconds': time.perf_counter() - start})
        if not scored:
            summary.update({'valid': None,
                            'status': 'evaluation_error' if 'error' in (parallel_status, serial_status)
                                      else 'evaluation_timeout',
                            'evaluation_timeout_seconds': evaluation_timeout})
            save(out / 'summary.json', summary)
            print(summary, flush=True)
            return summary
        label, winner, result = min(scored, key=lambda item:
                                    (item[2]['makespan'], item[2]['data_movement_bytes']['added_copy_bytes']))
        save(out / 'plan.json', winner)
        for source, target in ((f'{label}_result.json', 'result.json'),
                               (f'{label}_trace.json', 'trace.json'),
                               (f'{label}_log.txt', 'official_log.txt')):
            shutil.copyfile(out / source, out / target)
        summary.update({'valid': True,
                        'status': 'evaluated' if not compare_serial or serial_status == 'valid'
                                  else 'evaluated_serial_unavailable',
                        'selected': label + '_by_official_score',
                        'final_task_count': len(set(winner['node_to_subgraph'].values())),
                        'makespan': result['makespan'],
                        'data_movement_bytes': result['data_movement_bytes']})
    save(out / 'summary.json', summary)
    print(summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='case_001')
    parser.add_argument('--cores', type=int, choices=[2, 3, 4, 5], default=4)
    parser.add_argument('--plan-only', action='store_true', help='只构造和校验方案，不调用评分仿真')
    parser.add_argument('--eval-timeout', type=float, default=120, help='官方评分进程的秒数上限，默认120秒')
    args = parser.parse_args()
    if args.eval_timeout <= 0:
        parser.error('--eval-timeout must be positive')
    summary = run(args.case, args.cores, not args.plan_only, args.eval_timeout)
    if summary.get('status') in ('evaluation_timeout', 'evaluation_error'):
        sys.exit(2)

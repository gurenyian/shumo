"""A small, deterministic FM refiner for acyclic task hypergraphs.

The evaluator is deliberately absent from this module.  It refines a quotient
partition using the usual FM best-prefix rule while checking the quotient DAG
after every proposed move.  Tensor records are treated as hyperedges, so a
fanout is scored once as a boundary rather than as a collection of guessed
pair edges.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq


@dataclass(frozen=True)
class Move:
    op: int
    source: int
    target: int
    gain: float


@dataclass
class FMResult:
    groups: list
    moves: list
    best_prefix: int
    initial_objective: float
    final_objective: float


class Hypergraph:
    """Task graph plus tensor hyperedges used by the FM objective."""
    def __init__(self, graph, groups, bandwidth=1.0, memory_limit=None):
        self.graph = graph
        self.groups = [list(sorted(g)) for g in groups]
        self.bandwidth = float(bandwidth) or 1.0
        if isinstance(memory_limit, dict):
            memory_limit = memory_limit.get('cache_capacity_bytes')
        self.memory_limit = memory_limit
        all_ops = {int(o['id']): o for o in graph.get('ops', [])}
        self.ops = {op_id: op for op_id, op in all_ops.items()
                    if op.get('op') not in {'COPY_IN', 'COPY_OUT'}}
        self.pred = {v: set() for v in self.ops}
        self.succ = {v: set() for v in self.ops}
        raw_succ = {v: set() for v in all_ops}
        tensor_producers = defaultdict(set)
        tensor_consumers = defaultdict(set)
        for edge in graph.get('edges', []):
            a, b = edge['source'], edge['target']
            if a in all_ops and b in all_ops:
                raw_succ[a].add(b)
            elif a in all_ops:
                tensor_producers[b].add(a)
            elif b in all_ops:
                tensor_consumers[a].add(b)
        for tid, producers in tensor_producers.items():
            for a in producers:
                raw_succ[a].update(b for b in tensor_consumers[tid] if a != b)
        # Match the official plan validator: create operation dependencies
        # through tensor records, then contract COPY_IN/COPY_OUT paths.
        # Official inputs are overwhelmingly tensor mediated, so retaining only
        # literal op-to-op records would leave FM with no active boundaries.
        for source in self.ops:
            pending = list(raw_succ[source])
            visited_copy = set()
            while pending:
                target = pending.pop()
                if target in self.ops:
                    if target != source:
                        self.succ[source].add(target)
                        self.pred[target].add(source)
                elif target not in visited_copy:
                    visited_copy.add(target)
                    pending.extend(raw_succ[target])
        self.hyperedges = []
        for tensor in graph.get('tensors', []):
            tid = tensor['id']
            producers = {op for op in tensor_producers[tid] if op in self.ops}
            consumers = {op for op in tensor_consumers[tid] if op in self.ops}
            if producers or consumers:
                self.hyperedges.append((tid, int(tensor.get('size', 0)),
                                        frozenset(producers), frozenset(consumers)))
        self.pipe_totals = Counter(o.get('pipe', '') for o in self.ops.values())
        self.criticality = self._criticalities()

    def _criticalities(self):
        """Return longest predecessor paths without recursive depth limits."""
        indegree = {op: len(parents) for op, parents in self.pred.items()}
        ready = list(op for op, degree in indegree.items() if degree == 0)
        heapq.heapify(ready)
        criticality = {}
        while ready:
            op = heapq.heappop(ready)
            criticality[op] = self.ops[op].get('cycles', 0) + max(
                (criticality[parent] for parent in self.pred[op]), default=0)
            for child in self.succ[op]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(criticality) != len(self.ops):
            raise ValueError('operation dependency graph must be acyclic')
        return criticality

    def mapping(self, groups=None):
        groups = self.groups if groups is None else groups
        return {op: i for i, group in enumerate(groups) for op in group}

    def quotient_edges(self, groups=None):
        mapping = self.mapping(groups)
        edges = {i: set() for i in range(len(groups or self.groups))}
        for a, children in self.succ.items():
            for b in children:
                ga, gb = mapping[a], mapping[b]
                if ga != gb:
                    edges[ga].add(gb)
        # Hyperedges are data dependencies between all producer and consumer
        # groups.  This also covers fanout where no single pair is dominant.
        for _, _, producers, consumers in self.hyperedges:
            ps = {mapping[x] for x in producers if x in mapping}
            cs = {mapping[x] for x in consumers if x in mapping}
            for a in ps:
                edges[a].update(b for b in cs if b != a)
        return edges

    def acyclic(self, groups=None):
        edges = self.quotient_edges(groups)
        indegree = [0] * len(edges)
        for children in edges.values():
            for b in children:
                indegree[b] += 1
        ready = [i for i, degree in enumerate(indegree) if not degree]
        seen = 0
        while ready:
            a = ready.pop()
            seen += 1
            for b in edges[a]:
                indegree[b] -= 1
                if indegree[b] == 0:
                    ready.append(b)
        return seen == len(edges)

    def boundary_bytes(self, groups=None):
        groups = self.groups if groups is None else groups
        mapping = self.mapping(groups)
        total = 0
        for _, size, producers, consumers in self.hyperedges:
            owners = {mapping[x] for x in producers | consumers if x in mapping}
            if len(owners) > 1:
                total += size
        return total

    def pressure(self, groups=None):
        """Approximate live pressure per group from tensor values crossing it."""
        groups = self.groups if groups is None else groups
        mapping = self.mapping(groups)
        pressure = [0] * len(groups)
        for _, size, producers, consumers in self.hyperedges:
            owners = {mapping[x] for x in producers | consumers if x in mapping}
            if len(owners) > 1:
                for owner in owners:
                    pressure[owner] += size
        return pressure

    def objective(self, groups=None):
        groups = self.groups if groups is None else groups
        loads = [Counter() for _ in groups]
        for i, group in enumerate(groups):
            for op in group:
                loads[i][self.ops[op].get('pipe', '')] += self.ops[op].get('cycles', 0)
        peak = max((max(x.values(), default=0) for x in loads), default=0)
        pressure = self.pressure(groups)
        memory_penalty = sum(max(0, p - self.memory_limit) for p in pressure) \
            if self.memory_limit is not None else 0
        return self.boundary_bytes(groups) / self.bandwidth + peak + memory_penalty


class AcyclicFM:
    """Boundary restricted FM passes with deterministic locking and rollback."""
    def __init__(self, hypergraph, max_active=256, max_passes=2,
                 memory_weight=1.0):
        self.hg = hypergraph
        self.max_active = max(1, int(max_active))
        self.max_passes = max(1, int(max_passes))
        self.memory_weight = float(memory_weight)

    def _adjacent_targets(self, op, groups, source):
        mapping = self.hg.mapping(groups)
        targets = set()
        for neighbour in self.hg.pred[op] | self.hg.succ[op]:
            group = mapping.get(neighbour)
            if group is not None and group != source:
                targets.add(group)
        return sorted(targets)

    def _criticality(self, op):
        return self.hg.criticality[op]

    def _gain(self, groups, op, target):
        source = self.hg.mapping(groups)[op]
        # Empty quotient groups make plan IDs ambiguous and are not legal
        # finalists, so FM preserves at least one operation in every group.
        if source == target or len(groups[source]) <= 1:
            return float('-inf')
        before = self.hg.objective(groups)
        trial = [list(g) for g in groups]
        trial[source].remove(op)
        trial[target].append(op)
        trial = [sorted(g) for g in trial]
        if not self.hg.acyclic(trial):
            return float('-inf')
        after = self.hg.objective(trial)
        # Keep critical, high traffic boundaries active; the objective remains
        # generic and the official evaluator still judges all emitted plans.
        return before - after + self._criticality(op) * 1e-9

    def _active(self, groups):
        mapping = self.hg.mapping(groups)
        scored = []
        pressure = self.hg.pressure(groups)
        for source, group in enumerate(groups):
            for op in group:
                adjacent = self._adjacent_targets(op, groups, source)
                if adjacent:
                    critical = self._criticality(op)
                    scored.append((critical + pressure[source], op))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [op for _, op in scored[:self.max_active]]

    def run(self, groups=None):
        current = [list(g) for g in (self.hg.groups if groups is None else groups)]
        if any(not g for g in current) or not self.hg.acyclic(current):
            raise ValueError('groups must be nonempty and quotient must be acyclic')
        initial = self.hg.objective(current)
        all_moves = []
        best_prefix = 0
        for _ in range(self.max_passes):
            locked = set()
            start = [list(g) for g in current]
            start_value = self.hg.objective(start)
            prefix = []
            cumulative_best = 0.0
            pass_best_prefix = 0
            best_state = [list(g) for g in start]
            for op in self._active(current):
                if op in locked:
                    continue
                source = self.hg.mapping(current)[op]
                offers = []
                for target in self._adjacent_targets(op, current, source):
                    gain = self._gain(current, op, target)
                    if gain != float('-inf'):
                        offers.append((gain, target))
                if not offers:
                    continue
                gain, target = max(offers, key=lambda item: (item[0], -item[1]))
                move = Move(op, source, target, gain)
                current[source].remove(op)
                current[target].append(op)
                current = [sorted(g) for g in current]
                locked.add(op)
                prefix.append(move)
                value = start_value - self.hg.objective(current)
                if value > cumulative_best:
                    cumulative_best = value
                    best_state = [list(g) for g in current]
                    pass_best_prefix = len(prefix)
            current = best_state
            # The rejected suffix is useful only while evaluating this pass.
            # Report moves that are actually retained, so FMResult describes
            # its final partition even when a negative tail was explored.
            all_moves.extend(prefix[:pass_best_prefix])
            best_prefix = len(all_moves)
            if pass_best_prefix == 0:
                break
        return FMResult(current, all_moves, best_prefix, initial,
                        self.hg.objective(current))


def refine(graph, groups, bandwidth=1.0, memory_limit=None, **kwargs):
    """Convenience entry point used by the five core runner and tests."""
    hg = Hypergraph(graph, groups, bandwidth, memory_limit)
    return AcyclicFM(hg, **kwargs).run()

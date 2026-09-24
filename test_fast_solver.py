import random
import json
import unittest
from pathlib import Path

from fast_solver import (GraphModel, Groups, coarsen, coalesce_on_cores, construct, bound_fragmentation,
                         derive_multicore_plan, validate_task_order)

CFG = {'bandwidth': 60, 'capacity': {'L1': 524288, 'UB': 131072}}
WAITS = {'task_same_core_wait_cycles': 100, 'task_cross_core_wait_cycles': 1000}


def graph(costs, edges):
    return {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': w}
                    for i, w in enumerate(costs)], 'tensors': [],
            'edges': [{'source': a, 'target': b} for a, b in edges]}


class FastSolverTests(unittest.TestCase):
    def test_case003_proxy_does_not_discard_parallel_plan(self):
        path = Path(__file__).resolve().parent / 'official/data/case_003.json'
        graph_data = json.loads(path.read_text(encoding='utf-8'))
        plan, info = construct(graph_data, 4, CFG, WAITS)
        self.assertTrue(info['proxy_prefers_serial'])
        self.assertGreater(len(set(plan['node_to_subgraph'].values())), 1)
        self.assertGreater(sum(bool(order) for order in plan['core_schedules']), 1)

    def test_independent_components_are_not_fragmented(self):
        gm = GraphModel(graph([1] * 100, []))
        groups, info = bound_fragmentation(gm, [[i] for i in range(100)], 50, 2)
        self.assertFalse(info['used'])
        self.assertEqual(len(groups), 100)

    def test_fragmentation_guard_covers_graph_without_cycle(self):
        gm = GraphModel(graph([1] * 100, [(i, i + 1) for i in range(99)]))
        groups, info = bound_fragmentation(gm, [[i] for i in range(100)], 10, 2)
        self.assertTrue(info['used'])
        self.assertEqual(len(groups), 10)
        view = derive_multicore_plan(gm.graph, {'node_to_subgraph': {str(v): i for i,g in enumerate(groups) for v in g},
                                              'core_schedules': [list(range(10)), []]})
        validate_task_order(view)

    def test_join_is_not_partially_absorbed(self):
        gm = GraphModel(graph([50, 50, 10], [(0, 2), (1, 2)]))
        groups, _ = coarsen(gm, 2, WAITS)
        self.assertEqual(sorted(map(sorted, groups)), [[0], [1], [2]])

    def test_same_core_merge_preserves_external_wait_boundary(self):
        gm = GraphModel(graph([50, 50, 10], [(0, 2), (1, 2)]))
        groups, schedules, log = coalesce_on_cores(gm, [[0], [1], [2]], [[0, 2], [1]], 60)
        self.assertEqual(len(groups), 3)
        self.assertEqual(log, [])

    def test_contraction_rejects_alternative_path(self):
        # 0 -> 2 之外还有 0 -> 1 -> 2，收缩 0 和 2 会成环。
        groups = Groups({i: [i] for i in range(3)}, {0: set(), 1: {0}, 2: {0, 1}},
                        {0: {1, 2}, 1: {2}, 2: set()}, {i: {'PIPE_V': 1} for i in range(3)})
        self.assertFalse(groups.safe(0, 2))

    def test_generated_dags_cover_nodes_and_have_no_wait_cycles(self):
        for seed in range(30):
            rng = random.Random(seed)
            g = graph([rng.randint(0, 200) for _ in range(16)],
                      [(i, j) for i in range(16) for j in range(i + 1, 16) if rng.random() < .15])
            for cores in (2, 3, 4, 5):
                plan, _ = construct(g, cores, CFG, WAITS)
                view = derive_multicore_plan(g, plan)
                validate_task_order(view)
                self.assertEqual(len(plan['node_to_subgraph']), 16)
                self.assertEqual(len(plan['core_schedules']), cores)


if __name__ == '__main__':
    unittest.main()

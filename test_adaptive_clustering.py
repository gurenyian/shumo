import random
import unittest

from adaptive_clustering import adaptive_partition, schedule_groups
from fast_solver import GraphModel, list_schedule, estimate
from test_fast_solver import graph, CFG, WAITS


class AdaptiveClusteringTests(unittest.TestCase):
    def test_insertion_uses_idle_gap_and_shortens_schedule(self):
        gm = GraphModel(graph([29, 14, 5, 6, 25, 20, 4, 18, 23, 11],
                              [(0, 3), (0, 5), (1, 5), (1, 7), (1, 9),
                               (2, 7), (3, 4), (4, 9), (5, 6), (5, 9)]))
        groups = [[v] for v in gm.order]
        waits = {'task_same_core_wait_cycles': 1, 'task_cross_core_wait_cycles': 3}
        old, _ = list_schedule(gm, groups, 2, CFG, waits)
        plan, detail = schedule_groups(gm, groups, 2, CFG, waits)
        self.assertEqual(estimate(gm, groups, old, 60, waits), 93)
        self.assertEqual(estimate(gm, groups, plan['core_schedules'], 60, waits), 88)
        self.assertGreater(detail['insertions_before_existing_tasks'], 0)

    def test_dynamic_contractions_preserve_coverage_and_acyclicity(self):
        # Every constructed plan goes through the independent contest validator.
        for seed in range(25):
            rng = random.Random(seed)
            gm = GraphModel(graph([rng.randint(1, 500) for _ in range(24)],
                                  [(i, j) for i in range(24) for j in range(i + 1, 24)
                                   if rng.random() < .15]))
            plan, detail = adaptive_partition(gm, 5, CFG, WAITS)
            self.assertEqual(set(map(int, plan['node_to_subgraph'])), set(gm.ops))
            assigned = [v for order in plan['core_schedules'] for v in order]
            self.assertEqual(len(assigned), len(set(assigned)))
            self.assertEqual(set(assigned), set(plan['node_to_subgraph'].values()))
            self.assertEqual(len(assigned), detail['tasks'])


if __name__ == '__main__':
    unittest.main()

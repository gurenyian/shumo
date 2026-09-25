"""Focused contract checks for Scene-B subgraph granularity."""

import unittest

from solver_problem2 import validate_plan
from multicore_cut_evaluate_problem_2 import _build_scene_b_tasks


class SceneBGranularityTests(unittest.TestCase):
    def test_same_core_subgraph_order_changes_local_sequence_without_cross_copy(self):
        graph = {
            'ops': [
                {'id': 1, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10},
                {'id': 2, 'op': 'MUL', 'pipe': 'PIPE_V', 'cycles': 20},
            ],
            'tensors': [], 'edges': [],
        }
        capacity = {'L1': 524288, 'UB': 131072}
        merged = {'node_to_subgraph': {'1': 0, '2': 0},
                  'core_schedules': [[0], []]}
        validate_plan(graph, merged, 2)
        merged_tasks, merged_links, _, merged_traffic, _ = _build_scene_b_tasks(
            graph, merged, 60, capacity)
        raw_order = merged_tasks[0]['seq']
        ids = {raw_order[0]: 0, raw_order[1]: 1}
        split = {'node_to_subgraph': {str(op): sg for op, sg in ids.items()},
                 'core_schedules': [[1, 0], []]}
        validate_plan(graph, split, 2)
        split_tasks, split_links, _, split_traffic, _ = _build_scene_b_tasks(
            graph, split, 60, capacity)
        self.assertEqual(split_tasks[0]['seq'], list(reversed(raw_order)))
        self.assertEqual(merged_links, split_links)
        self.assertEqual(merged_traffic['added_copy_bytes'],
                         split_traffic['added_copy_bytes'])


if __name__ == '__main__':
    unittest.main()

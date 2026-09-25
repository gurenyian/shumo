import unittest
from unittest.mock import patch

from experiment import ROOT, read_evaluation_config, read_scene_a_config
from fast_solver import GraphModel
from multilevel_partition import (_legal_edges, _local_variants, _reachable_without,
                                  candidates)
from stub_multicore_cut_and_schedule import MulticoreCutError


class MultilevelPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_evaluation_config(str(ROOT / 'official/data/config.txt'))
        cls.waits = read_scene_a_config(str(ROOT / 'official/data/config.txt'))

    def graph(self):
        # 1 -> 2 has the alternate route 1 -> 3 -> 2.  Task 4 is an
        # independent component, so scheduling must still cover it once.
        ops = [{'id': op, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10 + op}
               for op in range(1, 5)]
        tensors = [{'id': 100 + index, 'pos': 'DDR', 'size': 64}
                   for index in range(4)]
        edges = []
        for tensor, source, target in zip(tensors, (1, 1, 3, 1), (2, 3, 2, 4)):
            edges.extend(({'source': source, 'target': tensor['id']},
                          {'source': tensor['id'], 'target': target}))
        return {'ops': ops, 'tensors': tensors, 'edges': edges}

    def test_exact_alternate_path_and_deterministic_coverage(self):
        graph = self.graph()
        gm = GraphModel(graph)
        groups = [[1], [2], [3], [4]]
        edges, *_ = _legal_edges(gm, groups, self.cfg['bandwidth'], 5, self.waits)
        self.assertNotIn((0, 1), edges)
        self.assertTrue(_reachable_without([{1, 2, 3}, set(), {1}, set()], 0, 1))
        incumbent = {'node_to_subgraph': {str(op): op - 1 for op in range(1, 5)},
                     'core_schedules': [[0, 2, 1, 3], [], [], [], []]}
        first = candidates(graph, incumbent, self.cfg, self.waits, 5, 3)
        second = candidates(graph, incumbent, self.cfg, self.waits, 5, 3)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 3)
        for plan in first:
            self.assertEqual({int(op) for op in plan['node_to_subgraph']}, {1, 2, 3, 4})
            scheduled = [task for core in plan['core_schedules'] for task in core]
            self.assertEqual(sorted(scheduled), list(range(len(set(scheduled)))) )

    def test_invalid_local_move_is_rejected(self):
        graph = self.graph()
        gm = GraphModel(graph)
        groups = [[1], [2], [3], [4]]
        base = {'node_to_subgraph': {str(op): op - 1 for op in range(1, 5)},
                'core_schedules': [[0, 2, 1, 3], [], [], [], []]}
        with patch('multilevel_partition.derive_multicore_plan',
                   side_effect=MulticoreCutError('dependency order violation')):
            self.assertEqual(_local_variants(gm, groups, base, 5, self.cfg, self.waits), [])


if __name__ == '__main__':
    unittest.main()

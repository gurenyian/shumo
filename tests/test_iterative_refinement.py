import unittest

from experiment import ROOT, read_evaluation_config, read_scene_a_config
from fast_solver import GraphModel
from iterative_refinement import _boundary_options, _edge_info, _partition_copy_bytes, analyze, candidates


class IterativeRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_evaluation_config(str(ROOT / 'official/data/config.txt'))
        cls.waits = read_scene_a_config(str(ROOT / 'official/data/config.txt'))

    def graph(self):
        ops = [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V' if i % 2 else 'PIPE_M', 'cycles': 10 + i} for i in range(1, 9)]
        tensors, edges = [], []
        for i in range(1, 8):
            tid = 100 + i; tensors.append({'id': tid, 'pos': 'DDR', 'size': 64 * i})
            edges.extend([{'source': i, 'target': tid}, {'source': tid, 'target': i + 1}])
        return {'ops': ops, 'tensors': tensors, 'edges': edges}

    def plan(self):
        return {'node_to_subgraph': {str(i): (0 if i <= 4 else i - 4) for i in range(1, 9)},
                'core_schedules': [[0, 1], [2], [3], [4], []]}

    def result(self):
        return {'per_core_timeline': [{'core_id': c, 'tasks': [
            {'subgraph_id': g, 'duration': 20 + g, 'start': 0, 'end': 20 + g}
            for g in order]} for c, order in enumerate(self.plan()['core_schedules'])]}

    def test_typed_delays(self):
        graph = self.graph(); plan = self.plan(); gm = GraphModel(graph)
        groups = [[1, 2, 3, 4], [5], [6], [7], [8]]
        durations = [20] * 5
        info = _edge_info(gm, groups, plan['core_schedules'], durations, self.waits)
        kinds = {(e['source'], e['target']): e['kind'] for e in info['typed_edges']}
        self.assertEqual(kinds[(0, 1)], 'same_core_adjacent')
        self.assertEqual(kinds[(1, 2)], 'cross_core')

        waits = {**self.waits, 'task_same_core_wait_cycles': 17,
                 'task_cross_core_wait_cycles': 29}
        configured = _edge_info(gm, groups, plan['core_schedules'], durations, waits)
        delays = {(e['source'], e['target']): e['delay'] for e in configured['typed_edges']}
        self.assertEqual(delays[(0, 1)], 17)
        self.assertEqual(delays[(1, 2)], 29)

    def test_schedule_only_edge_has_same_core_delay(self):
        graph = self.graph(); graph['edges'] = graph['edges'][:2]
        gm = GraphModel(graph)
        info = _edge_info(gm, [[1], [2], [3], [4], [5], [6], [7], [8]],
                          [[0, 7], [1], [2], [3], [4]], [1] * 8, self.waits)
        kinds = {(e['source'], e['target']): e for e in info['typed_edges']}
        self.assertEqual(kinds[(0, 7)]['delay'], 100)

    def test_boundary_accounting_and_determinism(self):
        graph = self.graph(); gm = GraphModel(graph)
        options = _boundary_options(gm, [1, 2, 3, 4])
        self.assertEqual(options[0]['boundary'], 128)
        groups = [[1, 2, 3, 4], [5], [6], [7], [8]]
        split_groups = [[1], [2, 3, 4], [5], [6], [7], [8]]
        self.assertEqual(_partition_copy_bytes(gm, split_groups) - _partition_copy_bytes(gm, groups), 128)
        plan = self.plan(); result = self.result()
        a, _ = candidates(graph, plan, result, self.cfg, self.waits)
        b, _ = candidates(graph, plan, result, self.cfg, self.waits)
        self.assertEqual([x['plan'] for x in a], [x['plan'] for x in b])
        self.assertLessEqual(len(a), 8)
        self.assertLessEqual(len({p['family'] for p in a}), 4)
        self.assertTrue(any(p['family'] == 'legal_partition_merge' for p in a))

    def test_candidates_are_legal(self):
        graph = self.graph(); proposals, _ = candidates(graph, self.plan(), self.result(), self.cfg, self.waits)
        self.assertTrue(proposals)
        for proposal in proposals:
            self.assertEqual(sorted(g for s in proposal['plan']['core_schedules'] for g in s),
                             list(range(len(set(proposal['plan']['node_to_subgraph'].values())))))


if __name__ == '__main__':
    unittest.main()

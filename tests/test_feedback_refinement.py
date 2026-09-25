import unittest

from experiment import ROOT, read_evaluation_config, read_scene_a_config
from feedback_refinement import _split_options, analyze, candidates
from fast_solver import GraphModel


class FeedbackRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_evaluation_config(str(ROOT / 'official/data/config.txt'))
        cls.waits = read_scene_a_config(str(ROOT / 'official/data/config.txt'))

    def graph(self):
        ops = [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10 + i} for i in range(1, 7)]
        tensors, edges = [], []
        for i in range(1, 6):
            tid = 100 + i; tensors.append({'id': tid, 'pos': 'DDR', 'size': 64 * i})
            edges.extend([{'source': i, 'target': tid}, {'source': tid, 'target': i + 1}])
        return {'ops': ops, 'tensors': tensors, 'edges': edges}

    def result(self, count):
        return {'per_core_timeline': [{'core_id': 0, 'tasks': [
            {'subgraph_id': i, 'duration': 20 + i, 'start': i * 20, 'end': i * 20 + 20 + i}
            for i in range(count)]}]}

    def test_augmented_critical_path_and_delays(self):
        graph = self.graph(); plan = {'node_to_subgraph': {'1': 0, '2': 0, '3': 0, '4': 0, '5': 1, '6': 2},
                                       'core_schedules': [[0, 1, 2], [], [], [], []]}
        info = analyze(GraphModel(graph), plan, self.result(3), self.cfg['bandwidth'], self.waits)
        self.assertEqual(info['critical_tasks'][-1], 2)
        self.assertTrue(any(edge['source'] == 0 and edge['target'] == 1 for edge in info['critical_edges']))

    def test_candidate_families_are_valid_and_distinct(self):
        graph = self.graph(); plan = {'node_to_subgraph': {'1': 0, '2': 0, '3': 0, '4': 0, '5': 1, '6': 2},
                                       'core_schedules': [[0, 1, 2], [], [], [], []]}
        proposals, info = candidates(graph, plan, self.result(3), self.cfg, self.waits)
        self.assertLessEqual(len(proposals), 4)
        self.assertEqual(len({str(p['plan']) for p in proposals}), len(proposals))
        self.assertTrue(any(p['family'].startswith('B_') for p in proposals))
        self.assertTrue(any(p['family'].startswith('C_') for p in proposals))
        for proposal in proposals:
            flat = [task for schedule in proposal['plan']['core_schedules'] for task in schedule]
            self.assertEqual(sorted(flat), list(range(len(set(flat)))))

    def test_split_options_use_pipe_cycle_work_and_prefix_boundaries(self):
        graph = self.graph()
        graph['ops'][0]['pipe'] = 'PIPE_M'; graph['ops'][0]['cycles'] = 100
        graph['ops'][1]['pipe'] = 'PIPE_M'; graph['ops'][1]['cycles'] = 50
        plan = {'node_to_subgraph': {str(i): 0 for i in range(1, 7)},
                'core_schedules': [[0], [], [], [], []]}
        info = analyze(GraphModel(graph), plan, self.result(1), self.cfg['bandwidth'], self.waits)
        members, options = _split_options(GraphModel(graph), info, 0)
        self.assertEqual(members, list(range(1, 7)))
        first = options[0]
        self.assertEqual(first['left_work']['PIPE_M'], 100)
        self.assertEqual(first['right_work']['PIPE_M'], 50)
        # The first cut crosses only tensor 101 (1 -> 2).
        self.assertEqual(first['boundary'], 64)


if __name__ == '__main__':
    unittest.main()

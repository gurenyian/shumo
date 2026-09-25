import unittest

from fm_candidates import candidate_variants


CFG = {'bandwidth': 60, 'capacity': {'L1': 10000, 'UB': 10000}}
WAITS = {'task_same_core_wait_cycles': 100, 'task_cross_core_wait_cycles': 1000}


class FMCandidateTests(unittest.TestCase):
    def test_adaptive_seed_can_split_a_single_task_incumbent(self):
        ops = [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10}
               for i in range(8)]
        edges = ([{'source': 0, 'target': i} for i in (1, 2, 3, 4)] +
                 [{'source': i, 'target': 7} for i in (1, 2, 3, 4)])
        graph = {'ops': ops, 'tensors': [], 'edges': edges}
        incumbent = {'node_to_subgraph': {str(i): 0 for i in range(8)},
                     'core_schedules': [[0], []]}

        variants = candidate_variants(graph, incumbent, 2, CFG, WAITS)

        self.assertGreaterEqual(len(variants), 2)
        self.assertTrue(any(len(set(plan['node_to_subgraph'].values())) > 1
                            and sum(bool(schedule) for schedule in plan['core_schedules']) == 2
                            for plan in variants[1:]))


if __name__ == '__main__':
    unittest.main()

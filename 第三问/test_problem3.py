"""Small, deterministic checks of the L2 proxy and official Scene-B contract."""

import unittest

from solver_problem3 import PredictiveL2FIFO
from multicore_cut_evaluate_problem_3 import evaluate_problem_3


CAPACITY = {'L1': 524288, 'UB': 131072}


class PredictiveFIFOTests(unittest.TestCase):
    def test_first_miss_then_hit(self):
        fifo = PredictiveL2FIFO(10)
        self.assertEqual(fifo.access(1, 4), (False, []))
        self.assertEqual(fifo.access(1, 4), (True, []))

    def test_hit_does_not_refresh_fifo(self):
        fifo = PredictiveL2FIFO(10)
        fifo.access(1, 4)
        fifo.access(2, 4)
        fifo.access(1, 4)
        self.assertEqual(fifo.access(3, 4), (False, [(1, 4)]))
        self.assertEqual(list(fifo.entries), [2, 3])

    def test_oversize_tensor_is_never_cached(self):
        fifo = PredictiveL2FIFO(10)
        self.assertEqual(fifo.access(1, 11), (False, []))
        self.assertEqual(fifo.access(1, 11), (False, []))
        self.assertEqual(fifo.used_bytes, 0)

    def test_multiple_fifo_evictions(self):
        fifo = PredictiveL2FIFO(10)
        for tid in (1, 2, 3):
            fifo.access(tid, 3)
        self.assertEqual(fifo.access(4, 8), (False, [(1, 3), (2, 3), (3, 3)]))
        self.assertEqual(list(fifo.entries), [4])


class OfficialCacheContractTests(unittest.TestCase):
    def test_delayed_second_core_reads_shared_tensor_from_l2(self):
        # Core 1 first reads a cold large input, giving Core 0 time to fill L2.
        graph = {'ops': [
            {'id': 1, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 3},
            {'id': 2, 'op': 'MUL', 'pipe': 'PIPE_V', 'cycles': 4}],
            'tensors': [{'id': 3, 'pos': 'UB', 'size': 12000},
                        {'id': 4, 'pos': 'UB', 'size': 120}],
            'edges': [{'source': 4, 'target': 1},
                      {'source': 3, 'target': 2},
                      {'source': 4, 'target': 2}]}
        plan = {'node_to_subgraph': {'1': 0, '2': 1},
                'core_schedules': [[0], [1]]}
        result = evaluate_problem_3(graph, plan, 60, CAPACITY, 500, 1048576, 250)
        shared = [(e['event'], e['core_id']) for e in result['cache_events']
                  if e['tensor_id'] == 4 and e['event'] in ('miss', 'hit')]
        self.assertEqual(shared, [('miss', 0), ('hit', 1)])
        self.assertEqual(result['cache_stats']['hit_bytes'], 120)

    def test_official_hit_does_not_refresh_fifo_order(self):
        graph = {'ops': [
            {'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 3}
            for i in range(1, 5)],
            'tensors': [{'id': 5, 'pos': 'UB', 'size': 100},
                        {'id': 6, 'pos': 'UB', 'size': 200},
                        *[{'id': i, 'pos': 'UB', 'size': 4} for i in (7, 8, 9)]],
            'edges': [{'source': 7, 'target': 1}, {'source': 8, 'target': 2},
                      {'source': 5, 'target': 3}, {'source': 7, 'target': 3},
                      {'source': 6, 'target': 4}, {'source': 9, 'target': 4}]}
        plan = {'node_to_subgraph': {str(i): i-1 for i in range(1, 5)},
                'core_schedules': [[0], [1], [2], [3]]}
        result = evaluate_problem_3(graph, plan, 60, CAPACITY, 500, 8, 250)
        events = result['cache_events']
        self.assertTrue(any(e['event'] == 'hit' and e['tensor_id'] == 7 for e in events))
        self.assertEqual([e['evicted_tensor_ids'] for e in events
                          if e['event'] == 'insert' and e['tensor_id'] == 9], [[7]])
        self.assertEqual([e['tensor_id'] for e in result['cache_final_entries']], [8, 9])

    def test_official_oversize_tensor_is_not_cached(self):
        graph = {'ops': [
            {'id': 1, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 3},
            {'id': 2, 'op': 'MUL', 'pipe': 'PIPE_V', 'cycles': 4}],
            'tensors': [{'id': 3, 'pos': 'UB', 'size': 1200000}],
            'edges': [{'source': 3, 'target': 1}, {'source': 3, 'target': 2}]}
        plan = {'node_to_subgraph': {'1': 0, '2': 1},
                'core_schedules': [[0], [1]]}
        capacity = {'L1': 4000000, 'UB': 4000000}
        result = evaluate_problem_3(graph, plan, 60, capacity, 500, 1048576, 250)
        self.assertEqual(result['cache_stats']['copy_in_misses'], 2)
        self.assertEqual(result['cache_used_bytes_final'], 0)

    def test_official_insertion_can_evict_multiple_fifo_entries(self):
        graph = {'ops': [
            {'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 3}
            for i in range(1, 5)],
            'tensors': [{'id': 5, 'pos': 'UB', 'size': 100},
                        *[{'id': i, 'pos': 'UB', 'size': 3} for i in (6, 7, 8)],
                        {'id': 9, 'pos': 'UB', 'size': 8}],
            'edges': [{'source': 6, 'target': 1}, {'source': 7, 'target': 2},
                      {'source': 8, 'target': 3}, {'source': 5, 'target': 4},
                      {'source': 9, 'target': 4}]}
        plan = {'node_to_subgraph': {str(i): i-1 for i in range(1, 5)},
                'core_schedules': [[0], [1], [2], [3]]}
        result = evaluate_problem_3(graph, plan, 60, CAPACITY, 500, 9, 250)
        self.assertEqual([e['evicted_tensor_ids'] for e in result['cache_events']
                          if e['event'] == 'insert' and e['tensor_id'] == 9],
                         [[6, 7, 8]])

    def test_same_core_subgraphs_keep_one_task_and_no_cross_copy(self):
        graph = {'ops': [
            {'id': 1, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 3},
            {'id': 2, 'op': 'MUL', 'pipe': 'PIPE_V', 'cycles': 4},
        ], 'tensors': [], 'edges': []}
        plan = {'node_to_subgraph': {'1': 0, '2': 1},
                'core_schedules': [[0, 1], []]}
        result = evaluate_problem_3(graph, plan, 60, CAPACITY, 500, 1048576, 250)
        self.assertEqual(result['task_count'], 2)
        self.assertEqual(len(result['cross_core_transfers']), 0)
        self.assertEqual(result['cache_stats']['hit_rate'], 0)

    def test_concurrent_copies_can_both_miss_before_insertion(self):
        graph = {'ops': [
            {'id': 1, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 3},
            {'id': 2, 'op': 'MUL', 'pipe': 'PIPE_V', 'cycles': 4},
        ], 'tensors': [{'id': 3, 'pos': 'UB', 'size': 120}],
            'edges': [{'source': 3, 'target': 1},
                      {'source': 3, 'target': 2}]}
        plan = {'node_to_subgraph': {'1': 0, '2': 1},
                'core_schedules': [[0], [1]]}
        result = evaluate_problem_3(graph, plan, 60, CAPACITY, 500, 1048576, 250)
        self.assertEqual(result['cache_stats']['copy_in_misses'], 2)
        self.assertEqual(result['cache_stats']['copy_in_hits'], 0)
        self.assertEqual(result['cache_stats']['miss_bytes'], 240)
        self.assertEqual(result['cache_used_bytes_final'], 120)


if __name__ == '__main__':
    unittest.main()

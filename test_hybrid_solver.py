import unittest

from hybrid_solver import groups_from_plan, neighbors, plan_from_groups, signature
from fast_solver import GraphModel, independent_component_plan
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order


class HybridTests(unittest.TestCase):
    def test_single_task_can_split_and_move_work_to_another_core(self):
        graph = {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10000}
                         for i in range(4)], 'tensors': [], 'edges': []}
        gm = GraphModel(graph)
        timeline = {'per_core_timeline': [
            {'core_id': 0, 'ops': [{'end': 40000}],
             'tasks': [{'subgraph_id': 0, 'duration': 40000}]},
            {'core_id': 1, 'ops': [], 'tasks': []}]}
        proposed = neighbors(gm, [[0, 1, 2, 3]], [[0], []], timeline, 60,
                             {'task_same_core_wait_cycles': 100,
                              'task_cross_core_wait_cycles': 1000})
        joint = next(item for item in proposed if item[0] == 'split_reschedule')
        self.assertEqual(sum(bool(order) for order in joint[3]), 2)
        validate_task_order(derive_multicore_plan(graph, plan_from_groups(joint[2], joint[3])))

    def test_independent_components_use_multiple_cores_without_new_dependencies(self):
        graph = {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 100}
                         for i in range(6)], 'tensors': [],
                 'edges': [{'source': 0, 'target': 1}, {'source': 2, 'target': 3},
                           {'source': 4, 'target': 5}]}
        plan = independent_component_plan(GraphModel(graph), 2)
        validate_task_order(derive_multicore_plan(graph, plan))
        self.assertEqual(len(set(plan['node_to_subgraph'].values())), 3)
        self.assertTrue(all(plan['core_schedules']))

    def test_neighbor_representation_and_legal_filter(self):
        graph = {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 100} for i in range(4)],
                 'tensors': [],
                 'edges': [{'source': 0, 'target': 2}, {'source': 1, 'target': 2},
                           {'source': 2, 'target': 3}]}
        plan = {'node_to_subgraph': {str(i): i for i in range(4)}, 'core_schedules': [[0, 2, 3], [1]]}
        gm = GraphModel(graph)
        groups, schedules = groups_from_plan(plan)
        self.assertEqual(signature(plan_from_groups(groups, schedules)), signature(plan))
        timeline = {'per_core_timeline': [{'core_id': 0, 'ops': [{'end': 300}]},
                                          {'core_id': 1, 'ops': [{'end': 100}]}]}
        proposed = neighbors(gm, groups, schedules, timeline, 60)
        self.assertTrue(any(item[0] == 'move' for item in proposed))
        legal = 0
        for _, _, new_groups, new_schedules, _ in proposed:
            candidate = plan_from_groups(new_groups, new_schedules)
            try:
                validate_task_order(derive_multicore_plan(graph, candidate))
                legal += 1
            except (RuntimeError, ValueError):
                pass
        self.assertGreater(legal, 0)


if __name__ == '__main__':
    unittest.main()

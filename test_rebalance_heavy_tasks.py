import json
import unittest
from pathlib import Path

from fast_solver import GraphModel
from rebalance_heavy_tasks import branch_chain_plan, split_heavy_groups


CFG = {'bandwidth': 60, 'capacity': {'L1': 524288, 'UB': 131072}}
WAITS = {'task_same_core_wait_cycles': 100, 'task_cross_core_wait_cycles': 1000}
ROOT = Path(__file__).resolve().parent


class RebalanceTests(unittest.TestCase):
    def test_fork_join_work_is_exposed_without_dependency_cycle(self):
        case = 'case_016'
        graph = json.loads((ROOT / 'official/data' / f'{case}.json').read_text(encoding='utf-8'))
        plan = json.loads((ROOT / 'results_all100/multicore' / f'{case}_5cores' /
                           'best_plan.json').read_text(encoding='utf-8'))
        gm = GraphModel(graph)
        branch, info = branch_chain_plan(gm, 5, CFG, WAITS, 1.0)
        self.assertEqual(len(branch['node_to_subgraph']), len(gm.ops))
        self.assertGreater(info['tasks'], 100)
        self.assertGreater(sum(bool(order) for order in branch['core_schedules']), 1)
        split, detail = split_heavy_groups(gm, plan, 5, CFG, WAITS, 1.0)
        self.assertEqual(len(split['node_to_subgraph']), len(gm.ops))
        self.assertGreater(detail['tasks_after'], detail['tasks_before'])


if __name__ == '__main__':
    unittest.main()

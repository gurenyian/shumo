"""针对依赖和等待时间的回归检查，不以代理模型代替官方评分。"""
import unittest
from experiment import model, schedule, evaluate_scene_a


def diamond():
    return {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10} for i in range(4)],
            'tensors': [], 'edges': [{'source': a, 'target': b} for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]]}


class DependencyTests(unittest.TestCase):
    def test_contraction_must_not_create_cycle(self):
        # 原图无环不保证随意分组后无环：{0,3} -> {1,2} -> {0,3}。
        with self.assertRaisesRegex(ValueError, 'cycle'):
            schedule(diamond(), [[0, 3], [1, 2]], 2, {'bandwidth': 60},
                     {'task_same_core_wait_cycles': 100, 'task_cross_core_wait_cycles': 1000})

    def test_copy_contraction_preserves_dependency(self):
        g = diamond()
        g['ops'][1]['op'] = 'COPY_IN'
        ops, pred, *_ = model(g)
        self.assertNotIn(1, ops)
        self.assertIn(0, pred[3])

    def test_same_core_and_cross_core_wait(self):
        g = {'ops': diamond()['ops'][:2], 'tensors': [], 'edges': [{'source': 0, 'target': 1}]}
        config = dict(bandwidth=60, capacity={'L1': 524288, 'UB': 131072}, cross_core_wait=1000, same_core_wait=100)
        for schedules, expected in [([[0, 1], []], 120), ([[0], [1]], 1020)]:
            result = evaluate_scene_a(g, {'node_to_subgraph': {'0': 0, '1': 1}, 'core_schedules': schedules}, **config)
            self.assertEqual(result['makespan'], expected)


if __name__ == '__main__':
    unittest.main()

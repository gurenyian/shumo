import unittest

from acyclic_fm_refinement import AcyclicFM, Hypergraph, refine


def graph():
    return {
        'ops': [
            {'id': 1, 'op': 'A', 'pipe': 'p0', 'cycles': 10},
            {'id': 2, 'op': 'B', 'pipe': 'p1', 'cycles': 8},
            {'id': 3, 'op': 'C', 'pipe': 'p1', 'cycles': 7},
            {'id': 4, 'op': 'D', 'pipe': 'p0', 'cycles': 4},
        ],
        'edges': [
            {'source': 1, 'target': 10}, {'source': 10, 'target': 2},
            {'source': 10, 'target': 3},
            {'source': 2, 'target': 11}, {'source': 11, 'target': 4},
            {'source': 3, 'target': 12}, {'source': 12, 'target': 4},
        ],
        'tensors': [{'id': 10, 'size': 100}, {'id': 11, 'size': 10},
                    {'id': 12, 'size': 10}],
    }


class FMTests(unittest.TestCase):
    def test_hyperedge_fanout_counted_once(self):
        hg = Hypergraph(graph(), [[1], [2], [3], [4]], bandwidth=10)
        self.assertEqual(hg.boundary_bytes(), 120)

    def test_tensor_edges_activate_real_boundaries(self):
        hg = Hypergraph(graph(), [[1, 2], [3], [4]], bandwidth=10)
        fm = AcyclicFM(hg)
        self.assertEqual(hg.succ[1], {2, 3})
        self.assertEqual(hg.pred[4], {2, 3})
        self.assertEqual(fm._adjacent_targets(2, hg.groups, 0), [2])
        self.assertIn(2, fm._active(hg.groups))

    def test_copy_paths_are_contracted_like_the_official_validator(self):
        copy_path = {
            'ops': [
                {'id': 1, 'op': 'A', 'pipe': 'p0', 'cycles': 1},
                {'id': 2, 'op': 'COPY_OUT', 'pipe': 'p0', 'cycles': 0},
                {'id': 3, 'op': 'COPY_IN', 'pipe': 'p0', 'cycles': 0},
                {'id': 4, 'op': 'B', 'pipe': 'p0', 'cycles': 1},
            ],
            'edges': [
                {'source': 1, 'target': 10}, {'source': 10, 'target': 2},
                {'source': 2, 'target': 11}, {'source': 11, 'target': 3},
                {'source': 3, 'target': 12}, {'source': 12, 'target': 4},
            ],
            'tensors': [{'id': 10, 'size': 1}, {'id': 11, 'size': 1},
                        {'id': 12, 'size': 1}],
        }
        hg = Hypergraph(copy_path, [[1], [4]])
        self.assertEqual(hg.succ[1], {4})
        self.assertEqual(hg.pred[4], {1})

    def test_criticality_handles_a_long_tensor_chain_iteratively(self):
        count = 1_201
        chain = {
            'ops': [{'id': op, 'op': 'A', 'pipe': 'p0', 'cycles': 1}
                    for op in range(1, count + 1)],
            'edges': [edge for op in range(1, count) for edge in (
                {'source': op, 'target': 10_000 + op},
                {'source': 10_000 + op, 'target': op + 1})],
            'tensors': [{'id': 10_000 + op, 'size': 1}
                        for op in range(1, count)],
        }
        fm = AcyclicFM(Hypergraph(chain, [list(range(1, count)), [count]]))
        self.assertEqual(fm._criticality(count), count)

    def test_legal_and_illegal_acyclic_moves(self):
        hg = Hypergraph(graph(), [[1], [2], [3], [4]], bandwidth=10)
        self.assertTrue(hg.acyclic([[1, 2], [3], [4]]))
        self.assertFalse(hg.acyclic([[1, 4], [2], [3]]))

    def test_determinism_and_nonempty_groups(self):
        first = refine(graph(), [[1], [2], [3], [4]], bandwidth=10)
        second = refine(graph(), [[1], [2], [3], [4]], bandwidth=10)
        self.assertEqual(first.groups, second.groups)
        self.assertTrue(all(first.groups))

    def test_memory_pressure_changes_objective(self):
        hg = Hypergraph(graph(), [[1], [2], [3], [4]], bandwidth=10, memory_limit=10)
        self.assertGreater(hg.objective(), Hypergraph(graph(), [[1], [2], [3], [4]],
                                                      bandwidth=10).objective())

    def test_best_prefix_rolls_back_negative_tail(self):
        hg = Hypergraph(graph(), [[1], [2], [3], [4]], bandwidth=10)
        result = refine(graph(), hg.groups, bandwidth=10, max_passes=1)
        self.assertLessEqual(result.final_objective, result.initial_objective)
        self.assertEqual(result.best_prefix, len(result.moves))
        self.assertTrue(all(result.groups))


if __name__ == '__main__':
    unittest.main()

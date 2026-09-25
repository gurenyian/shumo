"""Shared FM candidate construction for the core-count refinement runners."""

from acyclic_fm_refinement import Hypergraph, refine
from adaptive_clustering import adaptive_partition, schedule_groups
from hybrid_solver import groups_from_plan, signature
from stub_multicore_cut_and_schedule import MulticoreCutError, derive_multicore_plan
from evaluation_validation import validate_task_order
from fast_solver import GraphModel


def candidate_variants(graph, incumbent, cores, cfg, waits, max_finals=8):
    """Build a bounded list of unique, dependency-valid FM reschedules."""
    groups, _ = groups_from_plan(incumbent)
    gm = GraphModel(graph)
    hg = Hypergraph(graph, groups, cfg['bandwidth'], cfg.get('capacity'))
    refined = refine(graph, groups, cfg['bandwidth'], cfg.get('capacity'), max_active=256)
    # Preserve the original candidate prefix so resumable candidate labels
    # continue to refer to the same plans already recorded in history files.
    pools = [groups, refined.groups]
    for a, children in sorted(hg.quotient_edges(groups).items()):
        for b in sorted(children):
            merged = [list(group) for group in groups]
            merged[a].extend(merged[b])
            merged[b] = []
            compact = [group for group in merged if group]
            if hg.acyclic(compact):
                pools.append(compact)
            if len(pools) >= 5:
                break
        if len(pools) >= 5:
            break

    adaptive_plan, _ = adaptive_partition(gm, cores, cfg, waits)
    adaptive_groups, _ = groups_from_plan(adaptive_plan)
    adaptive_refined = refine(graph, adaptive_groups, cfg['bandwidth'],
                              cfg.get('capacity'), max_active=256)
    pools.extend((adaptive_groups, adaptive_refined.groups))

    plans, seen = [incumbent], {signature(incumbent)}
    for candidate_groups in pools:
        try:
            plan, _ = schedule_groups(gm, candidate_groups, cores, cfg, waits)
            validate_task_order(derive_multicore_plan(graph, plan))
        except (ValueError, RuntimeError, MulticoreCutError):
            continue
        token = signature(plan)
        if token not in seen:
            seen.add(token)
            plans.append(plan)
    return plans[:max_finals]

"""Check that all saved Problem-3 plans and official metrics are consistent."""

import csv
import gzip
import json
from pathlib import Path

from solver_problem2 import validate_plan


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results_problem3'


def main():
    table_path = OUTPUT / 'comparison_100cases.csv'
    with table_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    failures = []
    checked = 0
    graph_cache = {}
    for row in rows:
        case, cores = row['case'], int(row['cores'])
        key = f'{case}_{cores}cores'
        if row['status'] != 'scored':
            failures.append(f'{key}: {row["status"]}')
            continue
        folder = OUTPUT / key
        try:
            assert (ROOT / row['plan_file']).is_file()
            assert (ROOT / row['result_file']).is_file()
            plan = json.loads((folder / 'best_plan.json').read_text(encoding='utf-8'))
            raw_result = folder / 'best_result.json'
            if raw_result.is_file():
                result = json.loads(raw_result.read_text(encoding='utf-8'))
            else:
                with gzip.open(folder / 'best_result.json.gz', 'rt', encoding='utf-8') as stream:
                    result = json.load(stream)
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            if case not in graph_cache:
                graph_cache[case] = json.loads(
                    (ROOT / 'official/data' / f'{case}.json').read_text(encoding='utf-8'))
            validate_plan(graph_cache[case], plan, cores)
            movement = result['data_movement_bytes']
            cache = result['cache_stats']
            assert result['problem'] == 3
            assert result['cache_mode'] == 'read_only'
            assert len(plan['core_schedules']) == cores
            assert int(row['no_l2_makespan']) == summary['problem2_baseline_makespan']
            assert int(row['cache_makespan']) == summary['best']['makespan'] == result['makespan']
            assert (int(row['cache_added_copy_bytes']) ==
                    summary['best']['added_copy_bytes'] == movement['added_copy_bytes'])
            assert abs(float(row['cache_hit_rate']) - cache['hit_rate']) < 1e-9
            assert abs(float(row['l2_relative_speedup']) -
                       int(row['no_l2_makespan']) / result['makespan']) < 1e-9
            checked += 1
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            failures.append(f'{key}: {exc.__class__.__name__}: {exc}')
    report = {'expected_rows': 500, 'table_rows': len(rows),
              'validated_rows': checked, 'failures': failures}
    (OUTPUT / 'validation_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'table_rows': len(rows), 'validated': checked,
                      'failures': len(failures)}, ensure_ascii=False))
    for message in failures[:20]:
        print(message)
    if len(rows) != 500 or failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

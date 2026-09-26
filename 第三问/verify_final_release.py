"""Verify the confirmed V2 release without rerunning optimization or evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'FINAL_RELEASE.json'
RESULTS = ROOT / 'results_problem3_v2_final'


def digest(path):
    hasher = hashlib.sha256()
    if path.suffix in {'.py', '.md', '.csv', '.json', '.txt'}:
        hasher.update(path.read_bytes().replace(b'\r\n', b'\n'))
        return hasher.hexdigest()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def release_files():
    files = list(ROOT.glob('*.py'))
    files += [ROOT / name for name in
              ('README.md', '算法与结果说明.md', '最终结果与论文写法.md')]
    files += list((ROOT / 'official/code').glob('*.py'))
    files += list((ROOT / 'official/data').glob('case_???.json'))
    files += [ROOT / 'official/data/config.txt']
    folders = sorted(RESULTS.glob('case_*_*cores'))
    if len(folders) != 500:
        raise ValueError(f'expected 500 result folders, found {len(folders)}')
    for folder in folders:
        files += [folder / 'best_plan.json', folder / 'best_result.json.gz']
    files += list(RESULTS.glob('*.csv')) + list(RESULTS.glob('*.png'))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', action='store_true',
                        help='record an explicitly confirmed release; default only verifies')
    args = parser.parse_args()
    if args.create:
        report = json.loads((RESULTS / 'validation_report.json').read_text(encoding='utf-8'))
        if report['validated_rows'] != 500 or report['failures']:
            raise ValueError('the final result set must pass 500/500 validation first')
        with (RESULTS / 'average_speedup_curves.csv').open(encoding='utf-8-sig') as stream:
            speeds = list(csv.DictReader(stream))
        manifest = {
            'release': 'PROBLEM3_V2_FINAL_CONFIRMED',
            'confirmed_date': '2026-09-26',
            'final_entrypoint': 'optimize_problem3_v2.py',
            'final_results_directory': 'results_problem3_v2_final',
            'original_results_directory': 'results_problem3_v1_original',
            'cases': 100, 'cores': [1, 2, 3, 4, 5],
            'plans': 500, 'official_results': 500, 'validated_rows': 500,
            'mean_cache_speedup_by_core': {
                row['cores']: row['mean_cache_speedup'] for row in speeds},
            'hash_scope': 'final code, official evaluator and inputs, documentation, plans, compressed official results, tables, figures; portable summary paths excluded; text line endings normalized to LF',
            'sha256': {path.relative_to(ROOT).as_posix(): digest(path)
                       for path in release_files()},
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    failures = []
    for relative, expected in manifest['sha256'].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            failures.append(relative)
    print(json.dumps({'release': manifest['release'],
                      'checked_files': len(manifest['sha256']),
                      'failures': failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

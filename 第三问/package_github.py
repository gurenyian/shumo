"""Export the labeled V1 original and V2 final Problem-3 package."""

import argparse
import csv
import gzip
import json
import re
import shutil
from pathlib import Path


SOURCE = Path(__file__).resolve().parent
CODE = (
    'adaptive_clustering.py', 'experiment.py', 'fast_solver.py',
    'hybrid_solver.py', 'region_solver.py', 'solver_problem2.py',
    'solver_problem3.py', 'run_problem3_all.py', 'optimize_problem3_v2.py',
    'validate_results.py',
    'test_problem3.py',
)
DOCUMENTS = ('README.md', '算法与结果说明.md', '最终结果与论文写法.md',
             'README_V1_原始方案.md', '算法与结果说明_V1_原始方案.md',
             '最终结果与论文写法_V1_原始方案.md')


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def save_json(destination, value):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')


def compress_json(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as input_stream, destination.open('wb') as output_stream:
        with gzip.GzipFile(filename='', fileobj=output_stream, mode='wb',
                           compresslevel=6, mtime=0) as compressed:
            shutil.copyfileobj(input_stream, compressed, length=1024 * 1024)


def portable_summary(source, folder):
    value = json.loads(source.read_text(encoding='utf-8'))
    case, cores = value['case'], value['cores']
    value['problem2_baseline_plan'] = (
        None if cores == 1 else
        f'results_problem2/{case}_{cores}cores/best_plan.json')
    value['best_plan_path'] = f'{folder}/best_plan.json'
    value['best_result_path'] = f'{folder}/best_result.json.gz'
    if value.get('version') == 'V2_FINAL':
        value['original_result_folder'] = (
            f'results_problem3_v1_original/{case}_{cores}cores')
    return value


def export_baseline_table(destination):
    source = SOURCE / 'results_problem2/batch_results.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    for row in rows:
        case, cores = row['case'], int(row['cores'])
        row['plan_file'] = ('' if cores == 1 else
                            f'results_problem2/{case}_{cores}cores/best_plan.json')
        row['result_file'] = ''  # Q2 raw evaluator outputs are not part of this package.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(destination):
    if destination.resolve() == SOURCE.resolve():
        raise ValueError('destination must differ from the source package')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'destination is not empty: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    for name in CODE + DOCUMENTS:
        copy_file(SOURCE / name, destination / name)
    github_note = (
        '\n## GitHub 压缩结果说明\n\n'
        '为完整保留官方原始结果并控制仓库体积，本仓库的每个 '
        '`best_result.json` 均以无损 `best_result.json.gz` 保存。'
        '`run_problem3_all.py` 会跳过已有的压缩结果；'
        '`validate_results.py` 可直接读取并验证，无需先解压。'
        '若需要展开单个文件，可用 Python 标准库 `gzip`。'
        '候选过程中的临时计划和追踪文件未纳入交付，'
        '最终方案、官方原始结果、搜索历史和汇总统计均已保留。\n'
    )
    with (destination / 'README.md').open('a', encoding='utf-8') as stream:
        stream.write(github_note)
    paper = destination / '最终结果与论文写法.md'
    paper.write_text(paper.read_text(encoding='utf-8').replace(
        '`best_result.json`', '`best_result.json.gz`（无损压缩的官方原始 JSON）'),
        encoding='utf-8')

    for source in (SOURCE / 'official/code').glob('*.py'):
        copy_file(source, destination / 'official/code' / source.name)
    for source in (SOURCE / 'official/data').glob('case_???.json'):
        if re.fullmatch(r'case_\d{3}\.json', source.name):
            copy_file(source, destination / 'official/data' / source.name)
    copy_file(SOURCE / 'official/data/config.txt',
              destination / 'official/data/config.txt')

    export_baseline_table(destination / 'results_problem2/batch_results.csv')
    copy_file(SOURCE / 'results_problem2/average_speedup.csv',
              destination / 'results_problem2/average_speedup.csv')
    baseline_plans = list((SOURCE / 'results_problem2').glob('case_*_*cores/best_plan.json'))
    if len(baseline_plans) != 400:
        raise ValueError(f'expected 400 Problem-2 plans, found {len(baseline_plans)}')
    for source in baseline_plans:
        copy_file(source, destination / 'results_problem2' /
                  source.parent.name / source.name)

    original_bytes = compressed_bytes = 0
    counts = {}
    for version in ('results_problem3_v1_original', 'results_problem3_v2_final'):
        results_source = SOURCE / version
        for source in results_source.iterdir():
            if source.is_file() and (source.suffix in {'.csv', '.png', '.txt'} or
                                     source.name == 'validation_report.json'):
                copy_file(source, destination / version / source.name)
        if version == 'results_problem3_v1_original':
            for name in ('comparison_100cases.csv', 'appendix_100cases.csv'):
                table = destination / version / name
                with table.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    columns, rows = reader.fieldnames, list(reader)
                for row in rows:
                    row['result_file'] = row['result_file'].replace(
                        'best_result.json', 'best_result.json.gz')
                with table.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
        folders = sorted(p for p in results_source.iterdir()
                         if p.is_dir() and re.fullmatch(r'case_\d{3}_[1-5]cores', p.name))
        if len(folders) != 500:
            raise ValueError(f'expected 500 {version} folders, found {len(folders)}')
        counts[version] = len(folders)
        for index, folder in enumerate(folders, 1):
            rel = Path(version) / folder.name
            target = destination / rel
            copy_file(folder / 'best_plan.json', target / 'best_plan.json')
            copy_file(folder / 'search_history.json', target / 'search_history.json')
            save_json(target / 'summary.json', portable_summary(folder / 'summary.json', rel.as_posix()))
            raw = folder / 'best_result.json'
            packed = target / 'best_result.json.gz'
            if raw.is_file():
                compress_json(raw, packed)
                original_bytes += raw.stat().st_size
            else:
                copy_file(folder / 'best_result.json.gz', packed)
            compressed_bytes += packed.stat().st_size
            if index % 100 == 0:
                print(f'Packed {version} {index}/500 official results', flush=True)

    for name in ('refine_case_067_2cores', 'refine_case_073_2cores'):
        source = SOURCE / 'experiments' / name
        target = destination / 'experiments' / name
        if not source.is_dir():
            continue
        copy_file(source / 'best_plan.json', target / 'best_plan.json')
        copy_file(source / 'search_history.json', target / 'search_history.json')
        save_json(target / 'summary.json', portable_summary(
            source / 'summary.json', (Path('experiments') / name).as_posix()))
        compress_json(source / 'best_result.json', target / 'best_result.json.gz')

    manifest = {
        'official_cases': 100,
        'problem2_baseline_plans': len(baseline_plans),
        'problem3_v1_original_plans': counts['results_problem3_v1_original'],
        'problem3_v2_final_plans': counts['results_problem3_v2_final'],
        'problem3_official_results_gzip': sum(counts.values()),
        'official_result_original_bytes': original_bytes,
        'official_result_compressed_bytes': compressed_bytes,
        'lossless': True,
        'validation': 'See results_problem3_v2_final/validation_report.json',
    }
    save_json(destination / 'PACKAGE_MANIFEST.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.destination.resolve())


if __name__ == '__main__':
    main()

"""按用例顺序批量运行问题一：官方单核基准和 2～5 核混合求解器。

可中断后用同一命令继续；每完成一个配置就更新 results.csv。
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "official" / "data" / "config.txt"
CASES = tuple(f"case_{index:03d}" for index in range(1, 101))
FIELDS = ("case", "cores", "status", "makespan_cycles", "added_copy_bytes",
          "speedup", "official_score_attempts", "seconds_budget", "initial_score_timeout", "method",
          "result_file", "plan_file")


def newest_results_table(output):
    candidates = [path for path in (output / "results.csv", output / "results_latest.csv")
                  if path.exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def single_worker(case, output, fast_evaluator=False):
    code_root = "optimized_official" if fast_evaluator else "official"
    sys.path.insert(0, str(ROOT / code_root / "code"))
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_1 import read_scene_a_config
    from singlecore_evaluate import evaluate_singlecore

    graph = read_json(ROOT / "official" / "data" / f"{case}.json")
    settings = read_evaluation_config(str(CONFIG))
    waits = read_scene_a_config(str(CONFIG))
    result = evaluate_singlecore(
        graph, bandwidth=settings["bandwidth"], capacity=settings["capacity"],
        cross_core_wait=waits["task_cross_core_wait_cycles"],
        same_core_wait=waits["task_same_core_wait_cycles"],
    )
    result["input_graph"] = f"{case}.json"
    write_json(output, result)


def scored_multicore(folder):
    summary_file, result_file, plan_file = (folder / name for name in
                                            ("summary.json", "best_result.json", "best_plan.json"))
    if not all(path.exists() for path in (summary_file, result_file, plan_file)):
        return False
    try:
        return read_json(summary_file).get("status") == "scored"
    except (OSError, ValueError):
        return False


def refresh_tables(output, errors):
    rows = []
    for case in CASES:
        base_file = output / "singlecore" / f"{case}_result.json"
        base = read_json(base_file) if base_file.exists() else None
        baseline = base["makespan"] if base else None
        for cores in range(1, 6):
            folder = output / "multicore" / f"{case}_{cores}cores"
            result_file = base_file if cores == 1 else folder / "best_result.json"
            plan_file = "" if cores == 1 else str(folder / "best_plan.json")
            if cores == 1 and base is not None:
                result, status, method, attempts, budget, score_timeout = (
                    base, "scored", "singlecore", 1, "", "")
            elif cores > 1 and scored_multicore(folder):
                result = read_json(result_file)
                summary = read_json(folder / "summary.json")
                status, method = "scored", "hybrid"
                attempts = summary["official_score_attempts"]
                budget = summary.get("seconds_budget", "")
                score_timeout = summary.get("initial_score_timeout", "")
            else:
                result = None
                status = errors.get(f"{case}:{cores}", {}).get("status", "pending")
                method, attempts, budget, score_timeout = "", "", "", ""
            rows.append({
                "case": case, "cores": cores, "status": status,
                "makespan_cycles": result["makespan"] if result else "",
                "added_copy_bytes": result["data_movement_bytes"]["added_copy_bytes"] if result else "",
                "speedup": baseline / result["makespan"] if result and baseline else "",
                "official_score_attempts": attempts,
                "seconds_budget": budget,
                "initial_score_timeout": score_timeout,
                "method": method,
                "result_file": str(result_file) if result else "",
                "plan_file": plan_file if result else "",
            })
    temporary = output / "results.csv.writing"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    target = output / "results.csv"
    for attempt in range(3):
        try:
            temporary.replace(target)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.25)
            else:
                # Windows spreadsheet applications can keep results.csv locked.
                # Preserve the new complete snapshot at a separate stable path.
                temporary.replace(output / "results_latest.csv")
                print("results.csv 被占用；最新汇总已写入 results_latest.csv", flush=True)
    write_json(output / "errors.json", errors)
    averages = []
    for cores in range(1, 6):
        values = [row["speedup"] for row in rows
                  if row["cores"] == cores and row["speedup"] != ""]
        averages.append({"cores": cores, "finished_cases": len(values),
                         "average_speedup_so_far": sum(values) / len(values) if values else None,
                         "complete_100_cases": len(values) == 100})
    write_json(output / "average_speedup_progress.json", averages)


def run_job(command, timeout):
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", f"超过 {timeout:g} 秒外层上限"
    if process.returncode:
        return "error", (process.stderr.strip() or process.stdout.strip())[-1200:]
    return "ok", process.stdout.strip()[-300:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1, help="起始用例编号，默认 1")
    parser.add_argument("--end", type=int, default=100, help="结束用例编号，默认 100")
    parser.add_argument("--cores", type=int, nargs="+", choices=(2, 3, 4, 5),
                        default=(2, 3, 4, 5), help="要运行的多核配置，默认 2 3 4 5")
    parser.add_argument("--seconds", type=float, default=120,
                        help="每个多核配置的求解预算秒数，默认 120")
    parser.add_argument("--single-timeout", type=float, default=600,
                        help="每例单核官方评分最长秒数，默认 600")
    parser.add_argument("--initial-score-timeout", type=float, default=120,
                        help="每个多核初始候选的单次官方评分最长秒数，默认 120")
    parser.add_argument("--max-evals", type=int, default=8,
                        help="每个多核配置最多官方评分次数，默认 8")
    parser.add_argument("--starts", type=int, choices=(1, 3, 5), default=3,
                        help="多核构造起点数，默认 3")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_all100",
                        help="结果目录，默认本工程的 results_all100")
    parser.add_argument("--fast-evaluator", action="store_true",
                        help="使用经过等价性对照的事件模拟加速副本")
    parser.add_argument("--single-worker", metavar="CASE", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_worker:
        single_worker(args.single_worker, args.worker_output, args.fast_evaluator)
        return 0
    if not (1 <= args.start <= args.end <= 100):
        parser.error("用例范围必须满足 1 <= --start <= --end <= 100")
    if (args.seconds <= 0 or args.single_timeout <= 0 or
            args.initial_score_timeout <= 0 or args.max_evals < 1):
        parser.error("时间预算和 --max-evals 必须为正数")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {"config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                "max_evals": args.max_evals, "starts": args.starts}
    settings_file = output / "run_settings.json"
    if settings_file.exists():
        previous_settings = read_json(settings_file)
        previous_settings.pop("single_timeout", None)
        previous_settings.pop("seconds", None)
        if previous_settings != settings:
            parser.error("此结果目录使用了不同参数；请保持原参数继续，或改用新的 --output-root")
    write_json(settings_file, settings)
    errors_file = output / "errors.json"
    errors = read_json(errors_file) if errors_file.exists() else {}
    refresh_tables(output, errors)
    total = (args.end - args.start + 1) * (1 + len(set(args.cores)))
    step = 0
    for index in range(args.start, args.end + 1):
        case = f"case_{index:03d}"
        base_file = output / "singlecore" / f"{case}_result.json"
        step += 1
        if base_file.exists():
            errors.pop(f"{case}:1", None)
            print(f"[{step}/{total}] {case} 单核：已有结果，跳过", flush=True)
        else:
            print(f"[{step}/{total}] {case} 单核：开始", flush=True)
            base_file.parent.mkdir(parents=True, exist_ok=True)
            worker_command = [sys.executable, str(Path(__file__).resolve()), "--single-worker", case,
                              "--worker-output", str(base_file)]
            if args.fast_evaluator:
                worker_command.append("--fast-evaluator")
            status, message = run_job(worker_command, args.single_timeout)
            if status == "ok" and base_file.exists():
                errors.pop(f"{case}:1", None)
                print(f"    完成，Makespan={read_json(base_file)['makespan']}", flush=True)
            else:
                errors[f"{case}:1"] = {"status": status, "message": message}
                print(f"    {status}: {message}", flush=True)
            refresh_tables(output, errors)
        for cores in sorted(set(args.cores)):
            step += 1
            folder = output / "multicore" / f"{case}_{cores}cores"
            if scored_multicore(folder):
                errors.pop(f"{case}:{cores}", None)
                print(f"[{step}/{total}] {case} {cores}核：已有结果，跳过", flush=True)
                continue
            print(f"[{step}/{total}] {case} {cores}核：开始", flush=True)
            command = [sys.executable, str(ROOT / "hybrid_solver.py"),
                       "--case", case, "--cores", str(cores),
                       "--max-evals", str(args.max_evals), "--starts", str(args.starts),
                       "--seconds", str(args.seconds),
                       "--initial-score-timeout", str(args.initial_score_timeout),
                       "--output-root", str(output / "multicore")]
            if args.fast_evaluator:
                command.append("--fast-evaluator")
            status, message = run_job(command, args.seconds + 60)
            if status == "ok" and scored_multicore(folder):
                errors.pop(f"{case}:{cores}", None)
                result = read_json(folder / "best_result.json")
                print(f"    完成，Makespan={result['makespan']}，额外搬运="
                      f"{result['data_movement_bytes']['added_copy_bytes']} 字节", flush=True)
            else:
                summary_file = folder / "summary.json"
                if summary_file.exists():
                    detail = read_json(summary_file)
                    if detail.get("status") == "initial_score_unavailable":
                        history_file = folder / "search_history.json"
                        history = read_json(history_file) if history_file.exists() else []
                        first_status = detail.get("initial_score_status") or (
                            history[0].get("status") if history else "unknown")
                        status = "score_timeout" if first_status == "timeout" else "score_unavailable"
                        message = (f"首次官方评分未完成：{first_status}；"
                                   f"本配置预算 {args.seconds:g} 秒，无有效 Makespan")
                if status == "ok":
                    status = "no_score"
                errors[f"{case}:{cores}"] = {"status": status, "message": message}
                print(f"    {status}: {message}", flush=True)
            refresh_tables(output, errors)
    refresh_tables(output, errors)
    print("本次指定范围运行结束。查看：", output / "results.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

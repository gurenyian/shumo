# 第一问最终交付包

这是一个可独立运行的第一问工程目录，包含最终算法代码、运行入口、官方评估器、100 个用例数据、2～5 核方案和最终统计结果。

## 最终结果

| 核数 | 100 例平均加速比 |
|---:|---:|
| 1 | 1.000× |
| 2 | 1.600× |
| 3 | 2.161× |
| 4 | 2.634× |
| 5 | **4.031×** |

五核结果相较原方案有 66 个用例的 Makespan 降低。最终 500 行评分表在 `final_results/final_results_100cases.csv`；100 个最终五核方案和 2～4 核方案在 `final_results/plans/`，文件名形如 `case_003_5cores.json`。方案表中的路径是相对交付包目录的路径。

算法瓶颈、分阶段对比、通信与并行权衡见 `final_results/reports/优化结果说明.md` 和 `final_results/reports/ablation.csv`。加速比折线图见 `final_results/reports/average_speedup_before_after.png`。

## 文件结构

- `adaptive_clustering.py`：通信成本与并行损失共同驱动的 DAG 聚类。
- `hybrid_solver.py`：单用例混合求解器，含切图候选、列表调度和局部搜索。
- `refine_partition_schedule.py`：按实测任务时长和搬运量精调五核方案。
- `fast_solver.py`、`region_solver.py`：基础聚类、依赖区域划分和 HEFT 风格分核。
- `rebalance_heavy_tasks.py`、`rescue_fivecore.py`、`search_low_speedup.py`：前序通用候选生成与五核评分流程。
- `run_problem1_all.py`、`complete_missing_scores.py`：批量基础评分和补齐缺失评分。
- `experiment.py`：图读取、依赖处理、配置读取及评估辅助代码。
- `official/code/`：未修改的官方代码。
- `optimized_official/code/`：经代表用例等价对照的事件模拟加速副本；规则、缓存容量和 DDR 约束未改。
- `official/data/`：全部 100 个题目数据和配置。
- `final_results/plans/`：400 个 2～5 核最终方案；其中五核采用最新优化结果。

## 单例运行

在此目录打开 PowerShell。使用 Python 3.10 或更新版本；核心算法只依赖 Python 标准库。可先对一个用例评分：

```powershell
python -X utf8 -u hybrid_solver.py --case case_003 --cores 5 --starts 3 --max-evals 8 --seconds 120 --fast-evaluator
```

结果写入 `results_hybrid/`。也可以直接复核已整理的五核方案：

```powershell
python optimized_official/code/multicore_cut_evaluate_problem_1.py official/data/case_003.json final_results/plans/case_003_5cores.json --config official/data/config.txt -o results_hybrid/case_003_check.json
```

## 从头批量复现

运行 `run_final_pipeline.ps1` 会按顺序评分 1～100 例、补跑已知超时配置、生成五核候选、执行局部精调并更新结果。官方完整评分可能运行较久；程序会逐步保存结果，可中断后用同一命令续跑。

```powershell
powershell -ExecutionPolicy Bypass -File .\run_final_pipeline.ps1
```

## 对 2～5 核统一重新优化

`optimize_all_cores.py` 会读取本目录现有的 100 例评分和 2～5 核方案，对每个用例、每种核数分别运行通信感知自适应切图候选，并用评估器正式评分；只有 Makespan 更优（相同时额外搬运更少）才会替换该配置的方案。单核仍保留为题目要求的单核基准。该实验写入新的 `results_allcores_adaptive/`，不会覆盖当前最终结果。

在交付目录打开 PowerShell，运行完整的 100 例 × 2～5 核批处理：

```powershell
python -X utf8 -u optimize_all_cores.py --cores 2 3 4 5 --penalties 1.0 2.0 --timeout 120
```

每个配置有两个候选（共最多 800 次候选评分）。脚本会逐项保存进度，可用同一命令续跑；超时和评估错误的候选会在续跑时重试。结果写到 `results_allcores_adaptive/final_results_all_cores.csv` 和 `results_allcores_adaptive/average_speedup_all_cores.csv`。先小规模验证时可加 `--cases case_001 case_002`。

若只想复现最终精调阶段，先确保本目录已有 `results_all100/` 基线评分以及 `results_opt5/` 候选结果，再运行：

```powershell
python -X utf8 -u refine_partition_schedule.py --threshold 4.5 --penalties 1.0 2.0 --timeout 45
python -X utf8 -u refine_partition_schedule.py --threshold 4.5 --penalties 1.0 2.0 --timeout 45 --traffic-feedback
```

脚本只在正式评分变好时更新最佳方案。`optimized_official` 用于提高事件模拟速度；原版评估器仍保留，便于独立复核。



# 第二问：场景 B 多核切图与调度

本目录是可独立运行的第二问工程，包含求解器、官方评估器、100 个输入用例、第一问基准方案和第二问最终评分结果。运行环境建议 Python 3.10 或更新版本；求解器主体使用 Python 标准库，生成加速比图需要 `matplotlib`。

## 当前结果

| 核数 | 100 例平均加速比 |
|---:|---:|
| 1 | 1.000× |
| 2 | 2.011× |
| 3 | 2.924× |
| 4 | 3.722× |
| 5 | **4.439×** |

汇总表 [`results_problem2/final_results_100cases.csv`](results_problem2/final_results_100cases.csv) 包含 100 个用例、1～5 核的 Makespan、加速比和额外搬运量。多核方案在 `results_problem2/case_NNN_Kcores/best_plan.json`，共 400 份。平均加速比 CSV 和图也保存在 `results_problem2/`。

加速比逐例按“第一问单核 Makespan ÷ 第二问对应核数 Makespan”计算，再对 100 例取算术平均。表中额外搬运量含图划分通信和缓存溢出搬运。评估器结果按 Makespan 优先、额外搬运量其次选择方案。

## 代码与目录

- `solver_problem2.py`：单个用例和核数的场景 B 求解器；提供候选生成、有界局部搜索、合法性验证和官方评估。
- `run_problem2_all.py`：批量运行 100 个用例、2～5 核，保存逐配置汇总和平均加速比图。
- `adaptive_clustering.py`、`fast_solver.py`、`hybrid_solver.py`、`region_solver.py`、`experiment.py`：求解器依赖的切图、图建模和公共辅助模块。
- `official/`：题目评估器、评估辅助模块、100 个用例和固定配置。
- `final_results/final_results_100cases.csv`：第一问 1 核基准 Makespan，供第二问计算加速比。
- `results_allcores_adaptive/case_NNN_Kcores/best_plan.json`：第二问求解器使用的第一问自适应多核起点。
- `results_problem2/`：第二问当前汇总、400 份最终方案及加速比图。

最终方案与评分表已保留；完整搜索日志、每个候选的中间文件和展开的逐操作模拟时间线没有打包进仓库。需要时可用随附评估器对最终方案重新评分，生成详细时间线。

## 运行

在本目录打开终端，运行全部 100 个用例的 2～5 核配置：

```powershell
python -X utf8 -u run_problem2_all.py --cores 2 3 4 5 --workers 5 --max-evals 12 --seconds 300 --eval-timeout 120
```

脚本会跳过已经同时存在 `summary.json`、`best_plan.json` 和 `best_result.json` 的配置。运行结果默认写到 `results_problem2/`。调试单个配置可以运行：

```powershell
python -X utf8 -u solver_problem2.py official/data/case_010.json -n 4 --config official/data/config.txt --max-evals 12 --seconds 300 --eval-timeout 120
```

如需从头重新评分，可在批量命令中加 `--force`；这会重新评估已完成的配置。

## 复核单个最终方案

例如用题目官方问题二评估器复核 case_010 的四核方案：

```powershell
python official/code/multicore_cut_evaluate_problem_2.py official/data/case_010.json results_problem2/case_010_4cores/best_plan.json --config official/data/config.txt -o results_problem2/case_010_4cores_check.json
```

评估器输出的 `makespan` 是总完成周期数，`added_copy_bytes` 是相对原图额外增加的搬运量，`spill_added_copy_bytes` 是缓存溢出导致的额外搬运。场景 B 的跨核复制延迟固定为 500 cycle，DDR 带宽为 60 B/cycle；具体值以 `official/data/config.txt` 为准。

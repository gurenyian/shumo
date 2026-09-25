# 第一问求解与结果复现

本目录的求解器、优化实验和最终结果聚焦赛题第一问。目录内同时保留了官方提供的第一、二、三问评测器代码和它们共用的输入、调度及校验模块；这不表示本目录已经包含第二问、第三问各自的求解方案。

## 当前最终结果

最终逐例表：`results/final/final_results_100cases.csv`，共 100 个用例、1～5 核共 500 行，包含官方 Makespan、总额外数据搬运、相对固定单核基准的加速比、算法来源和方案路径。表中的 `added_copy_bytes` 是官方 `data_movement_bytes` 结果，等于切分增加的搬运量加上 spill 增加的搬运量。

| 核数 | 100 例平均加速比 |
|---:|---:|
| 1 | 1.000× |
| 2 | 1.833× |
| 3 | 2.621× |
| 4 | 3.302× |
| 5 | **4.080×** |

平均加速比：`results/final/final_average_speedup.csv`；曲线：`results/final/reports/average_speedup.svg`。平均值是 100 个逐例加速比的算术平均；图中虚线 `y=k` 表示理想线性加速参考。由于单核基准来自启发式调度而非最优调度，这条线是参考上限而非严格数学上界，个别逐例加速比可以超过核数。单核基准固定，2～5 核结果来自 FM 精调。逐例方案在 `results/final/plans/`，路径相对于 `results/final/`。

算法讨论和阶段对照见 `results/final/reports/优化结果说明.md`、`results/final/reports/ablation.csv`；文献与建模说明见 `算法文献与建模说明.md` 和 `docs/`。

## 复现最终结果

要求 Python 3.10 或更新版本。进入本目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_final_pipeline.ps1
```

脚本依次对 2～4 核和 5 核做有界 FM 精调，使用未修改的官方评测器评分，再导出表格、均值和曲线。默认 50 个 worker、每次官方评分超时 600 秒；两个 FM 阶段均可续跑。

每个核数独立生成自己的图划分和候选：k 核候选只用输入图、k 值和本核数已有方案构造，不依赖其他核数结果。共享候选生成器同时保留原方案局部精调，并加入可以从 DAG 重新形成更细任务组的自适应切分候选，避免 FM 只能在过粗初始分组中移动。最终仍由官方评测结果选择 Makespan 更优的方案；单核基准固定不变。

仓库保留最终结果，不上传实验中间产物。要继续执行精调流水线，需在本地准备 `results/experiments/results_iterative_5core/results.csv` 及其引用的五核方案；最终表中的固定单核基准和 2～4 核方案也必须可读。五核精调结果会写入 `results/experiments/results_fm_5core/`。

直接运行精调阶段：

```powershell
python -X utf8 -u optimize_fm_all_cores.py --cores 2 3 4 --workers 50 --timeout 600
```

可用 `--cases case_001 case_002` 仅运行指定用例；汇总会保留未选中用例已有的结果。汇总导出：

```powershell
python -X utf8 -u generate_final_results.py
```

## 代码分层

### 第一问主求解与最终链路

- `run_final_pipeline.ps1`：最终 FM 精调及报表的唯一推荐批处理入口。
- `optimize_fm_all_cores.py`、`optimize_fm_5core.py`：2～4 核及五核 FM 运行器。
- `fm_candidates.py`、`acyclic_fm_refinement.py`：共用候选构造和无环 FM 边界细化。
- `generate_final_results.py`：合并结果、复制最终方案、计算逐例加速比和均值，并生成 SVG 曲线。
- `hybrid_solver.py`：单用例候选搜索入口；`adaptive_clustering.py`、`fast_solver.py`、`region_solver.py` 提供聚类、调度及区域划分能力。
- `experiment.py`：第一问实验共用的读取、配置、官方评分调用和文件辅助函数。

### 可选优化实验与兼容入口

`feedback_refinement.py`、`iterative_refinement.py`、`multilevel_partition.py` 及其 `optimize_*` 运行器用于比较其他候选方法；`optimize_all_cores.py`、`rescue_fivecore.py`、`rebalance_heavy_tasks.py`、`search_low_speedup.py`、`refine_partition_schedule.py`、`run_problem1_all.py`、`complete_missing_scores.py` 和 `summarize_speedups.py` 是前序流程、单独实验或结果补算入口。它们不是当前最终 FM 汇总链的一部分，但部分模块互相导入，且仍可作为独立 CLI 使用；不要只因未被主链调用就删除。

回归测试集中在 `tests/`，运行：

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

### 赛题评测器

- `official/`：官方评测器、100 个用例和配置；保持原样。
- `optimized_official/`：事件模拟加速副本，供其他实验使用；最终 FM 链调用 `official/`。
- 评测器代码包含第一、二、三问及共享校验/调度模块。本目录的算法脚本目前以第一问为研究对象。

## 结果和文档目录

- `results/final/`：最终逐例表、平均加速比、最终方案和说明报告。
- `results/experiments/`：本地基础评分、候选搜索、FM 运行历史、方案及其他实验产物；不提交到远端。
- `docs/`：数学建模和版本差异说明。
- `算法文献与建模说明.md`：算法参考文献和建模对应关系。

生成的数据、候选方案和日志集中在 `results/`，测试集中在 `tests/`。根目录保留可直接导入的求解模块和运行入口；它们使用同目录模块导入，整体移动前需要一并改成包导入并验证所有命令行入口。当前盘点没有发现可仅凭未被主链引用就安全删除的算法脚本。

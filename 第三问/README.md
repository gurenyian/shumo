# 第三问交付说明：V2 是最终结果

**最终版本已于 2026-09-26 经用户确认锁定为 V2。** 本次交付不启用后续数学建模草案，也不重新搜索或替换已验证成绩。`FINAL_RELEASE.json` 记录最终代码与结果的 SHA-256 指纹，可用 `python -X utf8 verify_final_release.py` 检查文件是否被更改。

**请只从 `results_problem3_v2_final` 读取第三问最终成绩和方案。**
`results_problem3_v1_original` 是本次优化之前的原始结果，用于复核改进幅度，不能与 V2 混作同一批最终成绩。两套目录都含 100 个 case × 1～5 核，共 500 组；每组有最终切图分核方案、官方评估结果、汇总和搜索记录。

| 内容 | 原始版本 V1 | 最终版本 V2 |
|---|---|---|
| 结果目录 | `results_problem3_v1_original/` | **`results_problem3_v2_final/`** |
| 搜索入口 | `solver_problem3.py`、`run_problem3_all.py` | **`optimize_problem3_v2.py`** |
| 逐组方案 | `case_NNN_Kcores/best_plan.json` | **`case_NNN_Kcores/best_plan.json`** |
| 官方评估结果 | `best_result.json`，GitHub 包内为 `.gz` | **`best_result.json.gz`** |
| 汇总表 | `comparison_100cases.csv` | **`comparison_100cases.csv`** |
| 版本对比 | — | **`V1_vs_V2_comparison.csv`** |

`solver_problem3.py`、`run_problem3_all.py` 同时为 V2 提供官方评估接口与汇总函数，`solver_problem2.py` 提供方案验证和公共图模型；这些公共依赖必须保留，不能因为文件名含旧问题或历史搜索入口就删除。最终运行入口仍为 `optimize_problem3_v2.py`。

V2 以每组 V1 方案和官方评分为保底。从 V1 官方时间线里找出最晚结束核心及其高 DDR miss 子图，尝试将这些子图在同核顺序中提前或延后，或在有明显负载余量时移到其他核。变动前先检查全局依赖图是否仍然无环；候选再交给赛题第三问官方评估器。只有官方 `(Makespan, 额外搬运字节数)` 按字典序严格改善，才会替换保底方案。因此 V2 中未改善的组明确标为 `V1_ORIGINAL_UNCHANGED`，并直接保留原方案和官方结果；改善的组标为 `V2_FEEDBACK_SEARCH`。缓存命中率是诊断指标，不是接受条件。

V2 对超过 10,000 个非 COPY 算子的超大图保留已经正式评分的 V1 方案，避免邻域构造占用过多内存；其 `search_profile` 会标记 `baseline_only_large_graph`。这类配置也完整保存在 V2 最终目录中。

## 从哪里读最终结果

- 正文曲线：`results_problem3_v2_final/average_makespan_curves.png`、`average_speedup_curves.png`、`average_l2_relative_speedup.png`，对应 CSV 在同目录。
- 附录逐用例：`results_problem3_v2_final/appendix_100cases.csv`，包含无 L2 与只读 L2 的 Makespan、额外搬运量和官方缓存命中字节率。
- 原始与优化逐项对照：`results_problem3_v2_final/V1_vs_V2_comparison.csv`。
- 各核平均优化幅度：`results_problem3_v2_final/V1_vs_V2_average_by_core.csv`。
- 某一用例的实际切图、分核顺序：`results_problem3_v2_final/case_NNN_Kcores/best_plan.json`，其中 `node_to_subgraph` 是算子到子图的映射，`core_schedules` 是各核的子图顺序。
- 某一用例的第三问官方逐算子时间线和缓存事件：同组的 `best_result.json.gz`，可用 Python 标准库 `gzip` 打开。
- 该组结果来自 V1 还是新搜索：同组 `summary.json` 的 `selected_source`；它同时列出 `original_v1`、`best` 和新增官方评估次数。

`cache_hit_rate` 指官方 `hit_bytes / (hit_bytes + miss_bytes)`，按字节计算；没有 L2 的基线不适用命中率。额外搬运量使用官方 `data_movement_bytes.added_copy_bytes`。同一核数下 L2 相对无 L2 加速比为 `M_noL2 / M_L2`；各核平均值先逐 case 求比，再对 100 个 case 取算术平均。1～5 核的跨核加速比以单核无 L2 Makespan 为共同基准。

## 如何复现与核验

在本文件夹执行：

```powershell
python -X utf8 -m unittest test_problem3 -v
python -X utf8 -u optimize_problem3_v2.py --cores 1 2 3 4 5 --workers 5 --max-evals 4 --seconds 120 --eval-timeout 45
python -X utf8 validate_results.py
```

优化脚本默认跳过完整的 V2 结果目录；只有 `--force` 才会重新搜索。若只想根据已保存结果重建曲线和对照表：

```powershell
python -X utf8 optimize_problem3_v2.py --aggregate-only
```

`official/code/` 和 `official/data/` 为赛题评估程序与 100 个输入；`results_problem2/` 是第二问的无 L2 基线。V1 的方法说明保存在带 `_V1_原始方案` 后缀的文档中；最新方法和论文取数见 `算法与结果说明.md`、`最终结果与论文写法.md`。

V2 是有限预算的启发式优化，不是全局最优证明。最终成绩均以保存的官方评估结果为准；代理估计和缓存事件只用于决定尝试顺序。

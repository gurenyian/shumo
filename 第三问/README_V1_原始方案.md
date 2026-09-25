> **历史存档：V1 原始方案，不是最终优化结果。最终代码与数字请见 [README.md](README.md) 和 results_problem3_v2_final。此文中的旧目录名 results_problem3 现为 results_problem3_v1_original。**

# 第三问：共享只读 L2 的多核切图与调度

本目录独立保存第三问所需代码、赛题评估程序与数据、第二问基线方案，以及第三问的逐用例方案和结果。`solver_problem3.py` 生成并比较方案；最终成绩始终取自 `official/code/multicore_cut_evaluate_problem_3.py`，不把启发式估计当作真实 Makespan。

## 问题和评分口径

每个方案包含 `node_to_subgraph`（每个非 COPY 算子所属子图）和 `core_schedules`（各核的子图执行顺序）。场景 B 中，同一核的子图组成一个 Task，同核子图间可以继续利用 L1/UB；跨核中间数据仍需 COPY_OUT、500 cycle 同步等待及 COPY_IN。第三问在此基础上加入所有核心共享的只读 L2。当前赛题配置为容量 1,048,576 B、L2 读取带宽 250 B/cycle、DDR 读取带宽 60 B/cycle，两个带宽池独立。

官方评估器在 COPY_IN 发射时查 L2；miss 在 DDR 读取完成后插入，容量不足时按 FIFO 淘汰，hit 不刷新 FIFO 次序。同一数据的并发读取可能同时 miss。只读 L2 并不免除跨核 COPY_OUT 或 500 cycle 等待。程序在候选筛选中使用简化 FIFO 预测器，但只接受官方评估的 `(Makespan, added_copy_bytes)` 字典序改善。

## 算法流程

1. 读取第二问各 `(case, core)` 的 `best_plan.json` 作为基线，在第三问官方评估器中重新计分。1 核使用整图单 Task 方案，并单独用第二问评估器计算无 L2 基线。
2. 从图中提取子图依赖、算子 Pipe 工作量、L1/UB 压力、张量大小及消费者，识别适合跨核复用的 Hot Tensor。潜在收益近似为 `(读取核心数−1) × 大小 × (1/60−1/250)`，并结合关键路径权重排序。
3. 用轻量 FIFO 预测模型估计 hit/miss 字节、缓存淘汰和后续读取损失。该模型按顺序模拟，无法完全表示真实并发，因此仅用于候选排序。
4. 生成问题二的通信亲和/私有缓存方案，以及面向共享 L2 的保守和积极 Scatter、Cluster、FIFO 复用窗口重排、替代切图。Scatter 用 L2 换并行；Cluster 用同核私有缓存减少不友好的重复读取。
5. 复用第二问的 Move、Swap、ChainMove、Recluster 等邻域，增加复用窗口及 Cluster/Scatter 邻域；每轮只把少量高排名和多样性候选交给官方评估器。全程受评估次数、时间和停滞轮数约束，不宣称全局最优。

缓存命中率是诊断量而非最终目标。即使命中率升高，只要官方 Makespan 变差，候选就不会成为最终方案。`search_history.json` 记录每次正式评估的策略、代理预测、官方成绩及是否接受，便于论文解释。

## 运行

在本目录中运行，Python 需能导入 `matplotlib` 才会生成 PNG 曲线；CSV 输出和求解不依赖它。

```powershell
python -X utf8 -m unittest test_problem3 -v
python -X utf8 -u run_problem3_all.py --cores 1 2 3 4 5 --workers 5 --max-evals 8 --seconds 120 --eval-timeout 45
python -X utf8 validate_results.py
```

再次执行同一命令会跳过已有的完整 `(case, core)` 结果，并补跑缺失项。若需要更充分搜索，可将单配置预算调到 `--max-evals 16 --seconds 300 --eval-timeout 120`；使用 `--force` 才会覆盖已完成结果。单个用例示例：

```powershell
python -X utf8 solver_problem3.py official/data/case_010.json -n 4 --config official/data/config.txt --max-evals 16 --seconds 300 --eval-timeout 120
```

若只需根据已保存的 500 份结果重新生成汇总表和曲线：

```powershell
python -X utf8 run_problem3_all.py --aggregate-only
```

## 文件与指标

- `results_problem2/batch_results.csv` 和各 `best_plan.json`：第二问无 L2 基线，作为第三问起点。
- `results_problem3/case_NNN_Kcores/best_plan.json`：该用例该核数的最终切图和分核方案。
- 同目录 `best_result.json`：第三问官方评估器原始结果，包括 `per_core_timeline`、`cache_events`、`data_movement_bytes` 和 `cache_stats`。
- 同目录 `summary.json`、`search_history.json`：对比指标、计算预算及候选接受历史。
- `results_problem3/comparison_100cases.csv`：每用例每核的无 L2 / L2 Makespan、相对加速比、额外搬运量、命中率。
- `results_problem3/appendix_100cases.csv`：可直接筛选生成论文附录的逐用例数据。
- `results_problem3/average_l2_relative_speedup.csv`：对每个核数先逐用例计算 `M_noL2 / M_L2`，再对用例取算术平均。
- `results_problem3/average_l2_relative_speedup.png`：上述相同核数的 L2 相对加速比曲线。
- `results_problem3/average_makespan_curves.csv`、`average_speedup_curves.csv` 和同名 PNG：1～5 核下两配置的对比曲线。
- `results_problem3/average_transfer_and_hit_curves.csv`：逐核数的平均额外搬运字节数与平均官方 L2 命中字节率。
- `results_problem3/ablation_records.csv`：初始阶段各策略经官方评估的候选记录。
- `results_problem3/validation_report.json`：500 行方案、官方原始结果与汇总表的一致性检查。

`cache_hit_rate` 使用官方的**命中字节率** `hit_bytes/(hit_bytes+miss_bytes)`，不是命中次数比例；没有 L2 时该指标留空。`added_copy_bytes` 是官方统计的额外数据搬运字节数；L2 命中降低 DDR 读流量和执行时间，但不一定改变同一切图方案的额外 COPY 字节数。`best_result.json` 保留官方更细的分项，论文中应注明指标定义。

## 论文解释重点

第二问通常把共享输入的消费者聚到同核以节省 DDR 读取。第三问多了共享 L2，某些可缓存的共享输入允许消费者分散到多个核，同时取得并行性和缓存复用；但过度分散会带来跨核传输、同步等待、DDR miss、FIFO 淘汰和 L1/UB spill。算法对 Cluster 与 Scatter 都生成候选，按官方 Makespan 判定优劣。

代理模型的 FIFO 顺序是近似的：官方在 COPY_IN 发射时查缓存、在读取完成时插入，并发 miss 可能彼此重叠。代理计算的“预测命中”不能写作最终命中率。所谓 `optimistic_lower_bound` 只作为宽松比较尺度，不是最优性的证明。

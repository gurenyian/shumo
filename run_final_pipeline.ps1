$ErrorActionPreference = 'Stop'

function Invoke-TaskPython {
    param([string[]]$Arguments)
    & python -X utf8 -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令运行失败，退出码 $LASTEXITCODE ：python $($Arguments -join ' ')"
    }
}

Write-Host '第 1/8 步：为 100 个用例生成基础评分'
Invoke-TaskPython @('run_problem1_all.py', '--fast-evaluator')

Write-Host '第 2/8 步：补齐此前超时的配置'
Invoke-TaskPython @('complete_missing_scores.py', '--cases', '14', '16', '41', '58', '72', '76', '79', '91', '--timeout', '180')

Write-Host '第 3/8 步：生成统一五核候选并评估'
Invoke-TaskPython @('rescue_fivecore.py', '--threshold', '100', '--timeout', '60', '--region-threshold', '4.5', '--sink-threshold', '2')

Write-Host '第 4/8 步：对低加速比用例进行有界局部搜索'
Invoke-TaskPython @('search_low_speedup.py', '--threshold', '2', '--seconds', '45', '--max-evals', '8')

Write-Host '第 5/8 步：比较下游区域与共同来源区域'
Invoke-TaskPython @('rescue_fivecore.py', '--threshold', '100', '--timeout', '60', '--region-threshold', '4.5', '--sink-threshold', '2', '--source-threshold', '2')

Write-Host '第 6/8 步：统一尝试重任务拆分和分叉支路候选'
Invoke-TaskPython @('rebalance_heavy_tasks.py', '--threshold', '2', '--timeout', '60', '--branch-chains', '--branch-min-fraction', '0.15')

Write-Host '第 7/8 步：刷新五核候选结果'
Invoke-TaskPython @('rescue_fivecore.py', '--threshold', '100', '--timeout', '60', '--region-threshold', '4.5', '--sink-threshold', '2', '--source-threshold', '2')

Write-Host '第 8/8 步：执行自适应切图、通信反馈和实测时长重排'
Invoke-TaskPython @('refine_partition_schedule.py', '--threshold', '4.5', '--penalties', '1.0', '2.0', '--timeout', '45')
Invoke-TaskPython @('refine_partition_schedule.py', '--threshold', '4.5', '--penalties', '1.0', '2.0', '--timeout', '45', '--traffic-feedback')

Write-Host '全部完成。最终结果见 final_results/ 和 results_opt5/。'

$ErrorActionPreference = 'Stop'

function Invoke-TaskPython {
    param([string[]]$Arguments)
    & python -X utf8 -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令运行失败，退出码 $LASTEXITCODE ：python $($Arguments -join ' ')"
    }
}

Write-Host '第 1/3 步：对 2～4 核方案执行 FM 精调并使用官方评测器评分'
Invoke-TaskPython @('optimize_fm_all_cores.py', '--cores', '2', '3', '4', '--timeout', '600', '--workers', '50')

Write-Host '第 2/3 步：对五核方案执行共享 FM 候选精调并使用官方评测器评分'
Invoke-TaskPython @('optimize_fm_5core.py', '--timeout', '600', '--workers', '50')

Write-Host '第 3/3 步：汇总固定单核基准与 FM 多核结果，生成逐例表和加速曲线'
Invoke-TaskPython @('generate_final_results.py')

Write-Host '全部完成。最终结果见 results/final/；中间结果见 results/experiments/。'

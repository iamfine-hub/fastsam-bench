# FastSAM 检测台启动脚本
# 用法：右键“使用 PowerShell 运行”，或在终端执行 .\start.ps1 [端口]
param([int]$Port = 8000)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path (Split-Path -Parent $root) ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "找不到 venv：$py" -ForegroundColor Red
    Write-Host "请先按 README 的「环境」一节建好 .venv" -ForegroundColor Yellow
    exit 1
}

Write-Host "启动 http://127.0.0.1:$Port" -ForegroundColor Green
& $py (Join-Path $root "server.py") $Port

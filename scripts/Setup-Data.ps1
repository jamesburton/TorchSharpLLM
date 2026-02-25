<#
.SYNOPSIS
    Setup script: install Python dependencies, fetch datasets, and tokenize.

.DESCRIPTION
    Downloads and prepares training data for TorchSharpLLM MoE model.

.PARAMETER Core
    Download core 8 datasets (one per expert, IDs 0-7).

.PARAMETER All
    Download all 10 datasets.

.PARAMETER Select
    Download specific dataset IDs (e.g., -Select 0,1,7).

.PARAMETER Resume
    Resume interrupted downloads.

.PARAMETER DataDir
    Override the data directory (default: data/ in project root).

.EXAMPLE
    .\scripts\Setup-Data.ps1 -Core
    .\scripts\Setup-Data.ps1 -All
    .\scripts\Setup-Data.ps1 -Select 0,1,2,7
    .\scripts\Setup-Data.ps1 -Select 0,7 -Resume
#>
[CmdletBinding(DefaultParameterSetName = 'Interactive')]
param(
    [Parameter(ParameterSetName = 'Core')]
    [switch]$Core,

    [Parameter(ParameterSetName = 'All')]
    [switch]$All,

    [Parameter(ParameterSetName = 'Select')]
    [int[]]$Select,

    [switch]$Resume,

    [string]$DataDir
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if (-not $DataDir) {
    $DataDir = if ($env:TORCHSHARP_DATA_DIR) { $env:TORCHSHARP_DATA_DIR } else { Join-Path $ProjectRoot 'data' }
}
$env:TORCHSHARP_DATA_DIR = $DataDir

Write-Host ""
Write-Host "+----------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "|     TorchSharpLLM -- Dataset Setup                       |" -ForegroundColor Cyan
Write-Host "+----------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Write-Host "Data directory: $DataDir"
Write-Host ""

# ── Step 1: Install Python dependencies ──────────────────────────────────
Write-Host "-- Step 1: Checking Python dependencies --" -ForegroundColor Yellow

$python = $null
foreach ($cmd in @('python', 'python3', 'py')) {
    try {
        $ver = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $python = $cmd
            Write-Host "  Found: $cmd ($ver)"
            break
        }
    } catch { }
}

if (-not $python) {
    Write-Error "Python not found. Please install Python 3.8+ from https://python.org and add to PATH."
    exit 1
}

$reqFile = Join-Path $ScriptDir 'requirements.txt'
Write-Host "  Installing dependencies from $reqFile..."
& $python -m pip install -q -r $reqFile
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip install failed. Install manually: pip install -r scripts\requirements.txt"
}
Write-Host ""

# ── Step 2: Fetch datasets ───────────────────────────────────────────────
Write-Host "-- Step 2: Fetching datasets --" -ForegroundColor Yellow

$fetchScript = Join-Path $ScriptDir 'fetch_datasets.py'
$fetchArgs = @()

if ($Core) {
    $fetchArgs = @('--core')
} elseif ($All) {
    $fetchArgs = @('--all')
} elseif ($Select) {
    $fetchArgs = @('--select') + ($Select | ForEach-Object { $_.ToString() })
} else {
    # Interactive mode: show catalog and ask
    & $python $fetchScript --list
    Write-Host ""
    Write-Host "Which datasets would you like to download?"
    Write-Host "  Options: --core (0-7), --all (0-9), or --select <IDs>"
    Write-Host ""
    $choice = Read-Host "Enter choice (e.g., '--core' or '--select 0 1 2 7')"
    $fetchArgs = $choice -split '\s+'
}

if ($Resume) {
    $fetchArgs += '--resume'
}

& $python $fetchScript @fetchArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dataset download failed."
    exit 1
}
Write-Host ""

# ── Step 3: Tokenize ─────────────────────────────────────────────────────
Write-Host "-- Step 3: Tokenizing datasets --" -ForegroundColor Yellow
Write-Host "  This converts text to binary token files for fast C# loading."
Write-Host ""

$tokenizeScript = Join-Path $ScriptDir 'tokenize_datasets.py'
& $python $tokenizeScript --vocab-size 32000
if ($LASTEXITCODE -ne 0) {
    Write-Error "Tokenization failed."
    exit 1
}

Write-Host ""
Write-Host "+----------------------------------------------------------+" -ForegroundColor Green
Write-Host "|     Setup complete! Ready to train.                      |" -ForegroundColor Green
Write-Host "+----------------------------------------------------------+" -ForegroundColor Green
Write-Host ""
Write-Host "To train, run:"
Write-Host "  dotnet run --project src\TorchSharpLLM -- train --data-dir $DataDir" -ForegroundColor White
Write-Host ""

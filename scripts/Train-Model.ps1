<#
.SYNOPSIS
    Training launch script for TorchSharpLLM MoE model.

.DESCRIPTION
    Validates that tokenized data exists, then launches training with
    the specified configuration.

.PARAMETER Experts
    Train only specific expert indices (e.g., -Experts 0,1).

.PARAMETER LoRA
    Use LoRA for the trainable experts instead of full fine-tuning.

.PARAMETER MaxSteps
    Maximum number of training steps.

.PARAMETER Domain
    Train on a specific domain only (by dataset ID).

.PARAMETER Epochs
    Number of training epochs (default: 1).

.PARAMETER BatchSize
    Batch size (default: 4).

.PARAMETER SeqLen
    Sequence length (default: 512).

.PARAMETER LearningRate
    Learning rate (default: 3e-4).

.PARAMETER Mixing
    Data mixing strategy: Proportional, RoundRobin (default: Proportional).

.PARAMETER DataDir
    Override the data directory.

.PARAMETER CheckpointDir
    Directory for saving checkpoints (default: checkpoints).

.EXAMPLE
    .\scripts\Train-Model.ps1
    .\scripts\Train-Model.ps1 -Experts 0,1
    .\scripts\Train-Model.ps1 -LoRA -Experts 3,5
    .\scripts\Train-Model.ps1 -MaxSteps 1000
    .\scripts\Train-Model.ps1 -Domain 0 -Experts 0
#>
[CmdletBinding()]
param(
    [int[]]$Experts,
    [switch]$LoRA,
    [int]$MaxSteps,
    [int]$Domain = -1,
    [int]$Epochs,
    [int]$BatchSize,
    [int]$SeqLen,
    [double]$LearningRate,
    [string]$Mixing,
    [string]$DataDir,
    [string]$CheckpointDir
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if (-not $DataDir) {
    $DataDir = if ($env:TORCHSHARP_DATA_DIR) { $env:TORCHSHARP_DATA_DIR } else { Join-Path $ProjectRoot 'data' }
}

# Verify data exists
if (-not (Test-Path $DataDir)) {
    Write-Error "No tokenized data found in $DataDir`nRun .\scripts\Setup-Data.ps1 first to download and tokenize datasets."
    exit 1
}

$tokenFiles = Get-ChildItem -Path $DataDir -Recurse -Filter 'tokens.bin' -ErrorAction SilentlyContinue
if (-not $tokenFiles -or $tokenFiles.Count -eq 0) {
    Write-Error "No tokenized data found in $DataDir`nRun .\scripts\Setup-Data.ps1 first to download and tokenize datasets."
    exit 1
}

Write-Host ""
Write-Host "+----------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "|     TorchSharpLLM -- Training                            |" -ForegroundColor Cyan
Write-Host "+----------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

# Count available datasets
$datasetCount = $tokenFiles.Count

# Sum total tokens from metadata
$totalTokens = 0
$metaFiles = Get-ChildItem -Path $DataDir -Recurse -Filter 'tokens_meta.json' -ErrorAction SilentlyContinue
foreach ($mf in $metaFiles) {
    $meta = Get-Content $mf.FullName | ConvertFrom-Json
    $totalTokens += $meta.num_tokens
}

Write-Host "Data directory: $DataDir"
Write-Host "Datasets found: $datasetCount"
Write-Host ("Total tokens:   {0:N0}" -f $totalTokens)
Write-Host ""

# Build argument list for dotnet run
$dotnetArgs = @('run', '--project', (Join-Path $ProjectRoot 'src\TorchSharpLLM'), '--', 'train', '--data-dir', $DataDir)

if ($Experts) {
    $dotnetArgs += '--experts'
    $dotnetArgs += ($Experts | ForEach-Object { $_.ToString() })
}
if ($LoRA) { $dotnetArgs += '--lora' }
if ($PSBoundParameters.ContainsKey('MaxSteps') -and $MaxSteps -gt 0) {
    $dotnetArgs += @('--max-steps', $MaxSteps.ToString())
}
if ($Domain -ge 0) {
    $dotnetArgs += @('--domain', $Domain.ToString())
}
if ($PSBoundParameters.ContainsKey('Epochs') -and $Epochs -gt 0) {
    $dotnetArgs += @('--epochs', $Epochs.ToString())
}
if ($PSBoundParameters.ContainsKey('BatchSize') -and $BatchSize -gt 0) {
    $dotnetArgs += @('--batch-size', $BatchSize.ToString())
}
if ($PSBoundParameters.ContainsKey('SeqLen') -and $SeqLen -gt 0) {
    $dotnetArgs += @('--seq-len', $SeqLen.ToString())
}
if ($LearningRate -gt 0) {
    $dotnetArgs += @('--lr', $LearningRate.ToString())
}
if ($Mixing) {
    $dotnetArgs += @('--mixing', $Mixing)
}
if ($CheckpointDir) {
    $dotnetArgs += @('--checkpoint-dir', $CheckpointDir)
}

Write-Host "Running: dotnet $($dotnetArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& dotnet @dotnetArgs

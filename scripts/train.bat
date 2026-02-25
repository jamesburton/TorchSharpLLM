@echo off
setlocal EnableDelayedExpansion
REM
REM Training launch script for TorchSharpLLM MoE model.
REM
REM Usage:
REM   scripts\train.bat                                  Full training, all datasets
REM   scripts\train.bat --experts 0 1                    Train only experts 0,1
REM   scripts\train.bat --lora --experts 3 5             LoRA fine-tune experts 3,5
REM   scripts\train.bat --max-steps 1000                 Limit total steps
REM   scripts\train.bat --domain 0 --experts 0           Train expert 0 on code only
REM

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

if defined TORCHSHARP_DATA_DIR (
    set "DATA_DIR=%TORCHSHARP_DATA_DIR%"
) else (
    set "DATA_DIR=%PROJECT_ROOT%\data"
)

REM Verify data exists
if not exist "%DATA_DIR%" (
    echo ERROR: No tokenized data found in %DATA_DIR%
    echo Run scripts\setup_data.bat first to download and tokenize datasets.
    exit /b 1
)

set "FOUND_BIN="
for /r "%DATA_DIR%" %%f in (tokens.bin) do (
    set "FOUND_BIN=1"
)
if not defined FOUND_BIN (
    echo ERROR: No tokenized data found in %DATA_DIR%
    echo Run scripts\setup_data.bat first to download and tokenize datasets.
    exit /b 1
)

echo +----------------------------------------------------------+
echo ^|     TorchSharpLLM -- Training                            ^|
echo +----------------------------------------------------------+
echo.

REM Count available datasets
set "DATASET_COUNT=0"
for /r "%DATA_DIR%" %%f in (tokens.bin) do (
    set /a DATASET_COUNT+=1
)

echo Data directory: %DATA_DIR%
echo Datasets found: %DATASET_COUNT%
echo.

REM Pass all arguments through to the .NET application
dotnet run --project "%PROJECT_ROOT%\src\TorchSharpLLM" -- train --data-dir "%DATA_DIR%" %*
endlocal

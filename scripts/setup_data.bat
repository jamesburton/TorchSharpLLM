@echo off
setlocal EnableDelayedExpansion
REM
REM Setup script: install Python dependencies, fetch datasets, and tokenize.
REM
REM Usage:
REM   scripts\setup_data.bat                     Interactive: choose which datasets
REM   scripts\setup_data.bat --core              Download core 8 datasets (one per expert)
REM   scripts\setup_data.bat --all               Download all 10 datasets
REM   scripts\setup_data.bat --select 0 1 7      Download specific datasets
REM

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

if defined TORCHSHARP_DATA_DIR (
    set "DATA_DIR=%TORCHSHARP_DATA_DIR%"
) else (
    set "DATA_DIR=%PROJECT_ROOT%\data"
)

set "TORCHSHARP_DATA_DIR=%DATA_DIR%"

echo +----------------------------------------------------------+
echo ^|     TorchSharpLLM -- Dataset Setup                       ^|
echo +----------------------------------------------------------+
echo.
echo Data directory: %DATA_DIR%
echo.

REM -- Step 1: Install Python dependencies ---------------------------------
echo -- Step 1: Checking Python dependencies --
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Please install Python 3.8+ and add it to PATH.
    exit /b 1
)

pip install -q -r "%SCRIPT_DIR%requirements.txt" 2>nul
if errorlevel 1 (
    pip3 install -q -r "%SCRIPT_DIR%requirements.txt" 2>nul
    if errorlevel 1 (
        echo WARNING: Could not install Python deps. Install manually: pip install -r scripts\requirements.txt
    )
)
echo.

REM -- Step 2: Fetch datasets ----------------------------------------------
echo -- Step 2: Fetching datasets --

if "%~1"=="" (
    REM Interactive mode: show catalog and ask
    python "%SCRIPT_DIR%fetch_datasets.py" --list
    echo.
    echo Which datasets would you like to download?
    echo   Options: --core ^(0-7^), --all ^(0-9^), or --select ^<IDs^>
    echo.
    set /p CHOICE="Enter choice (e.g., '--core' or '--select 0 1 2 7'): "
    python "%SCRIPT_DIR%fetch_datasets.py" !CHOICE! --resume
) else (
    python "%SCRIPT_DIR%fetch_datasets.py" %* --resume
)

if errorlevel 1 (
    echo ERROR: Dataset download failed.
    exit /b 1
)
echo.

REM -- Step 3: Tokenize ----------------------------------------------------
echo -- Step 3: Tokenizing datasets --
echo This converts text to binary token files for fast C# loading.
echo.

python "%SCRIPT_DIR%tokenize_datasets.py" --vocab-size 32000

if errorlevel 1 (
    echo ERROR: Tokenization failed.
    exit /b 1
)

echo.
echo +----------------------------------------------------------+
echo ^|     Setup complete! Ready to train.                      ^|
echo +----------------------------------------------------------+
echo.
echo To train, run:
echo   dotnet run --project src\TorchSharpLLM -- train --data-dir %DATA_DIR%
echo.
endlocal

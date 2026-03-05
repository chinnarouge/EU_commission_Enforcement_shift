@echo off
REM ============================================================
REM  Data Preprocessing Pipeline
REM  Runs all four steps in sequence. Stops on first failure.
REM ============================================================

echo ======================================================================
echo   DATA PREPROCESSING PIPELINE
echo ======================================================================

set SCRIPT_DIR=%~dp0
set PYTHON=C:\Users\z004xh1j\AppData\Local\miniconda3\envs\rag\python.exe

echo.
echo [Step 1/4] Extracting .7z archives to raw HTML ...
"%PYTHON%" "%SCRIPT_DIR%data_extraction.py"
if %ERRORLEVEL% NEQ 0 (
    echo   [FAILED] data_extraction.py exited with code %ERRORLEVEL%
    goto :fail
)
echo   [OK] data_extraction.py

echo.
echo [Step 2/4] Parsing HTML into structured_data.csv ...
"%PYTHON%" "%SCRIPT_DIR%data_preprocessing.py"
if %ERRORLEVEL% NEQ 0 (
    echo   [FAILED] data_preprocessing.py exited with code %ERRORLEVEL%
    goto :fail
)
echo   [OK] data_preprocessing.py

echo.
echo [Step 3/4] Classifying cases (competition filter) ...
"%PYTHON%" "%SCRIPT_DIR%classify.py"
if %ERRORLEVEL% NEQ 0 (
    echo   [FAILED] classify.py exited with code %ERRORLEVEL%
    goto :fail
)
echo   [OK] classify.py

echo.
echo [Step 4/4] Post-processing (sector, decision stage, outcome) ...
"%PYTHON%" "%SCRIPT_DIR%data_overview_and _postprocessing.py"
if %ERRORLEVEL% NEQ 0 (
    echo   [FAILED] data_overview_and _postprocessing.py exited with code %ERRORLEVEL%
    goto :fail
)
echo   [OK] data_overview_and _postprocessing.py

echo.
echo ======================================================================
echo   PIPELINE COMPLETE
echo   Output: data\processesd_data\final_competition_cases.csv
echo ======================================================================
goto :eof

:fail
echo.
echo ======================================================================
echo   PIPELINE FAILED - see error above.
echo ======================================================================
exit /b 1

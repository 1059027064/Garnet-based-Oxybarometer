@echo off
chcp 65001 >nul

echo =====================================
echo Oxybarometer Prediction Tool
echo =====================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo.
echo Trying to activate conda environment: tabpfn_shap

REM Try common Anaconda locations
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" tabpfn_shap
    goto check_python
)

if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" tabpfn_shap
    goto check_python
)

if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat" tabpfn_shap
    goto check_python
)

if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\miniconda3\Scripts\activate.bat" tabpfn_shap
    goto check_python
)

echo.
echo ERROR: Conda installation was not found automatically.
echo Please run this tool from Anaconda Prompt instead:
echo.
echo   conda activate tabpfn_shap
echo   cd Prediction
echo   python predict.py
echo.
pause
exit /b 1

:check_python
echo.
echo Python path:
where python

echo.
echo Running prediction script...
python predict.py

if errorlevel 1 (
    echo.
    echo =====================================
    echo ERROR: Prediction failed
    echo =====================================
    pause
    exit /b 1
)

echo.
echo =====================================
echo DONE!
echo Results saved in:
echo   output\
echo   images\
echo =====================================
pause
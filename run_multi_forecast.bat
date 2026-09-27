@echo off

setlocal EnableDelayedExpansion

cd /d E:\QHHT



if not exist data\logs mkdir data\logs

echo.>> data\logs\task_multi_stdout.log

echo [%date% %time%] ===== START slot=%~1 =====>> data\logs\task_multi_stdout.log



REM 用法: run_multi_forecast.bat 0840

if not "%~1"=="" set RUN_SLOT=%~1

if not defined RUN_SLOT (

  echo [error] 请传入时段: 0840 / 1030 / 1415 / 2050>> data\logs\task_multi_stdout.log

  exit /b 1

)



set DAILY_NOTIFY=1

set MODEL_VERSION=5

set RUN_SOURCE=local



"C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" daily_multi_forecast.py >> data\logs\task_multi_stdout.log 2>&1

set EXIT_CODE=!ERRORLEVEL!

echo [%date% %time%] ===== END slot=%RUN_SLOT% exit=!EXIT_CODE! =====>> data\logs\task_multi_stdout.log

exit /b !EXIT_CODE!



@echo off
setlocal
cd /d E:\QHHT

REM 用法: run_multi_forecast.bat 0840
if not "%~1"=="" set RUN_SLOT=%~1
if not defined RUN_SLOT (
  echo [error] 请传入时段: 0840 / 1030 / 1415 / 2050
  exit /b 1
)

set DAILY_NOTIFY=1
set MODEL_VERSION=5
set RUN_SOURCE=local

REM 默认每次重新拉取 Tushare/akshare 最新数据（勿加 --no-refresh）
"C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" daily_multi_forecast.py >> data\logs\task_multi_stdout.log 2>&1

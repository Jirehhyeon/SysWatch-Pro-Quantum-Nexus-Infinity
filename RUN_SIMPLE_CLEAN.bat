@echo off
echo ================================================================
echo  SYSWATCH PRO - SIMPLE CLEAN MONITOR
echo  Clean and Simple System Monitor
echo ================================================================
echo.
echo Starting Simple Clean Monitor...
echo.
echo SIMPLE FEATURES:
echo - Real-time CPU and Memory Graphs
echo - Progress Bars for CPU/Memory/Disk
echo - Individual CPU Core Monitoring
echo - Top Process List
echo - Network Activity Monitor
echo - Clean Dark Theme Interface
echo - No Complex Effects - Just Data
echo.
echo KEY BENEFITS:
echo - Easy to Read Data
echo - Simple Color Coding (Green=Good, Red=Warning)
echo - Lightweight and Fast
echo - Auto Package Installation
echo - Perfect for Daily Monitoring
echo.
echo CONTROLS:
echo - ESC/Q: Exit
echo - SPACE: Screenshot
echo.
echo ================================================================
echo.

cd /d "%~dp0"
python SysWatch_Pro_SIMPLE_CLEAN_MONITOR.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo Error occurred. Installing packages automatically...
    echo ================================================================
    echo.
)

pause

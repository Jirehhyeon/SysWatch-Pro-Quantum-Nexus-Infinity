@echo off
echo ================================================================
echo  QUANTUM NEXUS DATA FOCUSED MONITOR
echo  Clean Data Visualization System Monitor
echo ================================================================
echo.
echo Starting Data-Focused Monitor...
echo.
echo DATA-FOCUSED Features:
echo - Real-time CPU/Memory/Network Charts
echo - Per-Core CPU Usage Bars (up to 16 cores)
echo - Top 15 Processes Real-time Table
echo - Performance-based Color Coding (Green=Good, Red=Danger)
echo - Clean UI with Clear Data Visibility
echo - Real-time Graphs and Animated Bars
echo - Detailed System Information Panel
echo.
echo Key Features:
echo - No Complex Background Effects
echo - Data Readability First Priority
echo - High-Contrast Colors for Clear Visualization
echo - Real-time Updates (200ms interval)
echo - Auto Package Installation
echo.
echo Controls:
echo - ESC/Q: Exit
echo - SPACE: Screenshot
echo - R: Reset Data History
echo.
echo ================================================================
echo.

cd /d "%~dp0"
python SysWatch_Pro_QUANTUM_NEXUS_INFINITY_DATA_FOCUSED.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo Error occurred. Auto-installing required packages...
    echo ================================================================
    echo.
)

pause

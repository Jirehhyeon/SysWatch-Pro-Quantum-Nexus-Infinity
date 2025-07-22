@echo off
echo ========================================
echo  SysWatch Pro QUANTUM NEXUS INFINITY
echo  Holographic Interface Launcher
echo ========================================
echo.
echo Starting Quantum Holographic Interface...
echo Press Ctrl+C to exit
echo.

cd /d "%~dp0"
python SysWatch_Pro_QUANTUM_NEXUS_INFINITY.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to run!
    echo.
    echo Possible solutions:
    echo 1. Run INSTALL.bat first
    echo 2. Check if Python is installed
    echo 3. Make sure all packages are installed
    echo.
)

pause
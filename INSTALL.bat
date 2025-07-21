@echo off
echo ========================================
echo  SysWatch Pro QUANTUM NEXUS INFINITY
echo  Installation Script
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found! Installing required packages...
echo.

echo Installing core packages...
python -m pip install --upgrade pip
python -m pip install psutil numpy pygame matplotlib colorama rich

echo.
echo Installing optional packages...
python -m pip install pillow requests pandas scikit-learn plotly

echo.
echo Installing Windows-specific packages...
python -m pip install py-cpuinfo wmi pywin32 pynvml

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run SysWatch Pro QUANTUM NEXUS INFINITY:
echo python SysWatch_Pro_QUANTUM_NEXUS_INFINITY.py
echo.
pause
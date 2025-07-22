@echo off
echo ============================================================
echo  QUANTUM NEXUS INFINITY ULTIMATE
echo  200+ FPS HOLOGRAPHIC SYSTEM MONITOR
echo ============================================================
echo.
echo Starting ULTIMATE Quantum Interface...
echo.
echo Features:
echo - 200+ FPS Performance
echo - Process Manager with Kill Function  
echo - Real-time System Optimization
echo - Deep Hardware Monitoring
echo - 3D Holographic Universe
echo - Floating Interactive Panels
echo.
echo Controls:
echo - ESC/Q: Exit
echo - SPACE: Screenshot
echo - R: Reset Universe
echo - 1-4: Quick Optimizations
echo - Click and drag panels
echo - Click processes to select, then kill
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python SysWatch_Pro_QUANTUM_NEXUS_INFINITY_ULTIMATE.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Failed to run ULTIMATE version!
    echo.
    echo Possible solutions:
    echo 1. Run INSTALL.bat first
    echo 2. Install missing packages:
    echo    pip install psutil pygame numpy matplotlib colorama
    echo    pip install pillow requests pandas scikit-learn
    echo    pip install wmi pywin32 py-cpuinfo
    echo 3. Run as Administrator for full features
    echo ============================================================
    echo.
)

pause

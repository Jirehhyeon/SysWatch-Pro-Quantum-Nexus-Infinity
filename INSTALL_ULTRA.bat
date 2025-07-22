@echo off
echo ================================================================
echo  QUANTUM NEXUS INFINITY ULTIMATE ULTRA
echo  Installation Script
echo ================================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found! Installing ULTRA packages...
echo.

echo [1/6] Installing core packages...
python -m pip install --upgrade pip
python -m pip install psutil numpy matplotlib colorama rich

echo.
echo [2/6] Installing PyGame Community Edition...
python -m pip install pygame-ce --upgrade

echo.
echo [3/6] Installing visualization packages...
python -m pip install pillow requests pandas plotly

echo.
echo [4/6] Installing Windows integration...
python -m pip install py-cpuinfo wmi pywin32 pynvml

echo.
echo [5/6] Installing AI/ML packages...
python -m pip install scikit-learn

echo.
echo [6/6] Installing optional audio support...
echo Note: PyAudio requires Microsoft Visual C++ 14.0 or greater
python -m pip install pyaudio
if errorlevel 1 (
    echo Warning: PyAudio installation failed - audio features will be disabled
    echo You can install it manually later if needed
)

echo.
echo Optional CUDA support for GPU acceleration...
echo Note: Requires NVIDIA GPU and CUDA toolkit
python -m pip install pycuda
if errorlevel 1 (
    echo Warning: PyCUDA installation failed - will use CPU fallback
    echo You can install CUDA toolkit and try again if you have NVIDIA GPU
)

echo.
echo ================================================================
echo Installation Complete!
echo ================================================================
echo.
echo Available versions:
echo - Standard: RUN.bat
echo - ULTIMATE: RUN_ULTIMATE.bat  
echo - ULTRA: RUN_ULTIMATE_ULTRA.bat
echo.
echo For best performance:
echo 1. Run as Administrator
echo 2. Close other applications
echo 3. Ensure Windows is updated
echo.
pause
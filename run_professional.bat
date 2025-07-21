@echo off
chcp 65001 >nul
title SysWatch Pro Professional - Enterprise System Monitor

echo.
echo ████████████████████████████████████████████████████████████████████████████████
echo ██                                                                            ██
echo ██    ███████ ██    ██ ███████ ██     ██  █████  ████████  ██████ ██   ██    ██
echo ██    ██       ██  ██  ██      ██     ██ ██   ██    ██    ██      ██   ██    ██
echo ██    ███████   ████   ███████ ██  █  ██ ███████    ██    ██      ███████    ██
echo ██         ██    ██         ██ ██ ███ ██ ██   ██    ██    ██      ██   ██    ██
echo ██    ███████    ██    ███████  ███ ███  ██   ██    ██     ██████ ██   ██    ██
echo ██                                                                            ██
echo ██                    PROFESSIONAL ENTERPRISE EDITION                        ██
echo ██                                                                            ██
echo ████████████████████████████████████████████████████████████████████████████████
echo.
echo                    🏢 Enterprise-grade System Performance Monitor
echo                    ⚡ Real-time Analytics ^& Advanced Monitoring
echo                    🎯 Professional Dashboard for IT Professionals
echo.
echo ████████████████████████████████████████████████████████████████████████████████
echo.

echo [INITIALIZING] Python environment check...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Python not found!
    echo.
    echo Please install Python from: https://python.org
    echo Make sure to add Python to your PATH during installation.
    echo.
    pause
    exit /b 1
)

echo ✅ Python environment ready
echo.
echo [LAUNCHING] SysWatch Pro Professional Edition...
echo.

python launch_professional.py

echo.
echo ████████████████████████████████████████████████████████████████████████████████
echo ██                                                                            ██
echo ██                        SESSION TERMINATED                                 ██
echo ██                                                                            ██
echo ██              Thank you for using SysWatch Pro Professional!              ██
echo ██                                                                            ██
echo ████████████████████████████████████████████████████████████████████████████████
echo.
pause
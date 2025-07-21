@echo off
title SysWatch Pro Quantum - AAA급 시스템 모니터링 스위트
color 0a

:: 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo ⚠️  관리자 권한이 필요합니다.
    echo    우클릭 후 "관리자 권한으로 실행"을 선택해주세요.
    echo.
    pause
    exit /b 1
)

:: 화면 클리어 및 헤더 출력
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   SYSWATCH PRO QUANTUM                      ║
echo ║                  🚀 AAA급 시스템 모니터링                     ║
echo ║                                                              ║
echo ║  Version: 3.0.0 Quantum Enterprise Edition                  ║
echo ║  Copyright (C) 2025 SysWatch Technologies Ltd.              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Python 설치 확인
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    echo.
    echo 📥 Python 설치 옵션:
    echo    1. Microsoft Store에서 Python 설치
    echo    2. python.org에서 다운로드
    echo.
    set /p choice="Python을 자동으로 다운로드하시겠습니까? (Y/N): "
    if /i "%choice%"=="Y" (
        echo.
        echo 🌐 Python 다운로드 페이지를 엽니다...
        start https://www.python.org/downloads/
    )
    pause
    exit /b 1
)

echo ✅ Python 확인 완료
echo.

:: 가상환경 확인 및 생성
if not exist "venv" (
    echo 📦 가상환경을 생성하는 중...
    python -m venv venv
    if %errorLevel% neq 0 (
        echo ❌ 가상환경 생성에 실패했습니다.
        pause
        exit /b 1
    )
    echo ✅ 가상환경 생성 완료
)

:: 가상환경 활성화
echo 🔄 가상환경 활성화 중...
call venv\Scripts\activate.bat

:: 필요한 패키지 설치
echo.
echo 📦 필수 패키지 설치 중...
echo    이 과정은 몇 분이 소요될 수 있습니다...
echo.

:: 기본 패키지
pip install --quiet --upgrade pip
pip install --quiet psutil numpy pandas matplotlib

:: GUI 패키지
pip install --quiet tkinter ttkbootstrap customtkinter

:: 고급 시각화
pip install --quiet plotly

:: AI/ML 패키지 (선택적)
echo 🧠 AI/ML 패키지 설치 중... (선택적)
pip install --quiet scikit-learn tensorflow torch --timeout 300 2>nul
if %errorLevel% neq 0 (
    echo ⚠️  AI/ML 패키지 설치 실패 - 기본 기능으로 계속 진행
)

:: 보안 패키지 (선택적)
echo 🛡️  보안 패키지 설치 중... (선택적)
pip install --quiet cryptography requests --timeout 300 2>nul

:: 고급 패키지 (선택적)
echo 🎨 고급 시각화 패키지 설치 중... (선택적)
pip install --quiet vtk opencv-python --timeout 300 2>nul
if %errorLevel% neq 0 (
    echo ⚠️  고급 시각화 패키지 설치 실패 - 기본 기능으로 계속 진행
)

echo.
echo ✅ 패키지 설치 완료
echo.

:: 실행 모드 선택
echo 🎯 실행 모드를 선택해주세요:
echo.
echo    1. 💫 Quantum GUI (홀로그래픽 3D 인터페이스)
echo    2. 🧠 AI Engine (터미널 기반 AI 분석)
echo    3. 🛡️  Security Engine (보안 스캐닝)
echo    4. 📊 Classic GUI (기존 인터페이스)
echo    5. ⚙️  System Optimizer (성능 최적화)
echo    6. 🚀 Full Suite (모든 기능)
echo.

set /p mode="모드를 선택하세요 (1-6): "

echo.
echo 🚀 SysWatch Pro Quantum을 시작합니다...
echo.

if "%mode%"=="1" (
    echo 💫 Quantum GUI 모드로 시작...
    python quantum_gui.py
) else if "%mode%"=="2" (
    echo 🧠 AI Engine 모드로 시작...
    python syswatch_quantum.py
) else if "%mode%"=="3" (
    echo 🛡️  Security Engine 모드로 시작...
    python security_engine.py
) else if "%mode%"=="4" (
    echo 📊 Classic GUI 모드로 시작...
    python syswatch_ultimate.py
) else if "%mode%"=="5" (
    echo ⚙️  System Optimizer 모드로 시작...
    python -c "from security_engine import quantum_optimizer; print('시스템 분석 중...'); import json; result = quantum_optimizer.analyze_system_performance(); print(json.dumps(result, indent=2, ensure_ascii=False)); input('Press Enter to continue...')"
) else if "%mode%"=="6" (
    echo 🚀 Full Suite 모드로 시작...
    echo    GUI와 AI Engine을 동시에 실행합니다...
    start "Quantum AI" python syswatch_quantum.py
    timeout /t 3 /nobreak >nul
    start "Security Engine" python security_engine.py
    timeout /t 2 /nobreak >nul
    python quantum_gui.py
) else (
    echo ❌ 잘못된 선택입니다. 기본 GUI 모드로 시작합니다...
    python quantum_gui.py
)

:: 종료 처리
echo.
echo 🛑 SysWatch Pro Quantum이 종료되었습니다.
echo.

:: 에러 확인
if %errorLevel% neq 0 (
    echo ❌ 오류가 발생했습니다. 오류 코드: %errorLevel%
    echo.
    echo 🔧 문제 해결 방법:
    echo    1. Python이 올바르게 설치되었는지 확인
    echo    2. 필요한 패키지가 설치되었는지 확인
    echo    3. 관리자 권한으로 실행했는지 확인
    echo    4. 바이러스 백신이 차단하지 않는지 확인
    echo.
    echo 📧 기술 지원: support@syswatch-pro.com
    echo 🌐 웹사이트: https://syswatch-pro.com
    echo.
) else (
    echo ✅ 정상적으로 종료되었습니다.
    echo.
    echo 🌟 SysWatch Pro Quantum을 사용해주셔서 감사합니다!
    echo    더 많은 기능과 업데이트를 위해 웹사이트를 방문해보세요.
    echo.
)

:: 정리
deactivate >nul 2>&1

echo 💡 팁: 바탕화면에 바로가기를 만들어 쉽게 실행하세요!
echo.
pause
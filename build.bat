@echo off
echo Building SysWatch Pro for Windows...

REM Create build directory
if not exist "build" mkdir build

REM Check for Visual Studio Build Tools
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Microsoft Visual C++ compiler not found.
    echo Please install Visual Studio Build Tools or Visual Studio Community.
    echo You can download from: https://visualstudio.microsoft.com/downloads/
    pause
    exit /b 1
)

REM Build the project
echo Compiling source files...
cl /nologo /I"include" /DWINDOWS src\main.c src\system_info.c src\process_monitor.c src\utils.c /Fe:build\syswatch.exe psapi.lib pdh.lib

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  SysWatch Pro built successfully!
    echo ========================================
    echo.
    echo Executable: build\syswatch.exe
    echo.
    echo To test the build:
    echo   build\syswatch.exe --test
    echo   build\syswatch.exe --help
    echo.
) else (
    echo.
    echo ========================================
    echo  Build failed!
    echo ========================================
    echo.
    echo Make sure you have:
    echo 1. Visual Studio Build Tools installed
    echo 2. Opened "Developer Command Prompt for VS"
    echo 3. Or run "vcvarsall.bat" to set up environment
    echo.
)

pause
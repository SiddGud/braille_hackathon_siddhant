@echo off
echo Installing YOLOv8 (Ultralytics) with SSL bypass...
echo.

set VENV_PIP="%~dp0venv\bin\pip.exe"

%VENV_PIP% install ultralytics --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

echo.
if %errorlevel% == 0 (
    echo ✅ Ultralytics installed successfully!
    echo    YOLOv8 is now available. Place best.pt in backend/models/
) else (
    echo ❌ Installation failed. Try running as Administrator.
    echo    Or install manually: pip install ultralytics
)
echo.
pause

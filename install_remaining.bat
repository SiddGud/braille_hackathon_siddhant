@echo off
:: Install remaining Python packages via MSYS2 pacman (after main install)
echo Installing additional Python packages via MSYS2...

set PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%
set MSYSTEM=UCRT64

:: Install uvicorn, gtts, requests via pacman
C:\msys64\usr\bin\pacman.exe -S --noconfirm --needed ^
    mingw-w64-ucrt-x86_64-python-uvicorn ^
    mingw-w64-ucrt-x86_64-python-requests ^
    mingw-w64-ucrt-x86_64-python-aiofiles 2>&1

:: Try pip for remaining packages (using system pip now)
C:\msys64\ucrt64\bin\python.exe -m pip install --break-system-packages --trusted-host pypi.org --trusted-host files.pythonhosted.org gtts python-multipart python-dotenv 2>&1

echo.
echo Done! Run start.bat to launch BrailleVision.
pause

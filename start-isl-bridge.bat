@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title ISL Bridge launcher

if not exist "backend\.venv\Scripts\python.exe" (
  echo Backend virtualenv not found.
  echo.
  echo From the backend folder run:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo Node.js / npm was not found on PATH.
  echo Install Node.js 20+ and try again.
  echo.
  pause
  exit /b 1
)

if not exist "backend\.env" if exist "backend\.env.example" (
  copy /y "backend\.env.example" "backend\.env" >nul
  echo Created backend\.env from .env.example
)

if not exist "frontend\.env" if exist "frontend\.env.example" (
  copy /y "frontend\.env.example" "frontend\.env" >nul
  echo Created frontend\.env from .env.example
)

if not exist "frontend\node_modules\" (
  echo Installing frontend packages...
  pushd frontend
  call npm install
  if errorlevel 1 (
    popd
    echo npm install failed.
    pause
    exit /b 1
  )
  popd
)

echo Starting backend on http://127.0.0.1:8000
start "ISL Bridge Backend" /D "%~dp0backend" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

echo Starting frontend Vite (usually http://localhost:5173)
start "ISL Bridge Frontend" /D "%~dp0frontend" cmd /k "npm run dev"

echo.
echo Two windows should stay open.
echo Backend health: http://127.0.0.1:8000/health
echo Translator:      http://localhost:5173
echo If 5173 is busy, use the Local URL printed in the frontend window.
echo.
timeout /t 5 /nobreak >nul
start "" "http://localhost:5173"
endlocal

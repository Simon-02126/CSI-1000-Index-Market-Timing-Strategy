@echo off
REM timing dashboard one-click launcher
cd /d %~dp0

REM already running? just open browser
netstat -ano | findstr "7100" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    start "" http://localhost:7100
    exit /b
)

REM wait 1s for server, then open browser; keep this window to see logs (close = stop server)
start "" cmd /c "ping -n 2 127.0.0.1 >nul & start http://localhost:7100"
node server.js --port 7100

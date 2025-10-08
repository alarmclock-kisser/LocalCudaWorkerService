@echo off
set PORT=32141
set IN_RULE=LocalCudaWorkerService_Inbound
set OUT_RULE=LocalCudaWorkerService_Outbound
set ASPNETCORE_URLS=https://0.0.0.0:%PORT%

:: Admin-Check
openfiles >nul 2>&1
if %errorlevel% NEQ 0 (
  echo Starte neu mit Admin...
  powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb RunAs"
  exit /b
)

echo == Firewall Inbound ==
netsh advfirewall firewall show rule name="%IN_RULE%" >nul 2>&1
if errorlevel 1 (
  netsh advfirewall firewall add rule name="%IN_RULE%" dir=in action=allow protocol=TCP localport=%PORT%
) else (
  echo Regel %IN_RULE% existiert bereits.
)

echo == Firewall Outbound ==
netsh advfirewall firewall show rule name="%OUT_RULE%" >nul 2>&1
if errorlevel 1 (
  netsh advfirewall firewall add rule name="%OUT_RULE%" dir=out action=allow protocol=TCP localport=%PORT%
) else (
  echo Regel %OUT_RULE% existiert bereits.
)

echo Benutzer-Umgebungsvariable setzen ASPNETCORE_URLS=%ASPNETCORE_URLS%
setx ASPNETCORE_URLS "%ASPNETCORE_URLS%" >nul

echo Optional System-weit setzen: setx ASPNETCORE_URLS "%ASPNETCORE_URLS%" /M
echo Fertig. Neue Konsole  ffnen damit Variable aktiv ist.
pause
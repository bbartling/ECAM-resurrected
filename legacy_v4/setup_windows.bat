@echo off
setlocal
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
if errorlevel 1 (
  echo Installation failed.
  pause
  exit /b 1
)
echo.
echo Installation complete. Launching ECAM Resurrected...
python -m ecam_resurrected

@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."
set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"

echo ============================================================================
echo   VISIONMATE GLASSES - PHASE 1.3a.1 (Offline Vietnamese Reading Foundation)
echo   Phim tat: [Space] Chup va Doc  [R] Doc lai  [Esc] Dung  [Ctrl+L] Xoa  [O] Mo anh  [F1] Tro giup
echo ============================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 -m visionmate.app %*
) else (
    python -X utf8 -m visionmate.app %*
)
endlocal

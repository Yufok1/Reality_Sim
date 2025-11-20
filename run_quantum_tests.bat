@echo off
echo ===============================================
echo   REALITY SIMULATOR - QUANTUM SUBSTRATE TESTS
echo ===============================================
echo.

cd /d "%~dp0"

echo Running quantum substrate tests...
echo.

python tests\test_quantum_substrate.py

echo.
echo ===============================================
echo   TESTS COMPLETE
echo ===============================================
echo.
pause


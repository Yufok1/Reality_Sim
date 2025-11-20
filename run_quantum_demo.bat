@echo off
echo ===============================================
echo   REALITY SIMULATOR - QUANTUM DEMO
echo ===============================================
echo.
echo This will show you quantum phenomena visually
echo Making the invisible VISIBLE
echo.
echo Press Ctrl+C to exit anytime
echo.
pause

cd /d "%~dp0\reality_simulator"

python demo_quantum.py

echo.
echo ===============================================
echo   DEMO COMPLETE
echo ===============================================
echo.
echo Check the generated PNG files to see:
echo   - Quantum superposition
echo   - Wave function collapse
echo   - Entanglement networks
echo   - Probability fields
echo   - Time evolution
echo.
pause


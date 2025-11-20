#!/usr/bin/env python3
"""
[ROCKET] Reality Simulator Setup Verification

Checks if all required dependencies are installed and system is ready to run.
"""

import sys
import subprocess
import importlib.util


def check_python_version():
    """Check Python version"""
    print("[PYTHON] Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"[SUCCESS] Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"[ERROR] Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False


def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"[SUCCESS] {package_name} - OK")
        return True
    except ImportError:
        print(f"[ERROR] {package_name} - NOT INSTALLED")
        print(f"   Install with: pip install {package_name}")
        return False


# Ollama check removed - AI agents are not used in this system


def check_reality_simulator_imports():
    """Check if Reality Simulator modules can be imported"""
    print("[ROCKET] Checking Reality Simulator modules...")

    # Add current directory to path for imports
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    modules_to_check = [
        'reality_simulator.quantum_substrate',
        'reality_simulator.subatomic_lattice',
        'reality_simulator.evolution_engine',
        'reality_simulator.symbiotic_network',
        'reality_simulator.symbiotic_network',
        'reality_simulator.agency.network_decision_agent',
        'reality_simulator.reality_renderer'
    ]

    all_good = True
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"[SUCCESS] {module} - OK")
        except ImportError as e:
            print(f"[ERROR] {module} - FAILED: {e}")
            all_good = False

    # Test main module separately (it has complex imports)
    try:
        # Just check if the file exists and is syntactically valid
        import ast
        # Use UTF-8 encoding to handle emoji characters
        with open(os.path.join(current_dir, 'reality_simulator', 'main.py'), 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("[SUCCESS] reality_simulator.main - SYNTAX OK")
    except Exception as e:
        print(f"[ERROR] reality_simulator.main - SYNTAX ERROR: {e}")
        all_good = False

    return all_good


def main():
    """Main setup check"""
    print("=" * 60)
    print("[ROCKET] REALITY SIMULATOR SETUP VERIFICATION")
    print("=" * 60)
    print()

    # Check Python version
    python_ok = check_python_version()
    print()

    # Check required packages
    print("[PACKAGES] Checking required packages...")
    packages_ok = all([
        check_package("numpy"),
        check_package("scipy"),
        check_package("networkx"),
        check_package("psutil")
    ])
    print()

    # Check optional packages
    print("[ART] Checking optional packages...")
    optional_ok = all([
        check_package("torch", "torch"),
        check_package("matplotlib", "matplotlib"),
    ])
    print()

    # Check Reality Simulator imports
    modules_ok = check_reality_simulator_imports()
    print()

    # Summary
    print("=" * 60)
    print("[CHART] SETUP SUMMARY")
    print("=" * 60)

    core_ready = python_ok and packages_ok and modules_ok

    print(f"Core System: {'[SUCCESS] READY' if core_ready else '[ERROR] ISSUES'}")
    print(f"Enhanced Features: {'[SUCCESS] AVAILABLE' if optional_ok else '[WARNING]  LIMITED'}")

    if core_ready:
        print()
        print("[CELEBRATION] Reality Simulator is ready to run!")
        print()
        print("Quick start:")
        print("  ./run_reality_simulator.bat")
        print("  python reality_simulator/main.py --mode observer")
        print()
    else:
        print()
        print("[ERROR] Core system has issues. Please fix above problems first.")
        print()
        print("Install missing packages:")
        print("  pip install numpy scipy networkx psutil")

    print()
    return 0 if core_ready else 1


if __name__ == "__main__":
    sys.exit(main())

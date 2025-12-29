#!/usr/bin/env python3
"""
Framework Validation Script
Ensures all experiment framework files are present and functional.
Run this to verify the framework is ready for use.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        size_kb = Path(filepath).stat().st_size / 1024
        print(f"✅ {description:<40} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"❌ {description:<40} (MISSING)")
        return False

def check_json_valid(filepath):
    """Check if JSON file is valid."""
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print(f"✅ JSON valid: {filepath}")
        return True
    except Exception as e:
        print(f"❌ JSON invalid: {filepath} - {e}")
        return False

def check_python_syntax(filepath):
    """Check if Python file has valid syntax."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", filepath],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✅ Syntax OK: {Path(filepath).name}")
        return True
    else:
        print(f"❌ Syntax ERROR: {Path(filepath).name}")
        print(f"   {result.stderr}")
        return False

def validate_imports():
    """Check if key imports are available."""
    imports = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "XGBoost"),
    ]
    
    print("\n📦 Checking required packages:")
    for module_name, display_name in imports:
        try:
            __import__(module_name)
            print(f"✅ {display_name:<20}")
        except ImportError:
            print(f"❌ {display_name:<20} (MISSING - install with pip)")
            return False
    return True

def main():
    """Run all validation checks."""
    os.chdir(Path(__file__).parent)
    
    print_section("STOCK PREDICTOR EXPERIMENT FRAMEWORK VALIDATION")
    
    all_good = True
    
    # Check core framework files
    print_section("1. Core Framework Files")
    files_to_check = [
        ("experiment_runner.py", "Core experiment orchestrator"),
        ("run_experiments.py", "JSON config CLI runner"),
        ("grid_search.py", "Hyperparameter search optimizer"),
    ]
    
    for filepath, desc in files_to_check:
        exists = check_file_exists(filepath, desc)
        if exists:
            if not check_python_syntax(filepath):
                all_good = False
        else:
            all_good = False
    
    # Check configuration files
    print_section("2. Configuration Files")
    config_files = [
        ("experiments_phase2b.json", "Phase 2b pre-built experiments"),
    ]
    
    for filepath, desc in config_files:
        exists = check_file_exists(filepath, desc)
        if exists:
            if not check_json_valid(filepath):
                all_good = False
        else:
            all_good = False
    
    # Check demo files
    print_section("3. Demo & Test Files")
    demo_files = [
        ("demo_experiments.py", "Interactive demonstration script"),
    ]
    
    for filepath, desc in demo_files:
        exists = check_file_exists(filepath, desc)
        if exists:
            if not check_python_syntax(filepath):
                all_good = False
        else:
            all_good = False
    
    # Check documentation
    print_section("4. Documentation Files")
    docs = [
        ("EXPERIMENT_FRAMEWORK_README.md", "User guide & API reference"),
        ("AUTOMATION_GUIDE.md", "Technical implementation guide"),
        ("QUICK_REFERENCE.md", "Command reference card"),
        ("FRAMEWORK_SUMMARY.md", "Project summary"),
    ]
    
    for filepath, desc in docs:
        check_file_exists(filepath, desc)
    
    # Check dependencies
    print_section("5. Package Dependencies")
    if not validate_imports():
        all_good = False
    
    # Final status
    print_section("VALIDATION SUMMARY")
    if all_good:
        print("""
✅ ALL CHECKS PASSED - Framework is ready to use!

Quick start commands:
  1. Run Phase 2b experiments:
     python run_experiments.py --config experiments_phase2b.json

  2. Run interactive demo:
     python demo_experiments.py

  3. Run grid search:
     python grid_search.py --ticker GLD --model xgb --search_type depth

See QUICK_REFERENCE.md for more commands.
        """)
        return 0
    else:
        print("""
❌ SOME CHECKS FAILED - Please fix issues above

Common fixes:
  1. Missing files? Re-download framework files
  2. Syntax errors? Check Python 3.11+ version
  3. Missing packages? Run: pip install xgboost scikit-learn yfinance
        """)
        return 1

if __name__ == "__main__":
    sys.exit(main())

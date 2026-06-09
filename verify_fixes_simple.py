#!/usr/bin/env python3
"""
Simple verification script for P0 fixes - checks code patterns without imports
"""

from pathlib import Path

def check_file_contains(filepath, patterns, description):
    """Check if file contains all patterns."""
    print(f"\nChecking {description}...")
    try:
        content = Path(filepath).read_text(encoding='utf-8')
        all_found = True
        for pattern in patterns:
            if pattern in content:
                print(f"  [PASS] Found: {pattern[:50]}...")
            else:
                print(f"  [FAIL] Missing: {pattern[:50]}...")
                all_found = False
        return all_found
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    print("=" * 70)
    print("P0 Fixes Verification - Code Pattern Check")
    print("=" * 70)

    tests = []
    
    # Test 1: Check MIN_TEMPERATURE in losses.py
    tests.append(check_file_contains(
        "fusion/losses.py",
        ["MIN_TEMPERATURE = 1e-3", "max(float(temperature), MIN_TEMPERATURE)"],
        "P0-2: MIN_TEMPERATURE constant (fusion/losses.py)"
    ))
    
    # Test 2: Check _safe_load method in dataset.py
    tests.append(check_file_contains(
        "fusion/dataset.py",
        ["def _safe_load(self, path: Path)", "weights_only=True", "mmap=True"],
        "P0-3: _safe_load method (fusion/dataset.py)"
    ))
    
    # Test 3: Check _safe_load usage in dataset.py
    tests.append(check_file_contains(
        "fusion/dataset.py",
        ["payload = self._safe_load(path)", "donor_payload = self._safe_load(donor_path)"],
        "P0-3: _safe_load usage (fusion/dataset.py)"
    ))
    
    # Test 4: Check load_config error handling in train.py
    tests.append(check_file_contains(
        "fusion/train.py",
        ["def load_config(path: str | Path)", 'if not path.exists():', 'example = path.parent'],
        "P0-1: Config error handling (fusion/train.py)"
    ))
    
    # Test 5: Check drop_last in _loader
    tests.append(check_file_contains(
        "fusion/train.py",
        ["drop_last = False", "if train and contrast_enabled:", "drop_last=True"],
        "P0-2: drop_last logic (fusion/train.py)"
    ))
    
    # Test 6: Check batch size validation in run()
    tests.append(check_file_contains(
        "fusion/train.py",
        ["contrast_weights =", "if any(w > 0 for w in contrast_weights) and batch_size < 2:"],
        "P0-2: Batch size validation (fusion/train.py)"
    ))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(tests)
    total = len(tests)
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[SUCCESS] All P0 fixes verified successfully!")
        print("\nModified files:")
        print("  - fusion/dataset.py (P0-3: Safe PT loading)")
        print("  - fusion/train.py (P0-1: Config errors, P0-2: Batch validation)")
        print("  - fusion/losses.py (P0-2: MIN_TEMPERATURE)")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} checks failed")
        print("Please review the code changes.")
        return 1

if __name__ == "__main__":
    exit(main())

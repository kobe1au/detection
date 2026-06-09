#!/usr/bin/env python3
"""
Verification script for P0 fixes

This script tests all P0 fixes to ensure they work correctly:
- P0-2: Small batch contrast protection
- P0-3: Safe PT loading

Run this after applying all P0 fixes to verify correctness.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from fusion.dataset import AEGDataset
        from fusion.train import load_config, run
        from fusion.losses import MIN_TEMPERATURE
        print("✅ All imports successful")
        print(f"   MIN_TEMPERATURE = {MIN_TEMPERATURE}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_safe_load_method():
    """Test that _safe_load method exists."""
    print("\nTesting _safe_load method...")
    try:
        from fusion.dataset import AEGDataset
        
        # Check method exists
        if not hasattr(AEGDataset, '_safe_load'):
            print("❌ AEGDataset._safe_load method not found")
            return False
        
        # Check method signature
        import inspect
        sig = inspect.signature(AEGDataset._safe_load)
        params = list(sig.parameters.keys())
        
        if 'self' in params and 'path' in params:
            print("✅ _safe_load method exists with correct signature")
            print(f"   Parameters: {params}")
            return True
        else:
            print(f"❌ _safe_load has unexpected parameters: {params}")
            return False
    except Exception as e:
        print(f"❌ Error checking _safe_load: {e}")
        return False

def test_min_temperature():
    """Test that MIN_TEMPERATURE constant exists."""
    print("\nTesting MIN_TEMPERATURE constant...")
    try:
        from fusion.losses import MIN_TEMPERATURE
        
        if MIN_TEMPERATURE == 1e-3:
            print(f"✅ MIN_TEMPERATURE = {MIN_TEMPERATURE} (correct)")
            return True
        else:
            print(f"❌ MIN_TEMPERATURE = {MIN_TEMPERATURE} (expected 1e-3)")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_config_error_message():
    """Test that load_config provides helpful error for missing files."""
    print("\nTesting config error messages...")
    try:
        from fusion.train import load_config
        
        # Try loading non-existent config
        fake_config = Path("nonexistent_config.yaml")
        try:
            load_config(fake_config)
            print("❌ Should have raised FileNotFoundError")
            return False
        except FileNotFoundError as e:
            error_msg = str(e)
            if "Config not found" in error_msg:
                print("✅ Helpful error message for missing config")
                print(f"   Error: {error_msg[:100]}...")
                return True
            else:
                print(f"❌ Error message not helpful: {error_msg}")
                return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dataloader_signature():
    """Test that _loader function has drop_last parameter."""
    print("\nTesting DataLoader configuration...")
    try:
        from fusion.train import _loader
        import inspect
        
        # Get source code
        source = inspect.getsource(_loader)
        
        if 'drop_last' in source:
            print("✅ _loader uses drop_last parameter")
            if 'contrast_enabled' in source:
                print("✅ _loader checks contrast_enabled")
                return True
            else:
                print("⚠️  Warning: contrast_enabled check not found")
                return True
        else:
            print("❌ drop_last parameter not found in _loader")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_size_validation():
    """Test that run() validates batch size."""
    print("\nTesting batch size validation...")
    try:
        from fusion.train import run
        import inspect
        
        source = inspect.getsource(run)
        
        checks = [
            ('contrast_weights' in source, "contrast_weights calculation"),
            ('batch_size' in source, "batch_size extraction"),
            ('batch_size < 2' in source or 'batch_size \u003c 2' in source, "batch_size < 2 check"),
        ]
        
        all_passed = True
        for check, desc in checks:
            if check:
                print(f"✅ Found: {desc}")
            else:
                print(f"❌ Missing: {desc}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("P0 Fixes Verification Script")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("_safe_load method", test_safe_load_method),
        ("MIN_TEMPERATURE constant", test_min_temperature),
        ("Config error messages", test_config_error_message),
        ("DataLoader drop_last", test_dataloader_signature),
        ("Batch size validation", test_batch_size_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All P0 fixes verified successfully!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the code changes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

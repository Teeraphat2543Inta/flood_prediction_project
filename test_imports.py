#!/usr/bin/env python3
"""
Simple test script to check if the code structure is correct
"""

def test_imports():
    """Test if all modules can be imported correctly"""
    try:
        # Test basic imports
        print("Testing basic imports...")
        import os
        import sys
        print("✓ Basic imports successful")
        
        # Test config import
        print("Testing config import...")
        from config import config
        print("✓ Config import successful")
        
        # Test utils import
        print("Testing utils import...")
        from src import utils
        print("✓ Utils import successful")
        
        # Test connection import
        print("Testing connection import...")
        from src.connection import connection2gee
        print("✓ Connection import successful")
        
        print("\n✓ All imports successful! Code structure is correct.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_config_values():
    """Test if config values are properly set"""
    try:
        from config import config
        
        print("\nTesting config values...")
        print(f"✓ MODEL_TYPES_TO_TRAIN: {config.MODEL_TYPES_TO_TRAIN}")
        print(f"✓ SEQUENCE_LENGTH: {config.SEQUENCE_LENGTH}")
        print(f"✓ FEATURES count: {len(config.FEATURES)}")
        print(f"✓ TARGET_CLASSIFICATION: {config.TARGET_CLASSIFICATION}")
        print(f"✓ MODEL_SAVE_PATH: {config.MODEL_SAVE_PATH}")
        
        return True
    except Exception as e:
        print(f"✗ Config test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING CODE STRUCTURE")
    print("=" * 50)
    
    import_success = test_imports()
    config_success = test_config_values()
    
    if import_success and config_success:
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("✓ Code is ready to run when dependencies are installed")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ SOME TESTS FAILED!")
        print("✗ Please check the errors above")
        print("=" * 50)

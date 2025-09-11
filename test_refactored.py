#!/usr/bin/env python3
"""Test script for refactored PhishGuard application."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.config import Config
        from src.utils import preprocess_text, load_model, predict_message
        from src.training import PhishGuardTrainer
        from src.app import create_app
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_preprocessing():
    """Test text preprocessing functionality."""
    try:
        from src.utils import preprocess_text
        
        test_cases = [
            ("Hello World!", "hello world"),
            ("Your account is LOCKED! Click here NOW!", "your account is locked click here now"),
            ("Meeting at 3:30 PM", "meeting at pm"),
            ("", ""),
            ("123-456-7890", "")
        ]
        
        for input_text, expected in test_cases:
            result = preprocess_text(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}'"
        
        print("✅ Text preprocessing works correctly")
        return True
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

def test_training():
    """Test model training functionality."""
    try:
        from src.training import PhishGuardTrainer
        
        trainer = PhishGuardTrainer()
        df = trainer.create_dataset()
        
        # Check dataset structure
        assert 'message' in df.columns
        assert 'label' in df.columns
        assert len(df) > 0
        assert df['label'].nunique() == 2  # Should have 2 unique labels
        
        print("✅ Dataset creation works correctly")
        return True
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_app_creation():
    """Test Flask app creation."""
    try:
        from src.app import create_app
        
        app = create_app()
        assert app is not None
        assert app.config['SECRET_KEY'] is not None
        
        print("✅ Flask app creation works correctly")
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing refactored PhishGuard application...\n")
    
    tests = [
        test_imports,
        test_preprocessing,
        test_training,
        test_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The refactored code is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Setup Validation Script for TW-MRT Model

This script validates that the reproducibility setup is working correctly
by running quick tests of all components.
"""

import os
import sys
import yaml
import torch
import numpy as np
import logging
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
        
        import yaml
        print(f"✓ PyYAML")
        
        from torch.utils.tensorboard import SummaryWriter
        print(f"✓ TensorBoard")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        if not os.path.exists('config.yaml'):
            print("✗ config.yaml not found")
            return False
            
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['experiment', 'model', 'training', 'data']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing required section: {section}")
                return False
        
        print("✓ Configuration file valid")
        return True
    except Exception as e:
        print(f"✗ Configuration loading error: {e}")
        return False

def test_cuda_setup():
    """Test CUDA setup if available."""
    print("\nTesting CUDA setup...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available (CPU-only mode)")
        return True
    
    try:
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test tensor operations
        x = torch.randn(10, 10).cuda()
        y = torch.matmul(x, x.t())
        print("✓ CUDA tensor operations working")
        
        return True
    except Exception as e:
        print(f"✗ CUDA test error: {e}")
        return False

def test_deterministic_setup():
    """Test deterministic operations."""
    print("\nTesting deterministic setup...")
    
    try:
        # Set seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Test reproducibility
        x1 = torch.randn(5, 5)
        torch.manual_seed(seed)
        x2 = torch.randn(5, 5)
        
        if torch.allclose(x1, x2):
            print("✓ Deterministic random number generation")
        else:
            print("✗ Non-deterministic random number generation")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Deterministic setup error: {e}")
        return False

def test_directory_structure():
    """Test that required directories can be created."""
    print("\nTesting directory structure...")
    
    try:
        # Test directory creation
        test_dirs = ['results', 'logs', 'data']
        for dir_name in test_dirs:
            os.makedirs(dir_name, exist_ok=True)
            if os.path.exists(dir_name):
                print(f"✓ Directory {dir_name} accessible")
            else:
                print(f"✗ Cannot create directory {dir_name}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Directory structure error: {e}")
        return False

def test_model_loading():
    """Test basic model loading."""
    print("\nTesting model components...")
    
    try:
        # Import our model components
        from testModel import DynamicGlucosePredictionModel
        
        # Create a small test model
        model = DynamicGlucosePredictionModel(
            input_features=3,
            d_model=32,
            nhead=2,
            num_layers=1,
            dim_feedforward=64,
            output_dim=12,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 3)
        t = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        with torch.no_grad():
            output = model(x, t)
        
        expected_shape = (batch_size, 12)
        if output.shape == expected_shape:
            print("✓ Model forward pass successful")
        else:
            print(f"✗ Model output shape mismatch: got {output.shape}, expected {expected_shape}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from utils import load_config, set_deterministic_mode
        
        # Test config loading
        config = load_config('config.yaml')
        print("✓ Utility config loading works")
        
        # Test deterministic mode setting
        set_deterministic_mode(config)
        print("✓ Deterministic mode setting works")
        
        return True
    except Exception as e:
        print(f"✗ Utilities error: {e}")
        return False

def run_dry_run_test():
    """Run a quick dry run test."""
    print("\nRunning dry run test...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'main_reproducible.py', '--dry-run'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Dry run completed successfully")
            return True
        else:
            print(f"✗ Dry run failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Dry run timed out")
        return False
    except Exception as e:
        print(f"✗ Dry run error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("="*60)
    print("TW-MRT SETUP VALIDATION")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("CUDA Setup", test_cuda_setup),
        ("Deterministic Setup", test_deterministic_setup),
        ("Directory Structure", test_directory_structure),
        ("Model Components", test_model_loading),
        ("Utilities", test_utilities),
        ("Dry Run Test", run_dry_run_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for reproducible experiments.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above and fix them.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
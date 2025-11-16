#!/usr/bin/env python3
"""
Test script for ensemble-based AFN implementation
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")
    
    try:
        from afn.afn_core import AFNCore, EnsembleRegressor
        print("✅ AFNCore import successful")
    except ImportError as e:
        print(f"❌ AFNCore import failed: {e}")
        return False
    
    try:
        from afn.comparison_algorithms import GA, PSO, ACO
        print("✅ Comparison algorithms import successful")
    except ImportError as e:
        print(f"❌ Comparison algorithms import failed: {e}")
        return False
    
    try:
        from afn.simple_test_functions import get_simple_test_function, get_simple_bounds
        print("✅ Simple test functions import successful")
    except ImportError as e:
        print(f"❌ Simple test functions import failed: {e}")
        return False
    
    try:
        from data.sample import load_bbob_function
        print("✅ BBOB data import successful")
    except ImportError as e:
        print(f"⚠️  BBOB data import failed (COCO not installed): {e}")
    
    return True

def test_ensemble_regressor():
    """Test EnsembleRegressor functionality"""
    print("\nTesting EnsembleRegressor...")
    
    try:
        from afn.afn_core import EnsembleRegressor
        
        # Create ensemble
        ensemble = EnsembleRegressor(input_dim=2, n_models=3, random_state=42)
        print("✅ EnsembleRegressor created successfully")
        
        # Generate test data
        X = np.random.randn(50, 2)
        y = np.sum(X**2, axis=1)  # Sphere function
        
        # Fit ensemble
        ensemble.fit(X, y)
        print("✅ Ensemble fitting successful")
        
        # Test prediction
        X_test = np.random.randn(10, 2)
        pred = ensemble.predict(X_test)
        print(f"✅ Prediction successful, shape: {pred.shape}")
        
        # Test uncertainty prediction
        mean_pred, std_pred = ensemble.predict_with_uncertainty(X_test)
        print(f"✅ Uncertainty prediction successful, shapes: {mean_pred.shape}, {std_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ EnsembleRegressor test failed: {e}")
        return False

def test_afn_core():
    """Test AFNCore functionality"""
    print("\nTesting AFNCore...")
    
    try:
        from afn.afn_core import AFNCore
        from afn.simple_test_functions import sphere
        
        # Create AFN instance
        bounds = [(-5, 5), (-5, 5)]
        afn = AFNCore(
            input_dim=2,
            bounds=bounds,
            max_evaluations=20,  # Small for testing
            n_models=3
        )
        print("✅ AFNCore created successfully")
        
        # Test optimization
        result = afn.optimize(sphere, verbose=False)
        print("✅ AFN optimization successful")
        print(f"   Best value: {result['best_y']:.6f}")
        print(f"   Evaluations: {result['evaluation_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AFNCore test failed: {e}")
        return False

def test_comparison_algorithms():
    """Test comparison algorithms"""
    print("\nTesting comparison algorithms...")
    
    try:
        from afn.comparison_algorithms import GA, PSO, ACO
        from afn.simple_test_functions import sphere
        
        bounds = [(-5, 5), (-5, 5)]
        
        # Test GA
        ga = GA(bounds=bounds, max_generations=10)
        ga_result = ga.optimize(sphere, verbose=False)
        print(f"✅ GA test successful, best: {ga_result['best_y']:.6f}")
        
        # Test PSO
        pso = PSO(bounds=bounds, max_iterations=10)
        pso_result = pso.optimize(sphere, verbose=False)
        print(f"✅ PSO test successful, best: {pso_result['best_y']:.6f}")
        
        # Test ACO
        aco = ACO(bounds=bounds, max_iterations=10)
        aco_result = aco.optimize(sphere, verbose=False)
        print(f"✅ ACO test successful, best: {aco_result['best_y']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison algorithms test failed: {e}")
        return False

def test_simple_functions():
    """Test simple test functions"""
    print("\nTesting simple test functions...")
    
    try:
        from afn.simple_test_functions import test_all_functions
        
        test_all_functions()
        print("✅ Simple test functions successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple test functions failed: {e}")
        return False

def test_bbob_functions():
    """Test BBOB functions (if available)"""
    print("\nTesting BBOB functions...")
    
    try:
        from data.sample import load_bbob_function
        
        # Test loading a BBOB function
        problem, info = load_bbob_function(func_id=1, dimension=2, instance=1)
        print(f"✅ BBOB function loaded: {info}")
        
        # Test function evaluation
        x = np.array([1.0, 2.0])
        y = problem(x)
        print(f"✅ BBOB function evaluation successful: f({x}) = {y:.6f}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  BBOB functions test failed (COCO not available): {e}")
        return True  # This is optional

def main():
    """Run all tests"""
    print("🧪 Testing ensemble-based AFN implementation...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_ensemble_regressor,
        test_afn_core,
        test_comparison_algorithms,
        test_simple_functions,
        test_bbob_functions
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! AFN is ready to use.")
        print("\nNext steps:")
        print("1. Run quick comparison: python run_afn_ga_pso_aco_comparison.py --functions 1 --dimensions 2 --n_runs 3")
        print("2. Run full comparison: python run_afn_ga_pso_aco_comparison.py")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

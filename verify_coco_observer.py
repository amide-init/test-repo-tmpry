#!/usr/bin/env python3
"""
Helper script to verify COCO Observer API usage
Run this to check if your Observer setup is correct
"""

import sys
import os

try:
    import cocoex
    from cocoex import Observer
    print("✓ COCO packages imported successfully")
except ImportError as e:
    print(f"✗ Failed to import COCO packages: {e}")
    print("  Install with: pip install coco-experiment cocopp")
    sys.exit(1)

def test_observer_api():
    """Test different Observer API variations"""
    print("\n" + "="*60)
    print("Testing COCO Observer API")
    print("="*60)
    
    # Create test directory
    test_dir = "test_observer_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # Get a test problem
    suite = cocoex.Suite("bbob", "", "")
    problem = suite.get_problem_by_function_dimension_instance(1, 2, 1)
    
    print(f"\nTest problem: {problem.name} (dim={problem.dimension})")
    print(f"Output directory: {test_dir}")
    
    # Test 1: Observer with 2 parameters
    print("\n[Test 1] Observer(suite_name, result_folder_string)")
    try:
        observer1 = Observer("bbob", f"result_folder: {test_dir}")
        print("  ✓ Constructor works with 2 parameters")
        
        # Try to set algorithm name
        if hasattr(observer1, 'set_algorithm_name'):
            observer1.set_algorithm_name("test_algorithm")
            print("  ✓ set_algorithm_name() method exists")
        elif hasattr(observer1, 'algorithm_name'):
            observer1.algorithm_name = "test_algorithm"
            print("  ✓ algorithm_name attribute exists")
        else:
            print("  ⚠ No way to set algorithm name found")
        
        # Try to attach
        problem_copy1 = suite.get_problem_by_function_dimension_instance(1, 2, 1)
        problem_copy1.observe_with(observer1)
        print("  ✓ observe_with() works")
        
        # Test evaluation
        result = problem_copy1([0.0, 0.0])
        print(f"  ✓ Problem evaluation works: f([0,0]) = {result}")
        
        # Finalize
        observer1.finalize()
        print("  ✓ finalize() works")
        
        print("  → Test 1 PASSED: 2-parameter constructor is correct")
        
    except Exception as e:
        print(f"  ✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Observer with 3 parameters
    print("\n[Test 2] Observer(suite_name, result_folder_string, algorithm_name)")
    try:
        observer2 = Observer("bbob", f"result_folder: {test_dir}", "test_algorithm")
        print("  ✓ Constructor works with 3 parameters")
        
        problem_copy2 = suite.get_problem_by_function_dimension_instance(1, 2, 1)
        problem_copy2.observe_with(observer2)
        print("  ✓ observe_with() works")
        
        result = problem_copy2([0.0, 0.0])
        print(f"  ✓ Problem evaluation works: f([0,0]) = {result}")
        
        observer2.finalize()
        print("  ✓ finalize() works")
        
        print("  → Test 2 PASSED: 3-parameter constructor is correct")
        
    except TypeError:
        print("  → Test 2: 3-parameter constructor not supported (this is OK)")
    except Exception as e:
        print(f"  ✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Check what files were created
    print("\n[Check] Generated files:")
    dat_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.dat'):
                full_path = os.path.join(root, file)
                dat_files.append(full_path)
                print(f"  ✓ {full_path}")
    
    if dat_files:
        print(f"\n✓ SUCCESS: {len(dat_files)} .dat file(s) generated")
        print("  Observer API is working correctly!")
    else:
        print("\n⚠ WARNING: No .dat files generated")
        print("  Observer might not be working correctly")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(test_dir)
        print(f"\n✓ Cleaned up test directory: {test_dir}")
    except:
        pass
    
    print("\n" + "="*60)
    print("Observer API Verification Complete")
    print("="*60)

if __name__ == '__main__':
    test_observer_api()


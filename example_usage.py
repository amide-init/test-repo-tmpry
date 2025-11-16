#!/usr/bin/env python3
"""
Example usage of ensemble-based AFN implementation
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

def example_basic_afn():
    """Example 1: Basic AFN optimization"""
    print("=" * 60)
    print("Example 1: Basic AFN Optimization")
    print("=" * 60)
    
    from afn.afn_core import AFNCore
    
    # Define objective function
    def sphere(x):
        return np.sum(x**2)
    
    # Set up AFN
    bounds = [(-5, 5), (-5, 5)]  # 2D problem
    afn = AFNCore(
        input_dim=2,
        bounds=bounds,
        max_evaluations=50,
        n_models=3
    )
    
    print("Optimizing Sphere function with AFN...")
    result = afn.optimize(sphere, verbose=True)
    
    print(f"\nResults:")
    print(f"  Best solution: {result['best_x']}")
    print(f"  Best value: {result['best_y']:.6f}")
    print(f"  Total evaluations: {result['evaluation_count']}")
    print(f"  Converged early: {result['converged']}")

def example_algorithm_comparison():
    """Example 2: Algorithm comparison"""
    print("\n" + "=" * 60)
    print("Example 2: Algorithm Comparison")
    print("=" * 60)
    
    from afn.afn_core import AFNCore
    from afn.comparison_algorithms import GA, PSO, ACO
    
    def rosenbrock(x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    bounds = [(-2.048, 2.048), (-2.048, 2.048)]
    max_evals = 50
    
    # Set up algorithms
    algorithms = {
        'AFN': AFNCore(input_dim=2, bounds=bounds, max_evaluations=max_evals),
        'GA': GA(bounds=bounds, max_generations=max_evals),
        'PSO': PSO(bounds=bounds, max_iterations=max_evals),
        'ACO': ACO(bounds=bounds, max_iterations=max_evals)
    }
    
    print("Comparing algorithms on Rosenbrock function...")
    results = {}
    
    for name, alg in algorithms.items():
        print(f"\nRunning {name}...")
        result = alg.optimize(rosenbrock, verbose=False)
        results[name] = result['best_y']
        print(f"  {name}: {result['best_y']:.6f}")
    
    # Find best algorithm
    best_alg = min(results, key=results.get)
    print(f"\n🏆 Best algorithm: {best_alg} with value {results[best_alg]:.6f}")

def example_simple_functions():
    """Example 3: Testing on simple functions"""
    print("\n" + "=" * 60)
    print("Example 3: Testing on Simple Functions")
    print("=" * 60)
    
    from afn.afn_core import AFNCore
    from afn.simple_test_functions import get_simple_test_function, get_simple_bounds, get_simple_function_name
    
    # Test on different functions
    functions_to_test = [1, 2, 3]  # Sphere, Rosenbrock, Rastrigin
    
    for func_id in functions_to_test:
        print(f"\nTesting {get_simple_function_name(func_id)}...")
        
        # Get function and bounds
        objective_func = get_simple_test_function(func_id, 2)
        bounds = get_simple_bounds(func_id, 2)
        
        # Set up AFN
        afn = AFNCore(
            input_dim=2,
            bounds=bounds,
            max_evaluations=30,
            n_models=3
        )
        
        # Optimize
        result = afn.optimize(objective_func, verbose=False)
        print(f"  Best value: {result['best_y']:.6f}")
        print(f"  Evaluations: {result['evaluation_count']}")

def example_bbob_functions():
    """Example 4: BBOB functions (if available)"""
    print("\n" + "=" * 60)
    print("Example 4: BBOB Functions (if available)")
    print("=" * 60)
    
    try:
        from data.sample import load_bbob_function
        from afn.afn_core import AFNCore
        
        # Load BBOB function
        problem, info = load_bbob_function(func_id=1, dimension=2, instance=1)
        print(f"Loaded: {info}")
        
        # Set up AFN with BBOB bounds
        bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) for i in range(2)]
        afn = AFNCore(
            input_dim=2,
            bounds=bounds,
            max_evaluations=50,
            n_models=3
        )
        
        print("Optimizing BBOB Sphere function with AFN...")
        result = afn.optimize(problem, verbose=True)
        
        print(f"\nResults:")
        print(f"  Best value: {result['best_y']:.6f}")
        print(f"  Total evaluations: {result['evaluation_count']}")
        
    except ImportError:
        print("⚠️  BBOB functions not available (COCO not installed)")
        print("   Install with: pip install coco-experiment cocopp")
    except Exception as e:
        print(f"❌ Error with BBOB functions: {e}")

def example_ensemble_analysis():
    """Example 5: Ensemble analysis"""
    print("\n" + "=" * 60)
    print("Example 5: Ensemble Analysis")
    print("=" * 60)
    
    from afn.afn_core import EnsembleRegressor
    import matplotlib.pyplot as plt
    
    # Create ensemble
    ensemble = EnsembleRegressor(input_dim=1, n_models=5, random_state=42)
    
    # Generate training data
    X_train = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_train = X_train.flatten()**2  # Simple quadratic
    
    # Fit ensemble
    ensemble.fit(X_train, y_train)
    
    # Generate test data
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    y_test = X_test.flatten()**2
    
    # Get predictions with uncertainty
    mean_pred, std_pred = ensemble.predict_with_uncertainty(X_test)
    
    print(f"Ensemble predictions:")
    print(f"  Mean prediction shape: {mean_pred.shape}")
    print(f"  Uncertainty shape: {std_pred.shape}")
    print(f"  Mean uncertainty: {np.mean(std_pred):.6f}")
    
    # Show some predictions
    print(f"\nSample predictions:")
    for i in [0, 50, 100, 150, 199]:
        print(f"  x={X_test[i,0]:.2f}, true={y_test[i]:.2f}, pred={mean_pred[i]:.2f}±{std_pred[i]:.2f}")

def main():
    """Run all examples"""
    print("🚀 AFN Examples")
    print("=" * 60)
    
    examples = [
        example_basic_afn,
        example_algorithm_comparison,
        example_simple_functions,
        example_bbob_functions,
        example_ensemble_analysis
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example {example.__name__} failed: {e}")
            print("   Continuing with next example...")
    
    print("\n" + "=" * 60)
    print("🎉 Examples completed!")
    print("\nNext steps:")
    print("1. Run the test suite: python test_afn.py")
    print("2. Run algorithm comparison: python run_afn_ga_pso_aco_comparison.py")
    print("3. Explore the code in afn/afn_core.py")

if __name__ == "__main__":
    main()
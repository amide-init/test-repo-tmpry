#!/usr/bin/env python3
"""
AFN vs CMA-ES Variants Comparison
Compares AFN algorithm with different CMA-ES variants with surrogate modeling:
- AFN: Standalone AFN algorithm
- CMA-ES: Standard CMA-ES
- AFN-CMA-ES: CMA-ES with AFN Random Forest ensemble
- LQ-CMA-ES: CMA-ES with Linear-Quadratic surrogate
- DTS-CMA-ES: CMA-ES with Dynamic Threshold Selection
- LMM-CMA-ES: CMA-ES with Local Meta-Model
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict

# Add paths
sys.path.append('.')
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

# Local imports
from afn.afn_core import AFNCore
from afn.cmaes_variants import (
    CMAEvolutionStrategy,
    AFN_CMA,
    LQ_CMA,
    DTS_CMA,
    LMM_CMA
)
from data.sample import load_bbob_function
from utils import MetricsCalculator, ComparisonPlotter, convert_to_json, parse_list_arg, setup_seeds, generate_seed


class OptimizationComparison:
    """Orchestrates comparison between AFN and CMA-ES variants"""
    
    ALL_ALGORITHMS = ["AFN", "CMA-ES", "AFN-CMA-ES", "LQ-CMA-ES", "DTS-CMA-ES", "LMM-CMA-ES"]
    
    def __init__(self,
                 test_functions: List[int] = [1, 2, 3],
                 dimensions: List[int] = [2, 5],
                 n_runs: int = 10,
                 max_evaluations: int = 200,
                 save_dir: str = "results",
                 algorithms: List[str] = None,
                 model_type: str = "random_forest"):
        """
        Initialize comparison framework
        
        Args:
            test_functions: List of BBOB function IDs to test
            dimensions: List of problem dimensions
            n_runs: Number of independent runs per test case
            max_evaluations: Maximum function evaluations per run
            save_dir: Directory to save results
            algorithms: List of algorithm names to run
            model_type: Surrogate model type ('random_forest' or 'mlp')
        """
        self.test_functions = test_functions
        self.dimensions = dimensions
        self.n_runs = n_runs
        self.max_evaluations = max_evaluations
        self.save_dir = save_dir
        self.model_type = model_type
        
        # Set algorithms to run
        if algorithms is None:
            self.algorithms = ["AFN", "CMA-ES"]  # Default
        else:
            self.algorithms = algorithms
        
        # Validate algorithms
        for alg in self.algorithms:
            if alg not in self.ALL_ALGORITHMS:
                raise ValueError(f"Unknown algorithm: {alg}. Available: {', '.join(self.ALL_ALGORITHMS)}")
        
        # Initialize utilities
        self.metrics_calculator = MetricsCalculator(max_evaluations)
        
        # Results storage
        self.results: Dict[str, List[Dict]] = {}
        self.metrics_summary: Dict[str, Dict] = {}
        
        # Setup output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(save_dir, f'cmaes_comparison_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize plotter with selected algorithms
        self.plotter = ComparisonPlotter(self.output_dir, algorithms=self.algorithms)
        
        print(f"Results will be saved to: {self.output_dir}")
    
    def instantiate_algorithm(self, algorithm_name: str, dimension: int, 
                             bounds: List, seed: int):
        """
        Instantiate an optimization algorithm
        
        Args:
            algorithm_name: Name of algorithm
            dimension: Problem dimension
            bounds: Search space bounds
            seed: Random seed
            
        Returns:
            Instantiated algorithm object
        """
        if algorithm_name == "AFN":
            # Original standalone AFN
            return AFNCore(
                input_dim=dimension,
                bounds=bounds,
                uncertainty_threshold=0.03,
                batch_size=8,
                max_evaluations=self.max_evaluations,
                convergence_threshold=1e-6,
                convergence_window=10,
                model_type=self.model_type,
                random_state=seed
            )
        
        # CMA-ES variants
        optimizer_classes = {
            "CMA-ES": CMAEvolutionStrategy,
            "AFN-CMA-ES": AFN_CMA,
            "LQ-CMA-ES": LQ_CMA,
            "DTS-CMA-ES": DTS_CMA,
            "LMM-CMA-ES": LMM_CMA
        }
        
        if algorithm_name not in optimizer_classes:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        OptimizerClass = optimizer_classes[algorithm_name]
        
        # AFN-CMA-ES supports model_type parameter
        if algorithm_name == "AFN-CMA-ES":
            return OptimizerClass(
                bounds=bounds,
                max_evaluations=self.max_evaluations,
                model_type=self.model_type,
                random_state=seed
            )
        else:
            return OptimizerClass(
                bounds=bounds,
                max_evaluations=self.max_evaluations,
                random_state=seed
            )
    
    def run_single_test(self, func_id: int, dimension: int, 
                       algorithm_name: str, run_idx: int, seed: int) -> Dict:
        """
        Run a single optimization test
        
        Args:
            func_id: BBOB function ID
            dimension: Problem dimension
            algorithm_name: Name of algorithm to test
            run_idx: Run index (for tracking)
            seed: Random seed
            
        Returns:
            Result dictionary
        """
        # Set random seeds
        setup_seeds(seed)
        
        # Load BBOB function
        problem, _ = load_bbob_function(func_id, dimension, instance=1)
        objective_function = problem
        bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) 
                 for i in range(dimension)]
        
        # Instantiate algorithm
        algorithm = self.instantiate_algorithm(algorithm_name, dimension, bounds, seed)
        
        # Run optimization
        start_time = time.time()
        result = algorithm.optimize(objective_function, verbose=False)
        end_time = time.time()
        
        # Add metadata
        result['execution_time'] = end_time - start_time
        result['algorithm'] = algorithm_name
        result['function_id'] = func_id
        result['dimension'] = dimension
        result['run'] = run_idx
        
        # Ensure evaluation_count is present
        if 'evaluation_count' not in result or not result['evaluation_count']:
            hist = result.get('history') or result.get('y_history') or []
            result['evaluation_count'] = len(hist) if hist else self.max_evaluations
        
        return result
    
    def run_comparison(self, verbose: bool = True) -> Dict:
        """
        Run full comparison across all test cases
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary of all results
        """
        import numpy as np
        
        total_tests = len(self.test_functions) * len(self.dimensions)
        test_counter = 0
        
        for func_id in self.test_functions:
            for dim in self.dimensions:
                test_counter += 1
                
                # Get function info
                problem, _ = load_bbob_function(func_id, dim, instance=1)
                func_name = problem.name if hasattr(problem, 'name') else f"BBOB Function {func_id}"
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"[{test_counter}/{total_tests}] Testing {func_name} (ID={func_id}, dimension={dim})")
                    print('='*80)
                
                for alg_idx, alg in enumerate(self.algorithms, 1):
                    key = f"func{func_id}_dim{dim}_{alg}"
                    self.results[key] = []
                    
                    if verbose:
                        print(f"\n  [{alg_idx}/{len(self.algorithms)}] {alg}:")
                    
                    for run_idx in range(self.n_runs):
                        seed = generate_seed(func_id, dim, run_idx, alg)
                        
                        try:
                            result = self.run_single_test(func_id, dim, alg, run_idx, seed)
                            self.results[key].append(result)
                            
                            if verbose:
                                print(f"    Run {run_idx+1:2d}/{self.n_runs}: "
                                      f"best={result['best_y']:12.6e}, "
                                      f"evals={result.get('evaluation_count', 0):4d}, "
                                      f"time={result.get('execution_time', 0):6.2f}s")
                        
                        except Exception as e:
                            print(f"    ✗ Run {run_idx+1}/{self.n_runs} failed: {e}")
                            # Add failed result
                            self.results[key].append({
                                'best_y': float('inf'),
                                'history': [],
                                'evaluation_count': 0,
                                'execution_time': 0,
                                'algorithm': alg,
                                'function_id': func_id,
                                'dimension': dim,
                                'run': run_idx
                            })
                    
                    # Show summary for this algorithm
                    if verbose:
                        valid_results = [r for r in self.results[key] if r.get('best_y', float('inf')) != float('inf')]
                        if valid_results:
                            best_vals = [r['best_y'] for r in valid_results]
                            mean_best = np.mean(best_vals)
                            std_best = np.std(best_vals)
                            min_best = np.min(best_vals)
                            print(f"    → Summary: mean={mean_best:.6e}, std={std_best:.6e}, min={min_best:.6e}")
        
        return self.results
    
    def compute_metrics(self) -> Dict:
        """
        Compute performance metrics for all results
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        for key, runs in self.results.items():
            # Parse key
            parts = key.split('_')
            func_id = int(parts[0].replace('func', ''))
            dimension = int(parts[1].replace('dim', ''))
            alg_name = '_'.join(parts[2:])
            
            # Compute metrics using calculator
            test_metrics = self.metrics_calculator.compute_all_metrics(runs, optimum=0.0)
            
            if test_metrics:
                test_key = f"func{func_id}_dim{dimension}"
                metrics.setdefault(alg_name, {})[test_key] = test_metrics
        
        self.metrics_summary = metrics
        return metrics
    
    def save_results(self) -> None:
        """Save all results and metrics to JSON files"""
        
        # Save detailed results
        with open(os.path.join(self.output_dir, 'runs.json'), 'w') as f:
            json.dump(convert_to_json(self.results), f, indent=2)
        
        # Save metrics summary
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(convert_to_json(self.metrics_summary), f, indent=2)
        
        # Save configuration
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump({
                'test_functions': self.test_functions,
                'dimensions': self.dimensions,
                'n_runs': self.n_runs,
                'max_evaluations': self.max_evaluations,
                'algorithms': self.algorithms,
                'model_type': self.model_type,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def run(self, verbose: bool = True):
        """
        Run complete comparison pipeline
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Tuple of (results, metrics)
        """
        if verbose:
            print("\n" + "="*80)
            print("STARTING AFN vs CMA-ES VARIANTS COMPARISON")
            print("="*80)
        
        # Run comparison
        self.run_comparison(verbose=verbose)
        
        # Compute metrics
        if verbose:
            print("\n" + "="*80)
            print("COMPUTING METRICS...")
            print("="*80)
        self.compute_metrics()
        if verbose:
            print("✓ Metrics computed successfully")
        
        # Create COCO-style CDF plots
        if verbose:
            print("\n" + "="*80)
            print("CREATING COCO-STYLE CDF PLOTS...")
            print("="*80)
        
        self.plotter.create_all_plots(results=self.results, 
                                      metrics_summary=self.metrics_summary,
                                      optimum=0.0)
        
        # Save results
        if verbose:
            print("\n" + "="*80)
            print("SAVING RESULTS...")
            print("="*80)
        self.save_results()
        
        return self.results, self.metrics_summary


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare AFN with CMA-ES variants on BBOB benchmark functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run AFN with Random Forest (default, fastest)
  python3 run_cmaes_comparison.py --algorithms AFN --functions 1,2,3 --dimensions 2,5
  
  # Run AFN with MLP (neural network surrogate)
  python3 run_cmaes_comparison.py --algorithms AFN --model_type mlp --functions 1,2,3 --dimensions 2,5
  
  # Run AFN and standard CMA-ES
  python3 run_cmaes_comparison.py --algorithms AFN,CMA-ES --functions 1,2,3 --dimensions 2,5
  
  # Compare Random Forest vs MLP for AFN
  python3 run_cmaes_comparison.py --algorithms AFN --model_type random_forest --functions 1,2,3 --dimensions 2,5
  python3 run_cmaes_comparison.py --algorithms AFN --model_type mlp --functions 1,2,3 --dimensions 2,5
  
  # Run only AFN-CMA-ES (AFN with CMA-ES)
  python3 run_cmaes_comparison.py --algorithms AFN-CMA-ES --functions 1,2,3 --dimensions 2,5 --max_evals 500
  
  # Run all available algorithms
  python3 run_cmaes_comparison.py --algorithms all --functions 1,2,3 --dimensions 2,5 --max_evals 200
  
  # Test on higher dimensions with more evaluations
  python3 run_cmaes_comparison.py --algorithms AFN,CMA-ES --functions 1-5 --dimensions 10 --n_runs 30 --max_evals 1000 --verbose
        """
    )
    
    parser.add_argument('--algorithms', type=str, default='AFN,CMA-ES',
                       help='Algorithms to run (comma-separated). Options: AFN, CMA-ES, AFN-CMA-ES, LQ-CMA-ES, DTS-CMA-ES, LMM-CMA-ES, or "all" (default: AFN,CMA-ES)')
    parser.add_argument('--functions', type=str, default='1,2,3',
                       help='Function IDs (comma-separated or range, e.g., "1,2,3" or "1-24") (default: 1,2,3)')
    parser.add_argument('--dimensions', type=str, default='2,5',
                       help='Dimensions (comma-separated, e.g., "2,5,10") (default: 2,5)')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs per test case (default: 10)')
    parser.add_argument('--max_evals', type=int, default=200,
                       help='Maximum function evaluations per run (default: 200)')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'mlp'],
                       help='Surrogate model type for AFN (random_forest or mlp) (default: random_forest)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress during optimization')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Parse lists
    test_functions = parse_list_arg(args.functions)
    dimensions = parse_list_arg(args.dimensions)
    
    # Parse algorithms
    if args.algorithms.lower() == 'all':
        algorithms = ["AFN", "CMA-ES", "AFN-CMA-ES", "LQ-CMA-ES", "DTS-CMA-ES", "LMM-CMA-ES"]
    else:
        algorithms = [alg.strip() for alg in args.algorithms.split(',')]
    
    # Print configuration
    print("\n" + "="*80)
    print("AFN vs CMA-ES VARIANTS COMPARISON")
    print("="*80)
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Surrogate Model: {args.model_type}")
    print(f"Test Functions: {test_functions}")
    print(f"Dimensions: {dimensions}")
    print(f"Runs per test: {args.n_runs}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"Output directory: {args.output_dir}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print("="*80)
    
    # Create comparison object
    comparison = OptimizationComparison(
        test_functions=test_functions,
        dimensions=dimensions,
        n_runs=args.n_runs,
        max_evaluations=args.max_evals,
        save_dir=args.output_dir,
        algorithms=algorithms,
        model_type=args.model_type
    )
    
    # Run comparison
    comparison.run(verbose=args.verbose)
    
    # Success message
    print("\n" + "="*80)
    print("✓ Comparison completed successfully!")
    print("="*80)
    print(f"\nResults saved to: {comparison.output_dir}")
    print("\nGenerated files:")
    print("  - runs.json                    (detailed results)")
    print("  - metrics_summary.json         (performance metrics)")
    print("  - config.json                  (experiment configuration)")
    print("  - convergence_curves.png       (convergence curves comparison)")
    print("  - cdf_1e-8.png                 (COCO CDF plot, target 1e-8)")
    print("  - cdf_1e-5.png                 (COCO CDF plot, target 1e-5)")
    print("  - cdf_1e-2.png                 (COCO CDF plot, target 1e-2)")
    print("  - cdf_multiple_targets.png     (COCO CDF plots, 4 targets)")
    print("  - performance_profile.png      (Performance profile plot)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import random
from datetime import datetime
from typing import List, Tuple, Dict, Callable
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

from afn.afn_core import AFNCore
from afn.cmaes_variants import AFN_CMA
from afn.comparison_algorithms import GA, PSO, ACO
from data.sample import load_bbob_function
from utils.metrics import MetricsCalculator
from utils.plotting import ComparisonPlotter
from utils.helpers import parse_list_arg, generate_seed

# Import Hansen and Bajer optimizers
try:
    import cma
    from skopt import gp_minimize
    from skopt.space import Real
    HANSEN_BAJER_AVAILABLE = True
except ImportError:
    HANSEN_BAJER_AVAILABLE = False
    print("Warning: cma or skopt not available. Hansen/Bajer optimizers will be skipped.")

def align_curve_to_budget(curve, budget):
    """Align curve to exact budget length"""
    if len(curve) >= budget:
        return curve[:budget]
    else:
        # Extend with last value
        last_val = curve[-1] if curve else 0.0
        return curve + [last_val] * (budget - len(curve))

def bajer_gp_minimize(func, lo_vec, hi_vec, dim, budget=100, seed=0):
    """
    Bajer-style GP surrogate baseline (GP + EI) ≈ DTS/S-CMA-ES spirit:
    we run a GP-EI loop (scikit-optimize) within the same evaluation budget.
    Reference: Bajer et al., 2019.  (GP uncertainty, EI-driven selection)
    """
    # scikit-optimize expects scalar bounds per dim
    lo = float(np.min(lo_vec))
    hi = float(np.max(hi_vec))
    space = [Real(lo, hi)] * dim

    # Run gp_minimize; it internally does initial points + EI acquisitions
    res = gp_minimize(
        func,
        space,
        n_calls=budget,
        n_initial_points=min(10, max(5, dim)),  # small warmup
        acq_func="EI",
        noise=0.0,
        random_state=seed,
    )

    # Build best-so-far curve from res.func_vals
    best = np.minimum.accumulate(np.array(res.func_vals, dtype=float))
    return align_curve_to_budget(best.tolist(), budget)

def hansen_cmaes(func, lo_vec, hi_vec, dim, budget=100, seed=0):
    """
    Hansen-style surrogate-assisted CMA-ES baseline (approx):
    use pycma's CMA-ES with bound constraints; pycma contains
    surrogate hooks (lq-CMA-ES family). Here we approximate with
    standard CMA-ES under identical budget—representative of the
    Hansen portfolio (CMA + surrogate gating).
    Reference: Hansen, 2019 (global linear/quad surrogate; rank-corr gating)
    """
    # Center start in the box; sigma ~ box size / 3
    lo = np.asarray(lo_vec, float)
    hi = np.asarray(hi_vec, float)
    x0 = (lo + hi) / 2.0
    sigma0 = float(np.mean((hi - lo) / 3.0))

    opts = {
        "bounds": [lo.tolist(), hi.tolist()],
        "popsize": 20,
        "maxfevals": budget,
        "seed": seed,
        "verb_disp": 0,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    best_curve = []
    fevals = 0
    while not es.stop():
        X = es.ask()
        y = [func(np.array(x, dtype=float)) for x in X]
        es.tell(X, y)
        fevals += len(y)
        # extend curve by the number of evaluations this gen consumed
        best = es.result.fbest
        best_curve.extend([best] * len(y))
        if fevals >= budget:
            break

    return align_curve_to_budget(best_curve, budget)

class AFNHansenBajerComparison:
    """Comparison between AFN, Hansen CMA-ES, and Bajer GP-EI"""
    
    def __init__(self, test_functions: List[int], dimensions: List[int], 
                 n_runs: int, max_evaluations: int, save_dir: str, 
                 target_precision: float = 1e-8, model_type: str = "random_forest"):
        self.test_functions = test_functions
        self.dimensions = dimensions
        self.n_runs = n_runs
        self.max_evaluations = max_evaluations
        self.target_precision = target_precision
        self.save_dir = save_dir
        self.model_type = model_type
        self.output_dir = os.path.join(save_dir, f"afn_hansen_bajer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Algorithms to compare
        self.algorithms = ["AFN-CMA-ES", "Hansen", "Bajer"]
        if not HANSEN_BAJER_AVAILABLE:
            self.algorithms = ["AFN-CMA-ES"]
            print("Warning: Only AFN-CMA-ES will be tested due to missing dependencies")
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(max_evaluations)
        
        # Initialize plotter with correct algorithm list
        self.plotter = ComparisonPlotter(self.output_dir, algorithms=self.algorithms)
        
        # Results storage
        self.results = {}
        self.metrics = {}
        
        # Save configuration
        config = {
            "test_functions": test_functions,
            "dimensions": dimensions,
            "n_runs": n_runs,
            "max_evaluations": max_evaluations,
            "target_precision": target_precision,
            "algorithms": self.algorithms,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def run_single_test(self, func_id: int, dimension: int, 
                       algorithm_name: str, run_idx: int, seed: int = None) -> Dict:
        """Run a single test case"""
        if seed is None:
            seed = run_idx * 1000 + func_id * 100 + dimension * 10
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Load function
        # Get COCO/BBOB function and bounds
        problem, info = load_bbob_function(func_id, dimension, instance=1)
        objective_function = problem
        bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) for i in range(dimension)]
        optimum = 0.0  # Most BBOB functions have optimum at 0.0
        
        start_time = time.time()
        
        try:
            if algorithm_name == "AFN":
                algorithm = AFNCore(
                    input_dim=dimension,
                    bounds=bounds,
                    uncertainty_threshold=0.03,
                    batch_size=8,
                    max_evaluations=self.max_evaluations,
                    convergence_threshold=1e-6,
                    convergence_window=10,
                    model_type=self.model_type
                )
                result = algorithm.optimize(objective_function, verbose=False)
                
                return {
                    'best_x': result['best_x'],
                    'best_y': result['best_y'],
                    'history': result['y_history'],
                    'evaluations': result['evaluation_count'],
                    'execution_time': time.time() - start_time,
                    'converged': result['converged'],
                    'optimum': optimum
                }
            
            elif algorithm_name == "AFN-CMA-ES":
                algorithm = AFN_CMA(
                    bounds=bounds,
                    max_evaluations=self.max_evaluations,
                    model_type=self.model_type,
                    random_state=seed
                )
                result = algorithm.optimize(objective_function, verbose=False)
                
                return {
                    'best_x': result['best_x'],
                    'best_y': result['best_y'],
                    'history': result['y_history'] if 'y_history' in result else result['history'],
                    'evaluations': result['evaluation_count'],
                    'execution_time': time.time() - start_time,
                    'converged': result.get('converged', False),
                    'optimum': optimum
                }
                
            elif algorithm_name == "Hansen" and HANSEN_BAJER_AVAILABLE:
                lo_vec = [b[0] for b in bounds]
                hi_vec = [b[1] for b in bounds]
                history = hansen_cmaes(objective_function, lo_vec, hi_vec, dimension, 
                                     self.max_evaluations, seed)
                
                best_y = history[-1] if history else float('inf')
                
                return {
                    'best_x': None,  # CMA-ES doesn't return best x easily
                    'best_y': best_y,
                    'history': history,
                    'evaluations': len(history),
                    'execution_time': time.time() - start_time,
                    'converged': False,  # CMA-ES convergence is complex
                    'optimum': optimum
                }
                
            elif algorithm_name == "Bajer" and HANSEN_BAJER_AVAILABLE:
                lo_vec = [b[0] for b in bounds]
                hi_vec = [b[1] for b in bounds]
                history = bajer_gp_minimize(objective_function, lo_vec, hi_vec, dimension,
                                          self.max_evaluations, seed)
                
                best_y = history[-1] if history else float('inf')
                
                return {
                    'best_x': None,  # GP minimize doesn't return best x easily
                    'best_y': best_y,
                    'history': history,
                    'evaluations': len(history),
                    'execution_time': time.time() - start_time,
                    'converged': False,  # GP minimize convergence is complex
                    'optimum': optimum
                }
                
        except Exception as e:
            print(f"Error in {algorithm_name} (f{func_id}, d{dimension}, run{run_idx}): {e}")
            return {
                'best_x': None,
                'best_y': float('inf'),
                'history': [],
                'evaluations': 0,
                'execution_time': time.time() - start_time,
                'converged': False,
                'optimum': optimum
            }

    def compute_metrics(self, results: List[Dict], optimum: float = 0.0) -> Dict:
        """
        Compute performance metrics from results using MetricsCalculator
        
        Args:
            results: List of result dictionaries
            optimum: Known optimum value (default: 0.0 for BBOB functions)
            
        Returns:
            Dictionary of computed metrics
        """
        if not results:
            return {
                'mean_best': float('inf'),
                'std_best': float('inf'),
                'mean_time': 0.0,
                'convergence_speed': 0.0,
                'success_rate': 0.0,
                'ert': float(self.max_evaluations),
                'resource_utilization': 0.0,
                'robustness': 0.0,
                'n_runs': 0
            }
        
        # Extract data
        best_values = [r['best_y'] for r in results if r['best_y'] != float('inf')]
        exec_times = [r['execution_time'] for r in results if r.get('execution_time', 0) > 0]
        
        # Extract histories
        histories = []
        for r in results:
            h = r.get('history', r.get('y_history', []))
            if isinstance(h, np.ndarray):
                h = h.tolist()
            elif not isinstance(h, list):
                h = list(h) if h is not None else []
            histories.append(h)
        
        if not best_values:
            return {
                'mean_best': float('inf'),
                'std_best': float('inf'),
                'mean_time': 0.0,
                'convergence_speed': 0.0,
                'success_rate': 0.0,
                'ert': float(self.max_evaluations),
                'resource_utilization': 0.0,
                'robustness': 0.0,
                'n_runs': len(results)
            }
        
        # Use MetricsCalculator for standard metrics
        all_metrics = self.metrics_calculator.compute_all_metrics(
            runs=results,
            optimum=optimum,
            target_precision=self.target_precision
        )
        
        # Basic statistics
        mean_best = np.mean(best_values)
        std_best = np.std(best_values)
        mean_time = np.mean(exec_times) if exec_times else 0.0
        
        return {
            'mean_best': float(mean_best),
            'std_best': float(std_best),
            'mean_time': float(mean_time),
            'convergence_speed': all_metrics['convergence_speed'],
            'success_rate': all_metrics['success_rate'],
            'ert': all_metrics['ert'],
            'resource_utilization': all_metrics['resource_utilization'],
            'robustness': all_metrics['robustness'],
            'n_runs': len(results)
        }
    

    def run_full_comparison(self, verbose: bool = True, save_plots: bool = True):
        """Run full comparison across all functions, dimensions, and algorithms"""
        total_tests = len(self.test_functions) * len(self.dimensions)
        test_counter = 0
        
        if verbose:
            print("\n" + "="*70)
            print("RUNNING COMPARISONS")
            print("="*70)
        
        for func_id in self.test_functions:
            for dimension in self.dimensions:
                test_counter += 1
                test_key = f"f{func_id}_d{dimension}"
                self.results[test_key] = {}
                self.metrics[test_key] = {}
                
                # Load function info for name
                try:
                    problem, info = load_bbob_function(func_id, dimension, instance=1)
                    func_name = info.get('name', f'f{func_id}')
                    optimum = 0.0  # BBOB functions have optimum at 0.0
                except:
                    func_name = f'f{func_id}'
                    optimum = 0.0
                
                if verbose:
                    print(f"\n[Test {test_counter}/{total_tests}] Function: {func_name} (f{func_id}) | Dimension: {dimension}D")
                
                for alg_idx, algorithm_name in enumerate(self.algorithms, 1):
                    if verbose:
                        print(f"\n  [{algorithm_name}]")
                    
                    algorithm_results = []
                    
                    for run_idx in range(self.n_runs):
                        result = self.run_single_test(func_id, dimension, algorithm_name, run_idx)
                        algorithm_results.append(result)
                        
                        if verbose:
                            status = "✓" if result['best_y'] < float('inf') else "✗"
                            print(f"  [{algorithm_name} - Run {run_idx + 1}/{self.n_runs}] "
                                  f"Evaluations: {result['evaluations']} | "
                                  f"Best: {result['best_y']:.6e} | "
                                  f"Time: {result['execution_time']:.2f}s | "
                                  f"Status: {status}")
                    
                    self.results[test_key][algorithm_name] = algorithm_results
                    
                    # Compute metrics with optimum value
                    metrics = self.compute_metrics(algorithm_results, optimum=optimum)
                    self.metrics[test_key][algorithm_name] = metrics
                    
                    if verbose:
                        print(f"  → {algorithm_name} Summary: "
                              f"Success: {metrics['success_rate']:.1f}% | "
                              f"ERT: {metrics['ert']:.0f} | "
                              f"Conv Speed: {metrics['convergence_speed']:.1f}%")
        
        # Save results
        self.save_results()
        
        # Create plots
        if save_plots:
            if verbose:
                print("\n" + "="*70)
                print("GENERATING PLOTS")
                print("="*70)
            self.create_comparison_plots()
        
        return self.results, self.metrics

    def save_results(self):
        """Save results to JSON files"""
        # Save detailed results
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save metrics summary
        with open(os.path.join(self.output_dir, "metrics_summary.json"), "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def create_comparison_plots(self):
        """Create COCO-style CDF comparison plots"""
        if not self.results:
            return
        
        print("✓ Generating CDF plots...")
        
        # Use ComparisonPlotter to create CDF plots
        # Note: Budgets must be expressed as multiples of the dimension (COCO standard)
        self.plotter.create_all_plots(
            results=self.results,
            metrics_summary=self.metrics,
            optimum=0.0  # BBOB functions have optimum at 0.0
        )
        
        print("✓ All plots generated successfully!")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='AFN-CMA-ES vs Hansen vs Bajer Comparison on BBOB Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark (f8 and f23 from paper) - Random Forest (default)
  python %(prog)s --functions 8,23 --dimensions 2,5,10,20 --n_runs 30 --max_evals 10000 --verbose
  
  # Full benchmark with MLP Deep Ensemble
  python %(prog)s --model_type mlp --functions 8,23 --dimensions 2,5,10,20 --n_runs 30 --max_evals 10000 --verbose
  
  # Quick test
  python %(prog)s --functions 1,8 --dimensions 2,5 --n_runs 5 --max_evals 2000 --verbose
  
  # Quick test with MLP
  python %(prog)s --model_type mlp --quick
  
  # Range support
  python %(prog)s --functions 1-24 --dimensions 2,5,10,20 --n_runs 30 --verbose
        """
    )
    parser.add_argument('--functions', type=str, default='8,23', 
                       help='BBOB function IDs (comma-separated or range like "1-24", default: 8,23)')
    parser.add_argument('--dimensions', type=str, default='2,5,10,20', 
                       help='Dimensions (comma-separated, default: 2,5,10,20)')
    parser.add_argument('--n_runs', type=int, default=30, 
                       help='Number of independent runs per test (default: 30)')
    parser.add_argument('--max_evals', type=int, default=10000, 
                       help='Maximum evaluations per run (default: 10000)')
    parser.add_argument('--target_precision', type=float, default=1e-8,
                       help='Target precision for success rate/ERT (default: 1e-8)')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'mlp'],
                       help='Surrogate model type for AFN-CMA-ES (random_forest or mlp) (default: random_forest)')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory for results (default: results)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable detailed progress output')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with minimal settings')
    
    return parser.parse_args()

def print_banner():
    """Print comparison banner"""
    print("=" * 80)
    print("AFN-CMA-ES vs Hansen vs Bajer Optimization Comparison")
    print("AFN-CMA-ES supports Random Forest (default) or MLP Deep Ensemble")
    print("=" * 80)

def main():
    """Main function"""
    args = parse_arguments()
    
    # Handle quick mode
    if args.quick:
        print("Quick mode enabled - using minimal settings")
        test_functions = [1]  # Only Sphere
        dimensions = [2]      # Only 2D
        args.n_runs = 2       # Only 2 runs
        args.max_evals = 50   # Only 50 evaluations
        args.verbose = True   # Always verbose in quick mode
    else:
        # Parse function and dimension lists (supports ranges like "1-24")
        test_functions = parse_list_arg(args.functions)
        dimensions = parse_list_arg(args.dimensions)
    
    # Print banner and configuration
    print_banner()
    print(f"\n╔{'═'*68}╗")
    print(f"║{'Configuration':^68}║")
    print(f"╠{'═'*68}╣")
    print(f"║  • Functions: {str(test_functions):<52}║")
    print(f"║  • Dimensions: {str(dimensions):<51}║")
    print(f"║  • Runs per test: {args.n_runs:<48}║")
    print(f"║  • Max evaluations: {args.max_evals:<46}║")
    print(f"║  • Target precision: {args.target_precision:<43}║")
    print(f"║  • Surrogate model: {args.model_type:<47}║")
    print(f"║  • Random seed: Deterministic (based on run index){' '*15}║")
    print(f"║  • Output: {args.output_dir:<54}║")
    print(f"╚{'═'*68}╝")
    
    # Calculate total runs
    algorithms = ["AFN-CMA-ES", "Hansen", "Bajer"] if HANSEN_BAJER_AVAILABLE else ["AFN-CMA-ES"]
    total_runs = len(test_functions) * len(dimensions) * len(algorithms) * args.n_runs
    print(f"\nAlgorithms: {', '.join(algorithms)}")
    print(f"Total evaluations: {total_runs} runs\n")
    
    # Validate inputs
    if len(test_functions) == 0:
        print("Error: No test functions specified")
        return 1
    
    if len(dimensions) == 0:
        print("Error: No dimensions specified")
        return 1
    
    if args.n_runs < 1:
        print("Error: Number of runs must be at least 1")
        return 1
    
    if args.max_evals < 1:
        print("Error: Max evaluations must be at least 1")
        return 1
    
    # Check if output directory exists or can be created
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create output directory '{args.output_dir}': {e}")
        return 1
    
    try:
        # Create comparison instance
        comparison = AFNHansenBajerComparison(
            test_functions=test_functions,
            dimensions=dimensions,
            n_runs=args.n_runs,
            max_evaluations=args.max_evals,
            save_dir=args.output_dir,
            target_precision=args.target_precision,
            model_type=args.model_type
        )
        
        # Run the comparison
        results, metrics = comparison.run_full_comparison(verbose=args.verbose, save_plots=True)
        
        # Print final summary
        print("\n" + "="*70)
        print("RESULTS SAVED")
        print("="*70)
        print(f"📁 Results directory: {comparison.output_dir}")
        print(f"📊 Metrics summary: metrics_summary.json")
        print(f"📈 Detailed results: results.json")
        print(f"🎨 Plots: CDF and performance profile visualizations")
        
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        for alg_name in algorithms:
            if any(alg_name in test_metrics for test_metrics in metrics.values()):
                # Calculate overall statistics
                all_success_rates = []
                all_erts = []
                all_conv_speeds = []
                all_robustness = []
                
                for test_data in metrics.values():
                    if alg_name in test_data:
                        all_success_rates.append(test_data[alg_name]['success_rate'])
                        all_erts.append(test_data[alg_name]['ert'])
                        all_conv_speeds.append(test_data[alg_name]['convergence_speed'])
                        all_robustness.append(test_data[alg_name]['robustness'])
                
                print(f"\n{alg_name}:")
                if all_success_rates:
                    print(f"  Success Rate:      {np.mean(all_success_rates):.1f}% ± {np.std(all_success_rates):.1f}%")
                if all_erts:
                    print(f"  ERT (avg evals):   {np.mean(all_erts):.0f} ± {np.std(all_erts):.0f}")
                if all_conv_speeds:
                    print(f"  Convergence Speed: {np.mean(all_conv_speeds):.1f}% ± {np.std(all_conv_speeds):.1f}%")
                if all_robustness:
                    print(f"  Robustness:        {np.mean(all_robustness):.1f}% ± {np.std(all_robustness):.1f}%")
        
        print("\n" + "="*70)
        return 0
        
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
